"""Phase 4: thread counts in /system/status and the thread alarm.

Before Phase 4 the only thread count was /system/threads/summary, which is
mounted only with MRD_ENABLE_DEBUG_ROUTES=1, and nothing warned while the
count climbed (the yfinance download leak went from about 100 to 1,494
threads in 13 minutes).  /system/status now carries a `threads` block, the
wedge watchdog logs a WARNING over MRD_THREAD_ALARM and an ERROR with the
top prefixes over twice that, and scripts/soak.py falls back to the block
when the debug routes are off.

These tests run offline and never import app.main.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import threading
import types
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routes import status as status_route
from app.services import status_service
from app.services import thread_monitor as tm
from app.services import wedge_watchdog

REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_KEYS = {'total', 'daemon', 'os_threads', 'top_prefixes', 'alarm_threshold', 'alarm_level', 'rss_mb'}


def _load_soak():
    spec = importlib.util.spec_from_file_location('qs_soak_script', REPO_ROOT / 'scripts' / 'soak.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


soak = _load_soak()


@pytest.fixture
def fresh_alarm(monkeypatch):
    monkeypatch.setattr(tm, '_alarm_state', {'level': 'ok', 'logged_at': float('-inf')})
    monkeypatch.setattr(tm, '_bad_alarm_values', set())
    monkeypatch.delenv('MRD_THREAD_ALARM', raising=False)


@pytest.fixture
def parked_threads():
    """Three live daemon threads named qs-parked-1..3, released afterwards."""
    release = threading.Event()
    threads = [threading.Thread(target=release.wait, name=f'qs-parked-{i}', daemon=True) for i in (1, 2, 3)]
    for t in threads:
        t.start()
    yield threads
    release.set()
    for t in threads:
        t.join(timeout=5)


def _monitor_records(caplog):
    return [r for r in caplog.records if r.name == 'app.thread_monitor']


# ---------------------------------------------------------------------------
# prefixes, threshold, levels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name, prefix', [
    ('snap-worker-3', 'snap-worker'),
    ('ThreadPoolExecutor-0_12', 'ThreadPoolExecutor'),
    ('Thread-7 (_run_via_pool)', 'Thread (_run_via_pool)'),
    ('yf-download_0', 'yf-download'),
    ('dh-worker-11', 'dh-worker'),
    ('MainThread', 'MainThread'),
    ('', '<unnamed>'),
    ('123', '<unnamed>'),
])
def test_thread_prefix_matches_the_soak_script(name, prefix):
    assert tm.thread_prefix(name) == prefix
    assert soak.thread_prefix(name) == prefix


def test_threshold_default_and_env(fresh_alarm, monkeypatch):
    assert tm.DEFAULT_THREAD_ALARM == 400
    assert tm.thread_alarm_threshold() == 400
    monkeypatch.setenv('MRD_THREAD_ALARM', ' 250 ')
    assert tm.thread_alarm_threshold() == 250
    monkeypatch.setenv('MRD_THREAD_ALARM', '1')
    assert tm.thread_alarm_threshold() == 1
    monkeypatch.setenv('MRD_THREAD_ALARM', '')
    assert tm.thread_alarm_threshold() == 400


@pytest.mark.parametrize('raw', ['abc', '0', '-5', '2.5'])
def test_bad_threshold_uses_the_default_and_warns_once(fresh_alarm, monkeypatch, caplog, raw):
    caplog.set_level(logging.WARNING, logger='app.thread_monitor')
    monkeypatch.setenv('MRD_THREAD_ALARM', raw)
    assert tm.thread_alarm_threshold() == 400
    assert tm.thread_alarm_threshold() == 400
    records = _monitor_records(caplog)
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert repr(raw) in records[0].getMessage()


@pytest.mark.parametrize('total, level', [
    (0, 'ok'), (400, 'ok'), (401, 'warning'), (800, 'warning'), (801, 'error'), (5000, 'error'),
])
def test_alarm_level_boundaries(total, level):
    assert tm.alarm_level(total, 400) == level


# ---------------------------------------------------------------------------
# thread_summary and process_rss_mb
# ---------------------------------------------------------------------------

def test_summary_counts_live_threads_by_prefix(fresh_alarm, parked_threads):
    summary = tm.thread_summary(top=100)
    assert set(summary) == SUMMARY_KEYS
    assert summary['top_prefixes']['qs-parked'] == 3
    assert summary['total'] >= 4
    assert summary['daemon'] >= 3
    assert sum(summary['top_prefixes'].values()) == summary['total']
    assert summary['alarm_threshold'] == 400


def test_summary_keeps_the_top_ten_prefixes_by_default():
    release = threading.Event()
    threads = [threading.Thread(target=release.wait, name=f'qs-group-{chr(97 + i)}-1', daemon=True)
               for i in range(12)]
    for t in threads:
        t.start()
    try:
        assert len(tm.thread_summary()['top_prefixes']) == 10
        assert len(tm.thread_summary(top=1)['top_prefixes']) == 1
    finally:
        release.set()
        for t in threads:
            t.join(timeout=5)


def test_summary_level_follows_the_env(fresh_alarm, monkeypatch, parked_threads):
    monkeypatch.setenv('MRD_THREAD_ALARM', '1')
    summary = tm.thread_summary()
    assert summary['alarm_threshold'] == 1
    assert summary['alarm_level'] == 'error'
    monkeypatch.setenv('MRD_THREAD_ALARM', '100000')
    assert tm.thread_summary()['alarm_level'] == 'ok'


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='reads /proc/self/status')
def test_rss_and_os_threads_on_linux(parked_threads):
    summary = tm.thread_summary()
    assert isinstance(summary['rss_mb'], float) and summary['rss_mb'] > 1
    assert isinstance(summary['os_threads'], int) and summary['os_threads'] >= 4


def test_rss_uses_the_windows_api_on_windows(monkeypatch):
    monkeypatch.setattr(tm, 'sys', types.SimpleNamespace(platform='win32'))
    monkeypatch.setattr(tm, '_windows_rss_bytes', lambda: 512 * 1024 * 1024)
    assert tm.process_rss_mb() == 512.0


def test_rss_falls_back_to_psutil_then_none(monkeypatch):
    monkeypatch.setattr(tm, 'sys', types.SimpleNamespace(platform='darwin'))
    fake = types.ModuleType('psutil')
    fake.Error = Exception
    fake.Process = lambda pid: types.SimpleNamespace(
        memory_info=lambda: types.SimpleNamespace(rss=300 * 1024 * 1024))
    monkeypatch.setitem(sys.modules, 'psutil', fake)
    assert tm.process_rss_mb() == 300.0
    monkeypatch.setitem(sys.modules, 'psutil', None)  # not installed
    assert tm.process_rss_mb() is None
    # Windows without a working API also ends at psutil.
    monkeypatch.setattr(tm, 'sys', types.SimpleNamespace(platform='win32'))
    monkeypatch.setattr(tm, '_windows_rss_bytes', lambda: None)
    assert tm.process_rss_mb() is None


@pytest.mark.skipif(sys.platform == 'win32', reason='checks the non-Windows path')
def test_windows_reader_returns_none_elsewhere():
    assert tm._windows_rss_bytes() is None


# ---------------------------------------------------------------------------
# check_thread_alarm
# ---------------------------------------------------------------------------

def _summary(total, threshold=400):
    return {
        'total': total,
        'alarm_threshold': threshold,
        'alarm_level': tm.alarm_level(total, threshold),
        'top_prefixes': {'Thread (_run_via_pool)': total - 20, 'snap-worker': 12},
    }


def test_alarm_logs_on_change_and_every_ten_minutes_while_raised(fresh_alarm, caplog):
    caplog.set_level(logging.INFO, logger='app.thread_monitor')
    assert tm.ALARM_REPEAT_SECONDS == 600.0

    assert tm.check_thread_alarm(_summary(100), now=0.0) == 'ok'
    assert _monitor_records(caplog) == []  # quiet while ok

    assert tm.check_thread_alarm(_summary(450), now=10.0) == 'warning'
    records = _monitor_records(caplog)
    assert [r.levelno for r in records] == [logging.WARNING]
    message = records[0].getMessage()
    assert '450 Python threads' in message and 'MRD_THREAD_ALARM (400)' in message
    assert 'Thread (_run_via_pool): 430' in message

    tm.check_thread_alarm(_summary(455), now=609.0)  # 599 s later: no repeat yet
    assert len(_monitor_records(caplog)) == 1
    tm.check_thread_alarm(_summary(460), now=610.0)  # 600 s later: repeat
    records = _monitor_records(caplog)
    assert [r.levelno for r in records] == [logging.WARNING, logging.WARNING]
    assert '460 Python threads' in records[1].getMessage()

    assert tm.check_thread_alarm(_summary(900), now=700.0) == 'error'  # change: logs at once
    records = _monitor_records(caplog)
    assert records[-1].levelno == logging.ERROR
    message = records[-1].getMessage()
    assert '900 Python threads, over twice MRD_THREAD_ALARM (400)' in message
    assert 'Top prefixes: Thread (_run_via_pool): 880, snap-worker: 12' in message

    tm.check_thread_alarm(_summary(950), now=750.0)
    assert len(_monitor_records(caplog)) == 3

    assert tm.check_thread_alarm(_summary(500), now=760.0) == 'warning'  # down a level
    assert _monitor_records(caplog)[-1].levelno == logging.WARNING

    assert tm.check_thread_alarm(_summary(300), now=770.0) == 'ok'
    records = _monitor_records(caplog)
    assert records[-1].levelno == logging.INFO
    assert 'back under MRD_THREAD_ALARM (400)' in records[-1].getMessage()

    tm.check_thread_alarm(_summary(300), now=2000.0)  # ok never repeats
    assert len(_monitor_records(caplog)) == 5


def test_alarm_without_a_summary_reads_the_live_threads(fresh_alarm, monkeypatch, caplog, parked_threads):
    caplog.set_level(logging.INFO, logger='app.thread_monitor')
    monkeypatch.setenv('MRD_THREAD_ALARM', '1')
    assert tm.check_thread_alarm() == 'error'
    records = _monitor_records(caplog)
    assert [r.levelno for r in records] == [logging.ERROR]
    assert 'Top prefixes:' in records[0].getMessage()


# ---------------------------------------------------------------------------
# /system/status
# ---------------------------------------------------------------------------

def test_status_payload_has_the_threads_block(fresh_alarm, parked_threads):
    block = status_service.get_system_status()['threads']
    assert set(block) == SUMMARY_KEYS
    assert block['total'] >= 4
    assert block['alarm_level'] == 'ok'


def test_status_survives_a_summary_failure(monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger='app.status')

    def boom(*_args, **_kwargs):
        raise RuntimeError('no threads today')

    monkeypatch.setattr(status_service, 'thread_summary', boom)
    assert status_service.get_system_status()['threads'] == {}
    messages = [r.getMessage() for r in caplog.records if r.name == 'app.status' and r.levelno >= logging.WARNING]
    assert any('thread summary failed: RuntimeError: no threads today' in m for m in messages)


def _status_client():
    app = FastAPI()
    app.include_router(status_route.router)
    return TestClient(app)


def test_system_status_route_returns_threads(fresh_alarm, parked_threads):
    body = _status_client().get('/system/status').json()
    threads = body['threads']
    assert set(threads) == SUMMARY_KEYS
    assert threads['total'] >= 4
    assert isinstance(threads['top_prefixes'], dict)
    # Additive only: the fields the envelope had before Phase 4 are still there.
    for key in ('backend_ok', 'provider', 'cache', 'warmer', 'gc_stats', 'yf_batch_executor',
                'mount_errors', 'state'):
        assert key in body


def test_system_status_route_with_a_failed_summary(monkeypatch):
    monkeypatch.setattr(status_service, 'thread_summary', lambda *a, **k: 1 / 0)
    resp = _status_client().get('/system/status')
    assert resp.status_code == 200
    assert resp.json()['threads'] == {}


# ---------------------------------------------------------------------------
# wedge watchdog
# ---------------------------------------------------------------------------

@pytest.fixture
def watchdog_path(monkeypatch, tmp_path):
    """Point the dump at tmp_path and stub the two scanner lookups."""
    fake_store = types.ModuleType('app.services.snapshot_store')
    fake_store.get_snapshot_meta = lambda market: {'market': market}
    fake_t10 = types.ModuleType('app.services.top10_priority_service')
    fake_t10.get_status = lambda: {'monitor_only': False}
    monkeypatch.setitem(sys.modules, 'app.services.snapshot_store', fake_store)
    monkeypatch.setitem(sys.modules, 'app.services.top10_priority_service', fake_t10)
    path = tmp_path / 'wedge_watchdog.json'
    monkeypatch.setattr(wedge_watchdog, '_DUMP_PATH', path)
    return path


def test_watchdog_frame_has_the_summary_and_raises_the_alarm(
        fresh_alarm, watchdog_path, monkeypatch, caplog, parked_threads):
    caplog.set_level(logging.INFO, logger='app.thread_monitor')
    monkeypatch.setenv('MRD_THREAD_ALARM', '1')
    wedge_watchdog._dump_once()
    frame = json.loads(watchdog_path.read_text(encoding='utf-8'))
    summary = frame['thread_summary']
    assert set(summary) == SUMMARY_KEYS
    assert summary['alarm_threshold'] == 1
    assert summary['alarm_level'] == 'error'
    assert summary['total'] >= 4
    assert frame['snapshot_meta_stocks'] == {'market': 'stocks'}
    assert isinstance(frame['threads'], list) and frame['threads']

    errors = [r for r in _monitor_records(caplog) if r.levelno == logging.ERROR]
    assert len(errors) == 1
    message = errors[0].getMessage()
    for name, count in summary['top_prefixes'].items():
        assert f'{name}: {count}' in message

    wedge_watchdog._dump_once()  # 30 s cadence: no second ERROR within 10 minutes
    assert len([r for r in _monitor_records(caplog) if r.levelno == logging.ERROR]) == 1


def test_watchdog_quiet_below_the_threshold(fresh_alarm, watchdog_path, caplog):
    caplog.set_level(logging.INFO, logger='app.thread_monitor')
    wedge_watchdog._dump_once()
    frame = json.loads(watchdog_path.read_text(encoding='utf-8'))
    assert frame['thread_summary']['alarm_level'] == 'ok'
    assert _monitor_records(caplog) == []


def test_watchdog_still_writes_when_the_summary_fails(watchdog_path, monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger='app.wedge_watchdog')

    def boom():
        raise RuntimeError('enumerate broke')

    monkeypatch.setattr(wedge_watchdog, 'thread_summary', boom)
    wedge_watchdog._dump_once()
    frame = json.loads(watchdog_path.read_text(encoding='utf-8'))
    assert frame['thread_summary'] == {'error': 'thread_summary_failed: RuntimeError: enumerate broke'}
    assert any('thread alarm check failed: RuntimeError: enumerate broke' in r.getMessage()
               for r in caplog.records if r.name == 'app.wedge_watchdog' and r.levelno == logging.WARNING)


# ---------------------------------------------------------------------------
# scripts/soak.py
# ---------------------------------------------------------------------------

STATUS_WITH_THREADS = {
    'backend_ok': True,
    'threads': {
        'total': 123, 'daemon': 120, 'alarm_level': 'ok', 'rss_mb': 456.7,
        'top_prefixes': {'snap-worker': 10, 'dh-worker': 8},
    },
}


def _soak_args(**overrides):
    values = {'base_url': 'http://127.0.0.1:9', 'token': None, 'timeout': 1.0, 'pid': None}
    values.update(overrides)
    return argparse.Namespace(**values)


def _fake_fetch(responses, calls):
    def fetch(base_url, path, token, timeout):
        calls.append(path)
        return responses.get(path)
    return fetch


def test_soak_falls_back_to_the_status_block(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(soak, 'fetch_json', _fake_fetch({'/system/status': STATUS_WITH_THREADS}, calls))
    row = soak.take_sample(_soak_args(), started=0.0)
    assert row['threads'] == 123
    assert row['top_thread_prefixes'] == 'snap-worker:10;dh-worker:8'
    assert row['rss_mb'] == 456.7
    assert calls.count('/system/status') == 1  # fetched once per sample, reused for RSS
    assert '/system/threads/summary' in calls


def test_soak_prefers_the_debug_summary(monkeypatch):
    summary = {'thread_count': 5, 'threads': [
        {'name': 'snap-worker-1'}, {'name': 'snap-worker-2'}, {'name': 'dh-worker-3'},
        {'name': 'dh-worker-4'}, {'name': 'MainThread'},
    ]}
    calls: list[str] = []
    monkeypatch.setattr(soak, 'fetch_json', _fake_fetch(
        {'/system/status': STATUS_WITH_THREADS, '/system/threads/summary': summary}, calls))
    row = soak.take_sample(_soak_args(), started=0.0)
    assert row['threads'] == 5
    assert row['top_thread_prefixes'] == 'snap-worker:2;dh-worker:2;MainThread:1'
    assert row['rss_mb'] == 456.7


def test_soak_leaves_blank_cells_when_nothing_answers(monkeypatch):
    monkeypatch.setattr(soak, 'fetch_json', _fake_fetch({}, []))
    row = soak.take_sample(_soak_args(), started=0.0)
    assert (row['threads'], row['top_thread_prefixes'], row['rss_mb']) == ('', '', '')


def _samples(points):
    """points: (seconds from the first sample, threads, rss_mb)."""
    return [
        {'_t': 1_000_000.0 + sec, 'elapsed_s': float(sec), 'threads': threads, 'rss_mb': rss,
         '_providers': {}}
        for sec, threads, rss in points
    ]


def test_soak_summary_separates_warm_up_from_steady_state():
    ramp = [(sec, 100 + sec, 200.0 + sec) for sec in range(0, 300, 30)]  # pools filling
    steady = [(sec, 400, 500.0) for sec in range(300, 1830, 30)]
    text = soak.summarize(_samples(ramp + steady), warmup_seconds=300.0)
    assert 'Max threads: 400' in text
    overall = next(line for line in text.splitlines() if line.startswith('Thread growth:'))
    assert float(overall.split()[2]) > 100  # the ramp dominates the whole-run slope
    assert 'Thread growth after 5 min warm-up: +0.0 threads/hour (51 samples, 400 -> 400)' in text
    assert 'RSS growth after 5 min warm-up: +0.0 MB/hour (500.0 -> 500.0 MB)' in text


def test_soak_summary_shows_a_leak_after_warm_up():
    points = [(sec, 100 + sec // 30 * 25, 300.0 + sec / 10) for sec in range(0, 1830, 30)]
    text = soak.summarize(_samples(points), warmup_seconds=300.0)
    assert 'Thread growth after 5 min warm-up: +3000.0 threads/hour' in text
    assert 'RSS growth after 5 min warm-up: +360.0 MB/hour' in text


def test_soak_summary_without_warm_up():
    points = [(sec, 100, 300.0) for sec in range(0, 600, 30)]
    text = soak.summarize(_samples(points), warmup_seconds=0.0)
    assert 'Thread growth: +0.0 threads/hour' in text
    assert 'warm-up' not in text


@pytest.mark.parametrize('extra, expected', [([], 300.0), (['--warmup-minutes', '0'], 0.0),
                                             (['--warmup-minutes', '2.5'], 150.0),
                                             (['--warmup-minutes', '-3'], 0.0)])
def test_soak_main_passes_the_warm_up(monkeypatch, tmp_path, capsys, extra, expected):
    seen = {}
    monkeypatch.setattr(soak, 'take_sample', lambda args, started: {
        col: '' for col in soak.COLUMNS} | {'utc': 'now', '_t': started, '_providers': {}})

    def fake_summarize(samples, warmup_seconds=0.0):
        seen['warmup_seconds'] = warmup_seconds
        return 'summary'

    monkeypatch.setattr(soak, 'summarize', fake_summarize)
    argv = ['--minutes', '0', '--interval', '0', '--out', str(tmp_path / 'soak.csv')] + extra
    assert soak.main(argv) == 0
    assert seen['warmup_seconds'] == expected
    assert 'summary' in capsys.readouterr().out
