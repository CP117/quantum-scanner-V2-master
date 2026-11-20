"""Phase 4: yfinance 1.8.0 and bounded download threads."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _pins() -> dict[str, str]:
    pins = {}
    for line in (REPO / 'requirements.txt').read_text(encoding='utf-8').splitlines():
        line = line.split('#', 1)[0].strip()
        match = re.fullmatch(r'([A-Za-z0-9_.\-\[\]]+)==([^\s;]+)', line)
        if match:
            name = re.sub(r'\[.*\]', '', match.group(1)).lower().replace('-', '_')
            pins[name] = match.group(2)
    return pins


def test_requirements_pin_yfinance_1_8_0_and_curl_cffi_0_17_0():
    pins = _pins()
    assert pins.get('yfinance') == '1.8.0'
    assert pins.get('curl_cffi') == '0.17.0'


def test_the_suite_runs_against_the_pinned_versions():
    from importlib.metadata import version

    pins = _pins()
    assert version('yfinance') == pins['yfinance']
    assert version('curl_cffi') == pins['curl_cffi']


# --- Bounded yf.download --------------------------------------------------

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest
import yfinance

from app.services import daily_history_service, scoring_service

_LIMIT_ENV = ('MRD_YF_DOWNLOAD_THREADS', 'MRD_YF_MAX_CONCURRENT_DOWNLOADS')


class FakeDownload:
    """Stands in for yf.download: sleeps `delay` seconds, or with no delay
    blocks until `gate` is set, recording how many calls overlap."""

    def __init__(self, delay: float = 0.0, frame=None):
        self.delay = delay
        self.frame = frame if frame is not None else object()
        self.gate = threading.Event()
        self.lock = threading.Lock()
        self.running = 0
        self.max_running = 0
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        with self.lock:
            self.calls.append(dict(kwargs, _thread=threading.current_thread().name))
            self.running += 1
            self.max_running = max(self.max_running, self.running)
        try:
            if self.delay:
                time.sleep(self.delay)
            else:
                self.gate.wait(10)
            return self.frame
        finally:
            with self.lock:
                self.running -= 1


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _stats() -> dict:
    return scoring_service.yf_batch_executor_stats()


@pytest.fixture()
def limits(monkeypatch):
    """Configure the download limits from the given environment, fake
    yf.download, and record provider failures. Teardown releases any
    blocked call, waits for abandoned ones to finish and restores the
    module's limits."""
    fakes: list[FakeDownload] = []
    failures: list[str] = []
    monkeypatch.setattr(scoring_service, 'mark_provider_failure', failures.append)

    def configure(fake: FakeDownload | None = None, **env):
        for name in _LIMIT_ENV:
            monkeypatch.delenv(name, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, str(value))
        scoring_service.configure_yf_download_limits()
        fake = fake or FakeDownload()
        fakes.append(fake)
        monkeypatch.setattr(scoring_service.yf, 'download', fake)
        return fake

    configure.failures = failures
    yield configure
    for fake in fakes:
        fake.gate.set()
    assert _wait_for(lambda: _stats()['abandoned_running'] == 0)
    monkeypatch.undo()
    scoring_service.configure_yf_download_limits()


def _download(**kwargs):
    kwargs.setdefault('tickers', 'AAPL MSFT')
    return scoring_service._yf_download_with_timeout(**kwargs)


def _in_thread(fn):
    out = {}
    thread = threading.Thread(target=lambda: out.setdefault('value', fn()), daemon=True)
    thread.start()
    return thread, out


@pytest.mark.parametrize('max_concurrent', [1, 2])
def test_concurrent_downloads_never_exceed_the_limit(limits, max_concurrent):
    fake = limits(FakeDownload(delay=0.15), MRD_YF_MAX_CONCURRENT_DOWNLOADS=max_concurrent,
                  MRD_YF_DOWNLOAD_THREADS=3)
    before = _stats()
    callers = [_in_thread(lambda: _download(threads=True, progress=True)) for _ in range(6)]
    for thread, _ in callers:
        thread.join(10)
    assert [out.get('value') for _, out in callers] == [fake.frame] * 6
    assert fake.max_running == max_concurrent
    after = _stats()
    assert after['completions'] - before['completions'] == 6
    assert after['yf_download_busy'] == before['yf_download_busy']
    # threads and progress are the wrapper's, whatever the caller passed.
    assert {(c['threads'], c['progress']) for c in fake.calls} == {(3, False)}
    assert all(c['_thread'].startswith('yf-download') for c in fake.calls)
    assert limits.failures == []


def test_busy_returns_none_without_marking_a_provider_failure(limits, monkeypatch):
    fake = limits()
    monkeypatch.setattr(scoring_service, 'YF_SLOT_TIMEOUT_SECONDS', 0.2)
    holder, held = _in_thread(_download)
    assert _wait_for(lambda: fake.running == 1)
    before = _stats()
    started = time.monotonic()
    assert _download() is None
    waited = time.monotonic() - started
    assert 0.15 <= waited < 3.0
    assert _stats()['yf_download_busy'] - before['yf_download_busy'] == 1
    assert len(fake.calls) == 1
    assert limits.failures == []
    fake.gate.set()
    holder.join(5)
    assert held.get('value') is fake.frame


def test_abandoned_downloads_are_counted_and_block_new_ones(limits, monkeypatch):
    fake = limits()
    monkeypatch.setattr(scoring_service, '_YF_BATCH_TIMEOUT_SECONDS', 0.2)
    before = _stats()
    assert _download() is None
    assert _download() is None
    mid = _stats()
    assert mid['timeouts'] - before['timeouts'] == 2
    assert mid['yf_download_abandoned'] - before['yf_download_abandoned'] == 2
    assert mid['abandoned_running'] == 2
    assert limits.failures == ['yf.download_timeout_or_error'] * 2

    # Two abandoned calls still run: the next download is refused at once.
    started = time.monotonic()
    assert _download() is None
    assert time.monotonic() - started < 1.0
    assert len(fake.calls) == 2
    after = _stats()
    assert after['yf_download_refused'] - mid['yf_download_refused'] == 1
    assert after['submits'] == mid['submits']
    assert len(limits.failures) == 2

    fake.gate.set()
    assert _wait_for(lambda: _stats()['abandoned_running'] == 0)
    assert _download() is fake.frame


def test_refusals_log_a_warning_at_most_once_a_minute(limits, monkeypatch, caplog):
    limits()
    monkeypatch.setattr(scoring_service, '_YF_BATCH_TIMEOUT_SECONDS', 0.2)
    monkeypatch.setattr(scoring_service, '_YF_REFUSAL_LOGGED_AT', float('-inf'))
    assert _download() is None
    assert _download() is None
    caplog.clear()
    with caplog.at_level('WARNING', logger='app.scoring'):
        assert _download() is None
        assert _download() is None
    refusals = [r for r in caplog.records if 'yf.download refused' in r.getMessage()]
    assert len(refusals) == 1
    assert refusals[0].levelname == 'WARNING'
    assert '2 abandoned downloads' in refusals[0].getMessage()


def test_one_abandoned_download_does_not_block_the_next(limits, monkeypatch):
    fake = limits()
    monkeypatch.setattr(scoring_service, '_YF_BATCH_TIMEOUT_SECONDS', 0.2)
    assert _download() is None
    assert _stats()['abandoned_running'] == 1
    # The spare worker runs the next call while the abandoned one hangs.
    monkeypatch.setattr(scoring_service, '_YF_BATCH_TIMEOUT_SECONDS', 5.0)
    caller, out = _in_thread(_download)
    assert _wait_for(lambda: fake.running == 2)
    fake.gate.set()
    caller.join(5)
    assert out.get('value') is fake.frame


def test_a_download_that_never_started_is_cancelled_not_abandoned(limits, monkeypatch):
    fake = limits()
    monkeypatch.setattr(scoring_service, '_YF_BATCH_TIMEOUT_SECONDS', 0.2)
    monkeypatch.setattr(scoring_service, 'YF_ABANDONED_LIMIT', 3)
    assert _download() is None
    assert _download() is None
    before = _stats()
    # Both workers hang, so the third call waits in the queue and is
    # cancelled when its caller gives up.
    assert _download() is None
    after = _stats()
    assert after['timeouts'] - before['timeouts'] == 1
    assert after['yf_download_abandoned'] == before['yf_download_abandoned']
    assert after['abandoned_running'] == 2
    fake.gate.set()
    assert _wait_for(lambda: _stats()['abandoned_running'] == 0)
    assert len(fake.calls) == 2


def test_an_error_returns_none_and_marks_a_provider_failure(limits, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError('Yahoo said no')

    limits()
    monkeypatch.setattr(scoring_service.yf, 'download', boom)
    before = _stats()
    assert _download() is None
    assert _stats()['errors'] - before['errors'] == 1
    assert limits.failures == ['yf.download_timeout_or_error']


def test_limits_come_from_the_environment(limits):
    limits(MRD_YF_DOWNLOAD_THREADS=2, MRD_YF_MAX_CONCURRENT_DOWNLOADS=3)
    stats = _stats()
    assert (stats['threads_per_download'], stats['max_concurrent'], stats['workers']) == (2, 3, 4)
    assert scoring_service._YF_BATCH_EXECUTOR._max_workers == 4

    limits(MRD_YF_DOWNLOAD_THREADS='lots', MRD_YF_MAX_CONCURRENT_DOWNLOADS='0')
    stats = _stats()
    assert (stats['threads_per_download'], stats['max_concurrent'], stats['workers']) == (4, 1, 2)


def test_defaults_are_four_threads_one_download_two_workers(limits):
    limits()
    stats = _stats()
    assert (stats['threads_per_download'], stats['max_concurrent'], stats['workers']) == (4, 1, 2)
    assert stats['abandoned_limit'] == 2
    assert stats['slot_timeout_seconds'] == 5.0
    assert scoring_service._YF_BATCH_EXECUTOR._thread_name_prefix == 'yf-download'
    for key in ('yf_download_busy', 'yf_download_abandoned', 'yf_download_refused', 'abandoned_running'):
        assert key in stats


def test_status_reports_the_download_bounds(limits):
    from app.services.status_service import get_system_status

    limits()
    reported = get_system_status()['yf_batch_executor']
    assert reported['max_concurrent'] == 1
    assert reported['threads_per_download'] == 4
    assert 'yf_download_busy' in reported


@pytest.fixture()
def no_budget_limits(monkeypatch):
    monkeypatch.setattr(scoring_service, 'provider_budget_allowance', lambda *a, **k: True)
    monkeypatch.setattr(scoring_service, 'warm_provider_session', lambda *a, **k: {})


def test_batch_quotes_download_with_bounded_threads(limits, no_budget_limits):
    fake = limits(FakeDownload(delay=0.01, frame=pd.DataFrame()))
    assert scoring_service.download_quotes_yfinance_only(['AAPL', 'MSFT'], 'stocks') == {}
    (call,) = fake.calls
    assert call['tickers'] == 'AAPL MSFT'
    assert (call['threads'], call['progress']) == (4, False)
    assert limits.failures == []


def test_busy_batch_quotes_fall_through_without_a_provider_failure(limits, no_budget_limits, monkeypatch):
    fake = limits()
    monkeypatch.setattr(scoring_service, 'YF_SLOT_TIMEOUT_SECONDS', 0.1)
    assert scoring_service._YF_DOWNLOAD_SLOTS.acquire(timeout=1)
    try:
        assert scoring_service.download_quotes_yfinance_only(['AAPL'], 'stocks') == {}
    finally:
        scoring_service._YF_DOWNLOAD_SLOTS.release()
    assert fake.calls == []
    assert limits.failures == []


# --- Daily history keeps its own executor ---------------------------------

class FakeTicker:
    gate = threading.Event()
    calls: list[str] = []

    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, **kwargs):
        FakeTicker.calls.append(threading.current_thread().name)
        FakeTicker.gate.wait(10)
        return pd.DataFrame()


@pytest.fixture()
def fake_ticker(monkeypatch):
    FakeTicker.gate = threading.Event()
    FakeTicker.calls = []
    monkeypatch.setattr(yfinance, 'Ticker', FakeTicker)
    monkeypatch.setattr(daily_history_service, '_YF_HISTORY_TIMEOUT_SECONDS', 0.2)
    yield FakeTicker
    FakeTicker.gate.set()
    daily_history_service._fail_until.pop('ZZPHASEFOUR', None)


def _fetch_history():
    # A non-default period, so nothing is cached or flushed to data/.
    return daily_history_service._fetch_daily_history(
        'ZZPHASEFOUR', True, True, None, '1d', '5d', 'raw', 'regular',
    )


def test_daily_history_runs_on_its_own_executor():
    executor = daily_history_service._YF_HISTORY_EXECUTOR
    assert executor._max_workers == daily_history_service._MAX_INFLIGHT
    assert executor._thread_name_prefix == 'dh-yf-history'
    assert executor is not scoring_service._YF_BATCH_EXECUTOR


def test_daily_history_timeout_is_counted(fake_ticker):
    before = daily_history_service.stats_snapshot()['yf_history_timeouts']
    started = time.monotonic()
    assert _fetch_history() is None
    assert time.monotonic() - started < 3.0
    assert daily_history_service.stats_snapshot()['yf_history_timeouts'] - before == 1
    assert fake_ticker.calls and fake_ticker.calls[0].startswith('dh-yf-history')


def test_daily_history_cancels_a_call_still_queued_at_its_timeout(fake_ticker, monkeypatch):
    one_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='dh-yf-history-test')
    monkeypatch.setattr(daily_history_service, '_YF_HISTORY_EXECUTOR', one_worker)
    assert _fetch_history() is None   # starts, then hangs on the gate
    assert _fetch_history() is None   # queued behind it, then cancelled
    fake_ticker.gate.set()
    one_worker.shutdown(wait=True)
    assert len(fake_ticker.calls) == 1
