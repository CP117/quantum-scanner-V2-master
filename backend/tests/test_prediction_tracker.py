"""Prediction tracker contract (Phase 3): schema, hit rules, list, delete, accuracy.

frontend/predictions.js is the consumer, so several tests read it and
check that every field it uses is produced here.
"""
from __future__ import annotations

import re
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.services import prediction_tracker_service as tracker

REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_JS = (REPO_ROOT / 'frontend' / 'predictions.js').read_text(encoding='utf-8')

T0 = datetime(2026, 7, 9, 20, 55, tzinfo=timezone.utc)


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / 'predictions.db'
    monkeypatch.setenv('MRD_PREDICTIONS_DB', str(path))
    monkeypatch.delenv('MRD_PREDICTION_MAGNITUDE_TOLERANCE_PCT', raising=False)
    return path


@pytest.fixture
def clock(monkeypatch):
    """Each save gets created_at one minute after the previous one."""
    ticks = iter(T0 + timedelta(minutes=i) for i in range(10_000))
    monkeypatch.setattr(tracker, '_utcnow', lambda: next(ticks))


def _save(**overrides):
    payload = {
        'symbol': 'AAPL', 'market': 'stocks', 'anchor_price': 100.0, 'target_price': 105.0,
        'direction': 'bull', 'forward_days': 10, 'source': 'auto_scan',
    }
    payload.update(overrides)
    return tracker.save_prediction(payload)


def _set(path, pred_id, **fields):
    assignments = ', '.join(f'{name} = ?' for name in fields)
    with sqlite3.connect(path) as conn:
        conn.execute(f'UPDATE saved_predictions SET {assignments} WHERE id = ?', (*fields.values(), pred_id))


def _columns(path):
    with sqlite3.connect(path) as conn:
        return {row[1] for row in conn.execute('PRAGMA table_info(saved_predictions)')}


# ---------------------------------------------------------------------------
# Hit rules
# ---------------------------------------------------------------------------

def test_return_and_error_pct():
    assert tracker.actual_return_pct(100, 110) == pytest.approx(10.0)
    assert tracker.actual_return_pct(100, 90) == pytest.approx(-10.0)
    assert tracker.error_pct(105, 110) == pytest.approx(100 * 5 / 105)
    assert tracker.error_pct(105, 100) == pytest.approx(-100 * 5 / 105)
    for bad in (None, 0, -1, 'x', float('nan'), float('inf')):
        assert tracker.actual_return_pct(bad, 110) is None
        assert tracker.actual_return_pct(100, bad) is None
        assert tracker.error_pct(bad, 110) is None
        assert tracker.error_pct(105, bad) is None


@pytest.mark.parametrize('direction, close, expected', [
    ('bull', 101, True), ('bull', 99, False), ('bull', 100, False),
    ('bear', 99, True), ('bear', 101, False), ('bear', 100, False),
    ('BULL', 101, True), (' bear ', 99, True),
    ('neutral', 120, None), ('neutral', 80, None), ('', 120, None), (None, 120, None),
    ('bull', None, None), ('bear', 0, None),
])
def test_directional_hit(direction, close, expected):
    assert tracker.directional_hit(direction, 100, close) is expected


def test_magnitude_hit_and_tolerance(monkeypatch):
    monkeypatch.delenv('MRD_PREDICTION_MAGNITUDE_TOLERANCE_PCT', raising=False)
    assert tracker.magnitude_tolerance_pct() == 2.0
    assert tracker.magnitude_hit(2.0) is True
    assert tracker.magnitude_hit(-2.0) is True
    assert tracker.magnitude_hit(2.0001) is False
    assert tracker.magnitude_hit(-2.0001) is False
    assert tracker.magnitude_hit(4.9, 5) is True
    assert tracker.magnitude_hit(5.1, 5) is False
    for bad in (None, 'x', float('nan'), float('inf')):
        assert tracker.magnitude_hit(bad) is None

    monkeypatch.setenv('MRD_PREDICTION_MAGNITUDE_TOLERANCE_PCT', '0')
    assert tracker.magnitude_tolerance_pct() == 0.0
    assert tracker.magnitude_hit(0.0) is True
    assert tracker.magnitude_hit(0.01) is False
    monkeypatch.setenv('MRD_PREDICTION_MAGNITUDE_TOLERANCE_PCT', '0.5')
    assert tracker.magnitude_tolerance_pct() == 0.5
    assert tracker.magnitude_hit(0.5) is True
    assert tracker.magnitude_hit(0.6) is False
    for raw in ('abc', '-1', 'nan', 'inf', '   '):
        monkeypatch.setenv('MRD_PREDICTION_MAGNITUDE_TOLERANCE_PCT', raw)
        assert tracker.magnitude_tolerance_pct() == 2.0, raw


def test_expiry_session_date_uses_the_market_timezone():
    late = '2026-07-20T02:00:00+00:00'  # 22:00 on July 19 in New York
    assert tracker.expiry_session_date(late, 'stocks') == date(2026, 7, 19)
    assert tracker.expiry_session_date(late, 'crypto') == date(2026, 7, 20)
    assert tracker.expiry_session_date('2026-07-20T02:00:00Z', 'stocks') == date(2026, 7, 19)
    assert tracker.expiry_session_date('2026-07-20T02:00:00', 'stocks') == date(2026, 7, 19)  # naive = UTC
    assert tracker.expiry_session_date(datetime(2026, 7, 20, 2, tzinfo=timezone.utc), None) == date(2026, 7, 19)
    assert tracker.expiry_session_date('2026-07-20T15:00:00+00:00', 'stocks') == date(2026, 7, 20)
    for bad in (None, '', 'not a date', 12345):
        assert tracker.expiry_session_date(bad, 'stocks') is None


def test_score_prediction_bundles_the_rules():
    row = {'anchor_price': 100.0, 'target_price': 105.0, 'direction': 'bull'}
    out = tracker.score_prediction(row, 104.0)
    assert out['actual_close'] == 104.0
    assert out['actual_return_pct'] == pytest.approx(4.0)
    assert out['error_pct'] == pytest.approx(-100 / 105)
    assert out['directional_hit'] is True
    assert out['magnitude_hit'] is True
    miss = tracker.score_prediction({**row, 'direction': 'bear'}, 110.0, tolerance_pct=1.0)
    assert miss['directional_hit'] is False
    assert miss['magnitude_hit'] is False
    neutral = tracker.score_prediction({**row, 'direction': 'neutral', 'target_price': 100.0}, 101.0)
    assert neutral['directional_hit'] is None
    assert neutral['magnitude_hit'] is True


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def test_db_path_follows_the_environment_on_every_call(tmp_path, monkeypatch):
    first, second = tmp_path / 'a.db', tmp_path / 'b.db'
    monkeypatch.setenv('MRD_PREDICTIONS_DB', str(first))
    _save(symbol='AAA')
    monkeypatch.setenv('MRD_PREDICTIONS_DB', str(second))
    _save(symbol='BBB')
    for path, symbol in ((first, 'AAA'), (second, 'BBB')):
        with sqlite3.connect(path) as conn:
            assert conn.execute('SELECT symbol FROM saved_predictions').fetchall() == [(symbol,)]
    monkeypatch.delenv('MRD_PREDICTIONS_DB')
    assert tracker.db_path() == REPO_ROOT / 'data' / 'saved_predictions.db'


def test_tests_never_use_the_tracked_database():
    assert tracker.db_path() != REPO_ROOT / 'data' / 'saved_predictions.db'


def test_save_list_filter_delete(db, clock):
    apple = _save(symbol='aapl', full_payload={'big': 'x' * 1000})
    coin = _save(symbol='BTC-USD', market='crypto', source='user', direction='bear', target_price=95.0)
    msft = _save(symbol='MSFT', direction='neutral', target_price=100.0)

    rows = tracker.list_predictions()
    assert [r['id'] for r in rows] == [msft['id'], coin['id'], apple['id']]  # newest first
    assert set(rows[0]) == set(tracker.LIST_COLUMNS)
    assert 'full_payload' not in rows[0]
    assert rows[2]['symbol'] == 'AAPL'
    assert all(r['status'] == 'open' for r in rows)
    assert all(r['directional_hit'] is None and r['magnitude_hit'] is None for r in rows)

    assert [r['id'] for r in tracker.list_predictions('crypto')] == [coin['id']]
    assert [r['id'] for r in tracker.list_predictions(None, None, 'user')] == [coin['id']]
    assert [r['id'] for r in tracker.list_predictions('stocks', 'open', 'auto_scan')] == [msft['id'], apple['id']]
    assert [r['id'] for r in tracker.list_predictions(limit=1)] == [msft['id']]
    assert tracker.list_predictions(None, 'evaluated') == []

    _set(db, apple['id'], status='evaluated', directional_hit=1, magnitude_hit=0, actual_close=104.0)
    (evaluated,) = tracker.list_predictions(None, 'evaluated')
    assert evaluated['id'] == apple['id']
    assert evaluated['directional_hit'] is True
    assert evaluated['magnitude_hit'] is False
    assert evaluated['actual_close'] == 104.0

    assert tracker.delete_prediction(coin['id']) is True
    assert tracker.delete_prediction(coin['id']) is False
    assert tracker.delete_prediction('') is False
    assert tracker.delete_prediction(None) is False
    assert [r['id'] for r in tracker.list_predictions()] == [msft['id'], apple['id']]


def test_saved_row_matches_what_save_returns(db, clock):
    saved = _save(symbol='NVDA', confidence_pct='61.5', notes='  watch earnings  ')
    (row,) = tracker.list_predictions()
    for key in ('id', 'symbol', 'market', 'anchor_price', 'target_price', 'direction',
                'confidence_pct', 'forward_days', 'expires_at', 'created_at', 'status', 'source'):
        assert row[key] == saved[key], key
    assert row['notes'] == 'watch earnings'
    assert saved['expires_at'] == (T0 + timedelta(days=10)).isoformat()


def test_a_duplicate_save_is_an_error_not_a_fake_row(db, monkeypatch):
    monkeypatch.setattr(tracker, '_utcnow', lambda: T0)
    _save()
    with pytest.raises(ValueError, match='duplicate'):
        _save()
    with sqlite3.connect(db) as conn:
        assert conn.execute('SELECT COUNT(*) FROM saved_predictions').fetchone() == (1,)


_JULY_SCHEMA = '''
    CREATE TABLE saved_predictions (
        id TEXT PRIMARY KEY, symbol TEXT NOT NULL, market TEXT NOT NULL,
        anchor_price REAL NOT NULL, target_price REAL NOT NULL, direction TEXT NOT NULL,
        confidence_pct REAL, forward_days INTEGER NOT NULL DEFAULT 10, expires_at TEXT NOT NULL,
        notes TEXT, created_at TEXT NOT NULL, full_payload TEXT,
        status TEXT NOT NULL DEFAULT 'open', actual_close REAL, error_pct REAL,
        directional_hit INTEGER, magnitude_hit INTEGER, evaluated_at TEXT,
        source TEXT NOT NULL DEFAULT 'user', resolved_at TEXT)
'''
_FIRST_RELEASE_SCHEMA = '''
    CREATE TABLE saved_predictions (
        id TEXT PRIMARY KEY, symbol TEXT NOT NULL, market TEXT NOT NULL DEFAULT 'stocks',
        anchor_price REAL NOT NULL, target_price REAL NOT NULL, direction TEXT NOT NULL,
        confidence_pct REAL, forward_days INTEGER NOT NULL DEFAULT 10, expires_at TEXT NOT NULL,
        notes TEXT, full_payload TEXT, created_at TEXT NOT NULL, evaluated_at TEXT,
        status TEXT NOT NULL DEFAULT 'open', source TEXT NOT NULL DEFAULT 'user',
        UNIQUE(symbol, anchor_price, created_at))
'''


@pytest.mark.parametrize('schema', [_JULY_SCHEMA, _FIRST_RELEASE_SCHEMA], ids=['july-db', 'first-release'])
def test_migration_upgrades_old_databases_and_runs_twice(db, schema):
    legacy = [
        ('a', 'bull', 'correct'), ('b', 'bear', 'incorrect'), ('c', 'neutral', 'correct'),
        ('d', 'bull', 'expired'), ('e', 'bear', 'open'),
    ]
    with sqlite3.connect(db) as conn:
        conn.execute(schema)
        for pid, direction, status in legacy:
            conn.execute(
                'INSERT INTO saved_predictions (id, symbol, market, anchor_price, target_price, direction, '
                'expires_at, created_at, status) VALUES (?, ?, ?, 100, 101, ?, ?, ?, ?)',
                (pid, pid.upper(), 'stocks', direction, '2026-07-19T20:55:00+00:00',
                 '2026-07-09T20:55:00+00:00', status),
            )
        if 'directional_hit' in schema:
            conn.execute("UPDATE saved_predictions SET directional_hit = 1 WHERE id = 'c'")

    tracker.init_db()
    tracker.init_db()

    needed = {name for name, _ in tracker._ADDED_COLUMNS}
    assert needed <= _columns(db)
    with sqlite3.connect(db) as conn:
        got = dict((r[0], (r[1], r[2])) for r in conn.execute(
            'SELECT id, status, directional_hit FROM saved_predictions'))
        indexes = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
    assert got == {
        'a': ('evaluated', 1), 'b': ('evaluated', 0), 'c': ('evaluated', None),
        'd': ('open', None), 'e': ('open', None),
    }
    assert 'idx_status_expires' in indexes
    assert len(tracker.list_predictions()) == 5


def test_maintenance_prunes_the_tracker_database(db, clock, monkeypatch, tmp_path):
    from app.services import maintenance_service as maintenance
    # Leave the tracked data/regulatory.db alone.
    monkeypatch.setattr(maintenance, '_REG_DB_CANDIDATES', [tmp_path / 'no-regulatory.db'])
    old = _save(symbol='OLD')
    _save(symbol='OPEN')
    gone = _save(symbol='GONE')
    _set(db, old['id'], status='evaluated', evaluated_at='2020-01-01T00:00:00+00:00')
    _set(db, gone['id'], status='unresolved', unresolved_reason='no_history',
         resolved_at='2020-01-01T00:00:00+00:00')
    assert maintenance.maintenance_status()['prediction_db_path'] == str(db)
    summary = maintenance.prune_databases()
    assert summary['predictions']['db'] == str(db)
    assert summary['predictions']['resolved_pruned'] == 1
    assert {r['symbol'] for r in tracker.list_predictions()} == {'OPEN', 'GONE'}


def test_a_fresh_database_has_every_column(db):
    tracker.init_db()
    assert {name for name, _ in tracker._ADDED_COLUMNS} | set(tracker.LIST_COLUMNS) <= _columns(db)


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def _evaluated(db, pred, directional, magnitude, err):
    _set(db, pred['id'], status='evaluated', directional_hit=directional, magnitude_hit=magnitude,
         error_pct=err, actual_close=100.0)


def test_accuracy_stats_math_and_shape(db, clock):
    a = _save(symbol='A')
    b = _save(symbol='B')
    c = _save(symbol='C', direction='bear', target_price=95.0)
    d = _save(symbol='D', direction='neutral', target_price=100.0)
    e = _save(symbol='E', source='user')
    f = _save(symbol='F', direction='bear', target_price=95.0)
    _save(symbol='G')
    h = _save(symbol='H-USD', market='crypto')
    _evaluated(db, a, 1, 1, 1.0)
    _evaluated(db, b, 0, 0, -5.0)
    _evaluated(db, c, 1, 0, 3.0)
    _evaluated(db, d, None, 1, -0.5)
    _evaluated(db, e, 1, 1, 0.2)
    _set(db, f['id'], status='unresolved', unresolved_reason='no_history')
    _evaluated(db, h, 0, 0, 10.0)

    s = tracker.accuracy_stats('stocks')
    assert (s['total_saved'], s['open'], s['evaluated'], s['unresolved']) == (7, 1, 5, 1)
    assert s['directional_accuracy_pct'] == 75.0      # A, C, E of A, B, C, E
    assert s['magnitude_accuracy_pct'] == 60.0        # A, D, E of five
    assert s['mean_abs_error_pct'] == 1.94            # (1 + 5 + 3 + 0.5 + 0.2) / 5
    assert s['median_abs_error_pct'] == 1.0
    assert s['magnitude_tolerance_pct'] == 2.0
    assert s['filters'] == {'market': 'stocks', 'source': None}
    assert s['by_direction'] == {
        'bull': {'total': 3, 'evaluated': 3, 'total_saved': 4,
                 'directional_accuracy_pct': 66.7, 'magnitude_accuracy_pct': 66.7},
        'bear': {'total': 1, 'evaluated': 1, 'total_saved': 2,
                 'directional_accuracy_pct': 100.0, 'magnitude_accuracy_pct': 0.0},
        'neutral': {'total': 1, 'evaluated': 1, 'total_saved': 1,
                    'directional_accuracy_pct': None, 'magnitude_accuracy_pct': 100.0},
    }
    assert s['by_source'] == {
        'auto_scan': {'total_saved': 6, 'open': 1, 'evaluated': 4, 'unresolved': 1,
                      'directional_accuracy_pct': 66.7, 'magnitude_accuracy_pct': 50.0},
        'user': {'total_saved': 1, 'open': 0, 'evaluated': 1, 'unresolved': 0,
                 'directional_accuracy_pct': 100.0, 'magnitude_accuracy_pct': 100.0},
    }
    assert (s['total_predictions'], s['correct'], s['incorrect']) == (4, 3, 1)
    assert (s['accuracy_pct'], s['bull_accuracy_pct'], s['bear_accuracy_pct']) == (75.0, 66.7, 100.0)
    assert s['expired_only'] is True

    user_only = tracker.accuracy_stats('stocks', 'user')
    assert (user_only['total_saved'], user_only['evaluated']) == (1, 1)
    assert user_only['by_source'] == s['by_source']  # by_source ignores the source filter
    assert user_only['by_direction']['bear']['total'] == 0

    everything = tracker.accuracy_stats()
    assert everything['total_saved'] == 8
    assert everything['by_source']['auto_scan']['evaluated'] == 5
    assert everything['directional_accuracy_pct'] == 60.0  # H misses


def test_accuracy_stats_on_an_empty_database(db):
    s = tracker.accuracy_stats()
    assert (s['total_saved'], s['open'], s['evaluated'], s['unresolved']) == (0, 0, 0, 0)
    for key in ('directional_accuracy_pct', 'magnitude_accuracy_pct', 'mean_abs_error_pct', 'median_abs_error_pct'):
        assert s[key] is None, key
    assert all(v['total'] == 0 and v['directional_accuracy_pct'] is None for v in s['by_direction'].values())
    assert all(v['evaluated'] == 0 and v['directional_accuracy_pct'] is None for v in s['by_source'].values())
    assert (s['accuracy_pct'], s['bull_accuracy_pct'], s['bear_accuracy_pct']) == (0.0, 0.0, 0.0)
    assert (s['total_predictions'], s['correct'], s['incorrect']) == (0, 0, 0)


def test_tolerance_reaches_the_stats(db, monkeypatch):
    monkeypatch.setenv('MRD_PREDICTION_MAGNITUDE_TOLERANCE_PCT', '3.5')
    assert tracker.accuracy_stats()['magnitude_tolerance_pct'] == 3.5


# ---------------------------------------------------------------------------
# Routes and the predictions.js contract
# ---------------------------------------------------------------------------

@pytest.fixture
def client(db):
    from app.routes.prediction_tracker import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_routes_save_list_accuracy_delete(client):
    payload = {'status': 'ok', 'symbol': 'AAPL', 'current_price': 190.5, 'target_price': 195.0,
               'direction': 'Bullish', 'confidence': 41.2, 'forward_days': 10}
    body = {  # what app.js savePrediction() posts
        'symbol': payload['symbol'], 'market': 'stocks', 'anchor_price': payload['current_price'],
        'target_price': payload['target_price'], 'direction': 'bull',
        'confidence_pct': payload['confidence'], 'forward_days': payload['forward_days'],
        'notes': 'from the detail panel', 'full_payload': payload,
    }
    res = client.post('/api/predictions/save', json=body)
    assert res.status_code == 200, res.text
    saved = res.json()
    assert (saved['symbol'], saved['source'], saved['status']) == ('AAPL', 'user', 'open')

    listing = client.get('/api/predictions/list').json()
    assert set(listing) == {'rows', 'total', 'filters'}
    assert listing['total'] == 1
    assert listing['rows'][0]['id'] == saved['id']
    assert client.get('/api/predictions/list', params={'status': 'evaluated'}).json()['rows'] == []
    assert client.get('/api/predictions/list', params={'status': 'correct'}).status_code == 422

    acc = client.get('/api/predictions/accuracy').json()
    assert acc['total_saved'] == 1
    assert acc['by_source']['user']['total_saved'] == 1

    assert client.delete('/api/predictions/nope').status_code == 404
    assert client.delete(f"/api/predictions/{saved['id']}").json() == {'ok': True, 'id': saved['id']}
    assert client.get('/api/predictions/list').json()['total'] == 0


def _fields(pattern):
    return set(re.findall(pattern, PREDICTIONS_JS))


def test_every_row_field_the_page_reads_is_listed():
    used = _fields(r'\br\.([a-z_]+)')
    assert {'directional_hit', 'magnitude_hit', 'actual_close', 'status'} <= used
    assert used <= set(tracker.LIST_COLUMNS)


def test_every_accuracy_field_the_page_reads_is_produced(db):
    stats = tracker.accuracy_stats()
    top = _fields(r'\ba\.([a-z_]+)')
    assert {'total_saved', 'magnitude_tolerance_pct'} <= top
    assert top <= set(stats)
    per_source = _fields(r'\b(?:auto|user)\.([a-z_]+)')
    assert per_source and all(per_source <= set(v) for v in stats['by_source'].values())
    per_direction = _fields(r'\bentry\.([a-z_]+)') | _fields(r'\(bd\[g\.key\] \|\| \{\}\)\.([a-z_]+)')
    assert 'total' in per_direction
    assert all(per_direction <= set(v) for v in stats['by_direction'].values())


def test_page_labels_match_the_rules():
    assert 'inside 95% band' not in PREDICTIONS_JS
    assert "'|actual - target| / target'" in PREDICTIONS_JS
    assert 'a.magnitude_tolerance_pct' in PREDICTIONS_JS
