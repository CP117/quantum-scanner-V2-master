"""Scanner prediction logging (Phase 3): auto_log_scan_predictions and the warmer call.

Until Phase 3 the warmer called auto_log_scan_predictions with keyword
arguments the function did not accept. The TypeError was logged at debug
level, so no scanner prediction had been logged since July.
"""
from __future__ import annotations

import inspect
import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import settings
from app.services import detail_service, price_prediction_service, snapshot_store, warmer_service
from app.services import prediction_tracker_service as tracker

T0 = datetime(2026, 11, 19, 15, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / 'predictions.db'
    monkeypatch.setenv('MRD_PREDICTIONS_DB', str(path))
    monkeypatch.setattr(tracker, '_utcnow', lambda: T0)
    monkeypatch.setattr(settings, 'auto_log_predictions_enabled', True)
    monkeypatch.setattr(settings, 'auto_log_predictions_max_new_per_cycle', 10)
    monkeypatch.setattr(settings, 'auto_log_predictions_forward_days', 10)
    return path


def _complete_row(sym, **overrides):
    row = {
        'symbol': sym, 'state': 'ready', 'data_source': 'yfinance', 'age_seconds': 4,
        'factor_breakdown': {'market': {
            'last_price': 100.0, 'trend_volume_delta': 1.5, 'institutional_confluence': 0.4,
            'options_positioning': 0.3, 'volume_sentiment': 0.2,
        }},
    }
    row.update(overrides)
    return row


@pytest.fixture
def snapshot(monkeypatch):
    """Snapshot rows by (symbol, market); a missing key means not scored yet."""
    rows: dict[tuple[str, str], dict] = {}
    calls: list[tuple[str, str]] = []

    def lookup(symbol, market='stocks'):
        calls.append((symbol, market))
        row = rows.get((symbol, market))
        return dict(row) if row is not None else None

    monkeypatch.setattr(snapshot_store, 'lookup_snapshot_row', lookup)
    lookup.rows = rows
    lookup.calls = calls
    return lookup


def _prediction(sym, current=100.0, target=104.0, composite='Bullish', confidence=42.5, **extra):
    pred = {
        'status': 'ok', 'symbol': sym, 'current_price': current, 'target_price': target,
        'direction': 'Bullish', 'composite_direction': composite, 'composite_score': 61.2,
        'confidence': confidence, 'low_price': 95.0, 'high_price': 113.0, 'capped': False,
    }
    pred.update(extra)
    return pred


@pytest.fixture
def predictor(monkeypatch):
    """predict_price stub: predictions by symbol; an Exception value is raised."""
    results: dict[str, object] = {}
    calls: list[tuple[str, int, str]] = []

    def predict(symbol, forward_days=10, market='stocks', forward_hours=None):
        calls.append((symbol, forward_days, market))
        result = results.get(symbol, _prediction(symbol))
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(price_prediction_service, 'predict_price', predict)
    predict.results = results
    predict.calls = calls
    return predict


def _rows(path):
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute('SELECT * FROM saved_predictions ORDER BY symbol')]


# ---------------------------------------------------------------------------
# What gets logged
# ---------------------------------------------------------------------------

def test_logs_a_prediction_for_each_complete_symbol(db, snapshot, predictor):
    snapshot.rows[('AAPL', 'stocks')] = _complete_row('AAPL')
    snapshot.rows[('MSFT', 'stocks')] = _complete_row('MSFT')
    msft = _prediction('MSFT', current=410.5, target=401.25, composite='Bearish', confidence=37.0)
    predictor.results['MSFT'] = msft

    result = tracker.auto_log_scan_predictions(['aapl', {'symbol': 'MSFT'}], market='stocks')

    assert result == {'enabled': True, 'logged': 2, 'already_open': 0, 'incomplete': 0,
                      'unavailable': 0, 'failed': 0}
    assert predictor.calls == [('AAPL', 10, 'stocks'), ('MSFT', 10, 'stocks')]
    aapl_row, msft_row = _rows(db)
    assert (aapl_row['symbol'], aapl_row['anchor_price'], aapl_row['target_price'],
            aapl_row['direction'], aapl_row['confidence_pct']) == ('AAPL', 100.0, 104.0, 'bull', 42.5)
    assert (msft_row['symbol'], msft_row['anchor_price'], msft_row['target_price'],
            msft_row['direction'], msft_row['confidence_pct']) == ('MSFT', 410.5, 401.25, 'bear', 37.0)
    for row in (aapl_row, msft_row):
        assert (row['market'], row['source'], row['status'], row['forward_days']) == ('stocks', 'auto_scan', 'open', 10)
        assert row['created_at'] == T0.isoformat()
        assert row['expires_at'] == (T0 + timedelta(days=10)).isoformat()
    assert json.loads(msft_row['full_payload']) == msft


@pytest.mark.parametrize('composite, direction', [
    ('Bullish', 'bull'), ('Bearish', 'bear'), ('Neutral', 'neutral'), (None, 'neutral'), ('', 'neutral'),
])
def test_direction_comes_from_the_composite_direction(db, snapshot, predictor, composite, direction):
    """July's rows stored the composite direction, so new rows stay comparable with them."""
    snapshot.rows[('AA', 'stocks')] = _complete_row('AA')
    # The price model calls the move Neutral (the target equals the anchor), the blend leans one way.
    predictor.results['AA'] = _prediction('AA', current=50.0, target=50.0, composite=composite)

    assert tracker.auto_log_scan_predictions(['AA'])['logged'] == 1
    assert _rows(db)[0]['direction'] == direction


def test_crypto_predictions_are_logged_for_the_crypto_market(db, snapshot, predictor):
    snapshot.rows[('BTC', 'crypto')] = _complete_row('BTC')
    predictor.results['BTC'] = _prediction('BTC', current=65000.0, target=66300.0)

    result = tracker.auto_log_scan_predictions(['btc'], market='CRYPTO')

    assert result['logged'] == 1
    assert snapshot.calls == [('BTC', 'crypto')]
    assert predictor.calls == [('BTC', 10, 'crypto')]
    assert _rows(db)[0]['market'] == 'crypto'


# ---------------------------------------------------------------------------
# What gets skipped
# ---------------------------------------------------------------------------

def test_a_symbol_with_an_open_scanner_prediction_is_skipped(db, snapshot, predictor):
    snapshot.rows[('AAPL', 'stocks')] = _complete_row('AAPL')

    assert tracker.auto_log_scan_predictions(['AAPL'])['logged'] == 1
    again = tracker.auto_log_scan_predictions(['AAPL'])

    assert (again['logged'], again['already_open']) == (0, 1)
    assert len(_rows(db)) == 1
    assert predictor.calls == [('AAPL', 10, 'stocks')]


def test_an_expired_prediction_waiting_for_evaluation_does_not_hold_up_the_next_window(
        db, snapshot, predictor, monkeypatch):
    snapshot.rows[('AAPL', 'stocks')] = _complete_row('AAPL')
    assert tracker.auto_log_scan_predictions(['AAPL'])['logged'] == 1
    expiry = T0 + timedelta(days=10)

    monkeypatch.setattr(tracker, '_utcnow', lambda: expiry - timedelta(seconds=1))
    assert tracker.auto_log_scan_predictions(['AAPL'])['already_open'] == 1
    monkeypatch.setattr(tracker, '_utcnow', lambda: expiry)
    assert tracker.auto_log_scan_predictions(['AAPL'])['logged'] == 1

    first, second = sorted(_rows(db), key=lambda r: r['created_at'])
    assert (first['status'], second['status']) == ('open', 'open')
    assert second['created_at'] == first['expires_at']


def test_only_an_open_scanner_prediction_for_the_same_market_blocks_a_symbol(db, snapshot, predictor):
    for sym in ('USR', 'CRY', 'DONE', 'GONE'):
        snapshot.rows[(sym, 'stocks')] = _complete_row(sym)
    tracker.save_prediction({'symbol': 'USR', 'anchor_price': 10, 'target_price': 11, 'direction': 'bull',
                             'forward_days': 10, 'source': 'user'})
    tracker.save_prediction({'symbol': 'CRY', 'market': 'crypto', 'anchor_price': 10, 'target_price': 11,
                             'direction': 'bull', 'forward_days': 10, 'source': 'auto_scan'})
    for sym, status in (('DONE', 'evaluated'), ('GONE', 'unresolved')):
        saved = tracker.save_prediction({'symbol': sym, 'anchor_price': 10, 'target_price': 11,
                                         'direction': 'bull', 'forward_days': 10, 'source': 'auto_scan'})
        with sqlite3.connect(db) as conn:
            conn.execute('UPDATE saved_predictions SET status = ? WHERE id = ?', (status, saved['id']))

    result = tracker.auto_log_scan_predictions(['USR', 'CRY', 'DONE', 'GONE'])

    assert (result['logged'], result['already_open']) == (4, 0)


@pytest.mark.parametrize('row', [
    None,
    _complete_row('X', state='preview'),
    _complete_row('X', state='degraded'),
    _complete_row('X', stale=True),
    _complete_row('X', data_source='unavailable'),
    _complete_row('X', age_seconds=600),
    _complete_row('X', factor_breakdown={'market': {'last_price': 100.0, 'trend_volume_delta': 1.0}}),
], ids=['not_scored', 'preview', 'degraded', 'stale', 'no_source', 'old', 'cheap_pass'])
def test_a_symbol_without_a_complete_snapshot_row_is_skipped(db, snapshot, predictor, row):
    """predict_price would fetch and rescore these, so they are left for a later pass."""
    if row is not None:
        snapshot.rows[('X', 'stocks')] = row

    result = tracker.auto_log_scan_predictions(['X'])

    assert (result['logged'], result['incomplete']) == (0, 1)
    assert predictor.calls == []
    assert _rows(db) == []


@pytest.mark.parametrize('prediction', [
    {'status': 'unavailable', 'reason': 'symbol_not_scored'},
    _prediction('X', status='degraded'),
    _prediction('X', target=None),
    _prediction('X', target=0.0),
    _prediction('X', current=0.0),
    _prediction('X', current=float('nan')),
    None,
], ids=['unavailable', 'not_ok', 'no_target', 'zero_target', 'zero_price', 'nan_price', 'none'])
def test_a_prediction_without_usable_prices_is_skipped(db, snapshot, predictor, prediction):
    """save_prediction would invent a 10% target when the target is missing; nothing is saved instead."""
    snapshot.rows[('X', 'stocks')] = _complete_row('X')
    predictor.results['X'] = prediction

    result = tracker.auto_log_scan_predictions(['X'])

    assert (result['logged'], result['unavailable']) == (0, 1)
    assert _rows(db) == []


def test_a_failing_symbol_is_counted_and_the_batch_goes_on(db, snapshot, predictor, caplog):
    for sym in ('BAD', 'GOOD'):
        snapshot.rows[(sym, 'stocks')] = _complete_row(sym)
    predictor.results['BAD'] = RuntimeError('boom')

    with caplog.at_level(logging.WARNING, logger=tracker.log.name):
        result = tracker.auto_log_scan_predictions(['BAD', 'GOOD'])

    assert (result['logged'], result['failed']) == (1, 1)
    assert [r['symbol'] for r in _rows(db)] == ['GOOD']
    assert 'BAD' in caplog.text and 'boom' in caplog.text


def test_a_save_failure_is_counted(db, snapshot, predictor, monkeypatch):
    snapshot.rows[('AAPL', 'stocks')] = _complete_row('AAPL')

    def broken_save(payload):
        raise RuntimeError('db insert failed: disk I/O error')

    monkeypatch.setattr(tracker, 'save_prediction', broken_save)

    assert tracker.auto_log_scan_predictions(['AAPL'])['failed'] == 1


def test_duplicates_and_blank_symbols_are_ignored(db, snapshot, predictor):
    snapshot.rows[('AAPL', 'stocks')] = _complete_row('AAPL')

    result = tracker.auto_log_scan_predictions(['AAPL', 'aapl', '', None, {'symbol': ' aapl '}, {}])

    assert result == {'enabled': True, 'logged': 1, 'already_open': 0, 'incomplete': 0,
                      'unavailable': 0, 'failed': 0}
    assert snapshot.calls == [('AAPL', 'stocks')]
    assert predictor.calls == [('AAPL', 10, 'stocks')]


# ---------------------------------------------------------------------------
# Limits and settings
# ---------------------------------------------------------------------------

def test_at_most_max_new_predictions_are_logged_per_call(db, snapshot, predictor, monkeypatch):
    symbols = [f'S{i}' for i in range(5)]
    for sym in symbols:
        snapshot.rows[(sym, 'stocks')] = _complete_row(sym)
    monkeypatch.setattr(settings, 'auto_log_predictions_max_new_per_cycle', 2)

    assert tracker.auto_log_scan_predictions(symbols)['logged'] == 2
    assert tracker.auto_log_scan_predictions(symbols, max_new=1)['logged'] == 1
    assert [r['symbol'] for r in _rows(db)] == ['S0', 'S1', 'S2']
    assert tracker.auto_log_scan_predictions(symbols, max_new=0)['logged'] == 0


def test_forward_days_come_from_the_settings(db, snapshot, predictor, monkeypatch):
    snapshot.rows[('AAPL', 'stocks')] = _complete_row('AAPL')
    snapshot.rows[('MSFT', 'stocks')] = _complete_row('MSFT')
    monkeypatch.setattr(settings, 'auto_log_predictions_forward_days', 5)

    tracker.auto_log_scan_predictions(['AAPL'])
    tracker.auto_log_scan_predictions(['MSFT'], forward_days=3)

    assert predictor.calls == [('AAPL', 5, 'stocks'), ('MSFT', 3, 'stocks')]
    aapl_row, msft_row = _rows(db)
    assert (aapl_row['forward_days'], aapl_row['expires_at']) == (5, (T0 + timedelta(days=5)).isoformat())
    assert (msft_row['forward_days'], msft_row['expires_at']) == (3, (T0 + timedelta(days=3)).isoformat())


def test_a_zero_day_window_logs_nothing(db, snapshot, predictor):
    """save_prediction would turn forward_days 0 into 10."""
    snapshot.rows[('AAPL', 'stocks')] = _complete_row('AAPL')

    assert tracker.auto_log_scan_predictions(['AAPL'], forward_days=0)['logged'] == 0
    assert predictor.calls == [] and not db.exists()


def test_logging_can_be_turned_off(db, snapshot, predictor, monkeypatch):
    snapshot.rows[('AAPL', 'stocks')] = _complete_row('AAPL')
    monkeypatch.setattr(settings, 'auto_log_predictions_enabled', False)

    result = tracker.auto_log_scan_predictions(['AAPL'])

    assert result['enabled'] is False and result['logged'] == 0
    assert predictor.calls == [] and snapshot.calls == []
    assert not db.exists()


# ---------------------------------------------------------------------------
# The real predict_price on a complete row makes no provider request
# ---------------------------------------------------------------------------

def test_a_complete_row_is_predicted_from_the_snapshot_without_a_fetch(db, snapshot, monkeypatch):
    fetches = []
    for name in ('download_quotes', 'fetch_live_snapshot', 'fetch_intraday_shape', 'fetch_coingecko_snapshot'):
        monkeypatch.setattr(detail_service, name, lambda *a, _name=name, **k: fetches.append(_name) or {})
    snapshot.rows[('AAPL', 'stocks')] = _complete_row(
        'AAPL', last_price=231.5, final_score=64.0, final_direction='Bullish',
    )

    result = tracker.auto_log_scan_predictions(['AAPL'])

    assert result['logged'] == 1, result
    assert fetches == []
    row = _rows(db)[0]
    assert row['anchor_price'] == 231.5
    assert row['direction'] == 'bull'
    assert json.loads(row['full_payload'])['current_price'] == 231.5


# ---------------------------------------------------------------------------
# The warmer
# ---------------------------------------------------------------------------

@pytest.fixture
def warmer_state(monkeypatch):
    monkeypatch.setitem(warmer_service._status, 'last_auto_log', None)
    monkeypatch.setattr(warmer_service, '_last_auto_log_error', [None])
    return warmer_service._status


def test_the_warmer_call_matches_the_signature():
    """The July bug: the warmer passed market= and max_new=, which the function did not take."""
    inspect.signature(tracker.auto_log_scan_predictions).bind(['AAPL'], market='stocks')


def test_the_warmer_logs_with_the_settings_and_stores_the_counts(db, snapshot, predictor, warmer_state,
                                                                  monkeypatch):
    snapshot.rows[('AAPL', 'crypto')] = _complete_row('AAPL')
    snapshot.rows[('ETH', 'crypto')] = _complete_row('ETH')
    monkeypatch.setattr(settings, 'auto_log_predictions_forward_days', 7)
    monkeypatch.setattr(settings, 'auto_log_predictions_max_new_per_cycle', 1)

    warmer_service._auto_log([{'symbol': 'AAPL'}, {'name': 'no symbol'}, {'symbol': 'ETH'}], 3, 'crypto')

    assert warmer_state['last_auto_log'] == {'enabled': True, 'logged': 1, 'already_open': 0,
                                             'incomplete': 0, 'unavailable': 0, 'failed': 0}
    assert predictor.calls == [('AAPL', 7, 'crypto')]


def test_the_warmer_loop_logs_after_scoring_a_batch(monkeypatch):
    stop = __import__('threading').Event()
    calls = []
    monkeypatch.setattr(warmer_service, '_stop', stop)
    for key in ('market', 'warmed_symbols', 'last_cycle_utc', 'last_batch'):
        monkeypatch.setitem(warmer_service._status, key, warmer_service._status[key])
    monkeypatch.setitem(warmer_service._status, 'market', 'crypto')
    monkeypatch.setattr(warmer_service, 'get_total_batches', lambda size, market: 4)
    monkeypatch.setattr(warmer_service, 'get_batch_slice', lambda batch, size, market: [{'symbol': 'BTC'}])
    monkeypatch.setattr(settings, 'warmer_interval_seconds', 0)

    def score(rows):
        calls.append(('score', rows))
        if len(calls) > 2:  # the loop never logged; stop instead of spinning
            stop.set()

    monkeypatch.setattr(warmer_service, 'score_symbol_rows', score)

    def auto_log(rows, batch, market):
        calls.append(('log', rows, batch, market))
        stop.set()

    monkeypatch.setattr(warmer_service, '_auto_log', auto_log)

    warmer_service._loop()

    assert calls == [('score', [{'symbol': 'BTC'}]), ('log', [{'symbol': 'BTC'}], 0, 'crypto')]


def test_a_warmer_failure_is_recorded_and_warned_about_once(warmer_state, monkeypatch, caplog):
    def broken(symbols, market='stocks', forward_days=None, max_new=None):
        raise sqlite3.OperationalError('database is locked')

    monkeypatch.setattr(tracker, 'auto_log_scan_predictions', broken)

    with caplog.at_level(logging.DEBUG, logger=warmer_service.log.name):
        warmer_service._auto_log([{'symbol': 'AAPL'}], 3, 'stocks')
        warmer_service._auto_log([{'symbol': 'AAPL'}], 4, 'stocks')

    assert warmer_state['last_auto_log'] == {'error': 'OperationalError: database is locked',
                                             'batch': 4, 'market': 'stocks'}
    levels = [r.levelno for r in caplog.records if 'auto_log_scan_predictions failed' in r.getMessage()]
    assert levels == [logging.WARNING, logging.DEBUG]


def test_the_warning_returns_after_a_success(db, snapshot, predictor, warmer_state, monkeypatch, caplog):
    real = tracker.auto_log_scan_predictions
    outcomes = iter([RuntimeError('boom'), None, RuntimeError('boom')])

    def flaky(symbols, market='stocks', forward_days=None, max_new=None):
        outcome = next(outcomes)
        if outcome is not None:
            raise outcome
        return real(symbols, market=market)

    monkeypatch.setattr(tracker, 'auto_log_scan_predictions', flaky)

    with caplog.at_level(logging.DEBUG, logger=warmer_service.log.name):
        for batch in range(3):
            warmer_service._auto_log([{'symbol': 'AAPL'}], batch, 'stocks')

    levels = [r.levelno for r in caplog.records if 'auto_log_scan_predictions failed' in r.getMessage()]
    assert levels == [logging.WARNING, logging.WARNING]


def test_system_status_shows_the_last_auto_log(warmer_state, monkeypatch):
    from app.routes import status as status_route

    app = FastAPI()
    app.include_router(status_route.router)
    client = TestClient(app)
    counts = {'enabled': True, 'logged': 3, 'already_open': 12, 'incomplete': 10, 'unavailable': 0, 'failed': 0}
    monkeypatch.setitem(warmer_state, 'market', 'crypto')
    monkeypatch.setitem(warmer_state, 'last_auto_log', counts)

    warmer = client.get('/system/status').json()['warmer']

    assert (warmer['market'], warmer['last_auto_log']) == ('crypto', counts)
    monkeypatch.setitem(warmer_state, 'last_auto_log', {'error': 'TypeError: boom', 'batch': 2, 'market': 'crypto'})
    assert client.get('/system/status').json()['warmer']['last_auto_log']['error'] == 'TypeError: boom'
