"""Prediction evaluation (Phase 3): close lookup, the evaluation pass and the background thread."""
from __future__ import annotations

import sqlite3
import threading
from datetime import date, datetime, timedelta, timezone
from datetime import time as clock_time
from zoneinfo import ZoneInfo

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.services import prediction_tracker_service as tracker
from app.services.prediction_tracker_service import CloseLookup

NY = ZoneInfo('America/New_York')
T0 = datetime(2026, 7, 9, 20, 55, tzinfo=timezone.utc)
# Saved 2026-07-09 16:55 New York; expires Sunday 2026-07-19, so the
# close that counts is Monday 2026-07-20.
ROW = {
    'id': 'x', 'symbol': 'AAPL', 'market': 'stocks', 'direction': 'bull',
    'anchor_price': 100.0, 'target_price': 105.0,
    'created_at': T0.isoformat(), 'expires_at': (T0 + timedelta(days=10)).isoformat(),
}
LATER = datetime(2026, 8, 1, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _fresh_retry_state():
    tracker._RETRY_AFTER.clear()
    yield
    tracker._RETRY_AFTER.clear()


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / 'predictions.db'
    monkeypatch.setenv('MRD_PREDICTIONS_DB', str(path))
    monkeypatch.delenv('MRD_PREDICTION_MAGNITUDE_TOLERANCE_PCT', raising=False)
    ticks = iter(T0 + timedelta(minutes=i) for i in range(10_000))
    monkeypatch.setattr(tracker, '_utcnow', lambda: next(ticks))
    return path


def _save(symbol, **overrides):
    payload = {
        'symbol': symbol, 'market': 'stocks', 'anchor_price': 100.0, 'target_price': 105.0,
        'direction': 'bull', 'forward_days': 10, 'source': 'auto_scan',
    }
    payload.update(overrides)
    return tracker.save_prediction(payload)


def _row(path, pred_id):
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        return dict(conn.execute('SELECT * FROM saved_predictions WHERE id = ?', (pred_id,)).fetchone())


def _history(*bars, splits=()):
    return {'bars': list(bars), 'splits': list(splits)}


def _ts(day, hour=13, minute=30):
    return datetime.combine(day, clock_time(hour, minute), tzinfo=timezone.utc).timestamp()


# ---------------------------------------------------------------------------
# Picking the close
# ---------------------------------------------------------------------------

def test_the_first_session_on_or_after_the_expiry_date_counts():
    history = _history((date(2026, 7, 16), 100.0), (date(2026, 7, 17), 101.0),
                       (date(2026, 7, 20), 102.0), (date(2026, 7, 21), 103.0))
    assert tracker.close_from_history(history, ROW, LATER) == CloseLookup(close=102.0, session_date=date(2026, 7, 20))
    on_the_day = {**ROW, 'expires_at': '2026-07-17T15:00:00+00:00'}  # Friday, mid-session
    assert tracker.close_from_history(history, on_the_day, LATER).close == 101.0
    unsorted = _history((date(2026, 7, 21), 103.0), (date(2026, 7, 20), 102.0))
    assert tracker.close_from_history(unsorted, ROW, LATER).close == 102.0


def test_bars_without_a_close_are_skipped():
    history = _history((date(2026, 7, 20), None), (date(2026, 7, 21), 103.0))
    assert tracker.close_from_history(history, ROW, LATER) == CloseLookup(close=103.0, session_date=date(2026, 7, 21))


def test_no_close_yet_waits_seven_days_then_gives_up():
    history = _history((date(2026, 7, 17), 101.0))
    expires = datetime.fromisoformat(ROW['expires_at'])
    assert tracker.close_from_history(history, ROW, expires + timedelta(days=1)) == CloseLookup(
        reason='no_close_yet', retry=True)
    assert tracker.close_from_history(history, ROW, expires + timedelta(days=6, hours=23)).retry is True
    assert tracker.close_from_history(history, ROW, expires + timedelta(days=7)) == CloseLookup(
        reason='no_close_after_expiry')


def test_a_close_long_after_the_expiry_does_not_count():
    seven_days = _history((date(2026, 7, 26), 99.0))
    assert tracker.close_from_history(seven_days, ROW, LATER).close == 99.0
    eight_days = _history((date(2026, 7, 27), 99.0))
    assert tracker.close_from_history(eight_days, ROW, LATER) == CloseLookup(reason='no_close_near_expiry')


def test_a_session_still_in_progress_means_retry():
    history = _history((date(2026, 7, 20), 102.0))
    before = datetime(2026, 7, 20, 16, 59, tzinfo=NY)
    assert tracker.close_from_history(history, ROW, before) == CloseLookup(reason='session_not_closed', retry=True)
    after = datetime(2026, 7, 20, 17, 0, tzinfo=NY)
    assert tracker.close_from_history(history, ROW, after).close == 102.0


def test_splits_after_the_save_put_the_close_on_the_anchor_basis():
    bar = (date(2026, 7, 20), 51.0)
    two_for_one = (_ts(date(2026, 7, 14)), 2.0)
    assert tracker.close_from_history(_history(bar, splits=[two_for_one]), ROW, LATER).close == 102.0
    before_save = (_ts(date(2026, 7, 1)), 2.0)
    assert tracker.close_from_history(_history(bar, splits=[before_save]), ROW, LATER).close == 51.0
    after_expiry = (_ts(date(2026, 7, 28)), 3.0)  # Yahoo adjusts the old bar for this one too
    assert tracker.close_from_history(_history(bar, splits=[two_for_one, after_expiry]), ROW, LATER).close == 306.0
    reverse = (_ts(date(2026, 7, 14)), 0.1)
    assert tracker.close_from_history(_history(bar, splits=[reverse]), ROW, LATER).close == pytest.approx(5.1)


def test_a_bad_expiry_is_unresolved():
    history = _history((date(2026, 7, 20), 102.0))
    assert tracker.close_from_history(history, {**ROW, 'expires_at': 'soon'}, LATER) == CloseLookup(reason='bad_expiry')


def test_crypto_days_are_utc():
    coin = {**ROW, 'symbol': 'BTC-USD', 'market': 'crypto', 'expires_at': '2026-07-20T02:00:00+00:00'}
    history = _history((date(2026, 7, 19), 50.0), (date(2026, 7, 20), 60.0))
    assert tracker.close_from_history(history, coin, LATER) == CloseLookup(close=60.0, session_date=date(2026, 7, 20))
    early = datetime(2026, 7, 21, 0, 59, tzinfo=timezone.utc)
    assert tracker.close_from_history(history, coin, early).reason == 'session_not_closed'
    assert tracker.close_from_history(history, coin, early + timedelta(minutes=1)).close == 60.0


def test_session_is_final():
    summer, winter = date(2026, 7, 20), date(2026, 11, 19)
    assert not tracker.session_is_final(summer, 'stocks', datetime(2026, 7, 20, 20, 59, tzinfo=timezone.utc))
    assert tracker.session_is_final(summer, 'stocks', datetime(2026, 7, 20, 21, 0, tzinfo=timezone.utc))
    assert not tracker.session_is_final(winter, None, datetime(2026, 11, 19, 21, 59, tzinfo=timezone.utc))
    assert tracker.session_is_final(winter, None, datetime(2026, 11, 19, 22, 0, tzinfo=timezone.utc))
    assert tracker.session_is_final(summer, 'stocks', datetime(2026, 7, 21, 1, 0, tzinfo=timezone.utc))
    assert not tracker.session_is_final(summer, 'crypto', datetime(2026, 7, 21, 0, 59, tzinfo=timezone.utc))
    assert tracker.session_is_final(summer, 'crypto', datetime(2026, 7, 21, 1, 0, tzinfo=timezone.utc))


def test_yahoo_history_symbol():
    assert tracker.yahoo_history_symbol('brk.b', 'stocks') == 'BRK-B'
    assert tracker.yahoo_history_symbol(' AAPL ', None) == 'AAPL'
    assert tracker.yahoo_history_symbol('BTC', 'crypto') == 'BTC-USD'
    assert tracker.yahoo_history_symbol('eth-usd', 'crypto') == 'ETH-USD'


def _chart_result(days_closes, *, tz='America/New_York', hour=9, minute=30, splits=None):
    zone = ZoneInfo(tz)
    result = {
        'meta': {'exchangeTimezoneName': tz},
        'timestamp': [int(datetime.combine(d, clock_time(hour, minute), tzinfo=zone).timestamp())
                      for d, _ in days_closes],
        'indicators': {'quote': [{'close': [c for _, c in days_closes]}]},
    }
    if splits is not None:
        result['events'] = {'splits': splits}
    return result


def test_parse_chart_history_dates_bars_and_reads_splits():
    result = _chart_result([(date(2026, 7, 17), 101.0), (date(2026, 7, 20), None)], splits={
        '1': {'date': 1784000000, 'numerator': 4, 'denominator': 1},
        '2': {'date': 1783000000, 'numerator': 1, 'denominator': 10},
        '3': {'date': 1785000000, 'numerator': 2},             # no denominator
        '4': {'date': 1786000000, 'numerator': 2, 'denominator': 0},
    })
    parsed = tracker.parse_chart_history(result, 'stocks')
    assert parsed['bars'] == [(date(2026, 7, 17), 101.0), (date(2026, 7, 20), None)]
    assert parsed['splits'] == [(1783000000.0, 0.1), (1784000000.0, 4.0)]


def test_parse_chart_history_dates_crypto_in_utc():
    # Midnight UTC is still the previous day in New York.
    result = _chart_result([(date(2026, 7, 20), 60.0)], tz='UTC', hour=0, minute=0)
    result['meta']['exchangeTimezoneName'] = 'America/New_York'
    assert tracker.parse_chart_history(result, 'crypto')['bars'] == [(date(2026, 7, 20), 60.0)]
    assert tracker.parse_chart_history(result, 'stocks')['bars'] == [(date(2026, 7, 19), 60.0)]


# ---------------------------------------------------------------------------
# The Yahoo price function
# ---------------------------------------------------------------------------

_NOT_JSON = object()


class FakeResponse:
    def __init__(self, status_code=200, body=None, error=None):
        self.status_code = status_code
        self._body = body
        self.error = error

    def json(self):
        if self._body is _NOT_JSON:
            raise ValueError('not json')
        return self._body


def _ok(days_closes, **kw):
    return FakeResponse(200, {'chart': {'result': [_chart_result(days_closes, **kw)], 'error': None}})


GOOD = _ok([(date(2026, 7, 17), 101.0), (date(2026, 7, 20), 102.0)])


class FakeYahoo:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, params):
        self.calls.append((url, dict(params)))
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(response, Exception):
            raise response
        return response


class Budget:
    def __init__(self, allow=True):
        self.allow = allow
        self.calls = []

    def __call__(self, cost, **kw):
        self.calls.append((cost, kw))
        return self.allow


def _price_fn(yahoo, budget=None, **kw):
    return tracker.make_history_price_fn(
        min_gap_seconds=0, http_get=yahoo, budget_fn=budget or Budget(), now_fn=lambda: LATER, **kw)


def test_price_fn_reads_the_close_and_asks_the_budget():
    yahoo, budget = FakeYahoo(GOOD), Budget()
    fn = _price_fn(yahoo, budget)
    assert fn(ROW) == CloseLookup(close=102.0, session_date=date(2026, 7, 20))
    ((url, params),) = yahoo.calls
    assert url == 'https://query1.finance.yahoo.com/v8/finance/chart/AAPL'
    assert params == {
        'period1': int((T0 - timedelta(days=2)).timestamp()),
        'period2': int(LATER.timestamp()),
        'interval': '1d', 'events': 'split',
    }
    assert budget.calls == [(1, {'priority': 3, 'wait_timeout_seconds': 5.0})]


def test_price_fn_fetches_each_symbol_once_unless_it_needs_older_bars():
    yahoo = FakeYahoo(GOOD)
    fn = _price_fn(yahoo)
    fn(ROW)
    fn({**ROW, 'id': 'y', 'created_at': (T0 + timedelta(hours=1)).isoformat()})
    assert len(yahoo.calls) == 1
    fn({**ROW, 'id': 'z', 'created_at': (T0 - timedelta(days=3)).isoformat()})
    assert len(yahoo.calls) == 2
    fn({**ROW, 'symbol': 'brk.b'})
    assert yahoo.calls[-1][0].endswith('/BRK-B')


@pytest.mark.parametrize('response, expected', [
    (FakeResponse(404, {'chart': {'result': None, 'error': {'code': 'Not Found'}}}), CloseLookup(reason='no_history')),
    (FakeResponse(400, {}), CloseLookup(reason='no_history')),
    (FakeResponse(200, {'chart': {'result': None, 'error': {'code': 'Not Found'}}}), CloseLookup(reason='no_history')),
    (FakeResponse(200, {'chart': {'result': [], 'error': None}}), CloseLookup(reason='no_history')),
    (FakeResponse(200, {'chart': {'result': [_chart_result([(date(2026, 7, 20), 102.0)])],
                                  'error': {'code': 'Bad Request'}}}), CloseLookup(reason='no_history')),
    (_ok([(date(2026, 7, 20), None)]), CloseLookup(reason='no_history')),
])
def test_permanent_failures_are_unresolved_and_cached(response, expected):
    yahoo = FakeYahoo(response)
    fn = _price_fn(yahoo)
    assert fn(ROW) == expected
    assert fn({**ROW, 'id': 'again'}) == expected
    assert len(yahoo.calls) == 1


@pytest.mark.parametrize('response, reason', [
    (FakeResponse(429, {}), 'rate_limited'),
    (FakeResponse(503, {}), 'http_503'),
    (FakeResponse(403, {}), 'http_403'),
    (FakeResponse(0, None, error='ReadTimeout: read timed out'), 'timeout'),
    (FakeResponse(0, None, error='ConnectionError: refused'), 'network_error'),
    (FakeResponse(200, _NOT_JSON), 'bad_response'),
    (RuntimeError('boom'), 'network_error:RuntimeError'),
])
def test_temporary_failures_retry_and_are_not_cached(response, reason):
    yahoo = FakeYahoo(response, GOOD)
    fn = _price_fn(yahoo)
    assert fn(ROW) == CloseLookup(reason=reason, retry=True)
    assert fn(ROW).close == 102.0
    assert len(yahoo.calls) == 2


def test_a_denied_budget_means_retry_without_a_request():
    yahoo = FakeYahoo(GOOD)
    fn = _price_fn(yahoo, Budget(allow=False))
    assert fn(ROW) == CloseLookup(reason='budget_denied', retry=True)
    assert yahoo.calls == []


def test_requests_are_spaced_by_the_minimum_gap(monkeypatch):
    now = [1000.0]
    slept = []

    def fake_sleep(seconds):
        slept.append(round(seconds, 6))
        now[0] += seconds

    monkeypatch.setattr(tracker.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(tracker.time, 'sleep', fake_sleep)
    yahoo = FakeYahoo(GOOD)
    fn = tracker.make_history_price_fn(min_gap_seconds=0.4, http_get=yahoo, budget_fn=Budget(),
                                       now_fn=lambda: LATER)
    fn({**ROW, 'symbol': 'A'})
    now[0] += 0.1
    fn({**ROW, 'symbol': 'B'})
    fn({**ROW, 'symbol': 'C'})
    assert slept == [0.3, 0.4]
    assert len(yahoo.calls) == 3


def test_a_row_without_a_symbol_or_date_is_unresolved():
    fn = _price_fn(FakeYahoo(GOOD))
    assert fn({**ROW, 'symbol': ''}) == CloseLookup(reason='bad_row')
    assert fn({**ROW, 'created_at': 'x', 'expires_at': 'y'}) == CloseLookup(reason='bad_row')


# ---------------------------------------------------------------------------
# The evaluation pass
# ---------------------------------------------------------------------------

def test_evaluate_end_to_end(db):
    hit = _save('AAPL')
    miss = _save('TSLA', direction='bear', target_price=95.0)
    flat = _save('KO', direction='neutral', target_price=100.0)
    fresh = _save('NEW', forward_days=30)
    gone = _save('GONE')
    slow = _save('SLOW')
    dead = _save('DEAD')
    answers = {
        'AAPL': CloseLookup(close=104.0, session_date=date(2026, 7, 20)),
        'TSLA': 103.0,
        'KO': 101.0,
        'GONE': None,
        'SLOW': CloseLookup(reason='rate_limited', retry=True),
        'DEAD': CloseLookup(reason='no_history'),
    }
    seen = []

    def price_fn(row):
        seen.append(row['symbol'])
        return answers[row['symbol']]

    now = T0 + timedelta(days=20)
    out = tracker.evaluate_expired_predictions(now=now, price_fn=price_fn)
    assert out == {
        'evaluated': 3, 'unresolved': 2, 'retry_later': 1, 'retry_reasons': {'rate_limited': 1},
        'checked': 6, 'still_open': 1, 'not_yet_expired': 1, 'busy': False,
    }
    assert 'NEW' not in seen

    a = _row(db, hit['id'])
    assert a['status'] == 'evaluated'
    assert (a['actual_close'], a['actual_close_date']) == (104.0, '2026-07-20')
    assert a['actual_return_pct'] == 4.0
    assert a['error_pct'] == round(-100 / 105, 4)
    assert (a['directional_hit'], a['magnitude_hit']) == (1, 1)
    assert a['evaluated_at'] == a['resolved_at'] == now.isoformat()
    assert a['unresolved_reason'] is None
    t = _row(db, miss['id'])
    assert (t['directional_hit'], t['magnitude_hit'], t['actual_close_date']) == (0, 0, None)
    k = _row(db, flat['id'])
    assert (k['directional_hit'], k['magnitude_hit']) == (None, 1)
    for pred, reason in ((gone, 'price_unavailable'), (dead, 'no_history')):
        r = _row(db, pred['id'])
        assert (r['status'], r['unresolved_reason']) == ('unresolved', reason)
        assert r['evaluated_at'] is None  # maintenance prunes on evaluated_at; unresolved rows stay
        assert r['resolved_at'] == now.isoformat()
    assert _row(db, slow['id'])['status'] == 'open'
    assert _row(db, fresh['id'])['status'] == 'open'

    stats = tracker.accuracy_stats()
    assert (stats['evaluated'], stats['unresolved'], stats['open']) == (3, 2, 2)
    assert stats['directional_accuracy_pct'] == 50.0
    assert stats['magnitude_accuracy_pct'] == 66.7
    listed = {r['symbol']: r for r in tracker.list_predictions(status='evaluated')}
    assert listed['AAPL']['directional_hit'] is True and listed['TSLA']['directional_hit'] is False
    assert listed['KO']['directional_hit'] is None

    seen.clear()
    again = tracker.evaluate_expired_predictions(now=now, price_fn=price_fn)
    assert (again['checked'], again['still_open'], seen) == (0, 1, [])

    assert tracker.delete_prediction(hit['id']) is True
    assert tracker.accuracy_stats()['evaluated'] == 2


def test_a_deferred_row_is_retried_after_the_backoff(db, monkeypatch):
    _save('SLOW')
    clock = [5000.0]
    monkeypatch.setattr(tracker.time, 'monotonic', lambda: clock[0])
    answers = [CloseLookup(reason='rate_limited', retry=True), 101.0]
    now = T0 + timedelta(days=20)

    def price_fn(row):
        return answers.pop(0)

    assert tracker.evaluate_expired_predictions(now=now, price_fn=price_fn, retry_backoff_seconds=60)['retry_later'] == 1
    clock[0] += 59
    assert tracker.evaluate_expired_predictions(now=now, price_fn=price_fn)['checked'] == 0
    clock[0] += 1
    assert tracker.evaluate_expired_predictions(now=now, price_fn=price_fn)['evaluated'] == 1


def test_a_zero_backoff_retries_on_the_next_pass(db):
    _save('SLOW')
    answers = [CloseLookup(reason='timeout', retry=True), 101.0]
    now = T0 + timedelta(days=20)
    first = tracker.evaluate_expired_predictions(now=now, price_fn=lambda row: answers.pop(0), retry_backoff_seconds=0)
    assert first['retry_reasons'] == {'timeout': 1}
    assert tracker.evaluate_expired_predictions(now=now, price_fn=lambda row: answers.pop(0))['evaluated'] == 1


def test_batches_take_the_oldest_expiry_first_and_skip_deferred_rows(db):
    for days, symbol in ((14, 'E'), (10, 'A'), (12, 'C'), (11, 'B'), (13, 'D')):
        _save(symbol, forward_days=days)
    now = T0 + timedelta(days=30)
    seen = []

    def price_fn(row):
        seen.append(row['symbol'])
        return CloseLookup(reason='rate_limited', retry=True) if row['symbol'] == 'A' else 101.0

    first = tracker.evaluate_expired_predictions(now=now, batch_size=2, price_fn=price_fn)
    assert seen == ['A', 'B']
    assert (first['checked'], first['evaluated'], first['still_open']) == (2, 1, 4)
    second = tracker.evaluate_expired_predictions(now=now, batch_size=2, price_fn=price_fn)
    assert seen[2:] == ['C', 'D']  # A is deferred, and the batch is still full
    assert second['still_open'] == 2


def test_a_failing_lookup_does_not_stop_the_pass(db):
    _save('BAD')
    _save('GOOD')

    def price_fn(row):
        if row['symbol'] == 'BAD':
            raise RuntimeError('provider exploded')
        return 101.0

    out = tracker.evaluate_expired_predictions(now=T0 + timedelta(days=20), price_fn=price_fn)
    assert out['evaluated'] == 1
    assert out['retry_reasons'] == {'lookup_error:RuntimeError': 1}


@pytest.mark.parametrize('price', [0, -3.0, float('nan'), 'abc'])
def test_an_unusable_price_is_unresolved(db, price):
    pred = _save('ODD')
    out = tracker.evaluate_expired_predictions(now=T0 + timedelta(days=20), price_fn=lambda row: price)
    assert out['unresolved'] == 1
    assert _row(db, pred['id'])['unresolved_reason'] == 'bad_price'


@pytest.mark.parametrize('answer', [101.0, None])
def test_a_row_changed_mid_pass_is_not_overwritten(db, answer):
    pred = _save('RACE')

    def price_fn(row):
        conn = sqlite3.connect(db)
        try:
            with conn:
                conn.execute("UPDATE saved_predictions SET status = 'evaluated' WHERE id = ?", (row['id'],))
        finally:
            conn.close()
        return answer

    out = tracker.evaluate_expired_predictions(now=T0 + timedelta(days=20), price_fn=price_fn)
    assert (out['checked'], out['evaluated'], out['unresolved']) == (1, 0, 0)
    row = _row(db, pred['id'])
    assert (row['status'], row['actual_close'], row['unresolved_reason']) == ('evaluated', None, None)


def test_a_second_pass_while_one_runs_does_nothing(db):
    _save('AAPL')
    assert tracker._EVAL_LOCK.acquire(blocking=False)
    try:
        out = tracker.evaluate_expired_predictions(now=T0 + timedelta(days=20), price_fn=lambda row: 101.0)
    finally:
        tracker._EVAL_LOCK.release()
    assert (out['busy'], out['checked'], out['still_open']) == (True, 0, 1)


def test_now_may_be_a_string_and_must_parse(db):
    _save('AAPL')
    out = tracker.evaluate_expired_predictions(now=(T0 + timedelta(days=20)).isoformat(), price_fn=lambda row: 101.0)
    assert out['evaluated'] == 1
    with pytest.raises(ValueError):
        tracker.evaluate_expired_predictions(now='later', price_fn=lambda row: 101.0)


def test_the_default_price_function_sees_the_pass_time(db, monkeypatch):
    _save('AAPL')
    seen = {}

    def factory(**kw):
        seen['now'] = kw['now_fn']()
        return lambda row: 101.0

    monkeypatch.setattr(tracker, 'make_history_price_fn', factory)
    now = T0 + timedelta(days=20)
    assert tracker.evaluate_expired_predictions(now=now)['evaluated'] == 1
    assert seen['now'] == now


def test_evaluate_now_route(db, monkeypatch):
    _save('AAPL')
    _save('LATER', forward_days=30)
    monkeypatch.setattr(tracker, '_utcnow', lambda: T0 + timedelta(days=20))
    monkeypatch.setattr(tracker, 'make_history_price_fn', lambda **kw: (lambda row: 101.0))
    from app.routes.prediction_tracker import router
    app = FastAPI()
    app.include_router(router)
    res = TestClient(app).post('/api/predictions/evaluate-now')
    assert res.status_code == 200
    body = res.json()
    assert (body['evaluated'], body['still_open'], body['not_yet_expired'], body['busy']) == (1, 0, 1, False)


# ---------------------------------------------------------------------------
# Background thread
# ---------------------------------------------------------------------------

def _evaluator_threads():
    return [t for t in threading.enumerate() if t.name == tracker.EVALUATOR_THREAD_NAME and t.is_alive()]


@pytest.fixture
def evaluator(monkeypatch):
    monkeypatch.setenv('MRD_START_BACKGROUND', '1')
    yield
    tracker.stop_evaluator()
    tracker._EVALUATOR['thread'] = None
    tracker._EVALUATOR_STOP.clear()


def test_the_evaluator_defaults_to_every_15_minutes_after_two():
    import inspect
    params = inspect.signature(tracker.start_evaluator).parameters
    assert (params['interval_seconds'].default, params['initial_delay_seconds'].default) == (900, 120)


def test_start_evaluator_respects_the_background_switch(monkeypatch):
    monkeypatch.setenv('MRD_START_BACKGROUND', '0')
    assert tracker.start_evaluator() is False
    assert _evaluator_threads() == []


def test_start_evaluator_runs_one_named_daemon_thread(evaluator, monkeypatch):
    ran = threading.Event()
    monkeypatch.setattr(tracker, 'evaluate_expired_predictions', lambda: ran.set())
    assert tracker.start_evaluator(interval_seconds=3600, initial_delay_seconds=0) is True
    assert ran.wait(5)
    (thread,) = _evaluator_threads()
    assert thread.daemon
    assert tracker.start_evaluator(interval_seconds=3600, initial_delay_seconds=0) is True
    assert _evaluator_threads() == [thread]
    tracker.stop_evaluator()
    assert not thread.is_alive()


def test_the_evaluator_waits_before_its_first_pass(evaluator, monkeypatch):
    ran = threading.Event()
    monkeypatch.setattr(tracker, 'evaluate_expired_predictions', lambda: ran.set())
    tracker.start_evaluator(interval_seconds=3600, initial_delay_seconds=30)
    assert not ran.wait(0.3)
    tracker.stop_evaluator()
    assert _evaluator_threads() == []


def test_the_evaluator_can_start_again_after_a_stop(evaluator, monkeypatch):
    ran = threading.Event()
    monkeypatch.setattr(tracker, 'evaluate_expired_predictions', lambda: ran.set())
    tracker.start_evaluator(interval_seconds=3600, initial_delay_seconds=30)
    tracker.stop_evaluator()
    assert tracker.start_evaluator(interval_seconds=3600, initial_delay_seconds=0) is True
    assert ran.wait(5)


def test_the_evaluator_survives_a_failing_pass(evaluator, monkeypatch):
    calls = []
    second = threading.Event()

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError('first pass fails')
        second.set()

    monkeypatch.setattr(tracker, 'evaluate_expired_predictions', flaky)
    tracker.start_evaluator(interval_seconds=0, initial_delay_seconds=0)
    assert second.wait(5)
