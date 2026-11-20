"""Phase 3: the detail panel's require_fresh refresh fetches a quote.

Before Phase 3, _refresh_snapshot_quote imported download_quotes and
utcnowiso from two modules that do not exist; the except around those
imports returned the cached row, so the "fresh" refresh never fetched
anything.  These tests pin the repaired behavior with download_quotes and
the quote cache replaced by fakes (no network).
"""
from __future__ import annotations

import copy
import logging
from datetime import datetime, timedelta, timezone

import pytest

from app.services import detail_service, snapshot_store


QUOTE_TIME = '2026-11-19T21:00:00+00:00'


def _iso_ago(seconds: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


def _snapshot_row() -> dict:
    return {
        'symbol': 'AAPL',
        'name': 'Apple Inc.',
        'exchange': 'NASDAQ',
        'final_score': 71.5,
        'state': 'ready',
        'data_source': 'yahoo-chart',
        'age_seconds': 12,
        'as_of_utc': '2026-11-19T20:00:00+00:00',
        'quote_as_of_utc': '2026-11-19T19:59:00+00:00',
        'forward_metrics': {'forward_1h': {'expected_pct': 0.4}},
        'lab_signals': {'kept': True},
        'factor_breakdown': {
            'market': {
                'last_price': 100.0,
                'previous_close': 99.0,
                'change_pct': 1.0101,
                'age_seconds': 12,
                'source': 'yahoo-chart',
                'trend_volume_delta': {'score': 55.0},
                'institutional_confluence': {'score': 61.0},
                'options_positioning': {'score': 48.0},
                'volume_sentiment': {'score': 52.0},
            },
            'fundamentals': {'sector': 'Technology'},
        },
    }


def _live_quote(last=110.0, prev=100.0, **extra) -> dict:
    quote = {
        'last_price': last,
        'previous_close': prev,
        'change_pct': None,
        'source': 'yahoo-chart',
        'captured_at_utc': _iso_ago(0),
        'quote_time_utc': QUOTE_TIME,
    }
    quote.update(extra)
    return quote


@pytest.fixture
def fakes(monkeypatch):
    """Replace the cascade and the quote cache; record what was called."""
    state = {
        'download_calls': [],
        'download_result': {'AAPL': _live_quote()},
        'download_exc': None,
        'cached': None,
        'saved': [],
    }

    def fake_download(symbols, market='stocks'):
        state['download_calls'].append((list(symbols), market))
        if state['download_exc'] is not None:
            raise state['download_exc']
        return state['download_result']

    monkeypatch.setattr(detail_service, 'download_quotes', fake_download)
    monkeypatch.setattr(detail_service, 'get_cached_quote', lambda sym: copy.deepcopy(state['cached']))
    monkeypatch.setattr(detail_service, 'save_quote', lambda sym, q: state['saved'].append((sym, dict(q))))
    monkeypatch.delenv('MRD_DETAIL_QUOTE_REUSE_SECONDS', raising=False)
    detail_service._refresh_warned_at.clear()
    yield state
    detail_service._refresh_warned_at.clear()


def test_live_quote_updates_price_change_and_quote_time(fakes):
    canonical = _snapshot_row()
    snap = dict(canonical)  # lookup_snapshot_row hands out a shallow copy

    out = detail_service._refresh_snapshot_quote(snap, 'AAPL', 'stocks')

    assert fakes['download_calls'] == [(['AAPL'], 'stocks')]
    mkt = out['factor_breakdown']['market']
    assert mkt['last_price'] == 110.0
    assert mkt['previous_close'] == 100.0
    assert mkt['change_pct'] == pytest.approx(10.0)
    assert mkt['age_seconds'] == 0
    assert mkt['source'] == 'yahoo-chart'
    # The other market fields and every overlay survive the refresh.
    assert mkt['institutional_confluence'] == {'score': 61.0}
    assert out['forward_metrics'] == {'forward_1h': {'expected_pct': 0.4}}
    assert out['lab_signals'] == {'kept': True}
    assert out['quote_as_of_utc'] == QUOTE_TIME
    assert out['detail_source'] == 'snapshot_refreshed'
    assert out['detail_quote_origin'] == 'live'
    assert out['fresh_rescore'] is True
    assert out['stale'] is False and out['freshness_label'] == 'fresh'
    assert 'detail_fallback_reason' not in out
    # The fetched quote is cached for the next tick and for the scanner.
    assert [sym for sym, _ in fakes['saved']] == ['AAPL']
    # The leaderboard's own row is not modified through the shared dicts.
    assert canonical['factor_breakdown']['market']['last_price'] == 100.0
    assert canonical['quote_as_of_utc'] == '2026-11-19T19:59:00+00:00'


def test_crypto_refresh_asks_the_crypto_cascade(fakes):
    fakes['download_result'] = {'BTC': _live_quote(last=91000.0, prev=90000.0, source='coingecko')}
    row = _snapshot_row()
    row['symbol'] = 'BTC'
    out = detail_service._refresh_snapshot_quote(row, 'BTC', 'crypto')
    assert fakes['download_calls'] == [(['BTC'], 'crypto')]
    assert out['factor_breakdown']['market']['last_price'] == 91000.0
    assert out['factor_breakdown']['market']['source'] == 'coingecko'


def test_source_label_comes_from_the_quote(fakes):
    fakes['download_result'] = {'AAPL': _live_quote(source='yfinance')}
    out = detail_service._refresh_snapshot_quote(_snapshot_row(), 'AAPL', 'stocks')
    assert out['factor_breakdown']['market']['source'] == 'yfinance'
    fakes['download_result'] = {'AAPL': _live_quote(source=None)}
    out = detail_service._refresh_snapshot_quote(_snapshot_row(), 'AAPL', 'stocks')
    # Before Phase 3 a missing label became 'yfinance'.
    assert out['factor_breakdown']['market']['source'] == 'unknown'


def test_recent_cached_quote_is_reused_without_a_cascade(fakes):
    fakes['cached'] = {
        'symbol': 'AAPL', 'last_price': 104.0, 'previous_close': 100.0,
        'captured_at_utc': _iso_ago(2), 'source': 'yahoo-chart',
        'quote_time_utc': QUOTE_TIME,
    }
    out = detail_service._refresh_snapshot_quote(_snapshot_row(), 'AAPL', 'stocks')
    assert fakes['download_calls'] == []
    assert fakes['saved'] == []
    mkt = out['factor_breakdown']['market']
    assert mkt['last_price'] == 104.0
    assert mkt['change_pct'] == pytest.approx(4.0)
    assert 1 <= mkt['age_seconds'] <= 5
    assert out['detail_quote_origin'] == 'recent_cache'
    assert out['quote_as_of_utc'] == QUOTE_TIME


def test_older_cached_quote_triggers_a_fetch(fakes):
    fakes['cached'] = {
        'symbol': 'AAPL', 'last_price': 104.0, 'previous_close': 100.0,
        'captured_at_utc': _iso_ago(30), 'source': 'yahoo-chart',
    }
    out = detail_service._refresh_snapshot_quote(_snapshot_row(), 'AAPL', 'stocks')
    assert len(fakes['download_calls']) == 1
    assert out['factor_breakdown']['market']['last_price'] == 110.0
    assert out['detail_quote_origin'] == 'live'


def test_reuse_window_is_configurable_and_zero_turns_it_off(fakes, monkeypatch):
    fakes['cached'] = {
        'symbol': 'AAPL', 'last_price': 104.0, 'previous_close': 100.0,
        'captured_at_utc': _iso_ago(20), 'source': 'yahoo-chart',
    }
    monkeypatch.setenv('MRD_DETAIL_QUOTE_REUSE_SECONDS', '60')
    out = detail_service._refresh_snapshot_quote(_snapshot_row(), 'AAPL', 'stocks')
    assert fakes['download_calls'] == []
    assert out['detail_quote_origin'] == 'recent_cache'

    fakes['cached']['captured_at_utc'] = _iso_ago(0)
    monkeypatch.setenv('MRD_DETAIL_QUOTE_REUSE_SECONDS', '0')
    out = detail_service._refresh_snapshot_quote(_snapshot_row(), 'AAPL', 'stocks')
    assert len(fakes['download_calls']) == 1
    assert out['detail_quote_origin'] == 'live'

    monkeypatch.setenv('MRD_DETAIL_QUOTE_REUSE_SECONDS', 'not-a-number')
    assert detail_service._detail_quote_reuse_seconds() == 5.0


def test_cached_quote_without_a_positive_pair_is_not_reused(fakes):
    fakes['cached'] = {
        'symbol': 'AAPL', 'last_price': 104.0, 'previous_close': 0.0,
        'captured_at_utc': _iso_ago(1), 'source': 'yahoo-chart',
    }
    out = detail_service._refresh_snapshot_quote(_snapshot_row(), 'AAPL', 'stocks')
    assert len(fakes['download_calls']) == 1
    assert out['detail_quote_origin'] == 'live'


def test_provider_error_falls_back_with_a_reason_and_one_warning(fakes, caplog):
    fakes['download_exc'] = RuntimeError('boom')
    caplog.set_level(logging.DEBUG, logger='app.detail')

    out = detail_service._refresh_snapshot_quote(_snapshot_row(), 'AAPL', 'stocks')

    assert out['detail_source'] == 'snapshot_fallback'
    assert out['detail_fallback_reason'] == 'provider_error: RuntimeError: boom'
    assert out['fresh_rescore'] is False
    # Snapshot values are kept untouched.
    mkt = out['factor_breakdown']['market']
    assert mkt['last_price'] == 100.0 and mkt['change_pct'] == 1.0101
    assert out['quote_as_of_utc'] == '2026-11-19T19:59:00+00:00'
    assert fakes['saved'] == []
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and 'AAPL' in warnings[0].getMessage()

    # The panel ticks every 2 s: the next failure inside a minute is DEBUG.
    caplog.clear()
    detail_service._refresh_snapshot_quote(_snapshot_row(), 'AAPL', 'stocks')
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []
    assert any('AAPL' in r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG)


def test_success_rearms_the_warning(fakes, caplog):
    caplog.set_level(logging.DEBUG, logger='app.detail')
    fakes['download_exc'] = RuntimeError('boom')
    detail_service._refresh_snapshot_quote(_snapshot_row(), 'AAPL', 'stocks')
    fakes['download_exc'] = None
    detail_service._refresh_snapshot_quote(_snapshot_row(), 'AAPL', 'stocks')
    caplog.clear()
    fakes['download_exc'] = RuntimeError('boom again')
    detail_service._refresh_snapshot_quote(_snapshot_row(), 'AAPL', 'stocks')
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


@pytest.mark.parametrize('result, reason', [
    ({}, 'no_quote'),
    (None, 'no_quote'),
    ({'MSFT': _live_quote()}, 'no_quote'),
    ({'AAPL': _live_quote(last=0.0)}, 'unusable_quote'),
    ({'AAPL': _live_quote(prev=None)}, 'unusable_quote'),
    ({'AAPL': _live_quote(last='n/a')}, 'unusable_quote'),
])
def test_empty_or_unusable_results_fall_back(fakes, result, reason):
    fakes['download_result'] = result
    out = detail_service._refresh_snapshot_quote(_snapshot_row(), 'AAPL', 'stocks')
    assert out['detail_source'] == 'snapshot_fallback'
    assert out['detail_fallback_reason'] == reason
    assert out['factor_breakdown']['market']['last_price'] == 100.0
    assert fakes['saved'] == []


def test_get_symbol_detail_require_fresh_runs_the_refresh(fakes, monkeypatch):
    monkeypatch.setattr(snapshot_store, 'lookup_snapshot_row', lambda sym, market='stocks': _snapshot_row())
    monkeypatch.setattr(detail_service, 'get_symbol_identity',
                        lambda sym, market='stocks': {'symbol': 'AAPL', 'name': 'Apple Inc.', 'exchange': 'NASDAQ'})

    out = detail_service.get_symbol_detail('AAPL', market='stocks', require_fresh=True)

    assert fakes['download_calls'] == [(['AAPL'], 'stocks')]
    assert out['detail_source'] == 'snapshot_refreshed'
    assert out['factor_breakdown']['market']['last_price'] == 110.0
    assert out['factor_breakdown']['market']['change_pct'] == pytest.approx(10.0)
    assert out['quote_as_of_utc'] == QUOTE_TIME
    assert out['forward_metrics'] == {'forward_1h': {'expected_pct': 0.4}}

    # Without require_fresh the snapshot is mirrored and nothing is fetched.
    out = detail_service.get_symbol_detail('AAPL', market='stocks')
    assert len(fakes['download_calls']) == 1
    assert out['detail_source'] == 'snapshot_mirror'
    assert out['factor_breakdown']['market']['last_price'] == 100.0
