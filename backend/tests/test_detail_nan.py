"""Phase 3: a NaN in a detail row no longer turns /stock/{symbol} into a 500.

After the close, Yahoo's daily bar for a thin name can carry an open,
high, low and volume but no close. The short selling pressure factor
took that NaN close into its price trend, the NaN reached
factor_breakdown.market.short_selling_pressure.price_trend_pct_per_wk,
and Starlette's JSONResponse refused the whole row (ValueError, 500).
"""
from __future__ import annotations

import json
import logging
import math
import sys
import types

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routes import detail as detail_route
from app.services.short_pressure_service import compute_short_selling_pressure
from app.utils.json_safe import replace_non_finite


# ---------------------------------------------------------------------------
# replace_non_finite
# ---------------------------------------------------------------------------

def test_replace_non_finite_replaces_nan_and_infinities_and_reports_paths():
    value = {
        'a': 1.5,
        'b': float('nan'),
        'c': {'d': float('inf'), 'e': [1, float('-inf'), 'x', None, True]},
        'f': (2.0, float('nan')),
        'g': np.float64('nan'),
        'h': 3,
    }
    found: list[str] = []
    clean = replace_non_finite(value, found)
    assert clean == {
        'a': 1.5,
        'b': None,
        'c': {'d': None, 'e': [1, None, 'x', None, True]},
        'f': [2.0, None],
        'g': None,
        'h': 3,
    }
    assert found == ['b', 'c.d', 'c.e[1]', 'f[1]', 'g']
    json.dumps(clean, allow_nan=False)
    # The input is left as it was.
    assert math.isnan(value['b']) and math.isinf(value['c']['d'])
    assert value['f'] == (2.0, value['f'][1])


def test_replace_non_finite_root_value_and_optional_found():
    assert replace_non_finite(float('nan')) is None
    found: list[str] = []
    assert replace_non_finite(float('inf'), found) is None
    assert found == ['<root>']
    assert replace_non_finite({'x': 1.0}) == {'x': 1.0}
    assert replace_non_finite('nan') == 'nan'


# ---------------------------------------------------------------------------
# short selling pressure: a session without a close is left out
# ---------------------------------------------------------------------------

def _sessions(count: int) -> pd.DataFrame:
    """`count` complete daily sessions with a mix of up and down days."""
    index = pd.bdate_range('2026-10-01', periods=count, tz='America/New_York')
    closes = [10.0 + 0.4 * i + (0.9 if i % 3 == 0 else -0.6) for i in range(count)]
    return pd.DataFrame({
        'Open': [c - 0.2 for c in closes],
        'High': [c + 0.5 for c in closes],
        'Low': [c - 0.7 for c in closes],
        'Close': closes,
        'Volume': [100_000.0 + 15_000.0 * (i % 4) for i in range(count)],
    }, index=index)


def _with_session_in_progress(frame: pd.DataFrame) -> pd.DataFrame:
    """`frame` plus a newest bar that has an open, high, low and volume but no close."""
    stamp = frame.index[-1] + pd.offsets.BDay(1)
    last = float(frame['Close'].iloc[-1])
    extra = pd.DataFrame({
        'Open': [last * 1.05], 'High': [last * 1.10], 'Low': [last * 0.90],
        'Close': [float('nan')], 'Volume': [900_000.0],
    }, index=pd.DatetimeIndex([stamp]))
    return pd.concat([frame, extra])


def _pressure(frame: pd.DataFrame) -> dict:
    return compute_short_selling_pressure('EUDA', 14.0, 13.5, info={}, daily_hist=frame)


def test_session_without_a_close_is_left_out():
    complete = _sessions(20)
    clean = _pressure(complete)
    with_open_bar = _pressure(_with_session_in_progress(complete))
    assert math.isfinite(with_open_bar['price_trend_pct_per_wk'])
    assert with_open_bar == clean
    found: list[str] = []
    replace_non_finite(with_open_bar, found)
    assert found == []
    # The fixture's trend is not zero, so a dropped trend would show.
    assert clean['price_trend_pct_per_wk'] != 0.0
    assert 'downside_persistence' in clean['components']


def test_a_session_without_a_close_in_the_middle_is_left_out():
    complete = _sessions(20)
    stamp = complete.index[9] + pd.Timedelta(hours=1)
    gap = pd.DataFrame({
        'Open': [30.0], 'High': [31.0], 'Low': [29.0],
        'Close': [float('nan')], 'Volume': [5_000_000.0],
    }, index=pd.DatetimeIndex([stamp]))
    halted = pd.concat([complete.iloc[:10], gap, complete.iloc[10:]])
    assert len(halted) == 21 and math.isnan(halted['Close'].iloc[10])
    assert _pressure(halted) == _pressure(complete)


def test_fifteen_sessions_with_a_close_are_needed():
    fourteen = _sessions(14)
    result = _pressure(_with_session_in_progress(fourteen))
    assert result == _pressure(fourteen)
    assert 'downside_persistence' not in result.get('components', {})
    fifteen = _pressure(_with_session_in_progress(_sessions(15)))
    assert 'downside_persistence' in fifteen['components']


# ---------------------------------------------------------------------------
# /stock/{symbol}
# ---------------------------------------------------------------------------

def _row_with_nan() -> dict:
    return {
        'symbol': 'EUDA',
        'last_price': 1.23,
        'detail_source': 'snapshot_mirror',
        'factor_breakdown': {'market': {'short_selling_pressure': {
            'score': 55.0, 'price_trend_pct_per_wk': float('nan')}}},
        'forward_metrics': {'bands': [1.0, float('inf')]},
    }


@pytest.fixture()
def client(monkeypatch):
    fake_tiers = types.ModuleType('app.services.tier_manager')
    fake_tiers.record_user_interaction = lambda symbol: None
    monkeypatch.setitem(sys.modules, 'app.services.tier_manager', fake_tiers)
    monkeypatch.setattr(detail_route, '_NON_FINITE_LOGGED', {})
    app = FastAPI()
    app.include_router(detail_route.router)
    return TestClient(app, raise_server_exceptions=False)


def test_detail_route_sends_nan_as_null_and_warns_once(client, monkeypatch, caplog):
    monkeypatch.setattr(detail_route, 'get_symbol_detail', lambda *a, **k: _row_with_nan())
    caplog.set_level(logging.WARNING, logger='app.routes.detail')
    res = client.get('/stock/EUDA')
    assert res.status_code == 200
    body = res.json()
    assert body['factor_breakdown']['market']['short_selling_pressure'] == {
        'score': 55.0, 'price_trend_pct_per_wk': None}
    assert body['forward_metrics']['bands'] == [1.0, None]
    assert body['last_price'] == 1.23
    warnings = [r.getMessage() for r in caplog.records if r.name == 'app.routes.detail']
    assert len(warnings) == 1
    assert 'EUDA' in warnings[0] and '2 non-finite' in warnings[0]
    assert 'factor_breakdown.market.short_selling_pressure.price_trend_pct_per_wk' in warnings[0]
    assert 'forward_metrics.bands[1]' in warnings[0]
    caplog.clear()
    assert client.get('/stock/EUDA?require_fresh=1').status_code == 200
    assert [r for r in caplog.records if r.name == 'app.routes.detail'] == []


def test_detail_route_warns_again_after_the_interval(client, monkeypatch, caplog):
    monkeypatch.setattr(detail_route, 'get_symbol_detail', lambda *a, **k: _row_with_nan())
    caplog.set_level(logging.WARNING, logger='app.routes.detail')
    clock = [1000.0]
    monkeypatch.setattr(detail_route.time, 'monotonic', lambda: clock[0])
    assert client.get('/stock/EUDA').status_code == 200
    clock[0] += detail_route.NON_FINITE_LOG_INTERVAL_SECONDS - 1
    assert client.get('/stock/EUDA').status_code == 200
    clock[0] += 1
    assert client.get('/stock/EUDA').status_code == 200
    warnings = [r for r in caplog.records if r.name == 'app.routes.detail']
    assert len(warnings) == 2


def test_detail_route_passes_a_clean_row_through(client, monkeypatch, caplog):
    row = {'symbol': 'AAPL', 'last_price': 349.21, 'factor_breakdown': {'market': {'x': 1.0}}, 'tags': ['a']}
    seen = {}

    def fake_detail(symbol, force_live=False, market='stocks', require_fresh=False):
        seen.update(symbol=symbol, force_live=force_live, market=market, require_fresh=require_fresh)
        return row

    monkeypatch.setattr(detail_route, 'get_symbol_detail', fake_detail)
    caplog.set_level(logging.WARNING, logger='app.routes.detail')
    res = client.get('/stock/aapl?require_fresh=true&market=stocks')
    assert res.status_code == 200
    assert res.json() == row
    assert seen == {'symbol': 'AAPL', 'force_live': False, 'market': 'stocks', 'require_fresh': True}
    assert [r for r in caplog.records if r.name == 'app.routes.detail'] == []
