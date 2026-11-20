"""Phase 2C: the daily change is measured against the previous session.

Every quote path must pair ``last_price`` with the previous session's
official close, take open, high, low and volume from ``last_price``'s own
session, and report the exchange timestamp as ``quote_time_utc``.

Two Yahoo chart fixtures pin the chart parser:

* ``yahoo_chart_aapl_5d_20261118.json`` is rebuilt from the AAPL
  ``range=5d`` reply recorded after the close on November 18, 2026 (the
  numbers in the playbook's Phase 2C context). The meta price, time and
  chartPreviousClose and every bar's open, close and volume are as
  recorded; the 11/12-11/17 highs and lows come from a ``range=10d`` reply
  fetched on November 19. The in-progress 11/18 bar's high and low were
  not recorded, so they are null and the parser falls back to
  meta.regularMarketDayHigh/Low, recorded from that evening's spark reply.
* ``yahoo_chart_aapl_5d_20261119.json`` is the real reply from the evening
  of November 19, 2026, unedited.

Everything else runs on in-process fakes; no socket is opened.
"""
from __future__ import annotations

import copy
import json
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest

from app.services import http_client, quote_math, scoring_service
from app.services.http_client import ResilientResponse
from app.services.providers import base as providers_base
from app.services.providers import coinpaprika_provider, cryptocompare_provider
from app.services.providers import finnhub_provider, yahoo_chart_provider
from app.services.providers import stooq_provider as stooq

FIXTURES = Path(__file__).parent / 'fixtures'
NAN = float('nan')
NY = 'America/New_York'
# Yahoo's bar values are float32-rounded (352.45001220703125).
BAR = 1e-4


def _chart(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())['chart']['result'][0]


def _utc(*parts: int) -> int:
    return int(datetime(*parts, tzinfo=timezone.utc).timestamp())


@pytest.fixture
def nov18() -> dict:
    return _chart('yahoo_chart_aapl_5d_20261118.json')


@pytest.fixture
def nov19() -> dict:
    return _chart('yahoo_chart_aapl_5d_20261119.json')


# ---------------------------------------------------------------------------
# Yahoo chart provider
# ---------------------------------------------------------------------------
def test_chart_nov18_uses_previous_session_close_not_window_close(nov18):
    q = yahoo_chart_provider.parse_chart_result(nov18, captured_at='2026-11-18T21:05:00+00:00')

    assert q['last_price'] == 352.45
    assert q['previous_close'] == pytest.approx(351.91, abs=BAR)
    assert q['change_pct'] == pytest.approx(0.1534, abs=0.001)
    # The old code divided by chartPreviousClose, the close before the
    # 5-day window, and reported -0.89%.
    assert nov18['meta']['chartPreviousClose'] == 355.63
    assert q['previous_close'] != nov18['meta']['chartPreviousClose']
    # Same-session OHLCV: open and volume from the 11/18 bar, high and low
    # from meta because that bar has none yet.
    assert q['open'] == pytest.approx(351.87, abs=BAR)
    assert q['volume'] == 33_393_854
    assert q['day_high'] == 356.42
    assert q['day_low'] == 351.471
    assert q['quote_time_utc'] == '2026-11-18T21:00:01+00:00'
    assert q['captured_at_utc'] == '2026-11-18T21:05:00+00:00'
    assert q['source'] == 'yahoo-chart'
    assert q['market_cap'] == 0.0
    assert q['provider_outcome'] == 'live_success' and q['preview_only'] is False
    assert set(quote_math.QUOTE_KEYS) <= set(q)


def test_chart_nov19_real_reply_matches_yahoos_change(nov19):
    q = yahoo_chart_provider.parse_chart_result(nov19)

    assert q['last_price'] == 349.21
    assert q['previous_close'] == pytest.approx(352.45, abs=BAR)
    assert q['change_pct'] == pytest.approx(-0.9193, abs=0.001)
    assert q['change_pct'] == pytest.approx(nov19['meta']['regularMarketChangePercent'], abs=0.01)
    # chartPreviousClose here is 11/12's close (354.88): -1.60%, the bug.
    assert nov19['meta']['chartPreviousClose'] == 354.88
    assert q['open'] == pytest.approx(352.91, abs=BAR)
    assert q['day_high'] == pytest.approx(355.28, abs=BAR)
    assert q['day_low'] == pytest.approx(346.47, abs=BAR)
    assert q['volume'] == 40_469_518
    assert q['quote_time_utc'] == '2026-11-19T21:00:01+00:00'


def test_chart_bar_order_does_not_matter(nov19):
    r0 = copy.deepcopy(nov19)
    r0['timestamp'] = r0['timestamp'][::-1]
    bars = r0['indicators']['quote'][0]
    for key in list(bars):
        bars[key] = bars[key][::-1]
    q = yahoo_chart_provider.parse_chart_result(r0)
    assert (q['last_price'], q['previous_close'], q['open'], q['volume']) == pytest.approx(
        (349.21, 352.45, 352.91, 40_469_518), abs=BAR)


def test_chart_range_1d_takes_chart_previous_close(nov18):
    # range=1d returns only the session's own bar, and then
    # chartPreviousClose is the previous session's close.
    r0 = copy.deepcopy(nov18)
    r0['meta'].update(range='1d', chartPreviousClose=351.91)
    r0['timestamp'] = r0['timestamp'][-1:]
    bars = r0['indicators']['quote'][0]
    for key in list(bars):
        bars[key] = bars[key][-1:]
    q = yahoo_chart_provider.parse_chart_result(r0)
    assert q['previous_close'] == 351.91
    assert q['change_pct'] == pytest.approx(0.1534, abs=0.001)


def test_chart_never_falls_back_to_window_close_when_earlier_bars_exist(nov18):
    # Earlier bars exist but have no close: there is no honest previous
    # close, so no quote (the next provider answers) rather than 355.63.
    r0 = copy.deepcopy(nov18)
    closes = r0['indicators']['quote'][0]['close']
    r0['indicators']['quote'][0]['close'] = [None] * len(closes)
    assert yahoo_chart_provider.parse_chart_result(r0) is None


def test_chart_without_meta_price_uses_newest_bar_with_a_close(nov19):
    r0 = copy.deepcopy(nov19)
    del r0['meta']['regularMarketPrice']
    del r0['meta']['regularMarketTime']
    q = yahoo_chart_provider.parse_chart_result(r0)
    assert q['last_price'] == pytest.approx(349.21, abs=BAR)
    assert q['previous_close'] == pytest.approx(352.45, abs=BAR)
    assert q['quote_time_utc'] == '2026-11-19T14:30:00+00:00'


def test_chart_columns_of_the_wrong_length(nov19):
    r0 = copy.deepcopy(nov19)
    del r0['meta']['regularMarketPrice']
    del r0['meta']['regularMarketTime']
    bars = r0['indicators']['quote'][0]
    bars['close'] = bars['close'] + [999.0]  # a value without a timestamp is ignored
    bars['volume'] = bars['volume'][:2]      # values that are not there are unknown
    q = yahoo_chart_provider.parse_chart_result(r0)
    assert q['last_price'] == pytest.approx(349.21, abs=BAR)
    assert q['previous_close'] == pytest.approx(352.45, abs=BAR)
    assert q['volume'] == r0['meta']['regularMarketVolume']


def test_chart_without_any_price_gives_no_quote(nov18):
    r0 = copy.deepcopy(nov18)
    del r0['meta']['regularMarketPrice']
    r0['indicators']['quote'][0]['close'] = [None] * 5
    assert yahoo_chart_provider.parse_chart_result(r0) is None


def test_chart_fetch_one_wires_the_parser(monkeypatch, nov18):
    body = (FIXTURES / 'yahoo_chart_aapl_5d_20261118.json').read_text()
    seen = []

    def fake_get(url, params=None, headers=None, cfg=None):
        seen.append((url, dict(params or {})))
        return ResilientResponse(status_code=200, text=body,
                                 headers={'Content-Type': 'application/json'}, url=url, attempts=1)

    monkeypatch.setattr(http_client, 'resilient_get', fake_get)
    sym, q = yahoo_chart_provider._fetch_one('AAPL', '2026-11-18T21:05:00+00:00')
    assert sym == 'AAPL'
    assert seen and seen[0][1] == {'range': '5d', 'interval': '1d'}
    assert q == yahoo_chart_provider.parse_chart_result(nov18, captured_at='2026-11-18T21:05:00+00:00')


# ---------------------------------------------------------------------------
# yfinance batch download (download_quotes_yfinance_only)
# ---------------------------------------------------------------------------
NOV18_DATES = ('2026-11-12', '2026-11-13', '2026-11-16', '2026-11-17', '2026-11-18')
NOV19_DATES = ('2026-11-13', '2026-11-16', '2026-11-17', '2026-11-18', '2026-11-19')

# Daily bars as yf.download labels them: naive session dates. After the
# close on 11/18 the session's bar had an open and a volume but no close.
AAPL_IN_PROGRESS = [
    (357.81, 359.37, 354.41, 354.88, 31_875_000),
    (357.96, 360.83, 355.75, 356.27, 40_027_500),
    (357.13, 357.13, 351.31, 352.78, 50_292_200),
    (352.83, 355.36, 349.67, 351.91, 37_407_500),
    (351.87, NAN, NAN, NAN, 33_393_854),
]
AAPL_FINAL = [
    (357.96, 360.83, 355.75, 356.27, 40_027_500),
    (357.13, 357.13, 351.31, 352.78, 50_292_200),
    (352.83, 355.36, 349.67, 351.91, 37_407_500),
    (351.94, 356.42, 351.47, 352.45, 34_209_900),
    (352.91, 355.28, 346.47, 349.21, 40_469_518),
]
COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']


def _daily(rows, dates) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)[-len(rows):]), name='Date')
    return pd.DataFrame(rows, index=idx, columns=COLUMNS, dtype=float)


def _grouped(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The yf.download(group_by='ticker') layout: (Ticker, Price) columns."""
    return pd.concat(frames, axis=1, names=['Ticker', 'Price'])


@pytest.fixture
def yf_download(monkeypatch):
    calls = []
    holder = SimpleNamespace(frame=None, calls=calls)

    def fake_download(**kwargs):
        calls.append(kwargs)
        return holder.frame

    monkeypatch.setattr(scoring_service, 'provider_budget_allowance', lambda *a, **k: True)
    monkeypatch.setattr(scoring_service, 'warm_provider_session', lambda *a, **k: {})
    monkeypatch.setattr(scoring_service, '_yf_download_with_timeout', fake_download)
    return holder


def test_yf_batch_in_progress_bar_gives_no_quote_for_that_symbol(yf_download):
    yf_download.frame = _grouped({
        'AAPL': _daily(AAPL_IN_PROGRESS, NOV18_DATES),
        'MSFT': _daily([(500.0, 507.0, 499.0, 506.24, 20e6), (506.5, 510.44, 501.08, 508.29, 21_349_348)], NOV18_DATES),
    })
    out = scoring_service.download_quotes_yfinance_only(['AAPL', 'MSFT'], 'stocks')
    # The old code returned AAPL 351.91 vs 352.78 next to today's open.
    assert 'AAPL' not in out
    msft = out['MSFT']
    assert (msft['last_price'], msft['previous_close']) == (508.29, 506.24)
    assert (msft['open'], msft['day_high'], msft['day_low'], msft['volume']) == (506.5, 510.44, 501.08, 21_349_348)
    assert msft['change_pct'] == pytest.approx((508.29 / 506.24 - 1) * 100, abs=1e-6)
    assert msft['quote_time_utc'] == '2026-11-18T05:00:00+00:00'
    assert msft['source'] == 'yfinance'


def test_yf_batch_complete_rows_pair_the_last_two_rows(yf_download):
    yf_download.frame = _grouped({
        'AAPL': _daily(AAPL_FINAL, NOV19_DATES),
        'MSFT': _daily([(1.0, 1.0, 1.0, 1.0, 1.0)], NOV19_DATES),  # one row: nothing to pair
    })
    out = scoring_service.download_quotes_yfinance_only(['AAPL', 'MSFT'], 'stocks')
    q = out['AAPL']
    assert (q['last_price'], q['previous_close']) == (349.21, 352.45)
    assert (q['open'], q['day_high'], q['day_low'], q['volume']) == (352.91, 355.28, 346.47, 40_469_518)
    assert q['change_pct'] == pytest.approx(-0.9193, abs=0.001)
    # Daily bars carry the session date only: local midnight as a marker.
    assert q['quote_time_utc'] == '2026-11-19T05:00:00+00:00'
    assert 'MSFT' not in out


def test_yf_batch_previous_close_skips_a_row_without_close(yf_download):
    rows = [list(r) for r in AAPL_FINAL]
    rows[3][3] = NAN  # 11/18 close missing
    yf_download.frame = _grouped({'AAPL': _daily(rows, NOV19_DATES), 'MSFT': _daily(AAPL_FINAL, NOV19_DATES)})
    q = scoring_service.download_quotes_yfinance_only(['AAPL', 'MSFT'], 'stocks')['AAPL']
    assert (q['last_price'], q['previous_close']) == (349.21, 351.91)


def test_yf_batch_single_symbol_multiindex_now_works(yf_download):
    # yfinance 0.2.66 keeps the (Ticker, Price) columns for one ticker. The
    # old single-symbol branch read data['Close'] (KeyError) and never
    # produced a quote.
    yf_download.frame = _grouped({'AAPL': _daily(AAPL_FINAL, NOV19_DATES)})
    q = scoring_service.download_quotes_yfinance_only(['AAPL'], 'stocks')['AAPL']
    assert (q['last_price'], q['previous_close'], q['volume']) == (349.21, 352.45, 40_469_518)
    assert yf_download.calls[0]['group_by'] == 'ticker'


def test_yf_batch_single_symbol_in_progress_gives_nothing(yf_download):
    yf_download.frame = _grouped({'AAPL': _daily(AAPL_IN_PROGRESS, NOV18_DATES)})
    assert scoring_service.download_quotes_yfinance_only(['AAPL'], 'stocks') == {}


def test_yf_batch_flat_columns_only_for_one_symbol(yf_download):
    yf_download.frame = _daily(AAPL_FINAL, NOV19_DATES)
    assert scoring_service.download_quotes_yfinance_only(['AAPL'], 'stocks')['AAPL']['last_price'] == 349.21
    # With several symbols a flat frame cannot be attributed to any of them.
    assert scoring_service.download_quotes_yfinance_only(['AAPL', 'MSFT'], 'stocks') == {}


def test_yf_batch_price_first_column_order(yf_download):
    # group_by='column' puts the field first: (Price, Ticker).
    frame = _grouped({'AAPL': _daily(AAPL_FINAL, NOV19_DATES), 'MSFT': _daily(AAPL_IN_PROGRESS, NOV18_DATES[-5:])})
    yf_download.frame = frame.swaplevel(0, 1, axis=1).sort_index(axis=1)
    out = scoring_service.download_quotes_yfinance_only(['AAPL', 'MSFT'], 'stocks')
    assert set(out) == {'AAPL'}
    assert out['AAPL']['previous_close'] == 352.45


def test_yf_batch_crypto_daily_bars_are_utc_days(yf_download, monkeypatch):
    # CoinGecko had nothing, so the crypto batch falls through to yf.download.
    monkeypatch.setattr(scoring_service, 'fetch_coingecko_snapshots', lambda symbols: {})
    yf_download.frame = _grouped({'BTC-USD': _daily(
        [(62987.3, 63210.6, 62382.5, 62758.1, 3.06e10), (62758.1, 63000.0, 62500.0, 62900.0, 1.1e10)],
        NOV19_DATES,
    )})
    q = scoring_service.download_quotes_yfinance_only(['BTC-USD'], 'crypto')['BTC-USD']
    assert (q['last_price'], q['previous_close']) == (62900.0, 62758.1)
    assert q['quote_time_utc'] == '2026-11-19T00:00:00+00:00'


def test_last_complete_daily_bar_rules():
    assert quote_math.last_complete_daily_bar(None) is None
    assert quote_math.last_complete_daily_bar(pd.DataFrame()) is None
    assert quote_math.last_complete_daily_bar(_daily(AAPL_IN_PROGRESS, NOV18_DATES)) is None
    assert quote_math.last_complete_daily_bar(_daily(AAPL_FINAL[:1], NOV19_DATES)) is None
    label, row, prev = quote_math.last_complete_daily_bar(_daily(AAPL_FINAL, NOV19_DATES))
    assert label == pd.Timestamp('2026-11-19') and row['Close'] == 349.21 and prev == 352.45
    assert quote_math.last_complete_daily_bar(pd.DataFrame({'Open': [1.0, 2.0]})) is None


# ---------------------------------------------------------------------------
# Session arithmetic
# ---------------------------------------------------------------------------
def test_previous_session_close_across_a_weekend(nov18):
    stamps = nov18['timestamp']
    closes = nov18['indicators']['quote'][0]['close']
    # Monday 11/16: the previous session is Friday 11/13.
    assert quote_math.previous_session_close(stamps, closes, NY, date(2026, 11, 16)) == pytest.approx(356.27, abs=BAR)
    # Wednesday 11/18: Tuesday 11/17; the 11/18 bar itself never counts.
    assert quote_math.previous_session_close(stamps, closes, NY, date(2026, 11, 18)) == pytest.approx(351.91, abs=BAR)
    # Nothing before the window.
    assert quote_math.previous_session_close(stamps, closes, NY, date(2026, 11, 12)) is None


def test_previous_session_close_across_a_holiday():
    # Thanksgiving, Thursday 2026-11-26: no session. Bars at 9:30 ET.
    stamps = [_utc(2026, 11, 24, 14, 30), _utc(2026, 11, 25, 14, 30), _utc(2026, 11, 27, 14, 30), _utc(2026, 11, 30, 14, 30)]
    closes = [101.0, 102.0, 103.0, 104.0]
    assert quote_math.previous_session_close(stamps, closes, NY, date(2026, 11, 27)) == 102.0
    # Monday after: Friday's half day, across the weekend.
    assert quote_math.previous_session_close(stamps, closes, NY, date(2026, 11, 30)) == 103.0
    # Any order; a null close is skipped for the one before it.
    assert quote_math.previous_session_close(stamps[::-1], closes[::-1], NY, date(2026, 11, 30)) == 103.0
    assert quote_math.previous_session_close(stamps, [101.0, 102.0, None, 104.0], NY, date(2026, 11, 30)) == 102.0
    assert quote_math.previous_session_close(stamps, closes, NY, None) is None


def test_session_dates_follow_the_exchange_timezone():
    btc_time = 1795054644  # 2026-11-19 02:17:24 UTC, from the Nov 18 spark reply
    assert quote_math.session_date(btc_time, NY) == date(2026, 11, 18)
    assert quote_math.session_date(btc_time, quote_math.CRYPTO_TZ) == date(2026, 11, 19)
    assert quote_math.session_date(btc_time, 'Not/AZone') == date(2026, 11, 18)
    for bad in (None, 'x', 0, -5, NAN, float('inf')):
        assert quote_math.session_date(bad) is None


def test_bar_date_handles_naive_aware_and_missing_stamps():
    assert quote_math.bar_date(None) is None
    assert quote_math.bar_date(pd.NaT) is None
    assert quote_math.bar_date(pd.Timestamp('2026-11-18')) == date(2026, 11, 18)  # naive = local
    assert quote_math.bar_date(pd.Timestamp('2026-11-19 02:17:24', tz='UTC')) == date(2026, 11, 18)
    assert quote_math.bar_date(pd.Timestamp('2026-11-19 02:17:24', tz='UTC'), 'UTC') == date(2026, 11, 19)
    assert quote_math.bar_date(date(2026, 11, 18)) == date(2026, 11, 18)
    assert quote_math.bar_date(1795012200) == date(2026, 11, 18)


def test_to_utc_iso_forms():
    assert quote_math.to_utc_iso(None) is None
    assert quote_math.to_utc_iso('') is None
    assert quote_math.to_utc_iso('not a time') is None
    assert quote_math.to_utc_iso(pd.NaT) is None
    assert quote_math.to_utc_iso(0) is None
    assert quote_math.to_utc_iso(1795035601) == '2026-11-18T21:00:01+00:00'
    assert quote_math.to_utc_iso(1795035601.75) == '2026-11-18T21:00:01+00:00'
    assert quote_math.to_utc_iso('2026-11-19T02:17:24Z') == '2026-11-19T02:17:24+00:00'
    assert quote_math.to_utc_iso('2026-11-19T02:17:24.123456+00:00') == '2026-11-19T02:17:24+00:00'
    # Naive values are exchange-local; a date is that session's midnight.
    assert quote_math.to_utc_iso('2023-12-29') == '2023-12-29T05:00:00+00:00'
    assert quote_math.to_utc_iso(date(2026, 7, 1)) == '2026-07-01T04:00:00+00:00'
    assert quote_math.to_utc_iso(pd.Timestamp('2026-11-18')) == '2026-11-18T05:00:00+00:00'
    assert quote_math.to_utc_iso(pd.Timestamp('2026-11-18'), 'UTC') == '2026-11-18T00:00:00+00:00'
    assert quote_math.to_utc_iso(pd.Timestamp('2026-11-18 15:55', tz=NY)) == '2026-11-18T20:55:00+00:00'


def test_price_amount_and_build_quote_edges():
    for bad in (None, '', 'abc', NAN, float('inf'), 0, -1):
        assert quote_math.price(bad) is None
    assert quote_math.price('352.45') == 352.45
    assert quote_math.amount(0) == 0.0
    assert quote_math.amount(-1) is None and quote_math.amount(NAN) is None

    assert quote_math.build_quote(last_price=0, previous_close=100, source='x') is None
    assert quote_math.build_quote(last_price=100, previous_close=None, source='x') is None
    q = quote_math.build_quote(last_price=101, previous_close=100, source='x', captured_at_utc='t')
    assert q['change_pct'] == 1.0
    assert (q['open'], q['day_high'], q['day_low'], q['volume'], q['quote_time_utc']) == (None,) * 5
    assert 'market_cap' not in q
    assert list(q) == list(quote_math.QUOTE_KEYS)
    assert quote_math.build_quote(last_price=101, previous_close=100, source='x', market_cap=None).get('market_cap', 'absent') == 'absent'
    assert quote_math.build_quote(last_price=101, previous_close=100, source='x', market_cap=0)['market_cap'] == 0.0
    assert quote_math.build_quote(last_price=101, previous_close=100, source='x', market_cap=NAN)['market_cap'] == 0.0


# ---------------------------------------------------------------------------
# yfinance single-symbol fallbacks (fetch_live_snapshot, fetch_intraday_shape)
# ---------------------------------------------------------------------------
def _intraday(minutes: int, start: str = '2026-11-18 09:30') -> pd.DataFrame:
    """Intraday bars for one session, tz-aware like yfinance's."""
    idx = pd.date_range(start, periods=4, freq=f'{minutes}min', tz=NY)
    return pd.DataFrame({
        'Open': [351.87, 352.40, 353.10, 352.90],
        'High': [352.50, 353.30, 356.42, 353.00],
        'Low': [351.47, 352.10, 352.80, 352.30],
        'Close': [352.41, 353.05, 352.95, 352.45],
        'Volume': [1_000_000.0, 600_000.0, 400_000.0, 2_000_000.0],
    }, index=idx)


class FakeTicker:
    """Stands in for yf.Ticker: fixed fast_info, history and metadata."""

    fast = None
    hist = None
    meta: dict = {}
    calls: list = []

    def __init__(self, symbol):
        self.symbol = symbol

    @property
    def fast_info(self):
        return type(self).fast

    def history(self, **kwargs):
        type(self).calls.append(kwargs)
        return type(self).hist

    @property
    def history_metadata(self):
        return dict(type(self).meta)


def _fast(**values):
    base = dict(last_price=0.0, previous_close=0.0, regular_market_previous_close=0.0, open=0.0,
                day_low=0.0, day_high=0.0, last_volume=0.0, market_cap=0.0)
    base.update(values)
    return SimpleNamespace(**base)


@pytest.fixture
def fake_yf(monkeypatch):
    class _T(FakeTicker):
        fast = _fast()
        hist = None
        meta = {}
        calls = []

    monkeypatch.setattr(scoring_service.yf, 'Ticker', _T)
    monkeypatch.setattr(scoring_service, 'provider_budget_allowance', lambda *a, **k: True)
    monkeypatch.setattr(scoring_service, 'warm_provider_session', lambda *a, **k: {})
    monkeypatch.setattr(scoring_service, 'download_quotes', lambda symbols, market='stocks': {})
    monkeypatch.setattr(scoring_service, 'mark_provider_failure', lambda *a, **k: None)
    return _T


def test_live_snapshot_fast_info_uses_regular_session_previous_close(fake_yf):
    # previous_close in yfinance 0.2.66 includes after-hours prints.
    fake_yf.fast = _fast(last_price=352.45, previous_close=352.10, regular_market_previous_close=351.91,
                         open=351.87, day_low=351.47, day_high=356.42, last_volume=33_393_854)
    q = scoring_service.fetch_live_snapshot('AAPL', 'stocks')
    assert q['source'] == 'yfinance-fast'
    assert q['previous_close'] == 351.91


def test_live_snapshot_1m_fallback_uses_the_previous_session(fake_yf):
    fake_yf.hist = _intraday(1)
    fake_yf.meta = {'range': '1d', 'chartPreviousClose': 351.91, 'regularMarketPrice': 352.45}
    q = scoring_service.fetch_live_snapshot('AAPL', 'stocks')
    assert q['source'] == 'yfinance-1m'
    # Not the first bar's close (352.41, this session's opening print).
    assert q['previous_close'] == 351.91
    assert q['last_price'] == 352.45
    assert q['change_pct'] == pytest.approx(0.1534, abs=0.001)
    assert (q['open'], q['day_high'], q['day_low']) == (351.87, 356.42, 351.47)
    assert q['volume'] == 4_000_000  # the session so far, not the last bar
    assert q['quote_time_utc'] == '2026-11-18T14:33:00+00:00'
    assert fake_yf.calls[-1] == {'period': '1d', 'interval': '1m', 'auto_adjust': False}


def test_live_snapshot_1m_fallback_without_range_1d_metadata_gives_nothing(fake_yf):
    fake_yf.hist = _intraday(1)
    # A 5d range's chartPreviousClose is the close before the window.
    fake_yf.meta = {'range': '5d', 'chartPreviousClose': 355.63}
    assert scoring_service.fetch_live_snapshot('AAPL', 'stocks') == {}


def test_intraday_shape_keeps_fast_info_previous_close_and_volume(fake_yf):
    fake_yf.fast = _fast(last_price=352.45, previous_close=352.10, regular_market_previous_close=351.91,
                         last_volume=33_393_854)
    fake_yf.hist = _intraday(5)
    fake_yf.meta = {'range': '1d', 'chartPreviousClose': 351.80}
    shape = scoring_service.fetch_intraday_shape('AAPL', 'stocks')
    assert fake_yf.calls[-1] == {'period': '1d', 'interval': '5m', 'auto_adjust': False}
    assert shape['previousClose'] == 351.91
    assert shape['volume'] == 33_393_854
    # The bars still fill what fast_info lacked.
    assert (shape['open'], shape['dayHigh'], shape['dayLow']) == (351.87, 356.42, 351.47)
    assert shape['currentPrice'] == 352.45


def test_intraday_shape_bars_fill_previous_close_from_metadata(fake_yf):
    fake_yf.hist = _intraday(5)
    fake_yf.meta = {'range': '1d', 'chartPreviousClose': 351.91}
    shape = scoring_service.fetch_intraday_shape('AAPL', 'stocks')
    assert shape['previousClose'] == 351.91  # not 352.41, the first bar's close
    assert shape['volume'] == 4_000_000
    assert shape['currentPrice'] == 352.45


def test_intraday_shape_never_uses_the_first_bar_as_previous_close(fake_yf):
    fake_yf.hist = _intraday(5)
    fake_yf.meta = {'range': '5d', 'chartPreviousClose': 355.63}
    shape = scoring_service.fetch_intraday_shape('AAPL', 'stocks')
    assert shape['previousClose'] == 0
    assert shape['open'] == 351.87


# ---------------------------------------------------------------------------
# Stooq, Finnhub, CryptoCompare, CoinPaprika
# ---------------------------------------------------------------------------
@pytest.fixture
def fresh_stooq(monkeypatch):
    fresh = {k: (None if k in ('last_block_reason', 'last_block_utc') else
                 False if k == 'half_open' else
                 0.0 if isinstance(v, float) else 0)
             for k, v in stooq._diag.items()}
    monkeypatch.setattr(stooq, '_diag', fresh)
    saved = providers_base._stats.pop('stooq', None)
    yield fresh
    providers_base._stats.pop('stooq', None)
    if saved is not None:
        providers_base._stats['stooq'] = saved


def _serve(monkeypatch, text: str, content_type: str = 'text/csv') -> None:
    def fake_get(url, params=None, headers=None, cfg=None):
        return ResilientResponse(status_code=200, text=text, headers={'Content-Type': content_type},
                                 url=url, attempts=1)
    monkeypatch.setattr(http_client, 'resilient_get', fake_get)


def test_stooq_pairs_the_newest_row_with_the_row_before(monkeypatch, fresh_stooq):
    _serve(monkeypatch, (FIXTURES / 'stooq_aapl_daily.csv').read_text())
    sym, q = stooq._fetch_one('AAPL', 'stocks', '2023-12-29T22:00:00+00:00', 'k' * 32)
    assert (q['last_price'], q['previous_close']) == (192.53, 193.58)
    assert (q['open'], q['day_high'], q['day_low'], q['volume']) == (192.90, 193.15, 191.72, 42_628_800)
    assert q['change_pct'] == pytest.approx((192.53 / 193.58 - 1) * 100, abs=1e-6)
    assert q['quote_time_utc'] == '2023-12-29T05:00:00+00:00'
    assert q['market_cap'] == 0.0 and q['source'] == 'stooq'
    assert fresh_stooq['successes'] == 1


def test_stooq_single_row_is_no_quote(monkeypatch, fresh_stooq):
    # It used to pair the row with itself: a made-up 0% change.
    _serve(monkeypatch, 'Date,Open,High,Low,Close,Volume\n2023-12-29,192.90,193.15,191.72,192.53,42628800\n')
    sym, q = stooq._fetch_one('AAPL', 'stocks', '2023-12-29T22:00:00+00:00', 'k' * 32)
    assert q is None
    assert fresh_stooq['parse_errors'] == 1 and fresh_stooq['successes'] == 0


def _finnhub(monkeypatch, body: dict) -> httpx.Client:
    monkeypatch.setattr(finnhub_provider, '_rate_limited_until', 0.0)
    monkeypatch.setattr(finnhub_provider, '_last_request_ts', 0.0)
    return httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)))


def test_finnhub_quote(monkeypatch):
    body = {'c': 352.45, 'pc': 351.91, 'o': 351.87, 'h': 356.42, 'l': 351.471, 't': 1795035601, 'd': 0.54, 'dp': 0.1534}
    q = finnhub_provider._fetch_one(_finnhub(monkeypatch, body), 'AAPL', 'token')
    assert (q['last_price'], q['previous_close'], q['open'], q['day_high'], q['day_low']) == (
        352.45, 351.91, 351.87, 356.42, 351.471)
    assert q['change_pct'] == pytest.approx(body['dp'], abs=0.001)
    assert q['volume'] == 0.0
    assert q['quote_time_utc'] == '2026-11-18T21:00:01+00:00'
    assert q['source'] == 'finnhub' and 'market_cap' not in q


def test_finnhub_zero_fields_stay_unknown(monkeypatch):
    # Before the open Finnhub sends o/h/l of 0. They used to be replaced
    # with the previous close and the current price.
    body = {'c': 100.0, 'pc': 99.0, 'o': 0, 'h': 0, 'l': 0, 't': 0}
    q = finnhub_provider._fetch_one(_finnhub(monkeypatch, body), 'AAPL', 'token')
    assert (q['open'], q['day_high'], q['day_low'], q['quote_time_utc']) == (None, None, None, None)
    assert finnhub_provider._fetch_one(_finnhub(monkeypatch, {'c': 100.0, 'pc': 0}), 'AAPL', 'token') is None


def test_cryptocompare_values_unchanged_plus_quote_time(monkeypatch):
    raw = {'RAW': {'BTC': {'USD': {
        'PRICE': 62758.1, 'OPEN24HOUR': 62987.285, 'HIGH24HOUR': 63210.613, 'LOW24HOUR': 62382.484,
        'VOLUME24HOUR': 12_345.6, 'MKTCAP': 1.24e12, 'LASTUPDATE': 1795054644,
    }}}}
    _serve(monkeypatch, json.dumps(raw), 'application/json')
    q = cryptocompare_provider.fetch(['BTC-USD'], 'crypto')['BTC-USD']
    # The same rolling 24-hour window as before 2C.
    assert (q['last_price'], q['previous_close'], q['open'], q['day_high'], q['day_low']) == (
        62758.1, 62987.285, 62987.285, 63210.613, 62382.484)
    assert (q['volume'], q['market_cap']) == (12_345.6, 1.24e12)
    assert q['quote_time_utc'] == '2026-11-19T02:17:24+00:00'
    assert q['change_pct'] == pytest.approx((62758.1 / 62987.285 - 1) * 100, abs=1e-6)
    assert q['source'] == 'cryptocompare'


def test_coinpaprika_values_unchanged_plus_quote_time(monkeypatch):
    payload = {'id': 'btc-bitcoin', 'last_updated': '2026-11-19T02:17:24Z', 'quotes': {'USD': {
        'price': 62758.1, 'percent_change_24h': -0.36, 'volume_24h': 3.06e10, 'market_cap': 1.24e12}}}
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)))
    monkeypatch.setattr(coinpaprika_provider, '_HTTPX_CLIENT', client)
    monkeypatch.setattr(coinpaprika_provider, '_throttle', lambda: None)
    monkeypatch.setattr(coinpaprika_provider, '_load_id_cache', lambda: {'BTC': 'btc-bitcoin'})
    q = coinpaprika_provider.fetch(['BTC-USD'], 'crypto')['BTC-USD']
    prev = 62758.1 / (1.0 - 0.0036)
    assert q['last_price'] == 62758.1
    assert q['previous_close'] == pytest.approx(prev) and q['open'] == pytest.approx(prev)
    assert q['day_low'] == pytest.approx(62758.1 * (1 - 0.36 / 200))
    assert q['day_high'] == pytest.approx(62758.1 * (1 + 0.36 / 200))
    assert (q['volume'], q['market_cap']) == (3.06e10, 1.24e12)
    assert q['quote_time_utc'] == '2026-11-19T02:17:24+00:00'
    assert q['change_pct'] == pytest.approx(-0.36, abs=1e-6)


def test_coingecko_batch_values_unchanged_plus_quote_time(monkeypatch):
    snap = {'BTC-USD': {'currentPrice': 62758.1, 'previousClose': 62987.285, 'open': 62987.285,
                        'dayLow': 62382.484, 'dayHigh': 63210.613, 'volume': 3.06e10, 'marketCap': 1.24e12,
                        'lastUpdated': '2026-11-19T02:17:24.512Z'}}
    monkeypatch.setattr(scoring_service, 'fetch_coingecko_snapshots', lambda symbols: snap)
    q = scoring_service.download_quotes_yfinance_only(['BTC-USD'], 'crypto')['BTC-USD']
    assert (q['last_price'], q['previous_close'], q['open']) == (62758.1, 62987.285, 62987.285)
    assert (q['day_low'], q['day_high'], q['volume'], q['market_cap']) == (62382.484, 63210.613, 3.06e10, 1.24e12)
    assert q['quote_time_utc'] == '2026-11-19T02:17:24+00:00'
    assert q['source'] == 'coingecko'


# ---------------------------------------------------------------------------
# quote_time_utc travels: cache, scored rows, normalized rows
# ---------------------------------------------------------------------------
@pytest.fixture
def quote_cache(monkeypatch, tmp_path):
    from app.config import settings
    from app.services import quote_cache as cache
    from app.services.memory_store import memory_store

    monkeypatch.setattr(settings, 'memory_cache_enabled', True)
    monkeypatch.setattr(settings, 'memory_cache_mode', 'enabled')
    monkeypatch.setattr(settings, 'cache_enable_quotes', True)
    monkeypatch.setattr(settings, 'cache_mode_quotes', 'enabled')
    monkeypatch.setattr(settings, 'cache_ttl_jitter_percent', 0.0)
    monkeypatch.setattr(settings, 'quote_cache_ttl_seconds', 60)
    monkeypatch.setattr(cache, '_SHARD_DIR', tmp_path / 'quote_cache')
    monkeypatch.setattr(cache, '_LEGACY_FILE', tmp_path / 'quote_cache.json')
    monkeypatch.setattr(cache, '_MIGRATION_DONE', True)
    monkeypatch.setattr(cache, '_DEBOUNCE_S', 1.0)
    memory_store.invalidate_domain('quotes')
    for value in (cache._SHARD_MEM, cache._SHARD_LOADED, cache._SHARD_VERSION,
                  cache._DIRTY_SINCE, cache._DIRTY_FAILURES, cache._DIRTY_RETRY_AT):
        value.clear()
    cache._DIRTY_SHARDS.clear()
    cache._PERSISTING_SHARDS.clear()
    yield cache
    memory_store.invalidate_domain('quotes')


def test_save_quote_keeps_quote_time(quote_cache):
    quote_cache.save_quote('AAPL', {'last_price': 352.45, 'previous_close': 351.91,
                                    'quote_time_utc': '2026-11-18T21:00:01+00:00'})
    quote_cache.save_quote('MSFT', {'last_price': 508.29, 'previous_close': 506.24})
    assert quote_cache.get_cached_quote('AAPL')['quote_time_utc'] == '2026-11-18T21:00:01+00:00'
    assert 'quote_time_utc' not in quote_cache.get_cached_quote('MSFT')
    assert quote_cache._SHARD_MEM['A']['AAPL']['quote_time_utc'] == '2026-11-18T21:00:01+00:00'


@pytest.fixture
def no_daily_history(monkeypatch, tmp_path):
    """Keep score_symbol_rows away from the daily-history cache.

    Its prefetch pool loops forever on non-daemon threads (the Phase 4
    lifecycle work), which would keep pytest from exiting, and its flush
    writes into data/daily_history_cache.
    """
    import threading
    from app.services import daily_history_service as dh

    monkeypatch.setattr(dh, '_SHARD_DIR', tmp_path / 'daily_history_cache')
    monkeypatch.setattr(dh, '_prefetch_started', True)
    monkeypatch.setattr(dh, 'prefetch_daily_history', lambda *a, **k: None)
    monkeypatch.setattr(dh, 'await_prefetch_for_batch', lambda *a, **k: 0)
    monkeypatch.setattr(dh, 'get_daily_history', lambda *a, **k: None)
    yield
    assert not [t.name for t in threading.enumerate() if t.name.startswith('dh-worker')]


def test_scored_rows_carry_quote_as_of_utc(monkeypatch, no_daily_history):
    live = quote_math.build_quote(last_price=352.45, previous_close=351.91, open=351.87, day_high=356.42,
                                  day_low=351.471, volume=33_393_854, quote_time_utc=1795035601,
                                  captured_at_utc='2026-11-18T21:05:00+00:00', source='yahoo-chart')
    cached = {'last_price': 508.29, 'previous_close': 506.24, 'captured_at_utc': '2026-11-18T21:04:00+00:00',
              'source': 'yahoo-chart', 'quote_time_utc': '2026-11-18T21:00:00+00:00'}
    monkeypatch.setattr(scoring_service, 'download_quotes',
                        lambda symbols, market='stocks': {s: dict(live) for s in symbols if s == 'AAPL'})
    monkeypatch.setattr(scoring_service, 'save_quote', lambda *a, **k: None)
    monkeypatch.setattr(scoring_service, 'get_cached_quote', lambda sym: dict(cached) if sym == 'MSFT' else None)
    monkeypatch.setattr(scoring_service, 'cached_quote_is_usable', lambda q: True)
    monkeypatch.setattr(scoring_service, 'quote_age_seconds', lambda q: 60)
    monkeypatch.setattr(scoring_service, 'fetch_intraday_shape', lambda sym, market='stocks': {})
    rows = scoring_service.score_symbol_rows([
        {'symbol': 'AAPL', 'name': 'Apple Inc.', 'exchange': 'NMS', 'asset_class': 'stock', 'market': 'stocks'},
        {'symbol': 'MSFT', 'name': 'Microsoft', 'exchange': 'NMS', 'asset_class': 'stock', 'market': 'stocks'},
    ])
    by_symbol = {r['symbol']: r for r in rows}
    assert by_symbol['AAPL']['quote_as_of_utc'] == '2026-11-18T21:00:01+00:00'
    assert by_symbol['AAPL']['as_of_utc'] == '2026-11-18T21:05:00+00:00'  # unchanged meaning
    assert by_symbol['MSFT']['quote_as_of_utc'] == '2026-11-18T21:00:00+00:00'


def test_normalize_carries_quote_as_of_utc():
    from app.utils import normalize

    out = normalize.normalize_result_row({'symbol': 'AAPL', 'as_of_utc': '2026-11-18T21:05:00+00:00',
                                          'quote_time_utc': '2026-11-18T21:00:01+00:00'})
    assert out['quote_as_of_utc'] == '2026-11-18T21:00:01+00:00'
    assert out['as_of_utc'] == '2026-11-18T21:05:00+00:00'
    assert normalize.normalize_result_row({'symbol': 'AAPL', 'quote_as_of_utc': 'A', 'quote_time_utc': 'B'})['quote_as_of_utc'] == 'A'
    assert normalize.normalize_result_row({'symbol': 'AAPL'})['quote_as_of_utc'] is None
    assert 'quote_as_of_utc' not in normalize.get_defaulted_fields_snapshot()['counts']
