"""
Direct Yahoo Finance chart-API provider.

Uses the public endpoint:
  GET https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=5d&interval=1d

No crumb required (unlike fast_info / .info on some yfinance versions).
Used as a redundancy layer when `yfinance` fast paths flap due to
rate-limit / crumb errors.

Phase 2C: `last_price` is meta.regularMarketPrice and `previous_close` is
the last close dated before that price's session. With range=5d,
meta.chartPreviousClose is the close before the 5-day window, not the
previous session's, so it is used only when the window holds no earlier
bar (always the case for range=1d). Open, high, low and volume come from
the bar dated the same session.
"""
from __future__ import annotations

import logging

import requests

from app.services.providers.base import (
    count_error,
    count_rate_limit,
    count_timeout,
)
from app.services import quote_math
from app.utils.time import utcnowiso

log = logging.getLogger('app.providers.yahoo_chart')

UA = 'Mozilla/5.0 (compatible; MarketRefinementDashboard/1.0)'
_BASE = 'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}'
_PARAMS = {'range': '5d', 'interval': '1d'}
_HEADERS = {'User-Agent': UA, 'Accept': 'application/json'}
_TIMEOUT = 6.0

# Phase 26.60: persistent module-level executor (mirrors stooq_provider).
# Replaces the per-batch ThreadPoolExecutor(max_workers=10) that was
# created and abandoned on every fetch() call. Sized at 2x the prior
# per-batch cap (10 -> 20) so hung sockets from previous batches don't
# starve new work — the pool naturally heals as OS-level socket
# timeouts release worker slots.
from concurrent.futures import ThreadPoolExecutor as _YahooChartTpe
_POOL = _YahooChartTpe(max_workers=20, thread_name_prefix='yahoo-chart')


def _fetch_one(sym: str, captured_at: str) -> tuple[str, dict | None]:
    """Fetch a single symbol from the Yahoo chart endpoint."""
    try:
        url = _BASE.format(symbol=sym)
        # Phase 26.18.c: resilient_get gives us retries + backoff + UA
        # rotation + Retry-After honoring on transient 5xx/429. Yahoo's
        # chart endpoint is particularly prone to 429-bursts during
        # market-open peaks, so this is high-value here.
        from app.services.http_client import resilient_get, ResilientGetConfig
        cfg = ResilientGetConfig(
            max_attempts=3,
            connect_timeout=3.0,
            read_timeout=_TIMEOUT,
            retry_after_cap_seconds=3.0,
        )
        res = resilient_get(url, params=_PARAMS, headers={'Accept': 'application/json'}, cfg=cfg)
        if res.status_code == 0 and res.error:
            # All retries exhausted on transient transport failure.
            err_l = res.error.lower()
            if 'timeout' in err_l:
                count_timeout('yahoo-chart', f'{sym}: {res.error}')
            else:
                count_error('yahoo-chart', f'{sym}: {res.error}')
            return sym, None
        if res.status_code == 429:
            count_rate_limit('yahoo-chart', f'HTTP 429 on {sym}')
            return sym, None
        if res.status_code != 200:
            return sym, None
        try:
            data = res.json() or {}
        except Exception as exc:  # noqa: BLE001
            count_error('yahoo-chart', f'{sym}: json_decode {exc}')
            return sym, None
        chart = (data.get('chart') or {})
        if chart.get('error'):
            return sym, None
        results = chart.get('result') or []
        if not results:
            return sym, None
        return sym, parse_chart_result(results[0], captured_at=captured_at)
    except Exception as exc:  # noqa: BLE001
        # Defensive net for any other unexpected error reaching here
        # (resilient_get itself shouldn't raise, but the parsing block
        # above could).
        count_error('yahoo-chart', f'{sym}: {exc}')
        log.debug('yahoo-chart fetch failed for %s: %s', sym, exc)
        return sym, None


def parse_chart_result(r0: dict, *, captured_at: str | None = None) -> dict | None:
    """One `chart.result[]` entry -> quote dict, or None without a usable pair."""
    meta = r0.get('meta') or {}
    tz = meta.get('exchangeTimezoneName') or quote_math.DEFAULT_TZ
    stamps = list(r0.get('timestamp') or [])
    bars = ((r0.get('indicators') or {}).get('quote') or [{}])[0] or {}

    def col(key: str) -> list:
        # One value per timestamp, whatever length Yahoo sent.
        values = list(bars.get(key) or [])
        return (values + [None] * len(stamps))[:len(stamps)]

    opens, highs, lows, closes, volumes = (col(k) for k in ('open', 'high', 'low', 'close', 'volume'))
    days = [quote_math.bar_date(t, tz) for t in stamps]

    last = quote_math.price(meta.get('regularMarketPrice'))
    quote_time = meta.get('regularMarketTime')
    session = quote_math.session_date(quote_time, tz)
    if last is None:
        # No meta price: fall back to the newest bar that has a close.
        newest = max((i for i, c in enumerate(closes) if quote_math.price(c) is not None),
                     key=lambda i: stamps[i], default=None)
        if newest is None:
            return None
        last, quote_time, session = float(closes[newest]), stamps[newest], days[newest]
    if session is None:
        dated = [d for d in days if d is not None]
        session = max(dated) if dated else None
    if session is None:
        return None

    prev = quote_math.previous_session_close(stamps, closes, tz, session)
    if prev is None and not any(d is not None and d < session for d in days):
        # The window starts at this session, so the close before the
        # window *is* the previous session's close. For range=1d that is
        # always so; for range=5d it only happens for a new listing.
        prev = quote_math.price(meta.get('chartPreviousClose'))
    if prev is None:
        return None

    idx = next((i for i in range(len(days) - 1, -1, -1) if days[i] == session), None)

    def at(values: list):
        return values[idx] if idx is not None else None

    return quote_math.build_quote(
        last_price=last,
        previous_close=prev,
        open=at(opens),
        day_high=quote_math.price(at(highs)) or meta.get('regularMarketDayHigh'),
        day_low=quote_math.price(at(lows)) or meta.get('regularMarketDayLow'),
        volume=at(volumes) if quote_math.amount(at(volumes)) is not None else meta.get('regularMarketVolume'),
        quote_time_utc=quote_time,
        captured_at_utc=captured_at,
        source='yahoo-chart',
        market_cap=0.0,
        tz_name=tz,
    )


def fetch(symbols: list[str], market: str) -> dict[str, dict]:
    if not symbols or market == 'crypto':
        return {}
    out: dict[str, dict] = {}
    captured_at = utcnowiso()
    # Parallelize across symbols — the endpoint is independent per-symbol
    # and fast (<200 ms each).  When the upstream yfinance batch returns
    # sparse data (often the case on residential ISPs), this cascade is
    # invoked with up to 25 symbols and used to serialize at 5 s per batch.
    #
    # Phase 26.60: use the persistent module-level pool `_POOL`.
    # `as_completed(timeout=20)` still enforces the batch deadline;
    # hung futures continue in the background pool without blocking new
    # submissions.
    from concurrent.futures import (
        as_completed,
        TimeoutError as _FuturesTimeoutError,
    )
    futures = [_POOL.submit(_fetch_one, sym, captured_at) for sym in symbols]
    try:
        for fut in as_completed(futures, timeout=20):
            try:
                sym, payload = fut.result()
                if payload:
                    out[sym] = payload
            except Exception:
                continue
    except _FuturesTimeoutError:
        log.debug('yahoo-chart fetch: as_completed timed out at 20s; %d/%d done',
                  len(out), len(futures))
    return out


def _legacy_sequential_fetch(symbols: list[str], market: str) -> dict[str, dict]:
    """Kept for fallback only — the new fetch() is the parallel version."""
    if not symbols or market == 'crypto':
        return {}
    out: dict[str, dict] = {}
    captured_at = utcnowiso()
    for sym in symbols:
        _, payload = _fetch_one(sym, captured_at)
        if payload:
            out[sym] = payload
    return out
