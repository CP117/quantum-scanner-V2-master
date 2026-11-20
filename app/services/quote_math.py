"""Session-aware quote arithmetic (Phase 2C).

A quote's daily change is ``last_price`` against the previous session's
official close, with open, high, low and volume taken from the same
session as ``last_price``. The code this replaces mixed days in two ways:

* Yahoo's chart endpoint with ``range=5d`` reports ``chartPreviousClose``
  as the close *before the 5-day window*, not yesterday's close. AAPL on
  November 18, 2026 showed -0.89% instead of +0.15%.
* After the close, the current day's daily bar can come back with a null
  close. ``Close.dropna().iloc[-1]`` then quietly returned yesterday's
  close as the "last price", next to today's open, high, low and volume.

previous_close is the official close as printed. On an ex-dividend date
the exchanges, and Yahoo's displayed change, measure from that close minus
the dividend, so change_pct differs by the dividend: MSFT on November 19,
2026 (ex-dividend $0.98) is -1.97% here and -1.78% on Yahoo. Adjusting
would need a corporate-actions source that every provider path can use.

Everything here is pure: no I/O, no clock reads, so the rules can be
tested from fixtures.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timezone
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from app.utils.time import utcnowiso

DEFAULT_TZ = 'America/New_York'
CRYPTO_TZ = 'UTC'

# The keys every quote row carries: the ten the cascade already produced,
# plus change_pct and quote_time_utc (new in Phase 2C).
QUOTE_KEYS = (
    'last_price', 'previous_close', 'change_pct', 'open', 'day_high',
    'day_low', 'volume', 'quote_time_utc', 'captured_at_utc', 'source',
    'provider_outcome', 'preview_only',
)


def _zone(tz_name: str | None) -> ZoneInfo:
    try:
        return ZoneInfo(tz_name or DEFAULT_TZ)
    except Exception:  # noqa: BLE001 - unknown name from a provider payload
        return ZoneInfo(DEFAULT_TZ)


def price(value: Any) -> float | None:
    """A usable price: finite and above zero, else None."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f) or f <= 0:
        return None
    return f


def amount(value: Any) -> float | None:
    """A usable volume: finite and zero or more, else None."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f) or f < 0:
        return None
    return f


def session_date(epoch_seconds: Any, tz_name: str | None = DEFAULT_TZ) -> date | None:
    """The calendar date of an instant in the exchange's timezone.

    Stocks default to America/New_York; pass ``CRYPTO_TZ`` (UTC) for
    crypto, whose daily sessions run midnight to midnight UTC.
    """
    try:
        ts = float(epoch_seconds)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(ts) or ts <= 0:
        return None
    return datetime.fromtimestamp(ts, tz=_zone(tz_name)).date()


def bar_date(stamp: Any, tz_name: str | None = DEFAULT_TZ) -> date | None:
    """Session date of a bar stamp: epoch seconds, a datetime or a date.

    A naive datetime is taken as already exchange-local, which is how
    ``yf.download`` labels daily bars.
    """
    if stamp is None:
        return None
    if isinstance(stamp, datetime):
        if stamp != stamp:  # pandas NaT
            return None
        if stamp.tzinfo is None:
            return stamp.date()
        return stamp.astimezone(_zone(tz_name)).date()
    if isinstance(stamp, date):
        return stamp
    return session_date(stamp, tz_name)


def to_utc_iso(stamp: Any, tz_name: str | None = DEFAULT_TZ) -> str | None:
    """ISO-8601 UTC string for epoch seconds, a datetime, a date or an ISO string.

    A naive datetime is localised to ``tz_name`` first; a bare date (or a
    date-only string) means local midnight of that session.
    """
    if stamp is None or stamp == '':
        return None
    if isinstance(stamp, str):
        try:
            parsed = datetime.fromisoformat(stamp.strip().replace('Z', '+00:00'))
        except ValueError:
            return None
        return to_utc_iso(parsed, tz_name)
    if isinstance(stamp, date) and not isinstance(stamp, datetime):
        stamp = datetime(stamp.year, stamp.month, stamp.day)
    if isinstance(stamp, datetime):
        if stamp != stamp:  # pandas NaT
            return None
        if hasattr(stamp, 'to_pydatetime'):  # pandas Timestamp
            stamp = stamp.to_pydatetime()
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=_zone(tz_name))
        return stamp.astimezone(timezone.utc).replace(microsecond=0).isoformat()
    try:
        ts = float(stamp)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(ts) or ts <= 0:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(microsecond=0).isoformat()


def previous_session_close(
    timestamps: Iterable[Any],
    closes: Iterable[Any],
    tz_name: str | None,
    session: date | None,
) -> float | None:
    """The last non-null close whose bar date is before ``session``.

    Bars may arrive in any order; the latest qualifying bar wins.
    """
    if session is None:
        return None
    best_date: date | None = None
    best: float | None = None
    for stamp, close in zip(timestamps or [], closes or []):
        d = bar_date(stamp, tz_name)
        value = price(close)
        if d is None or value is None or d >= session:
            continue
        if best_date is None or d >= best_date:
            best_date, best = d, value
    return best


def change_pct(last_price: Any, previous_close: Any) -> float | None:
    last, prev = price(last_price), price(previous_close)
    if last is None or prev is None:
        return None
    return round((last - prev) / prev * 100.0, 6)


def build_quote(
    *,
    last_price: Any,
    previous_close: Any,
    source: str,
    open: Any = None,  # noqa: A002 - matches the quote key
    day_high: Any = None,
    day_low: Any = None,
    volume: Any = None,
    quote_time_utc: Any = None,
    captured_at_utc: str | None = None,
    provider_outcome: str = 'live_success',
    preview_only: bool = False,
    market_cap: Any = None,
    tz_name: str | None = DEFAULT_TZ,
) -> dict | None:
    """The canonical quote dict, or None when there is no usable pair.

    ``open``, ``day_high``, ``day_low`` and ``volume`` stay None when the
    provider has no value for this session; callers already treat a falsy
    value as missing. ``market_cap`` is added only when given, so each
    provider keeps the keys it produced before.
    """
    last, prev = price(last_price), price(previous_close)
    if last is None or prev is None:
        return None
    quote = {
        'last_price': last,
        'previous_close': prev,
        'change_pct': change_pct(last, prev),
        'open': price(open),
        'day_high': price(day_high),
        'day_low': price(day_low),
        'volume': amount(volume),
        'quote_time_utc': to_utc_iso(quote_time_utc, tz_name),
        'captured_at_utc': captured_at_utc or utcnowiso(),
        'source': source,
        'provider_outcome': provider_outcome,
        'preview_only': bool(preview_only),
    }
    if market_cap is not None:
        quote['market_cap'] = amount(market_cap) or 0.0
    return quote


def _cell(row: Any, key: str) -> Any:
    try:
        return row[key]
    except Exception:  # noqa: BLE001 - missing column
        return None


def last_complete_daily_bar(frame: Any) -> tuple[Any, Any, float] | None:
    """(index label, row, previous close) for a daily OHLCV DataFrame.

    The row is the last one, and only if its Close is present. A last row
    without a Close is an in-progress session: the answer is None rather
    than yesterday's bar, so the next provider supplies the quote. The
    previous close comes from the nearest earlier row that has one.
    """
    if frame is None or getattr(frame, 'empty', True):
        return None
    try:
        closes = frame['Close']
    except Exception:  # noqa: BLE001 - no Close column
        return None
    if len(closes) == 0 or price(closes.iloc[-1]) is None:
        return None
    for i in range(len(closes) - 2, -1, -1):
        prev = price(closes.iloc[i])
        if prev is not None:
            return frame.index[-1], frame.iloc[-1], prev
    return None


def quote_from_daily_frame(
    frame: Any,
    *,
    source: str,
    captured_at_utc: str | None = None,
    tz_name: str | None = DEFAULT_TZ,
) -> dict | None:
    """A quote from a yfinance-style daily frame (Open/High/Low/Close/Volume).

    ``quote_time_utc`` is the last bar's timestamp. Daily bars are labelled
    with the session date only, so for them it marks the session (local
    midnight), not a trade time.
    """
    picked = last_complete_daily_bar(frame)
    if picked is None:
        return None
    label, row, prev = picked
    return build_quote(
        last_price=_cell(row, 'Close'),
        previous_close=prev,
        open=_cell(row, 'Open'),
        day_high=_cell(row, 'High'),
        day_low=_cell(row, 'Low'),
        volume=_cell(row, 'Volume'),
        quote_time_utc=label,
        captured_at_utc=captured_at_utc,
        source=source,
        tz_name=tz_name,
    )
