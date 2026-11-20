"""
Stooq EOD CSV provider. Off by default; opt in with an API key.

CSV endpoint:
  https://stooq.com/q/d/l/?s={symbol}.us&i=d&apikey={key}

Since March 2026 Stooq needs an API key for CSV downloads. Without one it
answers with a "Get your apikey" page instead of data, scripted requests
are often refused with "Access denied", and a spent daily quota comes back
as HTTP 200 with the text "Exceeded the daily hits limit". Some networks
cannot reach the host at all (connect timeouts on port 443). Treat it as
best-effort even with a key.

download_quotes() adds Stooq to the stock cascade only when availability()
allows it, which needs all three of:
  * MRD_STOOQ_ENABLED=1 (default 0);
  * a key: the "stooq" slot of the dashboard's API-key panel
    (data/user_api_keys.json) or STOOQ_API_KEY;
  * the dedicated circuit below to be closed.
fetch() repeats those checks, so calling it while Stooq is off does nothing,
and the worker pool is created on first use, so no Stooq thread exists
until then.

The key is sent as the `apikey` query parameter and masked as `apikey=***`
in every error string and log line this module writes.

Every reply is classified before parsing (classify_response). A refusal, an
HTML page or a quota message is a provider failure: it opens the circuit
and ends the batch, and the symbol is not blamed. A good reply yields the
two most recent daily closes as `last_price` and `previous_close`.
"""
from __future__ import annotations

import csv
import io
import logging
import os
import re
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as _FuturesTimeoutError,
    as_completed,
)
from datetime import datetime, timedelta, timezone
from threading import Lock

import requests

from app.services import quote_math
from app.services.providers.base import count_error, count_rate_limit, count_timeout
from app.utils.time import utcnowiso

log = logging.getLogger('app.providers.stooq')

_BASE = 'https://stooq.com/q/d/l/'
_HEADERS = {'User-Agent': 'Mozilla/5.0 (compatible; MarketRefinementDashboard/1.0)'}
_TIMEOUT = 3.0  # tightened from 6s — burning 6s per timeout on 943 attempts = 1.5 hrs of wasted clock

# Phase 2B: opt-in switch, read on every call.
_ENABLED_ENV = 'MRD_STOOQ_ENABLED'
_TRUTHY = {'1', 'true', 'yes', 'on'}

# Phase 26.60 replaced a fresh pool per batch with one persistent executor
# (thread churn). Phase 2B shrinks it from 20 workers to 2 and creates it on
# first use: Stooq is best-effort, and while it is off no Stooq thread
# should exist at all. Each batch still gets _BATCH_BUDGET_SECONDS; what has
# not started by then is cancelled (see fetch).
_POOL_WORKERS = 2
_BATCH_BUDGET_SECONDS = 20
_pool: ThreadPoolExecutor | None = None
_pool_lock = Lock()


def _get_pool() -> ThreadPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=_POOL_WORKERS, thread_name_prefix='stooq')
        return _pool


# Diagnostics so the operator can see WHY hit-rate is zero
# (timeouts vs http_errors vs no_data vs parse_errors).
_lock = Lock()
_diag = {
    'attempts': 0,
    'successes': 0,
    'timeouts': 0,
    'http_errors': 0,
    'no_data': 0,
    'parse_errors': 0,
    'network_errors': 0,
    # Phase 2B: provider-level refusals. None of these is a symbol failure.
    'blocked': 0,        # HTML page or "Access denied" instead of CSV
    'key_rejected': 0,   # "Get your apikey" page: the key is missing or not accepted
    'quota_hits': 0,     # "Exceeded the daily hits limit"
    'rate_limited': 0,   # HTTP 429 without the quota text
    'bad_payload': 0,    # 200 OK, but not the CSV the parser expects
    'skipped_circuit_open': 0,  # queued symbols dropped after the circuit opened
    'last_block_reason': None,  # quota | http_429 | key_rejected | blocked
    'last_block_utc': None,
    'consecutive_failures': 0,
    'circuit_open_until': 0.0,
    'circuit_trip_count': 0,
    'next_cooldown_seconds': 0.0,
    # Phase 26.9: half-open probe state. After cooldown elapses we flag this
    # `True` so the next failure trips immediately (with exponential backoff)
    # instead of accumulating another full _FAIL_THRESHOLD batch of failures.
    'half_open': False,
}

# Circuit breaker: after this many consecutive failures, stop trying for the
# duration of the cooldown so we don't keep burning a 3s round-trip per symbol.
# Lowered from 50 to 20 (Phase 26.7) so a fully-broken Stooq trips ~2.5x
# faster on each probationary re-open, reducing wasted timeouts/min.
_FAIL_THRESHOLD = 20
# Exponential cooldown — every subsequent trip doubles the cooldown so a
# persistently-unreachable Stooq host (typical from datacenter IPs) backs off
# to a 24-hr cadence instead of repeatedly tripping every 30 min.
_COOLDOWN_LADDER_SECONDS = [
    30 * 60,        # 1st trip: 30 min
    60 * 60,        # 2nd trip: 1 hr
    2 * 60 * 60,    # 3rd trip: 2 hr
    6 * 60 * 60,    # 4th trip: 6 hr
    24 * 60 * 60,   # 5th+ trips: 24 hr (cap)
]


def _next_cooldown_for_trip(trip_count: int) -> float:
    idx = min(max(trip_count, 1), len(_COOLDOWN_LADDER_SECONDS)) - 1
    return float(_COOLDOWN_LADDER_SECONDS[idx])


def _mono() -> float:
    """Monotonic clock for the circuit. Tests replace it."""
    return time.monotonic()


def _utcnow() -> datetime:
    """Wall clock for the quota reset at UTC midnight. Tests replace it."""
    return datetime.now(timezone.utc)


def _seconds_until_utc_midnight(now: datetime) -> float:
    nxt = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1.0, (nxt - now).total_seconds())


def _circuit_open_at(now_mono: float) -> bool:
    """Caller holds _lock."""
    until = _diag['circuit_open_until']
    return bool(until and now_mono < until)


# --- Phase 2B: opt-in gate ----------------------------------------------------

def is_enabled() -> bool:
    return (os.environ.get(_ENABLED_ENV) or '0').strip().lower() in _TRUTHY


def _api_key() -> str:
    """The "stooq" key from the API-key store, else STOOQ_API_KEY."""
    try:
        from app.services import api_keys
        return (api_keys.get('stooq') or '').strip()
    except Exception:  # noqa: BLE001
        return ''


def availability() -> tuple[bool, str | None]:
    """Whether the cascade may call fetch() now.

    Returns (True, None), or (False, reason) where reason is 'disabled'
    (MRD_STOOQ_ENABLED is not 1), 'no_api_key' or 'circuit_open'.
    """
    if not is_enabled():
        return False, 'disabled'
    if not _api_key():
        return False, 'no_api_key'
    now_mono = _mono()
    with _lock:
        if _circuit_open_at(now_mono):
            return False, 'circuit_open'
    return True, None


# --- Phase 2B: keep the key out of messages and logs -----------------------------

_APIKEY_RE = re.compile(r'(apikey=)[^&\s\'"<>)]+', re.IGNORECASE)


def redact(text: object, key: str | None = None) -> str:
    """Mask the API key in a URL, error string or log line."""
    out = str(text or '')
    if key and len(key) >= 8:
        out = out.replace(key, '***')
    return _APIKEY_RE.sub(r'\1***', out)


class _RedactApiKey(logging.Filter):
    """Masks apikey=... in log records. urllib3 logs every request URL with
    its query string at DEBUG, so the filter also sits on its logger."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return True
        if 'apikey=' in msg.lower():
            record.msg = redact(msg)
            record.args = ()
        return True


_REDACTOR = _RedactApiKey()
for _logger_name in ('urllib3.connectionpool', 'app.http_client', log.name):
    logging.getLogger(_logger_name).addFilter(_REDACTOR)


# --- Phase 2B: classify a reply before parsing ----------------------------------

_QUOTA_TEXT = 'exceeded the daily hits limit'
_KEY_PAGE_TEXT = 'get your apikey'
_DENIED_TEXT = 'access denied'
_CSV_HEADER = ['date', 'open', 'high', 'low', 'close']


def classify_response(status: int, content_type: str | None, text: str | None) -> tuple[str, str | None]:
    """Sort a Stooq reply into (outcome, reason).

    outcome        reason          effect
    ok             None            parse it
    not_found      'no_data'       symbol unknown to Stooq; counts toward the threshold
    rate_limited   'quota'         circuit open until the next UTC midnight
    rate_limited   'http_429'      circuit open, next ladder step
    provider_down  'key_rejected'  "Get your apikey" page; circuit open, next ladder step
    provider_down  'blocked'       HTML or "Access denied"; circuit open, next ladder step
    http_error     'http_<code>'   counts toward the threshold
    bad_payload    'empty_body' or 'no_csv_header'   counts toward the threshold

    Only ok and not_found say anything about the symbol itself.
    """
    body = (text or '').lstrip('\ufeff').strip()
    head = body[:4096].lower()
    ctype = (content_type or '').lower()
    if _QUOTA_TEXT in head:
        return 'rate_limited', 'quota'
    if status == 429:
        return 'rate_limited', 'http_429'
    if status not in (200, 401, 403):
        return 'http_error', f'http_{status}'
    if _KEY_PAGE_TEXT in head:
        return 'provider_down', 'key_rejected'
    if status != 200 or 'text/html' in ctype or body.startswith('<') or _DENIED_TEXT in head:
        return 'provider_down', 'blocked'
    if not body:
        return 'bad_payload', 'empty_body'
    if head.startswith('no data'):
        return 'not_found', 'no_data'
    first_line = head.splitlines()[0]
    if [c.strip() for c in first_line.split(',')][:len(_CSV_HEADER)] != _CSV_HEADER:
        return 'bad_payload', 'no_csv_header'
    return 'ok', None


_BLOCK_COUNTER = {
    'quota': 'quota_hits',
    'http_429': 'rate_limited',
    'key_rejected': 'key_rejected',
    'blocked': 'blocked',
}
_BLOCK_MESSAGE = {
    'quota': 'quota: Exceeded the daily hits limit',
    'http_429': 'rate_limited: HTTP 429',
    'key_rejected': 'key_rejected: Stooq answered with its "Get your apikey" page',
    'blocked': 'blocked: HTML or "Access denied" instead of CSV',
}


def _open_circuit(reason: str) -> None:
    """Open the circuit now because Stooq refused service.

    A quota reply keeps it open until the next UTC midnight, when Stooq's
    daily counter resets; any other refusal takes the next step of the
    cooldown ladder. When another worker already opened it, the ladder is
    not climbed again; only a later quota reopen time is applied.
    """
    now_mono = _mono()
    until_midnight = _seconds_until_utc_midnight(_utcnow()) if reason == 'quota' else None
    opened_for = None
    with _lock:
        _diag[_BLOCK_COUNTER.get(reason, 'blocked')] += 1
        _diag['last_block_reason'] = reason
        _diag['last_block_utc'] = _utcnow().isoformat()
        already_open = _circuit_open_at(now_mono)
        if until_midnight is not None:
            if not already_open or now_mono + until_midnight > _diag['circuit_open_until']:
                _diag['circuit_open_until'] = now_mono + until_midnight
                opened_for = until_midnight
        elif not already_open:
            _diag['circuit_trip_count'] = _diag.get('circuit_trip_count', 0) + 1
            opened_for = _next_cooldown_for_trip(_diag['circuit_trip_count'])
            _diag['circuit_open_until'] = now_mono + opened_for
            _diag['next_cooldown_seconds'] = opened_for
        _diag['half_open'] = False
    if reason in ('quota', 'http_429'):
        count_rate_limit('stooq', _BLOCK_MESSAGE[reason])
    else:
        count_error('stooq', _BLOCK_MESSAGE.get(reason, reason))
    if opened_for is not None:
        log.warning('stooq refused service (%s); circuit open for %d minutes', reason, int(opened_for // 60))


def stats_snapshot() -> dict:
    enabled = is_enabled()
    has_key = bool(_api_key())
    now_mono = _mono()
    with _lock:
        snap = dict(_diag)
    snap['circuit_open'] = bool(snap['circuit_open_until'] and now_mono < snap['circuit_open_until'])
    snap['circuit_remaining_seconds'] = max(0.0, snap['circuit_open_until'] - now_mono) if snap['circuit_open'] else 0.0
    snap['circuit_reopens_at_utc'] = (
        (_utcnow() + timedelta(seconds=snap['circuit_remaining_seconds'])).isoformat()
        if snap['circuit_open'] else None
    )
    # Phase 26.7: expose the trip threshold + next-cooldown so the
    # /api/providers/status endpoint can render the CB row correctly.
    snap['fail_threshold'] = _FAIL_THRESHOLD
    snap['next_cooldown_seconds'] = _next_cooldown_for_trip(
        int(snap.get('circuit_trip_count') or 0) + 1
    )
    # Phase 2B: say why Stooq is idle (None while it is in the cascade).
    snap['enabled'] = enabled
    snap['has_key'] = has_key
    if not enabled:
        snap['idle_reason'] = 'disabled'
    elif not has_key:
        snap['idle_reason'] = 'no_api_key'
    elif snap['circuit_open']:
        snap['idle_reason'] = 'circuit_open'
    else:
        snap['idle_reason'] = None
    snap['pool_workers'] = _POOL_WORKERS
    snap['pool_started'] = _pool is not None
    return snap


def _to_stooq_symbol(sym: str, market: str) -> str:
    s = sym.upper().strip()
    if market == 'crypto':
        return ''  # Stooq crypto symbols differ; defer to crypto-specific providers
    # Stooq prefers `aapl.us`, `msft.us` form for US equities.
    if '.' in s:
        return s.lower()
    return f'{s.lower()}.us'


def _parse_csv(text: str) -> list[dict]:
    rows: list[dict] = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        try:
            rows.append({
                'date': row.get('Date'),
                'open': float(row.get('Open') or 0),
                'high': float(row.get('High') or 0),
                'low': float(row.get('Low') or 0),
                'close': float(row.get('Close') or 0),
                'volume': float(row.get('Volume') or 0),
            })
        except Exception:
            continue
    # Stooq sends the oldest day first. Sorting on the ISO date keeps
    # last_price / previous_close right whatever the order.
    rows.sort(key=lambda r: r.get('date') or '')
    return rows


def _header_value(headers: dict | None, name: str) -> str:
    for key, value in (headers or {}).items():
        if str(key).lower() == name:
            return str(value or '')
    return ''


def _fetch_one(sym: str, market: str, captured_at: str, key: str) -> tuple[str, dict | None]:
    stooq_sym = _to_stooq_symbol(sym, market)
    if not stooq_sym:
        return sym, None
    # The circuit may have opened while this symbol sat in the queue (a
    # refusal on the other worker). Drop it unsent; it is not its fault.
    with _lock:
        if _circuit_open_at(_mono()):
            _diag['skipped_circuit_open'] += 1
            return sym, None
        _diag['attempts'] += 1
    try:
        # Phase 26.18.c: resilient_get hardens stooq against transient
        # 5xx + connection resets it commonly emits during peak hours.
        # Phase 2B: one attempt only. A connect timeout is the usual
        # failure here and retrying it just multiplies the wait; the
        # circuit decides when to try again.
        from app.services.http_client import resilient_get, ResilientGetConfig
        cfg = ResilientGetConfig(
            max_attempts=1, connect_timeout=3.0,
            read_timeout=_TIMEOUT, retry_after_cap_seconds=3.0,
        )
        res = resilient_get(_BASE, params={'s': stooq_sym, 'i': 'd', 'apikey': key}, cfg=cfg)
        if res.status_code == 0 and res.error:
            # Transport failure. requests puts the full URL, query string
            # included, into the message, hence redact().
            err = redact(res.error, key)
            err_l = err.lower()
            with _lock:
                if 'timeout' in err_l:
                    _diag['timeouts'] += 1
                else:
                    _diag['network_errors'] += 1
                _diag['consecutive_failures'] += 1
            if 'timeout' in err_l:
                count_timeout('stooq', f'{sym}: {err}')
            else:
                count_error('stooq', f'{sym}: {err}')
            return sym, None
        outcome, reason = classify_response(
            res.status_code, _header_value(res.headers, 'content-type'), res.text,
        )
        if outcome in ('provider_down', 'rate_limited'):
            _open_circuit(reason or 'blocked')
            return sym, None
        if outcome != 'ok':
            counter = {'http_error': 'http_errors', 'not_found': 'no_data'}.get(outcome, 'bad_payload')
            with _lock:
                _diag[counter] += 1
                _diag['consecutive_failures'] += 1
            if outcome == 'bad_payload':
                count_error('stooq', f'bad_payload: {reason}')
            return sym, None
        rows = _parse_csv(res.text.strip())
        if not rows:
            with _lock:
                _diag['parse_errors'] += 1
                _diag['consecutive_failures'] += 1
            return sym, None
        # Phase 2C: the newest row is the quote, its OHLCV is that row's,
        # and previous_close is the nearest earlier row's close. A single
        # row used to pair with itself (a made-up 0% change); now it is no
        # quote. quote_time_utc marks the session date (CSV has no time).
        last_row = rows[-1]
        prev_close = next((r['close'] for r in reversed(rows[:-1]) if r['close'] > 0), None)
        quote = quote_math.build_quote(
            last_price=last_row['close'],
            previous_close=prev_close,
            open=last_row['open'],
            day_high=last_row['high'],
            day_low=last_row['low'],
            volume=last_row['volume'],
            quote_time_utc=last_row.get('date'),
            captured_at_utc=captured_at,
            source='stooq',
            market_cap=0.0,
        )
        if quote is None:
            with _lock:
                _diag['parse_errors'] += 1
                _diag['consecutive_failures'] += 1
            return sym, None
        with _lock:
            _diag['successes'] += 1
            _diag['consecutive_failures'] = 0
            # A successful probe ends half-open mode -> circuit is healthy.
            _diag['half_open'] = False
        return sym, quote
    except Exception as exc:  # noqa: BLE001
        err = redact(exc, key)
        log.debug('stooq fetch failed for %s: %s', sym, err)
        with _lock:
            _diag['network_errors'] += 1
            _diag['consecutive_failures'] += 1
        count_error('stooq', f'{sym}: {err}')
        return sym, None


def fetch(symbols: list[str], market: str) -> dict[str, dict]:
    if not symbols or market == 'crypto':
        return {}
    # Phase 2B: the same gate as download_quotes(), so a direct call while
    # Stooq is off, or has no key, never touches the network.
    if not is_enabled():
        return {}
    key = _api_key()
    if not key:
        return {}
    # Circuit breaker: if we've had a flood of consecutive failures, skip this
    # provider entirely until the cooldown elapses. This stops Stooq from
    # burning ~3s per symbol when its host is unreachable from our network.
    now_mono = _mono()
    with _lock:
        if _diag['circuit_open_until'] and now_mono < _diag['circuit_open_until']:
            return {}
        # Trip check. Two paths can fire it:
        #   1) Normal: consecutive_failures hit the threshold AND we are not
        #      currently in a cooldown window.
        #   2) Half-open probe: cooldown just elapsed and the probationary
        #      probe failed even once -> re-trip immediately with the next
        #      ladder step (so a chronically broken host doesn't take another
        #      _FAIL_THRESHOLD round-trips to re-pause).
        normal_trip = (
            _diag['consecutive_failures'] >= _FAIL_THRESHOLD
            and not _diag['circuit_open_until']
        )
        half_open_retrip = (
            _diag.get('half_open')
            and _diag['consecutive_failures'] > 0
            and not _diag['circuit_open_until']
        )
        if normal_trip or half_open_retrip:
            _diag['circuit_trip_count'] = _diag.get('circuit_trip_count', 0) + 1
            cd = _next_cooldown_for_trip(_diag['circuit_trip_count'])
            _diag['circuit_open_until'] = now_mono + cd
            _diag['next_cooldown_seconds'] = cd
            _diag['half_open'] = False
            log.warning(
                'stooq circuit breaker tripped (#%d, %s) after %d consecutive failures; '
                'pausing for %d minutes (exponential backoff)',
                _diag['circuit_trip_count'],
                'half-open probe' if half_open_retrip else 'threshold',
                _diag['consecutive_failures'], int(cd // 60),
            )
            return {}
        # Re-enable after cooldown elapses -> enter half-open probationary mode.
        if _diag['circuit_open_until'] and now_mono >= _diag['circuit_open_until']:
            _diag['circuit_open_until'] = 0.0
            _diag['consecutive_failures'] = 0
            _diag['half_open'] = True
            log.info(
                'stooq circuit breaker cooldown elapsed; entering half-open probe '
                '(next failure will re-trip with %d-min cooldown).',
                int(_next_cooldown_for_trip(_diag.get('circuit_trip_count', 0) + 1) // 60),
            )
    out: dict[str, dict] = {}
    captured_at = utcnowiso()
    # Phase 26.32: do NOT use `with ThreadPoolExecutor() as pool:`.  The
    # context-manager exit calls `pool.shutdown(wait=True)` which blocks
    # until every submitted future completes, even if as_completed already
    # gave up at its 20-second deadline.  If any individual fetch hangs
    # past its requests.timeout (DNS deadlock, CLOSE_WAIT socket, etc.),
    # the entire batch wedges indefinitely.  The persistent pool plus
    # `as_completed(timeout=...)` lets us bail at exactly the budget.
    pool = _get_pool()
    futures = [pool.submit(_fetch_one, sym, market, captured_at, key) for sym in symbols]
    try:
        for fut in as_completed(futures, timeout=_BATCH_BUDGET_SECONDS):
            try:
                sym, payload = fut.result()
                if payload:
                    out[sym] = payload
            except Exception:
                continue
    except _FuturesTimeoutError:
        # Budget exhausted; whatever completed is in `out`.
        log.debug('stooq fetch: as_completed timed out at %ss; %d/%d done',
                  _BATCH_BUDGET_SECONDS, len(out), len(futures))
    finally:
        # Phase 2B: with two workers most of a large batch is still queued
        # when the budget runs out. Cancel what has not started so it does
        # not run on into the next batch and spend quota nobody waits for.
        for fut in futures:
            fut.cancel()
    return out
