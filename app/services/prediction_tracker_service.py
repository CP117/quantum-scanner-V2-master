"""
Prediction tracker: save, list, evaluate and score forward price predictions.

The warmer loop logs scanner predictions (source 'auto_scan') and the
detail panel's Save button logs manual ones (source 'user'). Each row
keeps the anchor price, the target price, the direction and an expiry
(created_at + forward_days calendar days). frontend/predictions.js reads
this module through app/routes/prediction_tracker.py, and its fields are
the contract.

Statuses
  open        saved and not evaluated yet.
  evaluated   the official close was found. actual_close, actual_close_date,
              actual_return_pct, error_pct, directional_hit and
              magnitude_hit are set.
  unresolved  no close can be found; unresolved_reason says why.

Rows from older builds used open / expired / correct / incorrect. The
schema check maps correct and incorrect to evaluated with
directional_hit 1 or 0, and expired back to open so it gets evaluated.

Hit rules (the pure functions below)
  actual_close       the official close of the first trading session dated
                     on or after the expiry date. The expiry date is taken
                     in America/New_York for stocks and in UTC for crypto.
  actual_return_pct  (actual_close / anchor_price - 1) * 100
  error_pct          (actual_close - target_price) / target_price * 100
  directional_hit    bull: actual_close > anchor_price.
                     bear: actual_close < anchor_price.
                     neutral: None, so neutral rows are left out of
                     directional accuracy.
  magnitude_hit      abs(error_pct) <= MRD_PREDICTION_MAGNITUDE_TOLERANCE_PCT
                     (default 2.0)

Evaluation (evaluate_expired_predictions) reads Yahoo's daily chart
through the shared provider budget: one request per symbol per pass,
period1 two days before created_at, splits included. A session counts
once it is final (17:00 New York for stocks, 01:00 UTC the next day for
crypto). Rate limits, network errors, a denied budget and sessions
still in progress leave the row open and skip it for
retry_backoff_seconds. A missing symbol, or no close within seven days
of the expiry, marks it unresolved. start_evaluator runs one batch
every 15 minutes in a daemon thread, and scripts/evaluate_predictions.py
runs the same passes from the command line.

The database is data/saved_predictions.db, or the path in
MRD_PREDICTIONS_DB. The variable is read on every call, so tests and the
evaluation script can point the module at a copy.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import statistics
import threading
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from datetime import time as clock_time
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterator
from zoneinfo import ZoneInfo

from app.services import quote_math

log = logging.getLogger('app.prediction_tracker')

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent.parent
_DEFAULT_DB = _REPO_ROOT / 'data' / 'saved_predictions.db'

DB_ENV = 'MRD_PREDICTIONS_DB'
TOLERANCE_ENV = 'MRD_PREDICTION_MAGNITUDE_TOLERANCE_PCT'
DEFAULT_MAGNITUDE_TOLERANCE_PCT = 2.0

STATUSES = ('open', 'evaluated', 'unresolved')
DIRECTIONS = ('bull', 'bear', 'neutral')
SOURCES = ('auto_scan', 'user')

_DB_LOCK = Lock()


def db_path() -> Path:
    """The tracker database: ``MRD_PREDICTIONS_DB`` when set, else data/saved_predictions.db."""
    override = os.environ.get(DB_ENV, '').strip()
    return Path(override).expanduser() if override else _DEFAULT_DB


def magnitude_tolerance_pct() -> float:
    """The magnitude-hit tolerance in percent of the target price.

    ``MRD_PREDICTION_MAGNITUDE_TOLERANCE_PCT`` overrides the default of
    2.0. A value that is not a finite number of zero or more falls back
    to the default.
    """
    raw = os.environ.get(TOLERANCE_ENV, '').strip()
    if not raw:
        return DEFAULT_MAGNITUDE_TOLERANCE_PCT
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_MAGNITUDE_TOLERANCE_PCT
    if not math.isfinite(value) or value < 0:
        return DEFAULT_MAGNITUDE_TOLERANCE_PCT
    return value


def _utcnow() -> datetime:
    """Current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    """Current UTC time as ISO 8601 string."""
    return _utcnow().isoformat()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_TABLE = '''
    CREATE TABLE IF NOT EXISTS saved_predictions (
      id TEXT PRIMARY KEY,
      symbol TEXT NOT NULL,
      market TEXT NOT NULL DEFAULT 'stocks',
      anchor_price REAL NOT NULL,
      target_price REAL NOT NULL,
      direction TEXT NOT NULL,
      confidence_pct REAL,
      forward_days INTEGER NOT NULL DEFAULT 10,
      expires_at TEXT NOT NULL,
      notes TEXT,
      full_payload TEXT,
      created_at TEXT NOT NULL,
      evaluated_at TEXT,
      status TEXT NOT NULL DEFAULT 'open',
      source TEXT NOT NULL DEFAULT 'user',
      actual_close REAL,
      actual_close_date TEXT,
      actual_return_pct REAL,
      error_pct REAL,
      directional_hit INTEGER,
      magnitude_hit INTEGER,
      unresolved_reason TEXT,
      resolved_at TEXT,
      UNIQUE(symbol, anchor_price, created_at)
    )
'''

# Columns added after the first release. Older databases get them through
# ALTER TABLE; resolved_at is also added by maintenance_service.
_ADDED_COLUMNS: tuple[tuple[str, str], ...] = (
    ('evaluated_at', 'TEXT'),
    ('actual_close', 'REAL'),
    ('actual_close_date', 'TEXT'),
    ('actual_return_pct', 'REAL'),
    ('error_pct', 'REAL'),
    ('directional_hit', 'INTEGER'),
    ('magnitude_hit', 'INTEGER'),
    ('unresolved_reason', 'TEXT'),
    ('resolved_at', 'TEXT'),
)

_INDEXES = (
    'CREATE INDEX IF NOT EXISTS idx_symbol ON saved_predictions(symbol)',
    'CREATE INDEX IF NOT EXISTS idx_status ON saved_predictions(status)',
    'CREATE INDEX IF NOT EXISTS idx_created ON saved_predictions(created_at DESC)',
    'CREATE INDEX IF NOT EXISTS idx_status_expires ON saved_predictions(status, expires_at)',
)

# Legacy statuses. A bull or bear row keeps a hit the old code stored;
# a neutral row gets none, whatever the old code recorded.
_LEGACY_STATUS_UPDATES = (
    "UPDATE saved_predictions SET status = 'evaluated', directional_hit = CASE WHEN "
    "direction IN ('bull', 'bear') THEN COALESCE(directional_hit, 1) END WHERE status = 'correct'",
    "UPDATE saved_predictions SET status = 'evaluated', directional_hit = CASE WHEN "
    "direction IN ('bull', 'bear') THEN COALESCE(directional_hit, 0) END WHERE status = 'incorrect'",
    "UPDATE saved_predictions SET status = 'open' WHERE status = 'expired'",
)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create or upgrade the table. Idempotent, and cheap enough to run on every connection.

    Running it every time (instead of once per process) keeps a database
    that was deleted or swapped while the server runs usable.
    """
    conn.execute(_CREATE_TABLE)
    present = {row[1] for row in conn.execute('PRAGMA table_info(saved_predictions)')}
    for name, kind in _ADDED_COLUMNS:
        if name not in present:
            conn.execute(f'ALTER TABLE saved_predictions ADD COLUMN {name} {kind}')
    for statement in _INDEXES:
        conn.execute(statement)
    legacy = conn.execute(
        "SELECT 1 FROM saved_predictions WHERE status IN ('correct', 'incorrect', 'expired') LIMIT 1"
    ).fetchone()
    if legacy:
        for statement in _LEGACY_STATUS_UPDATES:
            conn.execute(statement)


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    """A locked connection with the schema in place; commits on success and always closes."""
    path = db_path()
    with _DB_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            _ensure_schema(conn)
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()


def init_db() -> None:
    """Create or upgrade the schema. Safe to call repeatedly."""
    with _db():
        pass


# ---------------------------------------------------------------------------
# Hit rules (pure)
# ---------------------------------------------------------------------------

def _parse_utc(value: Any) -> datetime | None:
    """An ISO-8601 string or datetime as an aware UTC datetime (naive means UTC)."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def market_timezone(market: str | None) -> str:
    """The timezone that dates a market's sessions: UTC for crypto, New York otherwise."""
    return quote_math.CRYPTO_TZ if (market or '').lower() == 'crypto' else quote_math.DEFAULT_TZ


def expiry_session_date(expires_at: Any, market: str | None = 'stocks') -> date | None:
    """The calendar date of the expiry in the market's timezone.

    The evaluator takes the close of the first trading session dated on
    or after this date. An expiry of 2026-07-20 02:00 UTC is July 19 for
    a stock (22:00 in New York) and July 20 for a coin.
    """
    moment = _parse_utc(expires_at)
    if moment is None:
        return None
    return moment.astimezone(ZoneInfo(market_timezone(market))).date()


def actual_return_pct(anchor_price: Any, actual_close: Any) -> float | None:
    """Percent move from the anchor to the close: (close / anchor - 1) * 100."""
    anchor, close = quote_math.price(anchor_price), quote_math.price(actual_close)
    if anchor is None or close is None:
        return None
    return (close / anchor - 1.0) * 100.0


def error_pct(target_price: Any, actual_close: Any) -> float | None:
    """Signed miss against the target in percent of the target: (close - target) / target * 100."""
    target, close = quote_math.price(target_price), quote_math.price(actual_close)
    if target is None or close is None:
        return None
    return (close - target) / target * 100.0


def directional_hit(direction: Any, anchor_price: Any, actual_close: Any) -> bool | None:
    """Whether the close moved the predicted way from the anchor.

    bull hits when the close is above the anchor and bear when it is
    below; an unchanged close misses both. neutral, an unknown direction
    or a missing price gives None, which leaves the row out of
    directional accuracy.
    """
    anchor, close = quote_math.price(anchor_price), quote_math.price(actual_close)
    if anchor is None or close is None:
        return None
    kind = str(direction or '').strip().lower()
    if kind == 'bull':
        return close > anchor
    if kind == 'bear':
        return close < anchor
    return None


def magnitude_hit(error: Any, tolerance_pct: float | None = None) -> bool | None:
    """Whether abs(error_pct) is within the tolerance (default from magnitude_tolerance_pct()).

    None when the error is missing or not a finite number.
    """
    try:
        value = float(error)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    tol = magnitude_tolerance_pct() if tolerance_pct is None else float(tolerance_pct)
    return abs(value) <= tol


def score_prediction(
    row: dict[str, Any],
    actual_close: Any,
    tolerance_pct: float | None = None,
) -> dict[str, Any]:
    """The evaluated fields for one row and its close: return, error and both hits."""
    err = error_pct(row.get('target_price'), actual_close)
    return {
        'actual_close': quote_math.price(actual_close),
        'actual_return_pct': actual_return_pct(row.get('anchor_price'), actual_close),
        'error_pct': err,
        'directional_hit': directional_hit(row.get('direction'), row.get('anchor_price'), actual_close),
        'magnitude_hit': magnitude_hit(err, tolerance_pct),
    }


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    """Insert a new saved prediction.

    `payload` must include `symbol`, `anchor_price`, `target_price`,
    `direction`, `forward_days`.  Other fields are optional; we fill
    sensible defaults.  Returns the inserted row as a dict.

    Raises ValueError on malformed input, and on a duplicate (same
    symbol, anchor price and created_at), so the route handler can
    return 400. sqlite3 errors are logged and re-raised as RuntimeError.
    """
    if not payload:
        raise ValueError('empty payload')
    sym = (payload.get('symbol') or '').strip().upper()
    if not sym:
        raise ValueError('symbol required')
    market = (payload.get('market') or 'stocks').lower()
    if market not in ('stocks', 'crypto'):
        market = 'stocks'

    try:
        anchor = float(payload.get('anchor_price') or payload.get('current_price') or 0)
        target = float(payload.get('target_price') or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'invalid price field: {exc}') from exc
    
    # CRITICAL FIX: Ensure both anchor and target are valid prices.
    # If target_price is missing or zero, derive it from anchor + a default delta.
    if anchor <= 0:
        raise ValueError('anchor_price must be > 0')
    if target <= 0:
        # Auto-derive target_price if missing: use 10% move in direction of signal
        direction = (payload.get('direction') or '').lower()
        if direction == 'bull':
            target = anchor * 1.10  # 10% upside default
        elif direction == 'bear':
            target = anchor * 0.90  # 10% downside default
        else:
            # Neutral: pick a reasonable default (slightly above anchor)
            target = anchor * 1.05
        log.debug('prediction_tracker: derived target_price=%.2f from anchor=%.2f direction=%s',
                  target, anchor, direction)

    direction = (payload.get('direction') or '').lower()
    if direction not in DIRECTIONS:
        # Derive from sign of (target - anchor) if not explicit.
        delta = target - anchor
        direction = 'bull' if delta > 0 else ('bear' if delta < 0 else 'neutral')

    forward_days = int(payload.get('forward_days') or 10)
    confidence_pct = payload.get('confidence_pct')
    confidence_pct = float(confidence_pct) if confidence_pct is not None else None

    created_at = _utcnow()
    expires_dt = created_at + timedelta(days=forward_days)
    pred_id = uuid.uuid4().hex

    # Persist the full prediction payload so the tracker page can show
    # full audit context (factor contributions, narrative, etc.) without
    # re-running the model.
    full_payload_json = ''
    try:
        full_payload_json = json.dumps(
            payload.get('full_payload') or payload,
            default=str,
        )
    except Exception as exc:
        log.debug('prediction_tracker: payload json failed: %s', exc)
        full_payload_json = '{}'

    notes = (payload.get('notes') or '').strip()[:1000]  # cap to 1k chars
    source = (payload.get('source') or 'user').strip().lower()
    if source not in SOURCES:
        source = 'user'

    try:
        with _db() as c:
            c.execute('''
                INSERT INTO saved_predictions
                  (id, symbol, market, anchor_price, target_price, direction,
                   confidence_pct, forward_days, expires_at, notes,
                   created_at, full_payload, status, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)
            ''', (
                pred_id, sym, market, anchor, target, direction,
                confidence_pct, forward_days, expires_dt.isoformat(), notes,
                created_at.isoformat(), full_payload_json, source,
            ))
    except sqlite3.IntegrityError as exc:
        # Nothing was stored, so do not hand back a row that looks saved.
        raise ValueError(f'duplicate prediction for {sym}: {exc}') from exc
    except sqlite3.Error as exc:
        log.exception('prediction_tracker: insert failed: %s', exc)
        raise RuntimeError(f'db insert failed: {exc}') from exc
    log.debug('prediction_tracker: saved %s %s @ %.2f -> %.2f (%s, %dd)',
              sym, market, anchor, target, direction, forward_days)

    return {
        'id': pred_id,
        'symbol': sym,
        'market': market,
        'anchor_price': anchor,
        'target_price': target,
        'direction': direction,
        'confidence_pct': confidence_pct,
        'forward_days': forward_days,
        'expires_at': expires_dt.isoformat(),
        'created_at': created_at.isoformat(),
        'status': 'open',
        'source': source,
    }


# ---------------------------------------------------------------------------
# Reading and deleting
# ---------------------------------------------------------------------------

# What /api/predictions/list returns per row: everything predictions.js
# reads, without full_payload (several kB per row).
LIST_COLUMNS = (
    'id', 'symbol', 'market', 'direction', 'anchor_price', 'target_price',
    'confidence_pct', 'forward_days', 'notes', 'source', 'status',
    'created_at', 'expires_at', 'evaluated_at', 'actual_close',
    'actual_close_date', 'actual_return_pct', 'error_pct',
    'directional_hit', 'magnitude_hit', 'unresolved_reason',
)


def _as_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _public_row(row: sqlite3.Row) -> dict[str, Any]:
    out = {name: row[name] for name in LIST_COLUMNS}
    out['directional_hit'] = _as_bool(out['directional_hit'])
    out['magnitude_hit'] = _as_bool(out['magnitude_hit'])
    return out


def list_predictions(
    market: str | None = None,
    status: str | None = None,
    source: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Saved predictions, newest first, optionally filtered by market, status and source.

    Hits come back as True, False or None (predictions.js compares with
    ===). full_payload is left out. sqlite errors propagate so the route
    answers 500 instead of an empty list.
    """
    clauses: list[str] = []
    params: list[Any] = []
    for column, value in (('market', market), ('status', status), ('source', source)):
        if value:
            clauses.append(f'{column} = ?')
            params.append(str(value).lower())
    query = f"SELECT {', '.join(LIST_COLUMNS)} FROM saved_predictions"
    if clauses:
        query += ' WHERE ' + ' AND '.join(clauses)
    query += ' ORDER BY created_at DESC, id LIMIT ?'
    params.append(max(1, min(int(limit), 5000)))
    with _db() as c:
        return [_public_row(r) for r in c.execute(query, params).fetchall()]


def delete_prediction(prediction_id: str) -> bool:
    """Delete one prediction by id. True when a row was removed."""
    pid = str(prediction_id or '').strip()
    if not pid:
        return False
    with _db() as c:
        return c.execute('DELETE FROM saved_predictions WHERE id = ?', (pid,)).rowcount > 0


def get_open_predictions(symbol: str | None = None, market: str | None = None) -> list[dict]:
    """Retrieve all open predictions, optionally filtered by symbol + market."""
    query = 'SELECT * FROM saved_predictions WHERE status = ?'
    params = ['open']
    if symbol:
        query += ' AND symbol = ?'
        params.append(symbol.upper())
    if market:
        query += ' AND market = ?'
        params.append(market.lower())
    query += ' ORDER BY created_at DESC'
    try:
        with _db() as c:
            rows = c.execute(query, params).fetchall()
            return [dict(r) for r in rows]
    except sqlite3.Error as exc:
        log.warning('prediction_tracker: fetch open failed: %s', exc)
        return []


def get_all_predictions(symbol: str | None = None, limit: int = 1000) -> list[dict]:
    """Retrieve all predictions (any status), newest first."""
    query = 'SELECT * FROM saved_predictions'
    params: list[Any] = []
    if symbol:
        query += ' WHERE symbol = ?'
        params.append(symbol.upper())
    query += ' ORDER BY created_at DESC LIMIT ?'
    params.append(limit)
    try:
        with _db() as c:
            rows = c.execute(query, params).fetchall()
            return [dict(r) for r in rows]
    except sqlite3.Error as exc:
        log.warning('prediction_tracker: fetch all failed: %s', exc)
        return []


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def _pct(hits: int, total: int) -> float | None:
    return round(100.0 * hits / total, 1) if total else None


def _rates(rows: list[sqlite3.Row]) -> dict[str, Any]:
    """Counts and hit rates over a set of rows; percentages are None without a sample."""
    evaluated = [r for r in rows if r['status'] == 'evaluated']
    directional = [r['directional_hit'] for r in evaluated if r['directional_hit'] is not None]
    magnitude = [r['magnitude_hit'] for r in evaluated if r['magnitude_hit'] is not None]
    return {
        'total_saved': len(rows),
        'open': sum(1 for r in rows if r['status'] == 'open'),
        'evaluated': len(evaluated),
        'unresolved': sum(1 for r in rows if r['status'] == 'unresolved'),
        'directional_total': len(directional),
        'directional_hits': sum(1 for h in directional if h),
        'directional_accuracy_pct': _pct(sum(1 for h in directional if h), len(directional)),
        'magnitude_accuracy_pct': _pct(sum(1 for h in magnitude if h), len(magnitude)),
    }


def accuracy_stats(market: str | None = None, source: str | None = None) -> dict[str, Any]:
    """Accuracy over the saved predictions, in the shape predictions.js reads.

    Top-level counts and rates, and by_direction, use the market and
    source filters. by_source uses only the market filter, so the page
    can always show auto_scan (every scored symbol, no curation) next to
    user (manual picks, selection-biased).

    Directional accuracy covers evaluated bull and bear rows; magnitude
    accuracy and the error averages cover every evaluated row with a
    close. Percentages are None until there is a sample.

    The legacy keys (total_predictions, correct, incorrect, accuracy_pct,
    bull_accuracy_pct, bear_accuracy_pct, expired_only) describe the
    directional sample and use 0.0 for an empty one, as before.
    """
    query = ('SELECT status, direction, source, directional_hit, magnitude_hit, error_pct '
             'FROM saved_predictions')
    params: list[Any] = []
    if market:
        query += ' WHERE market = ?'
        params.append(market.lower())
    with _db() as c:
        market_rows = c.execute(query, params).fetchall()
    rows = [r for r in market_rows if not source or r['source'] == source]

    overall = _rates(rows)
    errors = [abs(r['error_pct']) for r in rows
              if r['status'] == 'evaluated' and r['error_pct'] is not None]

    by_direction: dict[str, dict[str, Any]] = {}
    for kind in DIRECTIONS:
        rates = _rates([r for r in rows if r['direction'] == kind])
        by_direction[kind] = {
            # predictions.js sums `total` to decide whether to draw the
            # chart, so it counts the evaluated rows the rates cover.
            'total': rates['evaluated'],
            'evaluated': rates['evaluated'],
            'total_saved': rates['total_saved'],
            'directional_accuracy_pct': rates['directional_accuracy_pct'],
            'magnitude_accuracy_pct': rates['magnitude_accuracy_pct'],
        }

    by_source: dict[str, dict[str, Any]] = {}
    for kind in SOURCES:
        rates = _rates([r for r in market_rows if r['source'] == kind])
        by_source[kind] = {
            key: rates[key] for key in (
                'total_saved', 'open', 'evaluated', 'unresolved',
                'directional_accuracy_pct', 'magnitude_accuracy_pct',
            )
        }

    return {
        'total_saved': overall['total_saved'],
        'open': overall['open'],
        'evaluated': overall['evaluated'],
        'unresolved': overall['unresolved'],
        'directional_accuracy_pct': overall['directional_accuracy_pct'],
        'magnitude_accuracy_pct': overall['magnitude_accuracy_pct'],
        'mean_abs_error_pct': round(statistics.fmean(errors), 2) if errors else None,
        'median_abs_error_pct': round(statistics.median(errors), 2) if errors else None,
        'magnitude_tolerance_pct': magnitude_tolerance_pct(),
        'by_direction': by_direction,
        'by_source': by_source,
        'filters': {'market': market, 'source': source},
        # Legacy keys.
        'total_predictions': overall['directional_total'],
        'correct': overall['directional_hits'],
        'incorrect': overall['directional_total'] - overall['directional_hits'],
        'accuracy_pct': overall['directional_accuracy_pct'] or 0.0,
        'bull_accuracy_pct': by_direction['bull']['directional_accuracy_pct'] or 0.0,
        'bear_accuracy_pct': by_direction['bear']['directional_accuracy_pct'] or 0.0,
        'expired_only': True,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

# A stock session's close is final an hour after the 16:00 New York close;
# a crypto day (UTC) an hour after midnight.
STOCK_CLOSE_FINAL_LOCAL = clock_time(17, 0)
CRYPTO_CLOSE_GRACE = timedelta(hours=1)
# How long after expiry a missing close is still worth waiting for, and
# how far after the expiry date a close may fall and still count.
NO_CLOSE_GIVE_UP = timedelta(days=7)
MAX_CLOSE_GAP_DAYS = 7

HISTORY_URL = 'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}'


@dataclass(frozen=True)
class CloseLookup:
    """What a price function found for one prediction.

    close         the official close on the anchor's price basis, or None.
    session_date  the session that close belongs to.
    reason        why there is no close.
    retry         True when the cause is temporary (rate limit, network,
                  session still open). The row stays open and is skipped
                  for retry_backoff_seconds. False without a close marks
                  the row unresolved with `reason`.
    """
    close: float | None = None
    session_date: date | None = None
    reason: str | None = None
    retry: bool = False


PriceFn = Callable[[dict], 'float | CloseLookup | None']


def session_is_final(session: date, market: str | None, now: datetime) -> bool:
    """Whether a daily bar's close can still change at `now`.

    Stocks: final from 17:00 New York time on the session date. Crypto:
    final from 01:00 UTC on the next day.
    """
    if (market or '').lower() == 'crypto':
        final_at = datetime.combine(session + timedelta(days=1), clock_time(0), tzinfo=timezone.utc)
        final_at += CRYPTO_CLOSE_GRACE
    else:
        final_at = datetime.combine(session, STOCK_CLOSE_FINAL_LOCAL, tzinfo=ZoneInfo(quote_math.DEFAULT_TZ))
    return now >= final_at


def close_from_history(history: dict, row: dict, now: datetime) -> CloseLookup:
    """The official close for one prediction, picked from a daily history.

    `history` holds ``bars``, a list of (session_date, close), and
    ``splits``, a list of (epoch_seconds, numerator / denominator).
    Yahoo's closes are adjusted for every split up to today, so the
    close is multiplied by each split ratio dated after created_at to put
    it back on the anchor's basis.

    The first bar dated on or after the expiry date is used. A bar from
    a session that is not final yet means retry. No bar at all means
    retry for NO_CLOSE_GIVE_UP after the expiry, then unresolved. A
    first bar more than MAX_CLOSE_GAP_DAYS after the expiry date
    (a halt or delisting) is unresolved.
    """
    market = row.get('market')
    target = expiry_session_date(row.get('expires_at'), market)
    expires = _parse_utc(row.get('expires_at'))
    if target is None or expires is None:
        return CloseLookup(reason='bad_expiry')
    bars = sorted(
        (day, float(close)) for day, close in (history.get('bars') or [])
        if day is not None and day >= target and quote_math.price(close) is not None
    )
    if not bars:
        if now - expires < NO_CLOSE_GIVE_UP:
            return CloseLookup(reason='no_close_yet', retry=True)
        return CloseLookup(reason='no_close_after_expiry')
    session, close = bars[0]
    if (session - target).days > MAX_CLOSE_GAP_DAYS:
        return CloseLookup(reason='no_close_near_expiry')
    if not session_is_final(session, market, now):
        return CloseLookup(reason='session_not_closed', retry=True)
    created = _parse_utc(row.get('created_at'))
    factor = 1.0
    if created is not None:
        for epoch, ratio in history.get('splits') or []:
            if epoch > created.timestamp():
                factor *= ratio
    return CloseLookup(close=close * factor, session_date=session)


def yahoo_history_symbol(symbol: str, market: str | None) -> str:
    """Yahoo's spelling of a symbol: BRK.B is BRK-B, and a bare coin gets -USD."""
    sym = (symbol or '').strip().upper()
    if (market or '').lower() == 'crypto':
        return sym if '-' in sym else f'{sym}-USD'
    return sym.replace('.', '-').replace('/', '-')


def parse_chart_history(result: dict, market: str | None) -> dict:
    """Daily bars and splits from one Yahoo ``chart.result`` entry.

    Bars are dated in the exchange's timezone (UTC for crypto).
    """
    meta = result.get('meta') or {}
    if (market or '').lower() == 'crypto':
        tz = quote_math.CRYPTO_TZ
    else:
        tz = meta.get('exchangeTimezoneName') or quote_math.DEFAULT_TZ
    stamps = list(result.get('timestamp') or [])
    quote = ((result.get('indicators') or {}).get('quote') or [{}])[0] or {}
    closes = list(quote.get('close') or [])
    bars = [(quote_math.session_date(t, tz), c) for t, c in zip(stamps, closes)]
    splits: list[tuple[float, float]] = []
    for event in ((result.get('events') or {}).get('splits') or {}).values():
        try:
            epoch = float(event['date'])
            numerator, denominator = float(event['numerator']), float(event['denominator'])
        except (KeyError, TypeError, ValueError):
            continue
        if numerator > 0 and denominator > 0:
            splits.append((epoch, numerator / denominator))
    return {'bars': bars, 'splits': sorted(splits)}


def _history_failure(res: Any) -> CloseLookup | None:
    """The CloseLookup for a failed chart response, or None when it is a 200."""
    status = int(getattr(res, 'status_code', 0) or 0)
    if status == 0:
        error = str(getattr(res, 'error', '') or '').lower()
        return CloseLookup(reason='timeout' if 'timeout' in error else 'network_error', retry=True)
    if status == 429:
        return CloseLookup(reason='rate_limited', retry=True)
    if status in (400, 404, 410, 422):
        return CloseLookup(reason='no_history')
    if status != 200:
        return CloseLookup(reason=f'http_{status}', retry=True)
    return None


def make_history_price_fn(
    min_gap_seconds: float = 1.0,
    *,
    budget_wait_seconds: float = 5.0,
    http_get: Callable[..., Any] | None = None,
    budget_fn: Callable[..., bool] | None = None,
    now_fn: Callable[[], datetime] | None = None,
) -> Callable[[dict], CloseLookup]:
    """A price function that reads closes from Yahoo's daily chart.

    One request per symbol, cached for the life of the function (a
    failure that is not temporary is cached too). Requests are at least
    `min_gap_seconds` apart and each needs a slot from the shared
    provider budget at background priority; a denied slot means retry.
    `http_get`, `budget_fn` and `now_fn` are for tests.
    """
    cache: dict[tuple[str, str], tuple[int, dict | CloseLookup]] = {}
    gate = Lock()
    last_request = [float('-inf')]

    def fetch(sym: str, market: str, period1: int, period2: int) -> dict | CloseLookup:
        get = http_get
        if get is None:
            from app.services.http_client import ResilientGetConfig, resilient_get

            cfg = ResilientGetConfig(max_attempts=3, connect_timeout=4.0, read_timeout=10.0,
                                     retry_after_cap_seconds=5.0)

            def get(url, params):
                return resilient_get(url, params=params, headers={'Accept': 'application/json'}, cfg=cfg)
        allow = budget_fn
        if allow is None:
            from app.services.provider_session import provider_budget_allowance as allow
        with gate:
            wait = last_request[0] + min_gap_seconds - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            if not allow(1, priority=3, wait_timeout_seconds=budget_wait_seconds):
                return CloseLookup(reason='budget_denied', retry=True)
            last_request[0] = time.monotonic()
        params = {'period1': period1, 'period2': period2, 'interval': '1d', 'events': 'split'}
        try:
            res = get(HISTORY_URL.format(symbol=sym), params)
        except Exception as exc:  # noqa: BLE001 - resilient_get should not raise; a test double might
            return CloseLookup(reason=f'network_error:{type(exc).__name__}', retry=True)
        failure = _history_failure(res)
        if failure is not None:
            return failure
        try:
            chart = (res.json() or {}).get('chart') or {}
        except Exception:  # noqa: BLE001 - not JSON
            return CloseLookup(reason='bad_response', retry=True)
        results = chart.get('result') or []
        if chart.get('error') or not results:
            return CloseLookup(reason='no_history')
        history = parse_chart_history(results[0], market)
        if not any(quote_math.price(close) is not None for _, close in history['bars']):
            return CloseLookup(reason='no_history')
        return history

    def price_fn(row: dict) -> CloseLookup:
        now = now_fn() if now_fn else _utcnow()
        market = str(row.get('market') or 'stocks').lower()
        sym = yahoo_history_symbol(str(row.get('symbol') or ''), market)
        created = _parse_utc(row.get('created_at')) or _parse_utc(row.get('expires_at'))
        if not sym or created is None:
            return CloseLookup(reason='bad_row')
        period1 = int((created - timedelta(days=2)).timestamp())
        cached = cache.get((sym, market))
        if cached is None or cached[0] > period1:
            found = fetch(sym, market, period1, int(now.timestamp()))
            if isinstance(found, CloseLookup) and found.retry:
                return found
            cached = (period1, found)
            cache[(sym, market)] = cached
        history = cached[1]
        if isinstance(history, CloseLookup):
            return history
        return close_from_history(history, row, now)

    return price_fn


# Prediction id -> time.monotonic() before which it is not retried.
_RETRY_AFTER: dict[str, float] = {}
_EVAL_LOCK = Lock()


def _as_lookup(found: Any) -> CloseLookup:
    if isinstance(found, CloseLookup):
        return found
    if found is None:
        return CloseLookup(reason='price_unavailable')
    close = quote_math.price(found)
    return CloseLookup(close=close) if close is not None else CloseLookup(reason='bad_price')


def _open_counts(conn: sqlite3.Connection, moment: datetime) -> dict[str, int]:
    stamp = moment.isoformat()
    row = conn.execute(
        "SELECT COALESCE(SUM(CASE WHEN expires_at <= ? THEN 1 ELSE 0 END), 0), "
        "COALESCE(SUM(CASE WHEN expires_at > ? THEN 1 ELSE 0 END), 0) "
        "FROM saved_predictions WHERE status = 'open'", (stamp, stamp),
    ).fetchone()
    return {'still_open': int(row[0]), 'not_yet_expired': int(row[1])}


def evaluate_expired_predictions(
    now: datetime | None = None,
    batch_size: int = 200,
    price_fn: PriceFn | None = None,
    *,
    retry_backoff_seconds: float = 600.0,
) -> dict[str, Any]:
    """Evaluate up to `batch_size` expired open predictions, oldest expiry first.

    `price_fn(row)` returns the close as a float, None (no price: the row
    becomes unresolved with 'price_unavailable') or a CloseLookup. It
    defaults to make_history_price_fn(). A row whose lookup says retry
    stays open and is skipped for `retry_backoff_seconds` in this
    process, so repeated passes move on to other rows.

    Returns evaluated, unresolved, retry_later, retry_reasons, checked
    (rows looked at), still_open (expired rows still open afterwards),
    not_yet_expired and busy (True when another pass was running, and
    this one did nothing).
    """
    moment = _parse_utc(now) if now is not None else _utcnow()
    if moment is None:
        raise ValueError(f'bad now: {now!r}')
    result: dict[str, Any] = {
        'evaluated': 0, 'unresolved': 0, 'retry_later': 0, 'retry_reasons': {},
        'checked': 0, 'busy': False,
    }
    if not _EVAL_LOCK.acquire(blocking=False):
        with _db() as c:
            result.update(_open_counts(c, moment))
        result['busy'] = True
        return result
    try:
        fn = price_fn or make_history_price_fn(now_fn=lambda: moment)
        clock = time.monotonic()
        for pid, until in list(_RETRY_AFTER.items()):
            if until <= clock:
                del _RETRY_AFTER[pid]
        deferred = set(_RETRY_AFTER)
        size = max(1, int(batch_size))
        with _db() as c:
            rows = c.execute(
                "SELECT * FROM saved_predictions WHERE status = 'open' AND expires_at <= ? "
                'ORDER BY expires_at, id LIMIT ?', (moment.isoformat(), size + len(deferred)),
            ).fetchall()
        batch = [dict(r) for r in rows if r['id'] not in deferred][:size]
        reasons: Counter[str] = Counter()
        stamp = moment.isoformat()
        for row in batch:
            result['checked'] += 1
            try:
                lookup = _as_lookup(fn(row))
            except Exception as exc:  # noqa: BLE001 - one bad lookup must not stop the pass
                log.warning('prediction evaluator: lookup failed for %s: %s', row.get('symbol'), exc)
                lookup = CloseLookup(reason=f'lookup_error:{type(exc).__name__}', retry=True)
            close = quote_math.price(lookup.close)
            if close is not None:
                scored = score_prediction(row, close)
                with _db() as c:
                    changed = c.execute(
                        "UPDATE saved_predictions SET status = 'evaluated', actual_close = ?, "
                        'actual_close_date = ?, actual_return_pct = ?, error_pct = ?, '
                        'directional_hit = ?, magnitude_hit = ?, unresolved_reason = NULL, '
                        "evaluated_at = ?, resolved_at = ? WHERE id = ? AND status = 'open'",
                        (
                            round(close, 6),
                            lookup.session_date.isoformat() if lookup.session_date else None,
                            _round(scored['actual_return_pct']), _round(scored['error_pct']),
                            _as_int(scored['directional_hit']), _as_int(scored['magnitude_hit']),
                            stamp, stamp, row['id'],
                        ),
                    ).rowcount
                result['evaluated'] += changed
            elif lookup.retry:
                _RETRY_AFTER[row['id']] = time.monotonic() + max(0.0, float(retry_backoff_seconds))
                result['retry_later'] += 1
                reasons[lookup.reason or 'unknown'] += 1
            else:
                # evaluated_at stays empty: maintenance pruning keys on it,
                # and unresolved rows are kept.
                with _db() as c:
                    changed = c.execute(
                        "UPDATE saved_predictions SET status = 'unresolved', unresolved_reason = ?, "
                        "resolved_at = ? WHERE id = ? AND status = 'open'",
                        ((lookup.reason or 'price_unavailable')[:200], stamp, row['id']),
                    ).rowcount
                result['unresolved'] += changed
        result['retry_reasons'] = dict(reasons)
        with _db() as c:
            result.update(_open_counts(c, moment))
    finally:
        _EVAL_LOCK.release()
    if result['checked']:
        log.info(
            'prediction evaluator: checked %d, evaluated %d, unresolved %d, retry later %d %s, still open %d',
            result['checked'], result['evaluated'], result['unresolved'], result['retry_later'],
            result['retry_reasons'] or '', result['still_open'],
        )
    return result


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _as_int(value: bool | None) -> int | None:
    return None if value is None else int(value)


# ---------------------------------------------------------------------------
# Background evaluator
# ---------------------------------------------------------------------------

_EVALUATOR: dict[str, threading.Thread | None] = {'thread': None}
_EVALUATOR_STOP = threading.Event()
_EVALUATOR_START = Lock()
EVALUATOR_THREAD_NAME = 'prediction-evaluator'


def _evaluator_loop(interval_seconds: float, initial_delay_seconds: float) -> None:
    if _EVALUATOR_STOP.wait(max(0.0, initial_delay_seconds)):
        return
    while True:
        try:
            evaluate_expired_predictions()
        except Exception:  # noqa: BLE001 - keep the thread alive; the next tick retries
            log.exception('prediction evaluator pass failed')
        if _EVALUATOR_STOP.wait(max(1.0, interval_seconds)):
            return


def start_evaluator(interval_seconds: float = 900, *, initial_delay_seconds: float = 120) -> bool:
    """Start the background evaluator unless it is already running. Returns whether it runs.

    One daemon thread named 'prediction-evaluator' waits
    `initial_delay_seconds` so startup traffic settles, then evaluates
    one batch every `interval_seconds`. MRD_START_BACKGROUND=0 (the test
    suite sets it) leaves it stopped. Phase 5 moves it into the
    background-task registry.
    """
    if os.environ.get('MRD_START_BACKGROUND', '1').strip() == '0':
        log.info('prediction evaluator not started: MRD_START_BACKGROUND=0')
        return False
    with _EVALUATOR_START:
        thread = _EVALUATOR['thread']
        if thread is not None and thread.is_alive():
            return True
        _EVALUATOR_STOP.clear()
        thread = threading.Thread(
            target=_evaluator_loop,
            args=(float(interval_seconds), float(initial_delay_seconds)),
            name=EVALUATOR_THREAD_NAME,
            daemon=True,
        )
        _EVALUATOR['thread'] = thread
        thread.start()
    return True


def stop_evaluator(timeout: float = 5.0) -> None:
    """Ask the background evaluator to stop and wait up to `timeout` seconds for it."""
    _EVALUATOR_STOP.set()
    thread = _EVALUATOR['thread']
    if thread is not None:
        thread.join(timeout)


# ---------------------------------------------------------------------------
# Auto-logging
# ---------------------------------------------------------------------------

def auto_log_scan_predictions(
    rows: list[dict],
    max_new_per_cycle: int = 10,
    forward_days: int = 10,
) -> None:
    """Auto-log top-N symbols from a completed scanner batch.

    Called by the warmer loop on each cycle to log systematic predictions
    from the background scan. These are tagged source='auto_scan' so the
    UI can distinguish user-manual predictions from algorithmic ones.

    `rows` should be the top-scoring symbols from the batch (usually the
    scan result envelope['results']). We pick the top N that haven't been
    logged today and insert them.
    """
    if not rows or max_new_per_cycle <= 0:
        return

    # Quick check: don't bother if the DB hasn't been initialized yet.
    try:
        if not db_path().exists():
            return
    except Exception:
        return

    now = _utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Collect symbols already logged today.
    try:
        with _db() as c:
            logged_today = {
                r[0] for r in c.execute('''
                    SELECT DISTINCT symbol FROM saved_predictions
                    WHERE source = 'auto_scan' AND created_at >= ?
                ''', (today_start.isoformat(),)).fetchall()
            }
    except sqlite3.Error as exc:
        log.warning('prediction_tracker: auto_log fetch failed: %s', exc)
        return

    logged_count = 0
    for row in rows:
        if logged_count >= max_new_per_cycle:
            break
        sym = (row.get('symbol') or '').upper()
        if not sym or sym in logged_today:
            continue

        # Extract the data we need from the row.
        score = float(row.get('final_score') or 0.0)
        direction = (row.get('final_direction') or 'Neutral').lower()
        if direction not in ('bull', 'bear', 'neutral'):
            direction = 'neutral'

        # Get the current/anchor price.  Try multiple sources.
        anchor_price = None
        market_fb = (row.get('factor_breakdown') or {}).get('market') or {}
        if market_fb.get('last_price'):
            anchor_price = float(market_fb['last_price'])
        if not anchor_price or anchor_price <= 0:
            # Try symbol identity lookup as fallback.
            try:
                from app.services.snapshot_store import lookup_snapshot_row
                snap_row = lookup_snapshot_row(sym, 'stocks')
                if snap_row:
                    snap_fb = (snap_row.get('factor_breakdown') or {}).get('market') or {}
                    anchor_price = float(snap_fb.get('last_price') or 0)
            except Exception:
                pass

        if not anchor_price or anchor_price <= 0:
            log.debug('prediction_tracker: auto_log skip %s (no price)', sym)
            continue

        # Derive target_price based on score and direction.
        # Higher score → more aggressive target (farther from anchor).
        # Bull: target = anchor * (1 + 0.05 * normalized_score)
        # Bear: target = anchor * (1 - 0.05 * normalized_score)
        normalized_score = max(0.0, min(1.0, score / 100.0))  # 0 to 1
        if direction == 'bull':
            # 5-15% upside range based on score
            target_price = anchor_price * (1.0 + 0.05 + 0.1 * normalized_score)
        elif direction == 'bear':
            # 5-15% downside range based on score
            target_price = anchor_price * (1.0 - 0.05 - 0.1 * normalized_score)
        else:
            # Neutral: tight range (2-3% move)
            target_price = anchor_price * (1.0 + 0.01 * (0.5 - normalized_score))

        try:
            save_prediction({
                'symbol': sym,
                'market': 'stocks',
                'anchor_price': anchor_price,
                'target_price': target_price,
                'direction': direction,
                'forward_days': forward_days,
                'confidence_pct': min(100.0, score),  # Use final_score as confidence
                'notes': f'auto_scan: score={score:.1f}',
                'source': 'auto_scan',
            })
            logged_count += 1
            logged_today.add(sym)
        except ValueError as exc:
            log.debug('prediction_tracker: auto_log skip %s: %s', sym, exc)
        except RuntimeError as exc:
            log.warning('prediction_tracker: auto_log failed for %s: %s', sym, exc)
            # Continue on DB errors — don't let one failure block the whole batch.

    if logged_count > 0:
        log.info('prediction_tracker: auto_log added %d new predictions', logged_count)
