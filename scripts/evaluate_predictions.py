"""Evaluate every expired open prediction, in batches, until none is left.

Runs the pass the server's background evaluator runs
(prediction_tracker_service.evaluate_expired_predictions), with one
Yahoo price function shared by every batch so each symbol is fetched
once. Requests go through the provider budget at background priority
and are at least --min-gap seconds apart.

Resumable: each row is written as soon as it is decided, and evaluated
or unresolved rows are never looked at again, so a stopped run carries
on where it left off. A row whose lookup failed for a temporary reason
(rate limit, network, a session that is not final yet) stays open. It
is retried after --retry-wait seconds, up to --retry-rounds times, and
otherwise on the next run. The server may keep running: both write
with a status = 'open' guard, so no row is evaluated twice.

Exit status: 0 when no expired prediction is left open, 2 when some
are (run again later), 1 when the database does not exist, 130 after
Ctrl-C.

Usage
-----
    python scripts/evaluate_predictions.py --dry-run
    python scripts/evaluate_predictions.py
    python scripts/evaluate_predictions.py --db /path/to/copy.db --min-gap 0.5
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

EXIT_DONE = 0
EXIT_NO_DB = 1
EXIT_OPEN_LEFT = 2
EXIT_INTERRUPTED = 130

# Within one retry round a row that must be retried is not looked at
# again; the round ends when only such rows are left.
_ROUND_BACKOFF_SECONDS = 1e9

# Legacy builds used 'expired' for rows the schema check maps back to open.
_OPEN_STATUSES = "('open', 'expired')"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate expired saved predictions against the official close.')
    parser.add_argument('--dry-run', action='store_true',
                        help='count the expired open predictions and stop; nothing is looked up or written')
    parser.add_argument('--max-batches', type=int, default=None, metavar='N',
                        help='stop after N batches (default: until none is left)')
    parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                        help='predictions per batch (default 200)')
    parser.add_argument('--db', type=Path, default=None, metavar='PATH',
                        help='database to evaluate (default: MRD_PREDICTIONS_DB, else data/saved_predictions.db)')
    parser.add_argument('--min-gap', type=float, default=1.0, metavar='SECONDS',
                        help='minimum seconds between Yahoo requests (default 1.0)')
    parser.add_argument('--retry-rounds', type=int, default=2, metavar='N',
                        help='extra rounds for predictions whose lookup must be retried (default 2)')
    parser.add_argument('--retry-wait', type=float, default=60.0, metavar='SECONDS',
                        help='seconds to wait before each retry round (default 60)')
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error('--batch-size must be at least 1')
    if args.max_batches is not None and args.max_batches < 0:
        parser.error('--max-batches cannot be negative')
    if args.min_gap < 0 or args.retry_wait < 0 or args.retry_rounds < 0:
        parser.error('--min-gap, --retry-wait and --retry-rounds cannot be negative')
    return args


def _connect_read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f'{path.resolve().as_uri()}?mode=ro', uri=True)


def _expired_open(path: Path, now: datetime) -> dict[str, Any]:
    """Counts of expired open predictions, read without changing the database."""
    stamp = now.isoformat()
    conn = _connect_read_only(path)
    try:
        groups = conn.execute(
            'SELECT market, source, COUNT(*), MIN(expires_at), MAX(expires_at) FROM saved_predictions '
            f'WHERE status IN {_OPEN_STATUSES} AND expires_at <= ? GROUP BY market, source '
            'ORDER BY market, source', (stamp,)).fetchall()
        symbols = conn.execute(
            "SELECT COUNT(DISTINCT market || ':' || symbol) FROM saved_predictions "
            f'WHERE status IN {_OPEN_STATUSES} AND expires_at <= ?', (stamp,)).fetchone()[0]
        later = conn.execute(
            f'SELECT COUNT(*) FROM saved_predictions WHERE status IN {_OPEN_STATUSES} AND expires_at > ?',
            (stamp,)).fetchone()[0]
    except sqlite3.OperationalError as exc:
        if 'no such table' not in str(exc):
            raise
        groups, symbols, later = [], 0, 0
    finally:
        conn.close()
    return {
        'total': sum(g[2] for g in groups),
        'groups': [{'market': g[0], 'source': g[1], 'count': g[2], 'first': g[3], 'last': g[4]} for g in groups],
        'symbols': symbols,
        'not_yet_expired': later,
    }


def dry_run(path: Path, min_gap: float, now: datetime | None = None) -> int:
    moment = now or datetime.now(timezone.utc)
    found = _expired_open(path, moment)
    print(f'Database: {path}')
    print(f"Expired and open: {found['total']:,}")
    for g in found['groups']:
        print(f"  {g['market']}/{g['source']}: {g['count']:,}, expiring {str(g['first'])[:10]} to {str(g['last'])[:10]}")
    print(f"Not expired yet: {found['not_yet_expired']:,}")
    minutes = found['symbols'] * min_gap / 60.0
    print(f"Symbols to look up: {found['symbols']:,} (about {minutes:.0f} min at --min-gap {min_gap:g})")
    print('Dry run: nothing was looked up or written.')
    return EXIT_DONE


def _pct(value: float | None) -> str:
    return 'n/a' if value is None else f'{value:.1f}%'


def _table_row(source: str, direction: str, evaluated: int, directional: float | None,
               magnitude: float | None) -> str:
    return f'  {source:<10}{direction:<10}{evaluated:>10,}{_pct(directional):>13}{_pct(magnitude):>11}'


def summary_lines(tracker: Any, path: Path) -> list[str]:
    """Counts and provisional accuracy by source and direction, for the end of a run."""
    stats = tracker.accuracy_stats()
    lines = [
        f"Saved {stats['total_saved']:,}: {stats['evaluated']:,} evaluated, "
        f"{stats['unresolved']:,} unresolved, {stats['open']:,} open.",
        'Accuracy (provisional). Directional: bull closes above the anchor, bear below; '
        f"neutral is left out. Magnitude: close within +/-{stats['magnitude_tolerance_pct']:g}% of the target.",
        f"  {'source':<10}{'direction':<10}{'evaluated':>10}{'directional':>13}{'magnitude':>11}",
    ]
    for source in (None, *tracker.SOURCES):
        part = stats if source is None else tracker.accuracy_stats(source=source)
        if source is not None and not part['evaluated']:
            continue
        name = source or 'all'
        lines.append(_table_row(name, 'all', part['evaluated'], part['directional_accuracy_pct'],
                                part['magnitude_accuracy_pct']))
        for direction in tracker.DIRECTIONS:
            cell = part['by_direction'][direction]
            if cell['evaluated']:
                lines.append(_table_row(name, direction, cell['evaluated'], cell['directional_accuracy_pct'],
                                        cell['magnitude_accuracy_pct']))
    if stats['mean_abs_error_pct'] is not None:
        lines.append(f"Absolute error against the target: mean {stats['mean_abs_error_pct']:.2f}%, "
                     f"median {stats['median_abs_error_pct']:.2f}%.")
    conn = _connect_read_only(path)
    try:
        up, down, total = conn.execute(
            'SELECT COALESCE(SUM(actual_return_pct > 0), 0), COALESCE(SUM(actual_return_pct < 0), 0), COUNT(*) '
            "FROM saved_predictions WHERE status = 'evaluated' AND actual_return_pct IS NOT NULL").fetchone()
        reasons = conn.execute(
            "SELECT COALESCE(unresolved_reason, 'unknown'), COUNT(*) FROM saved_predictions "
            "WHERE status = 'unresolved' GROUP BY 1 ORDER BY 2 DESC, 1").fetchall()
    finally:
        conn.close()
    if total:
        lines.append(
            f'Base rate: {100.0 * up / total:.1f}% of evaluated closes ended above the anchor and '
            f'{100.0 * down / total:.1f}% below, the directional accuracy of always calling bull or always bear.')
    if reasons:
        lines.append('Unresolved: ' + ', '.join(f'{reason} {count:,}' for reason, count in reasons) + '.')
    return lines


def evaluate(args: argparse.Namespace, tracker: Any, path: Path) -> int:
    price_fn = tracker.make_history_price_fn(min_gap_seconds=args.min_gap)
    totals: Counter[str] = Counter()
    round_reasons: Counter[str] = Counter()
    batches = 0
    rounds_left = args.retry_rounds
    stopped_early = interrupted = False
    started = time.monotonic()
    tracker.reset_retry_backoff()
    try:
        while True:
            if args.max_batches is not None and batches >= args.max_batches:
                stopped_early = True
                break
            out = tracker.evaluate_expired_predictions(
                batch_size=args.batch_size, price_fn=price_fn, retry_backoff_seconds=_ROUND_BACKOFF_SECONDS)
            if out['checked']:
                batches += 1
                for key in ('checked', 'evaluated', 'unresolved', 'retry_later'):
                    totals[key] += out[key]
                round_reasons.update(out['retry_reasons'])
                print(f"batch {batches}: checked {out['checked']}, evaluated {out['evaluated']}, "
                      f"unresolved {out['unresolved']}, retry later {out['retry_later']}, "
                      f"expired and open {out['still_open']:,}", flush=True)
                continue
            if out['still_open'] and rounds_left:
                rounds_left -= 1
                print(f"{out['still_open']:,} expired predictions need a retry "
                      f'({_reason_text(round_reasons)}); retrying in {args.retry_wait:g}s.', flush=True)
                time.sleep(args.retry_wait)
                tracker.reset_retry_backoff()
                round_reasons.clear()
                continue
            break
    except KeyboardInterrupt:
        interrupted = True
        print('\nInterrupted. Every decided prediction is saved; run the script again to continue.')
    finally:
        tracker.reset_retry_backoff()

    left = _expired_open(path, datetime.now(timezone.utc))['total']
    print()
    print(f"Checked {totals['checked']:,} in {batches} batches ({time.monotonic() - started:.0f}s): "
          f"{totals['evaluated']:,} evaluated, {totals['unresolved']:,} unresolved.")
    if left:
        why = f' Last retry reasons: {_reason_text(round_reasons)}.' if round_reasons else ''
        how = f'stopped after {batches} batches' if stopped_early else 'run the script again later'
        print(f'{left:,} expired predictions are still open ({how}).{why}')
    else:
        print('No expired prediction is left open.')
    for line in summary_lines(tracker, path):
        print(line)
    if interrupted:
        return EXIT_INTERRUPTED
    return EXIT_OPEN_LEFT if left else EXIT_DONE


def _reason_text(reasons: Counter[str]) -> str:
    return ', '.join(f'{reason} {count:,}' for reason, count in reasons.most_common()) or 'none recorded'


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.db is not None:
        os.environ['MRD_PREDICTIONS_DB'] = str(args.db)
    from app.services import prediction_tracker_service as tracker

    path = tracker.db_path()
    if not path.is_file():
        print(f'No database at {path}.', file=sys.stderr)
        return EXIT_NO_DB
    if args.dry_run:
        return dry_run(path, args.min_gap)
    tracker.init_db()
    return evaluate(args, tracker, path)


if __name__ == '__main__':
    sys.exit(main())
