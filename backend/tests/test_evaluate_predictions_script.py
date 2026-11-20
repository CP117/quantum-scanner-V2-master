"""scripts/evaluate_predictions.py: dry run, batches, retry rounds, resume and summary."""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.services import prediction_tracker_service as tracker
from app.services.prediction_tracker_service import CloseLookup

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    'evaluate_predictions_script', _REPO_ROOT / 'scripts' / 'evaluate_predictions.py')
script = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
sys.modules[_SPEC.name] = script
_SPEC.loader.exec_module(script)

T0 = datetime(2026, 7, 9, 20, 55, tzinfo=timezone.utc)
RETRY = CloseLookup(reason='rate_limited', retry=True)


@pytest.fixture(autouse=True)
def _fresh_retry_state():
    tracker.reset_retry_backoff()
    yield
    tracker.reset_retry_backoff()


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / 'predictions.db'
    # main() writes the variable itself; setting it here makes monkeypatch restore it.
    monkeypatch.setenv('MRD_PREDICTIONS_DB', str(path))
    monkeypatch.delenv('MRD_PREDICTION_MAGNITUDE_TOLERANCE_PCT', raising=False)
    return path


def _save_all(monkeypatch, *rows):
    """Save rows one minute apart from T0; returns their ids."""
    ticks = iter(T0 + timedelta(minutes=i) for i in range(10_000))
    ids = []
    with monkeypatch.context() as m:
        m.setattr(tracker, '_utcnow', lambda: next(ticks))
        for symbol, overrides in rows:
            payload = {
                'symbol': symbol, 'market': 'stocks', 'anchor_price': 100.0, 'target_price': 105.0,
                'direction': 'bull', 'forward_days': 10, 'source': 'auto_scan',
            }
            payload.update(overrides)
            ids.append(tracker.save_prediction(payload)['id'])
    return ids


class FakeYahoo:
    """Stands in for make_history_price_fn; answers per symbol, in order."""

    def __init__(self, answers):
        self.answers = {sym: list(a) if isinstance(a, list) else [a] for sym, a in answers.items()}
        self.seen = []
        self.factory_calls = []

    def factory(self, **kwargs):
        self.factory_calls.append(kwargs)
        return self.price_fn

    def price_fn(self, row):
        self.seen.append(row['symbol'])
        queue = self.answers[row['symbol']]
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(answer, BaseException):
            raise answer
        return answer


@pytest.fixture
def yahoo(monkeypatch):
    holder = {}

    def install(answers):
        fake = FakeYahoo(answers)
        monkeypatch.setattr(tracker, 'make_history_price_fn', fake.factory)
        holder['fake'] = fake
        return fake
    return install


def _statuses(path):
    with sqlite3.connect(path) as conn:
        return dict(conn.execute('SELECT symbol, status FROM saved_predictions').fetchall())


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def test_dry_run_counts_and_looks_nothing_up(db, tmp_path, monkeypatch, capsys):
    _save_all(monkeypatch, ('AAPL', {}), ('MSFT', {}), ('BTC', {'market': 'crypto', 'source': 'user'}),
              ('LATER', {}))
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE saved_predictions SET expires_at = '2099-01-01T00:00:00+00:00' WHERE symbol = 'LATER'")
    before = _statuses(db)
    monkeypatch.setenv('MRD_PREDICTIONS_DB', str(tmp_path / 'decoy.db'))  # --db wins
    monkeypatch.setattr(tracker, 'make_history_price_fn', lambda **kw: pytest.fail('dry run looked a price up'))
    assert script.main(['--dry-run', '--db', str(db), '--min-gap', '30']) == 0
    out = capsys.readouterr().out
    assert f'Database: {db}' in out
    assert 'Expired and open: 3\n' in out
    assert '  crypto/user: 1, expiring 2026-07-19 to 2026-07-19\n' in out
    assert '  stocks/auto_scan: 2, expiring 2026-07-19 to 2026-07-19\n' in out
    assert 'Not expired yet: 1\n' in out
    assert 'Symbols to look up: 3 (about 2 min at --min-gap 30)\n' in out
    assert 'Dry run: nothing was looked up or written.' in out
    assert _statuses(db) == before


def test_dry_run_leaves_a_legacy_database_alone(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'legacy.db'
    monkeypatch.setenv('MRD_PREDICTIONS_DB', str(path))
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE saved_predictions (id TEXT PRIMARY KEY, symbol TEXT, market TEXT, '
                     'source TEXT, status TEXT, expires_at TEXT)')
        conn.executemany('INSERT INTO saved_predictions VALUES (?, ?, ?, ?, ?, ?)', [
            ('1', 'AAPL', 'stocks', 'auto_scan', 'expired', '2026-07-19T20:55:00+00:00'),
            ('2', 'MSFT', 'stocks', 'auto_scan', 'open', '2026-07-20T20:55:00+00:00'),
            ('3', 'KO', 'stocks', 'auto_scan', 'correct', '2026-07-20T20:55:00+00:00'),
        ])
    assert script.main(['--dry-run', '--db', str(path)]) == 0
    assert 'Expired and open: 2\n' in capsys.readouterr().out
    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute('PRAGMA table_info(saved_predictions)')}
        statuses = sorted(r[0] for r in conn.execute('SELECT status FROM saved_predictions'))
    assert 'unresolved_reason' not in columns
    assert statuses == ['correct', 'expired', 'open']


def test_dry_run_on_an_empty_database(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'empty.db'
    monkeypatch.setenv('MRD_PREDICTIONS_DB', str(path))
    sqlite3.connect(path).close()
    assert script.main(['--dry-run', '--db', str(path)]) == 0
    assert 'Expired and open: 0\n' in capsys.readouterr().out


def test_a_missing_database_is_an_error_and_is_not_created(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'nope.db'
    monkeypatch.setenv('MRD_PREDICTIONS_DB', str(path))
    assert script.main(['--db', str(path)]) == 1
    assert script.main(['--db', str(path), '--dry-run']) == 1
    assert f'No database at {path}.' in capsys.readouterr().err
    assert not path.exists()


@pytest.mark.parametrize('argv', [
    ['--batch-size', '0'], ['--min-gap', '-1'], ['--retry-wait', '-1'], ['--retry-rounds', '-1'],
    ['--max-batches', '-1'],
])
def test_bad_options_are_rejected(argv):
    with pytest.raises(SystemExit) as exc:
        script.parse_args(argv)
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# Evaluating
# ---------------------------------------------------------------------------

def test_a_run_evaluates_everything_and_prints_the_summary(db, monkeypatch, capsys, yahoo):
    _save_all(
        monkeypatch,
        ('AAPL', {}),
        ('TSLA', {'direction': 'bear', 'target_price': 95.0}),
        ('KO', {'direction': 'neutral', 'target_price': 100.0}),
        ('MSFT', {'source': 'user'}),
        ('DEAD', {}),
    )
    fake = yahoo({
        'AAPL': CloseLookup(close=104.0, session_date=date(2026, 7, 20)),
        'TSLA': 103.0, 'KO': 101.0, 'MSFT': 106.0,
        'DEAD': CloseLookup(reason='no_history'),
    })
    assert script.main(['--db', str(db), '--batch-size', '2', '--min-gap', '0.25']) == 0
    out = capsys.readouterr().out
    assert fake.factory_calls == [{'min_gap_seconds': 0.25}]
    assert fake.seen == ['AAPL', 'TSLA', 'KO', 'MSFT', 'DEAD']
    assert 'batch 1: checked 2, evaluated 2, unresolved 0, retry later 0, expired and open 3\n' in out
    assert 'batch 3: checked 1, evaluated 0, unresolved 1, retry later 0, expired and open 0\n' in out
    assert 'Checked 5 in 3 batches' in out and '4 evaluated, 1 unresolved.' in out
    assert 'No expired prediction is left open.' in out
    assert 'Saved 5: 4 evaluated, 1 unresolved, 0 open.' in out
    assert 'within +/-2% of the target' in out
    table = [
        script._table_row('all', 'all', 4, 66.7, 75.0),
        script._table_row('all', 'bull', 2, 100.0, 100.0),
        script._table_row('all', 'bear', 1, 0.0, 0.0),
        script._table_row('all', 'neutral', 1, None, 100.0),
        script._table_row('auto_scan', 'all', 3, 50.0, 66.7),
        script._table_row('auto_scan', 'bull', 1, 100.0, 100.0),
        script._table_row('auto_scan', 'bear', 1, 0.0, 0.0),
        script._table_row('auto_scan', 'neutral', 1, None, 100.0),
        script._table_row('user', 'all', 1, 100.0, 100.0),
        script._table_row('user', 'bull', 1, 100.0, 100.0),
    ]
    lines = out.splitlines()
    header = lines.index(f"  {'source':<10}{'direction':<10}{'evaluated':>10}{'directional':>13}{'magnitude':>11}")
    assert lines[header + 1:header + 1 + len(table)] == table
    assert not lines[header + 1 + len(table)].startswith('  ')
    assert '       n/a     100.0%' in table[3]
    assert 'Base rate: 100.0% of evaluated closes ended above the anchor and 0.0% below' in out
    assert 'Unresolved: no_history 1.' in out
    assert set(_statuses(db).values()) == {'evaluated', 'unresolved'}


def test_rows_that_need_a_retry_get_more_rounds(db, monkeypatch, capsys, yahoo):
    _save_all(monkeypatch, ('SLOW', {}), ('FAST', {}))
    yahoo({'SLOW': [RETRY, 101.0], 'FAST': 102.0})
    slept = []
    monkeypatch.setattr(script.time, 'sleep', slept.append)
    assert script.main(['--db', str(db), '--retry-wait', '7', '--retry-rounds', '1']) == 0
    out = capsys.readouterr().out
    assert slept == [7.0]
    assert '1 expired predictions need a retry (rate_limited 1); retrying in 7s.' in out
    assert 'Checked 3 in 2 batches' in out and '2 evaluated, 0 unresolved.' in out
    assert _statuses(db) == {'SLOW': 'evaluated', 'FAST': 'evaluated'}


def test_retry_rounds_are_bounded(db, monkeypatch, capsys, yahoo):
    _save_all(monkeypatch, ('SLOW', {}))
    fake = yahoo({'SLOW': RETRY})
    monkeypatch.setattr(script.time, 'sleep', lambda s: None)
    assert script.main(['--db', str(db), '--retry-rounds', '2']) == 2
    assert fake.seen == ['SLOW'] * 3
    out = capsys.readouterr().out
    assert '1 expired predictions are still open (run the script again later). ' \
           'Last retry reasons: rate_limited 1.' in out
    assert _statuses(db) == {'SLOW': 'open'}


def test_without_retry_rounds_a_retry_waits_for_the_next_run(db, monkeypatch, capsys, yahoo):
    _save_all(monkeypatch, ('SLOW', {}))
    fake = yahoo({'SLOW': [RETRY, 101.0]})
    monkeypatch.setattr(script.time, 'sleep', lambda s: pytest.fail('slept without retry rounds'))
    assert script.main(['--db', str(db), '--retry-rounds', '0']) == 2
    assert script.main(['--db', str(db), '--retry-rounds', '0']) == 0
    assert fake.seen == ['SLOW', 'SLOW']


def test_a_run_starts_without_old_deferrals(db, monkeypatch, capsys, yahoo):
    _save_all(monkeypatch, ('SLOW', {}))
    tracker.evaluate_expired_predictions(price_fn=lambda row: RETRY)  # deferred for ten minutes
    fake = yahoo({'SLOW': 101.0})
    assert script.main(['--db', str(db), '--retry-rounds', '0']) == 0
    assert fake.seen == ['SLOW']


def test_max_batches_stops_and_the_next_run_resumes(db, monkeypatch, capsys, yahoo):
    _save_all(monkeypatch, *((sym, {}) for sym in ('A', 'B', 'C', 'D', 'E')))
    fake = yahoo({sym: 101.0 for sym in 'ABCDE'})
    assert script.main(['--db', str(db), '--batch-size', '2', '--max-batches', '1']) == 2
    assert '3 expired predictions are still open (stopped after 1 batches).' in capsys.readouterr().out
    assert fake.seen == ['A', 'B']
    assert script.main(['--db', str(db), '--batch-size', '2']) == 0
    assert fake.seen == ['A', 'B', 'C', 'D', 'E']


def test_an_interrupt_keeps_what_was_decided(db, monkeypatch, capsys, yahoo):
    _save_all(monkeypatch, ('A', {}), ('B', {}), ('C', {}))
    yahoo({'A': 101.0, 'B': KeyboardInterrupt(), 'C': 101.0})
    assert script.main(['--db', str(db)]) == 130
    out = capsys.readouterr().out
    assert 'Interrupted. Every decided prediction is saved' in out
    assert '2 expired predictions are still open' in out
    assert _statuses(db) == {'A': 'evaluated', 'B': 'open', 'C': 'open'}
    assert not tracker._EVAL_LOCK.locked()
