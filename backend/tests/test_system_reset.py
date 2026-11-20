"""Phase 3: POST /api/system/reset reports what it actually did.

Before Phase 3 the reset called two kick functions that do not exist,
"cleared" a priority-lane flag through module attributes that do not
exist, and "reset" seven circuit breakers by rewriting their thresholds
without closing any of them.  It reported success for all of it.
"""
from __future__ import annotations

import time
from collections import defaultdict

import pytest
from fastapi import HTTPException

from app.routes import future_mode
from app.services import snapshot_store as ss
from app.services.providers import base


TEST_NAMES = ('t-open', 't-half-open', 't-closed-failing', 't-healthy')


@pytest.fixture
def breakers():
    """Four providers in different breaker states; removed afterwards."""
    base.configure_circuit('t-open', threshold=2, cooldown_seconds=30.0)
    base.record_error('t-open', 'boom')
    base.record_error('t-open', 'boom')
    base.configure_circuit('t-half-open', threshold=1, cooldown_seconds=30.0)
    base.record_error('t-half-open', 'boom')
    with base._lock_for('t-half-open'):
        row = base._stats['t-half-open']
        # Pretend this breaker already climbed the ladder and is probing.
        row['circuit_state'] = 'half_open'
        row['circuit_trip_count'] = 4
        row['cb_current_cooldown_seconds'] = 3600.0
    base.record_error('t-closed-failing', 'boom')
    base.record_error('t-closed-failing', 'boom')
    base.record_call('t-healthy')
    base.record_hit('t-healthy')
    assert base._stats['t-open']['circuit_state'] == 'open'
    assert base._stats['t-closed-failing']['circuit_state'] == 'closed'
    yield
    with base._locks_table_lock:
        for name in TEST_NAMES:
            base._stats.pop(name, None)


def _row(name):
    return base.provider_stats_snapshot()[name]


def test_reset_circuits_closes_open_and_half_open_breakers(breakers):
    closed = base.reset_circuits()

    assert 't-open' in closed and 't-half-open' in closed
    assert 't-closed-failing' not in closed and 't-healthy' not in closed
    assert closed == sorted(closed)
    for name in TEST_NAMES:
        row = _row(name)
        assert row['circuit_state'] == 'closed'
        assert row['consecutive_failures'] == 0
        assert row['circuit_trip_count'] == 0
        assert row['circuit_remaining_seconds'] == 0.0
        assert row['circuit_open_until_utc'] is None
    # The ladder restarts at the provider's own base cooldown.
    assert _row('t-half-open')['cb_current_cooldown_seconds'] == 30.0
    assert _row('t-healthy')['cb_current_cooldown_seconds'] == base._CB_DEFAULT_COOLDOWN


def test_reset_circuits_keeps_tunables_and_counters(breakers):
    base.reset_circuits()
    assert _row('t-open')['cb_threshold'] == 2
    assert _row('t-open')['cb_cooldown_seconds'] == 30.0
    assert _row('t-closed-failing')['cb_threshold'] == base._CB_DEFAULT_THRESHOLD
    assert _row('t-open')['errors'] == 2
    assert _row('t-healthy')['hits'] == 1


def test_reset_circuits_does_not_invent_rows(monkeypatch):
    # A private table, so rows other tests created cannot hide a new one.
    table = defaultdict(base._new_stat_row)
    monkeypatch.setattr(base, '_stats', table)
    base.record_call('t-healthy')
    base.reset_circuits()
    assert set(table) == {'t-healthy'}


def test_breaker_needs_a_full_threshold_again_after_reset(breakers):
    base.reset_circuits()
    base.record_error('t-open', 'boom')
    assert _row('t-open')['circuit_state'] == 'closed'
    base.record_error('t-open', 'boom')
    row = _row('t-open')
    assert row['circuit_state'] == 'open'
    # First trip after the reset uses the base cooldown, not the ladder.
    assert row['circuit_trip_count'] == 1
    assert row['cb_current_cooldown_seconds'] == 30.0


@pytest.fixture
def seeded_snapshot():
    ss.clear_snapshot()
    rows = [
        {'symbol': f'T{i}', 'final_score': 50.0 + i, 'state': 'ready'}
        for i in range(3)
    ]
    ss.upsert_rows('stocks', rows)
    ss.upsert_rows('crypto', [{'symbol': 'TBTC', 'final_score': 60.0, 'state': 'ready'}])
    with ss._locks['stocks']:
        ss._snapshot_meta['stocks'].update(current_sweep_scanned=40, current_batch_index=7)
    yield
    ss.clear_snapshot()


def test_soft_reset_reports_what_it_did(breakers, seeded_snapshot):
    out = future_mode.system_reset(mode='soft', confirm=False)

    assert out['mode'] == 'soft'
    assert out['disk_caches'] is None
    for key in ('scan_kicked',):
        assert key not in out
    cleared = out['in_memory_cleared']
    for key in ('priority_lane_monitor_only_cleared', 'wedge_watchdog_state_reset'):
        assert key not in cleared
    assert cleared['snapshot_rows'] == 4
    assert 't-open' in cleared['circuit_breakers_closed']
    assert 't-half-open' in cleared['circuit_breakers_closed']
    assert cleared['circuit_breakers_reset'] == len(cleared['circuit_breakers_closed'])
    assert cleared['rate_limit_stats_reset'] is True
    for key in ('advanced_cache', 'lab_cache', 'strategy_cache', 'predictive_cache'):
        assert isinstance(cleared[key], int)
    assert _row('t-open')['circuit_state'] == 'closed'

    assert ss._snapshot['stocks'] == {} and ss._snapshot['crypto'] == {}
    meta = ss._snapshot_meta['stocks']
    assert meta['current_sweep_scanned'] == 0
    assert meta['current_batch_index'] == 0
    assert meta['highest_completed_batch'] == -1
    assert meta['rows_scored'] == 0


def test_soft_reset_leaves_other_providers_tunables_alone(breakers):
    before = {name: (row['cb_threshold'], row['cb_cooldown_seconds'])
              for name, row in base.provider_stats_snapshot().items()}
    future_mode.system_reset(mode='soft', confirm=False)
    after = {name: (row['cb_threshold'], row['cb_cooldown_seconds'])
             for name, row in base.provider_stats_snapshot().items()}
    # The old reset rewrote every listed provider to 8 failures / 120 s.
    assert after == before


def test_second_reset_closes_nothing(breakers):
    future_mode.system_reset(mode='soft', confirm=False)
    out = future_mode.system_reset(mode='soft', confirm=False)
    assert out['in_memory_cleared']['circuit_breakers_reset'] == 0
    assert out['in_memory_cleared']['circuit_breakers_closed'] == []


def test_hard_reset_still_requires_confirm():
    with pytest.raises(HTTPException) as err:
        future_mode.system_reset(mode='hard', confirm=False)
    assert err.value.status_code == 400


def test_reset_is_quick():
    t0 = time.monotonic()
    out = future_mode.system_reset(mode='soft', confirm=False)
    assert time.monotonic() - t0 < 2.0
    assert out['elapsed_ms'] < 2000
