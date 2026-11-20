"""Phase 26.49 — Future Mode manual refresh + system reset routes.

Two endpoints:

* `POST /api/future_mode/refresh/{symbol}`
    DEEP per-symbol refresh — force a live fetch through the full
    provider cascade, invalidate daily history & quote caches, rescore
    the row at full pass-2 depth, attach BOTH fast + GARCH tier
    forward_metrics, and upsert into the snapshot.  Returns the
    rebuilt detail payload so the dashboard can update in-place.

* `POST /api/system/reset`
    Operator-driven recovery from a stuck pipeline.  Two modes:

      ?mode=soft  →  clear in-memory state only:
        - snapshot buckets + thin caches
        - open / half-open provider circuit breakers + CBOE rate-limit stats
        - Future Mode caches (Advanced, Lab, Strategy, Predictive)
      ?mode=hard  →  soft reset PLUS delete on-disk shard caches
        (anything under `data/cache/*.json`).  Daily-history cache
        and the regulatory SQLite DB are PRESERVED — they're
        expensive to rebuild and not implicated in any wedge we've
        seen.

Neither mode needs to wake the scanner: the snapshot loop sleeps at most
a few seconds between batches (SNAPSHOT_INTER_BATCH_SLEEP 0.5 s,
SNAPSHOT_INTER_MARKET_SLEEP 0.75 s, 5 s when its worker pool is
saturated or a batch failed), so the next batch refills the cleared
snapshot on its own.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException, Query

from app.utils.input_tolerance import loose_bool, normalize_market, normalize_symbol

log = logging.getLogger('app.future_mode_routes')

router = APIRouter(tags=['future_mode'])


# ---------------------------------------------------------------------------
# 1) Per-symbol deep refresh
# ---------------------------------------------------------------------------

@router.post('/api/future_mode/refresh/{symbol}')
def future_mode_deep_refresh(
    symbol: str,
    market: str = Query('stocks'),
):
    """Deep refresh a single symbol through the full Future Mode
    pipeline: force a live quote, invalidate daily history, rescore
    at pass-2 depth, attach fast + GARCH tier forward_metrics, and
    upsert into the snapshot.

    Returns the freshly rebuilt detail payload so the dashboard can
    update the detail panel AND the leaderboard row in-place.
    """
    sym = normalize_symbol(symbol)
    if not sym:
        raise HTTPException(status_code=400, detail='empty_symbol')
    mkt = normalize_market(market, default='stocks')

    t0 = time.monotonic()

    # 1. Mark this market as user-active so live providers stay hot.
    try:
        from app.services.market_activity_service import stamp_active
        stamp_active(mkt)
    except Exception:  # noqa: BLE001
        pass

    # 2. Invalidate caches so the rescore picks fresh data.
    try:
        from app.services.daily_history_service import invalidate as invalidate_daily_history
        invalidate_daily_history(sym)
    except Exception as exc:  # noqa: BLE001
        log.debug('invalidate_daily_history failed for %s: %s', sym, exc)
    try:
        from app.services.quote_cache import invalidate_quote
        invalidate_quote(sym, market=mkt)
    except Exception:  # noqa: BLE001
        pass
    # 3. Drop the Future Mode caches FOR THIS SYMBOL so the next
    #    attach_forward_metrics_* call recomputes the advanced + lab
    #    + strategy bundles from the freshly-fetched daily history.
    try:
        from app.services.future_mode_service import (
            _adv_cache, _adv_cache_lock,
            _lab_cache, _lab_cache_lock,
            _strategy_cache, _strategy_cache_lock,
        )
        with _adv_cache_lock:
            for k in [k for k in _adv_cache if k[0] == sym.upper()]:
                _adv_cache.pop(k, None)
        with _lab_cache_lock:
            for k in [k for k in _lab_cache if k[0] == sym.upper()]:
                _lab_cache.pop(k, None)
        with _strategy_cache_lock:
            for k in [k for k in _strategy_cache if k[0] == sym.upper()]:
                _strategy_cache.pop(k, None)
    except Exception:  # noqa: BLE001
        pass

    # 4. Force-live fetch + full rescore via the detail service.
    from app.services.detail_service import get_symbol_detail
    payload = get_symbol_detail(sym, force_live=True, market=mkt)
    if not payload:
        raise HTTPException(status_code=404, detail=f'no_data_for_{sym}')

    # 4b. Phase 26.50 bugfix — explicitly *block* until daily history is
    #     populated.  The detail-service path uses
    #     `get_daily_history(allow_fetch=True, blocking=False)`, which
    #     silently returns None when YFinance throttle gates fire
    #     (inflight cap, min-gap, cooldown).  When that happens the
    #     subsequent `attach_forward_metrics_garch` call's
    #     `_load_closes_cached(...)` returns `[]`, trips the
    #     `len(closes) < 20` early-return, and the user ends up with
    #     `forward_metrics_garch=None` — which the frontend silently
    #     downgrades to fast-tier display.  The whole *point* of deep
    #     refresh is to escape that downgrade, so block here.
    try:
        from app.services.daily_history_service import get_daily_history
        get_daily_history(sym, allow_fetch=True, blocking=True)
    except Exception as exc:  # noqa: BLE001
        log.warning('deep-refresh daily-history prime failed for %s: %s', sym, exc)

    # 5. Attach FAST + GARCH tier Future Mode blocks.  detail_service
    #    returns a normalised row; rerun the score_symbol_rows pipeline
    #    so forward_metrics gets refreshed against the new data.
    try:
        from app.services.scoring_service import score_symbol_rows
        scored = score_symbol_rows([payload], force_full_pass2=True)
        if scored:
            payload = scored[0]
    except Exception as exc:  # noqa: BLE001
        log.warning('deep-refresh rescore failed for %s: %s', sym, exc)

    # 6. Run the GARCH-tier overlay on this row regardless of whether
    #    it's currently in the top-25.  This is the whole point of the
    #    "deep" refresh — the user wants the GARCH quality immediately.
    try:
        from app.services.future_mode_service import attach_forward_metrics_garch
        attach_forward_metrics_garch(payload, sym, market=mkt)
    except Exception as exc:  # noqa: BLE001
        log.warning('deep-refresh GARCH attach failed for %s: %s', sym, exc)

    # 7. Push the enriched row back into the snapshot so the leader-
    #    board picks it up on its next poll.
    try:
        from app.services.snapshot_store import upsert_rows
        upsert_rows(mkt, [payload])
    except Exception as exc:  # noqa: BLE001
        log.debug('snapshot upsert failed for %s: %s', sym, exc)

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    payload['deep_refresh_elapsed_ms'] = elapsed_ms
    payload['deep_refresh_at_utc'] = int(time.time())
    return payload


# ---------------------------------------------------------------------------
# 2) Soft / hard system reset
# ---------------------------------------------------------------------------

def _reset_in_memory_state() -> dict:
    """Clear the in-memory caches and close the provider circuit breakers.

    Returns what was actually done, for the response and the toast.

    Phase 3 removed three blocks that reported work they did not do:
    the scan kick (neither kick function exists, and the snapshot loop
    refills on its own within seconds), the priority-lane monitor-only
    reset (that flag is a local variable inside the lane's loop; the old
    fallback set module attributes that do not exist and still reported
    True, and the lane leaves monitor-only by itself after its 60 s
    pause and three fast ticks), and the wedge-watchdog reset (the
    watchdog keeps no state; it only rewrites data/wedge_watchdog.json
    every 30 s).
    """
    cleared = {
        'snapshot_rows': 0,
        'advanced_cache': 0,
        'lab_cache': 0,
        'strategy_cache': 0,
        'predictive_cache': 0,
        'circuit_breakers_reset': 0,
        'circuit_breakers_closed': [],
        'rate_limit_stats_reset': False,
    }
    # --- 1) Snapshot buckets ----------------------------------------
    try:
        from app.services import snapshot_store as ss
        for mkt in ('stocks', 'crypto'):
            try:
                with ss._locks.get(mkt, ss._locks['stocks']):
                    cleared['snapshot_rows'] += len(ss._snapshot.get(mkt, {}))
                # clear_snapshot also drops the sorted view, the thin-row
                # cache and the Redis copy, and zeroes rows_scored.
                ss.clear_snapshot(mkt)
                # Reset the sweep counters so the UI shows "rescanning".
                with ss._locks.get(mkt, ss._locks['stocks']):
                    meta = ss._snapshot_meta.get(mkt)
                    if isinstance(meta, dict):
                        meta.update({
                            'current_sweep_scanned': 0,
                            'current_batch_index': 0,
                            'highest_completed_batch': -1,
                        })
            except Exception as exc:  # noqa: BLE001
                log.warning('snapshot reset failed for %s: %s', mkt, exc)
    except Exception as exc:  # noqa: BLE001
        log.warning('snapshot reset partial: %s', exc)

    # --- 2) Future Mode caches --------------------------------------
    try:
        from app.services.future_mode_service import clear_future_mode_caches
        stats = clear_future_mode_caches()
        cleared['advanced_cache'] = stats.get('cleared_advanced', 0)
        cleared['lab_cache'] = stats.get('cleared_lab', 0)
        cleared['strategy_cache'] = stats.get('cleared_strategy', 0)
        cleared['predictive_cache'] = stats.get('cleared_predictive_expansion', 0)
    except Exception as exc:  # noqa: BLE001
        log.warning('future-mode cache reset partial: %s', exc)

    # --- 3) Provider circuit breakers --------------------------------
    # Close every open or half-open breaker in app.services.providers.base
    # and restart its backoff ladder.  (Before Phase 3 this called
    # configure_circuit for a hardcoded list of seven names, which only
    # rewrote their thresholds to 8 / 120 s, closed nothing, and missed
    # yahoo-chart and yfinance+coingecko; it reported 7 every time.)
    try:
        from app.services.providers import base as providers_base
        closed = providers_base.reset_circuits()
        cleared['circuit_breakers_closed'] = closed
        cleared['circuit_breakers_reset'] = len(closed)
    except Exception as exc:  # noqa: BLE001
        log.warning('circuit breaker reset failed: %s', exc)

    # --- 4) Rate-limit telemetry counters ---------------------------
    try:
        from app.services.providers import cboe_options_provider as cp
        with cp._lock:
            for k in cp._rl_stats:
                cp._rl_stats[k] = 0
            cp._last_request_ts = 0.0
        cleared['rate_limit_stats_reset'] = True
    except Exception as exc:  # noqa: BLE001
        log.warning('CBOE rate-limit stats reset failed: %s', exc)

    return cleared


def _wipe_disk_shards() -> dict:
    """Remove the on-disk shard caches.  Uses an explicit ALLOWLIST
    (not a glob) so we never accidentally nuke user-critical state
    files like `user_added_symbols.json`, `sec_ticker_cik.json`, or
    the wedge_watchdog dump (regenerated automatically but useful
    forensic evidence if a recent freeze happened).

    Files we WILL remove:
        - cached_crypto_universe.json     (cheap to refetch from /coins)
        - coingecko_coin_list_cache.json
        - coingecko_catalog_cache.json
        - any file under `data/cache/`     (true shard caches)

    Files we PRESERVE explicitly:
        - regulatory.db                    (expensive; not implicated)
        - daily_history_cache/             (expensive; not implicated)
        - user_added_symbols.json          (USER DATA — never delete)
        - known_bad_symbols.json           (slow learned blacklist)
        - sec_ticker_cik.json              (slow upstream pull)
        - saved_predictions.db             (user history)
        - public_url.txt                   (tunnel URL)
        - variant.json / *_universe.json   (build manifest)

    Returns the list of files actually removed for the response.
    """
    removed: list[str] = []
    # Hard allowlist of cache filenames that are safe to nuke.
    SAFE_TO_DELETE = {
        'cached_crypto_universe.json',
        'coingecko_coin_list_cache.json',
        'coingecko_catalog_cache.json',
    }
    candidates: list[Path] = []
    # 1) Anything under a `cache/` subdirectory is by definition a cache.
    for root in (Path('/app/app/data/cache'), Path('/app/data/cache')):
        if root.exists():
            candidates.extend(root.glob('*.json'))
    # 2) Explicit allowlisted files in the data root.
    for root in (Path('/app/app/data'), Path('/app/data')):
        if root.exists():
            for fname in SAFE_TO_DELETE:
                p = root / fname
                if p.exists():
                    candidates.append(p)
    for f in candidates:
        try:
            if f.is_file():
                f.unlink()
                removed.append(str(f))
        except Exception as exc:  # noqa: BLE001
            log.debug('could not remove %s: %s', f, exc)
    return {'removed_files': removed, 'count': len(removed)}


@router.post('/api/system/reset')
def system_reset(
    mode: str = Query('soft', pattern='^(soft|hard)$'),
    confirm: str | bool = Query(False),
):
    """Operator-driven recovery.

    `mode=soft`: clear in-memory state, leave disk caches intact.
    `mode=hard`: soft reset + wipe `data/cache/*.json`.  Requires
    `?confirm=true` so an accidental click can't nuke disk caches.
    """
    t0 = time.monotonic()
    mode = (mode or 'soft').lower()
    confirmed = loose_bool(confirm, default=False)

    if mode == 'hard' and not confirmed:
        raise HTTPException(
            status_code=400,
            detail='hard_reset_requires_confirm_true',
        )

    in_mem = _reset_in_memory_state()
    disk = None
    if mode == 'hard':
        disk = _wipe_disk_shards()

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    return {
        'mode': mode,
        'in_memory_cleared': in_mem,
        'disk_caches': disk,
        'elapsed_ms': elapsed_ms,
        'note': (
            'The snapshot refills as the scanner finishes its next '
            'batches (it pauses at most a few seconds between them).  '
            'Watch /api/scan/snapshot/meta for the next sweep tick.'
        ),
    }



# ---------------------------------------------------------------------------
# 3) Cache telemetry (Phase 26.50 P2)
#
# Surfaces the three Future Mode caches (Advanced / Lab / Strategy):
#   * size, hits, misses, hit_rate
#   * miss_latency_ms_ema (smoothed) and miss_latency_ms_last
#   * ttl_seconds
#
# Useful both for the operator UI and for regression tests verifying the
# advanced-signals memoisation is doing real work.
# ---------------------------------------------------------------------------
@router.get('/api/future_mode/cache_metrics')
def future_mode_cache_metrics():
    """Return cache hit-rate + latency stats for every Future Mode tier."""
    from app.services.future_mode_service import (
        get_advanced_cache_stats,
        get_lab_cache_stats,
        get_strategy_cache_stats,
        get_predictive_cache_stats,
    )
    adv = get_advanced_cache_stats()
    lab = get_lab_cache_stats()
    strat = get_strategy_cache_stats()
    pred = get_predictive_cache_stats()

    def _aggregate(*stats):
        total_hits = sum(int(s.get('hits', 0)) for s in stats)
        total_misses = sum(int(s.get('misses', 0)) for s in stats)
        total = total_hits + total_misses
        return {
            'hits': total_hits,
            'misses': total_misses,
            'hit_rate': round((total_hits / total) if total > 0 else 0.0, 4),
            'size': sum(int(s.get('size', 0)) for s in stats),
        }

    return {
        'advanced': adv,
        'lab': lab,
        'strategy': strat,
        'predictive_expansion': pred,
        'overall': _aggregate(adv, lab, strat, pred),
        'generated_at_ms': int(time.time() * 1000),
    }



# ---------------------------------------------------------------------------
# 4) Viewport-driven priority registration (Phase 26.51)
#
# Frontend POSTs the symbols currently visible on the user's first
# leaderboard page (post-filter, post-sort) to this endpoint every
# few seconds.  The priority lane reads the registry on every tick
# and ALWAYS includes those symbols in its re-score + GARCH-overlay
# pass — guaranteeing the user's first page is under continuous
# deep scan even when filters / sorts have promoted symbols that
# wouldn't make the global top-N priority cut.
#
# Request shape:
#   { "market": "stocks", "symbols": ["AAPL", "TSLA", ...] }
#
# Response shape:
#   { "ok": true, "accepted": N, "live_after": M, "ttl_seconds": 90 }
# ---------------------------------------------------------------------------
@router.post('/api/future_mode/visible_symbols')
def register_visible_symbols(payload: dict = Body(...)):
    """Register the symbols currently visible on the user's first page
    so the priority lane keeps them under continuous deep scan."""
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail='payload must be a JSON object')
    market = normalize_market(payload.get('market'), default='stocks')
    raw_symbols = payload.get('symbols')
    if raw_symbols is None or not isinstance(raw_symbols, list):
        raise HTTPException(status_code=400, detail='`symbols` must be a JSON array')
    from app.services.visible_symbols import register_visible, get_visible, get_stats
    accepted = register_visible(market, raw_symbols)
    return {
        'ok': True,
        'market': market,
        'accepted': accepted,
        'live_after': len(get_visible(market)),
        'stats': get_stats(),
    }


@router.get('/api/future_mode/visible_symbols')
def list_visible_symbols(market: str = Query('stocks')) -> dict:
    """Read-back for the registry — useful for the operator UI and
    for verifying the frontend ping path is healthy."""
    mkt = normalize_market(market, default='stocks')
    from app.services.visible_symbols import get_visible, get_stats
    return {
        'market': mkt,
        'symbols': get_visible(mkt),
        'stats': get_stats(),
    }


# ---------------------------------------------------------------------------
# Phase 26.60 — Predictive Expansion Pack
# ---------------------------------------------------------------------------
@router.get('/api/registry/phase_2660')
def registry_phase_2660() -> dict:
    """Return the Phase 26.60 metric registry (14 metrics + 5 composite
    multipliers) so the frontend can render labels, units, range hints,
    and ranking-role chips consistently with the existing popovers."""
    from app.services.predictive_expansion_registry import get_registry
    return get_registry()


@router.get('/api/future_mode/predictive_cache_metrics')
def predictive_cache_metrics() -> dict:
    """Telemetry: hit/miss/latency of the Phase 26.60 Predictive
    Expansion per-symbol cache.  Mirrors the existing lab/strategy
    cache_metrics endpoints — useful for sanity-checking that the
    leveraged variant isn't recomputing the bundle on every tick."""
    from app.services.future_mode_service import get_predictive_cache_stats
    return {'predictive_expansion': get_predictive_cache_stats()}
