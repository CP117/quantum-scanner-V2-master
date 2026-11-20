import logging
import time

from fastapi import APIRouter, Query, HTTPException
from app.services.detail_service import get_symbol_detail
from app.utils.input_tolerance import loose_bool, normalize_market, normalize_symbol
from app.utils.json_safe import replace_non_finite

router = APIRouter()
log = logging.getLogger('app.routes.detail')

# Path of a replaced non-finite value -> time.monotonic() it was last logged.
_NON_FINITE_LOGGED: dict[str, float] = {}
NON_FINITE_LOG_INTERVAL_SECONDS = 600.0


def _json_safe_detail(symbol: str, payload):
    """The detail payload with NaN and infinite floats replaced by null.

    One NaN anywhere in the row made the response a 500. Each path that
    needed a replacement is logged as a WARNING, at most once per path
    every NON_FINITE_LOG_INTERVAL_SECONDS, so the source stays visible.
    """
    found: list[str] = []
    clean = replace_non_finite(payload, found)
    if found:
        now = time.monotonic()
        fresh = [p for p in dict.fromkeys(found)
                 if now - _NON_FINITE_LOGGED.get(p, float('-inf')) >= NON_FINITE_LOG_INTERVAL_SECONDS]
        if fresh:
            for p in fresh:
                _NON_FINITE_LOGGED[p] = now
            log.warning('detail %s: %d non-finite value(s) sent as null, at %s',
                        symbol, len(found), ', '.join(fresh[:5]))
    return clean


@router.get('/stock/{symbol}')
def stock_detail(
    symbol: str,
    force_live: str | bool = Query(False),
    require_fresh: str | bool = Query(False),
    market: str = Query('stocks'),
):
    """Per-symbol detail endpoint.

    `require_fresh=True` (Phase 26.68) always triggers a re-fetch of the
    live quote + rescore against the cached daily-hist + intraday shape.
    This is the flag the detail panel's 2-second live tick sends so
    filter-elevated symbols don't stay stuck at partial batch-cheap-pass
    metrics.  Because it reuses the batched quote cache (short TTL) it
    doesn't hammer providers.

    `force_live=True` is heavier — full intraday + daily re-download.
    Reserved for the "Refresh now" button (unblock a lockup / clear a
    rate-limit stall).
    """
    sym = normalize_symbol(symbol)
    if not sym:
        raise HTTPException(status_code=400, detail='symbol must not be empty')
    mkt = normalize_market(market, default='stocks')
    fl = loose_bool(force_live, default=False)
    rf = loose_bool(require_fresh, default=False)
    # Phase 28: record user interaction so the symbol is promoted to at least
    # Tier 2 and protected from demotion for the next 5 minutes.
    try:
        from app.services.tier_manager import record_user_interaction
        record_user_interaction(sym)
    except Exception:  # noqa: BLE001
        pass
    return _json_safe_detail(sym, get_symbol_detail(sym, force_live=fl, market=mkt, require_fresh=rf))
