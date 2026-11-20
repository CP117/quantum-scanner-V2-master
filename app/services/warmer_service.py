
from __future__ import annotations
import logging
import threading
import time
from app.config import settings
from app.services.batch_service import get_total_batches, get_batch_slice
from app.services.scoring_service import score_symbol_rows
from app.utils.time import utcnow_iso

log = logging.getLogger('app.warmer')

_status = {
    'enabled': settings.warmer_enabled,
    'running': False,
    'interval_seconds': settings.warmer_interval_seconds,
    'last_cycle_utc': None,
    'warmed_symbols': 0,
    'last_batch': 0,
    'market': 'stocks',
    # The counts from the last auto_log_scan_predictions() call, or
    # {'error': ...} when it raised. This loop walks the whole universe
    # on a timer with no user involvement, so the 'auto_scan' rows it
    # logs are the unbiased sample; manual saves are selection-biased.
    'last_auto_log': None,
}
_thread = None
_stop = threading.Event()


_last_auto_log_error: list[str | None] = [None]


def _auto_log(rows, batch, market):
    """Log scanner predictions for the batch; a failure is recorded, never raised.

    auto_log_scan_predictions saves at most
    settings.auto_log_predictions_max_new_per_cycle rows per tick, and only
    for fully scored symbols, so predict_price reads the snapshot mirror.
    A new error is logged as a warning, a repeat at debug level.
    """
    try:
        from app.services.prediction_tracker_service import auto_log_scan_predictions
        symbols = [r.get('symbol') for r in rows if r.get('symbol')]
        _status['last_auto_log'] = auto_log_scan_predictions(symbols, market=market)
        _last_auto_log_error[0] = None
    except Exception as exc:  # noqa: BLE001 - the warmer must keep warming
        text = f'{type(exc).__name__}: {exc}'
        _status['last_auto_log'] = {'error': text, 'batch': batch, 'market': market}
        level = logging.DEBUG if text == _last_auto_log_error[0] else logging.WARNING
        log.log(level, 'auto_log_scan_predictions failed for batch %d (%s): %s', batch, market, text,
                exc_info=level == logging.WARNING)
        _last_auto_log_error[0] = text


def _loop():
    batch = 0
    last_market = _status.get('market', 'stocks')
    total_batches = get_total_batches(settings.batch_size, last_market)
    while not _stop.is_set():
        market = _status.get('market', 'stocks')
        if market != last_market:
            batch = 0
            total_batches = get_total_batches(settings.batch_size, market)
            last_market = market
        rows = get_batch_slice(batch, settings.batch_size, market)
        try:
            score_symbol_rows(rows)
            _status['warmed_symbols'] += len(rows)
            _status['last_cycle_utc'] = utcnow_iso()
            _status['last_batch'] = batch
            _auto_log(rows, batch, market)
        except Exception:
            _status['last_cycle_utc'] = utcnow_iso()
            _status['last_batch'] = batch
        batch = (batch + 1) % total_batches
        _stop.wait(settings.warmer_interval_seconds)


def start_warmer():
    global _thread
    if not settings.warmer_enabled or (_thread and _thread.is_alive()):
        return
    _status['running'] = True
    _thread = threading.Thread(target=_loop, daemon=True, name='quote-warmer')
    _thread.start()


def stop_warmer():
    _stop.set()
    _status['running'] = False


def warmer_status() -> dict:
    return dict(_status)


def set_warmer_market(market: str = 'stocks') -> None:
    _status['market'] = 'crypto' if str(market).lower() == 'crypto' else 'stocks'
