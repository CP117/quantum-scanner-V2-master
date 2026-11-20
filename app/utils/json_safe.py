"""Make values safe to send as JSON."""
from __future__ import annotations

import math
from typing import Any


def replace_non_finite(value: Any, found: list[str] | None = None, path: str = '') -> Any:
    """Return `value` with every NaN or infinite float replaced by None.

    JSON has no NaN or Infinity, and Starlette's JSONResponse raises
    ValueError on them, so one such float anywhere in a payload turns the
    whole response into a 500. Dicts and lists are copied, and tuples
    become lists, as JSON would make them; anything else is returned
    unchanged. `value` itself is never modified.

    When `found` is a list, the path of each replaced value is appended
    to it, for example
    ``factor_breakdown.market.short_selling_pressure.price_trend_pct_per_wk``
    or ``rows[3].score``.
    """
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if found is not None:
            found.append(path or '<root>')
        return None
    if isinstance(value, dict):
        return {
            key: replace_non_finite(item, found, f'{path}.{key}' if path else str(key))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [replace_non_finite(item, found, f'{path}[{i}]') for i, item in enumerate(value)]
    return value
