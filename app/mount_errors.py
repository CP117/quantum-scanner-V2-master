"""Startup mount failures (Phase 3).

app/main.py mounts most routers inside their own try/except so that one
broken module cannot take the whole app down.  Before Phase 3 a failure
left only a log line (or, for threads_debug, nothing), and the affected
pages and endpoints simply returned 404.  Each handler now calls
record_mount_error(), which keeps the failure in MOUNT_ERRORS
(name -> "ExceptionType: message"), logs it with its traceback, and,
when MRD_STRICT_MOUNT=1, stops startup instead.  /system/status reports
the dict as mount_errors.

This lives in its own module so the status route and the tests can read
it without importing app.main, which starts the scanner.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

MOUNT_ERRORS: dict[str, str] = {}

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MAX_TEXT = 500


def _shorten_paths(text: str) -> str:
    # /system/status is readable without a token, so local paths are
    # shortened: the repo first (it usually sits under the home folder).
    for prefix, label in ((str(_REPO_ROOT), '<repo>'), (str(Path.home()), '~')):
        if len(prefix) > 3:
            text = text.replace(prefix, label)
    return text


def strict_mount() -> bool:
    return os.environ.get('MRD_STRICT_MOUNT', '0').strip() == '1'


def record_mount_error(name: str, exc: BaseException, logger: str = 'app.startup') -> None:
    """Record, log and, in strict mode, re-raise one mount failure."""
    text = _shorten_paths(f'{type(exc).__name__}: {exc}')[:_MAX_TEXT]
    MOUNT_ERRORS[name] = text
    logging.getLogger(logger).exception('%s failed to mount: %s', name, text, exc_info=exc)
    if strict_mount():
        raise RuntimeError(f'{name} failed to mount (MRD_STRICT_MOUNT=1): {text}') from exc


def mount_errors() -> dict[str, str]:
    """A copy of MOUNT_ERRORS for /system/status."""
    return dict(MOUNT_ERRORS)
