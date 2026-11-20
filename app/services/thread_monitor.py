"""Phase 4: thread counts and process memory for /system/status, and the
thread alarm the wedge watchdog raises.

`thread_summary()` is cheap (names only, no stacks), so /system/status can
call it on every request. `check_thread_alarm()` logs a WARNING when the
Python thread count goes over MRD_THREAD_ALARM (default 400) and an ERROR,
with the most common thread-name prefixes, over twice that. It logs when
the level changes and every 10 minutes while it stays raised; the wedge
watchdog calls it on its 30 s cadence.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
from collections import Counter

log = logging.getLogger('app.thread_monitor')

DEFAULT_THREAD_ALARM = 400
ALARM_REPEAT_SECONDS = 600.0

_alarm_lock = threading.Lock()
_alarm_state = {'level': 'ok', 'logged_at': float('-inf')}
_bad_alarm_values: set[str] = set()


def thread_prefix(name: str) -> str:
    """'snap-worker-3' -> 'snap-worker'; 'ThreadPoolExecutor-0_12' -> 'ThreadPoolExecutor'.

    The same rule as scripts/soak.py, so both report the same prefixes.
    """
    stripped = re.sub(r'[-_ ]?\d+', '', name or '').strip(' -_')
    return stripped or '<unnamed>'


def thread_alarm_threshold() -> int:
    """MRD_THREAD_ALARM, default 400. Anything but a whole number of at
    least 1 uses the default (with one warning per bad value)."""
    raw = (os.environ.get('MRD_THREAD_ALARM') or '').strip()
    if not raw:
        return DEFAULT_THREAD_ALARM
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value >= 1:
        return value
    with _alarm_lock:
        first = raw not in _bad_alarm_values
        _bad_alarm_values.add(raw)
    if first:
        log.warning('MRD_THREAD_ALARM=%r is not a whole number of at least 1; using %d',
                    raw, DEFAULT_THREAD_ALARM)
    return DEFAULT_THREAD_ALARM


def alarm_level(total: int, threshold: int) -> str:
    """'error' over twice the threshold, 'warning' over it, else 'ok'."""
    if total > 2 * threshold:
        return 'error'
    if total > threshold:
        return 'warning'
    return 'ok'


def _linux_status_field(field: str) -> int | None:
    try:
        with open('/proc/self/status', encoding='ascii') as fh:
            for line in fh:
                if line.startswith(field + ':'):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def _windows_rss_bytes() -> int | None:
    """Working set of this process from GetProcessMemoryInfo (psapi)."""
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ('cb', wintypes.DWORD),
            ('PageFaultCount', wintypes.DWORD),
            ('PeakWorkingSetSize', ctypes.c_size_t),
            ('WorkingSetSize', ctypes.c_size_t),
            ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
            ('QuotaPagedPoolUsage', ctypes.c_size_t),
            ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
            ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
            ('PagefileUsage', ctypes.c_size_t),
            ('PeakPagefileUsage', ctypes.c_size_t),
        ]

    try:
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        try:
            get_info = kernel32.K32GetProcessMemoryInfo
        except AttributeError:
            get_info = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
        get_info.restype = wintypes.BOOL
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(ProcessMemoryCounters)
        if not get_info(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return None
        return int(counters.WorkingSetSize)
    except (OSError, AttributeError, ValueError):
        return None


def process_rss_mb() -> float | None:
    """Resident memory of this process in MB, or None if it can't be read.

    Linux reads /proc/self/status, Windows calls GetProcessMemoryInfo, and
    anything else uses psutil when it is installed (it is not a dependency).
    """
    rss_bytes = None
    if sys.platform.startswith('linux'):
        kib = _linux_status_field('VmRSS')
        rss_bytes = kib * 1024 if kib is not None else None
    elif sys.platform == 'win32':
        rss_bytes = _windows_rss_bytes()
    if rss_bytes is None:
        try:
            import psutil  # noqa: WPS433
        except ImportError:
            return None
        try:
            rss_bytes = psutil.Process(os.getpid()).memory_info().rss
        except (psutil.Error, OSError):
            return None
    return round(rss_bytes / (1024 * 1024), 1)


def thread_summary(top: int = 10) -> dict:
    """Python thread count, the `top` most common name prefixes, the alarm
    threshold and level, the OS thread count (Linux) and process RSS."""
    threads = threading.enumerate()
    total = len(threads)
    threshold = thread_alarm_threshold()
    counts = Counter(thread_prefix(t.name) for t in threads)
    return {
        'total': total,
        'daemon': sum(1 for t in threads if t.daemon),
        'os_threads': _linux_status_field('Threads') if sys.platform.startswith('linux') else None,
        'top_prefixes': dict(counts.most_common(top)),
        'alarm_threshold': threshold,
        'alarm_level': alarm_level(total, threshold),
        'rss_mb': process_rss_mb(),
    }


def check_thread_alarm(summary: dict | None = None, now: float | None = None) -> str:
    """Log the thread alarm when its level changes, and every
    ALARM_REPEAT_SECONDS while it stays raised. Returns the level."""
    summary = summary if summary is not None else thread_summary()
    level = summary.get('alarm_level', 'ok')
    now = time.monotonic() if now is None else now
    with _alarm_lock:
        previous = _alarm_state['level']
        due = level != previous or (
            level != 'ok' and now - _alarm_state['logged_at'] >= ALARM_REPEAT_SECONDS
        )
        _alarm_state['level'] = level
        if due:
            _alarm_state['logged_at'] = now
    if not due:
        return level
    total = summary.get('total')
    threshold = summary.get('alarm_threshold')
    prefixes = ', '.join(f'{name}: {count}' for name, count in (summary.get('top_prefixes') or {}).items())
    if level == 'error':
        log.error('%s Python threads, over twice MRD_THREAD_ALARM (%s). Top prefixes: %s',
                  total, threshold, prefixes)
    elif level == 'warning':
        log.warning('%s Python threads, over MRD_THREAD_ALARM (%s). Top prefixes: %s',
                    total, threshold, prefixes)
    else:
        log.info('%s Python threads, back under MRD_THREAD_ALARM (%s)', total, threshold)
    return level
