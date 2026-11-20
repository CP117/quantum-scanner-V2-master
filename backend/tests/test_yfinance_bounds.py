"""Phase 4: yfinance 1.8.0 and bounded download threads."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _pins() -> dict[str, str]:
    pins = {}
    for line in (REPO / 'requirements.txt').read_text(encoding='utf-8').splitlines():
        line = line.split('#', 1)[0].strip()
        match = re.fullmatch(r'([A-Za-z0-9_.\-\[\]]+)==([^\s;]+)', line)
        if match:
            name = re.sub(r'\[.*\]', '', match.group(1)).lower().replace('-', '_')
            pins[name] = match.group(2)
    return pins


def test_requirements_pin_yfinance_1_8_0_and_curl_cffi_0_17_0():
    pins = _pins()
    assert pins.get('yfinance') == '1.8.0'
    assert pins.get('curl_cffi') == '0.17.0'


def test_the_suite_runs_against_the_pinned_versions():
    from importlib.metadata import version

    pins = _pins()
    assert version('yfinance') == pins['yfinance']
    assert version('curl_cffi') == pins['curl_cffi']
