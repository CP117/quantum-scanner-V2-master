"""Every MRD_ setting the app reads is listed in .env.example.

.github/copilot-instructions.md asks for each new tunable to be listed in
.env.example with its default and a one-line description.  Phases 2C and
3 added five settings without entries, and ten older ones were never
listed; this test keeps the file complete from here on.

Names are found as exact string constants ('MRD_...') in app/ and
scripts/, which is how every os.environ lookup in the code spells them.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_EXAMPLE = REPO_ROOT / '.env.example'
NAME = re.compile(r'^MRD_[A-Z0-9_]+$')
ENTRY = re.compile(r'^(?:# )?(MRD_[A-Z0-9_]+)=')
LAUNCHERS = ('start.bat', 'start.sh')


def _names_in_code() -> dict[str, str]:
    found: dict[str, str] = {}
    for base in ('app', 'scripts'):
        for path in sorted((REPO_ROOT / base).rglob('*.py')):
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and NAME.match(node.value):
                    found.setdefault(node.value, str(path.relative_to(REPO_ROOT)))
    return found


def _entries() -> tuple[dict[str, int], list[str]]:
    lines = ENV_EXAMPLE.read_text(encoding='utf-8').splitlines()
    entries: dict[str, int] = {}
    for index, line in enumerate(lines):
        match = ENTRY.match(line)
        if match:
            assert match.group(1) not in entries, f'{match.group(1)} is listed twice'
            entries[match.group(1)] = index
    return entries, lines


def test_every_setting_read_in_code_is_listed():
    entries, _ = _entries()
    missing = {name: where for name, where in sorted(_names_in_code().items()) if name not in entries}
    assert not missing, f'add to .env.example with a one-line description: {missing}'


def test_every_entry_has_a_description_line_above_it():
    entries, lines = _entries()
    for name, index in entries.items():
        above = lines[index - 1] if index else ''
        assert above.startswith('# ') and not above.startswith('# ---') and not ENTRY.match(above), (
            f'{name} needs a one-line description comment directly above it')


def test_every_entry_is_still_read_somewhere():
    entries, _ = _entries()
    code = set(_names_in_code())
    launchers = ''.join((REPO_ROOT / name).read_text(encoding='utf-8', errors='replace')
                        for name in LAUNCHERS if (REPO_ROOT / name).exists())
    stale = [name for name in entries if name not in code and name not in launchers]
    assert not stale, f'.env.example lists settings nothing reads: {stale}'
