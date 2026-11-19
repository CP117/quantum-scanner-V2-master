"""Static check: every import of an ``app.*`` module or name must exist.

Several silent failures in this codebase came from imports of modules or
functions that were never written, hidden inside ``try/except`` blocks.
This test parses every ``.py`` file under ``app/`` with :mod:`ast` (it
never imports them) and checks:

* ``import app.x.y`` and ``from app.x import y``, including imports inside
  functions and ``try`` blocks, and relative imports: the module must be
  a file or package under ``app/``, and each imported name must be a
  submodule or be bound at the target module's top level (def, class,
  assignment, import, a ``global`` assignment, or a name in ``__all__``);
* attribute access on module aliases, e.g.
  ``from app.services import prediction_tracker_service as tracker`` and
  then ``tracker.list_predictions(...)``: the attribute must be bound at
  the aliased module's top level.

``KNOWN_BROKEN`` lists the failures that existed when this test was
added.  The suite fails on any failure that is not in the list, and also
when a listed entry starts resolving, so the list can only shrink.
"""
from __future__ import annotations

import ast
import functools
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = REPO_ROOT / 'app'

# (file relative to the repo root, dotted target that does not resolve)
KNOWN_BROKEN: set[tuple[str, str]] = {
    # detail_service imports modules that were never added to the repo.
    ('app/services/detail_service.py', 'app.services.provider_service'),
    ('app/services/detail_service.py', 'app.services.stateful_kit'),
    # future_mode's "scan now" path imports a function and a module that do not exist.
    ('app/routes/future_mode.py', 'app.services.active_scan_service.request_immediate_iteration'),
    ('app/routes/future_mode.py', 'app.services.scan_worker'),
    # future_mode's reset is hasattr-guarded, but neither helper exists: the lane's
    # monitor-only flag is a loop-local variable, so the reset reports success and clears nothing.
    ('app/routes/future_mode.py', 'app.services.top10_priority_service.reset_monitor_state'),
    ('app/routes/future_mode.py', 'app.services.wedge_watchdog.reset_state'),
    # main.py tries to start a prediction evaluator that the service never defined.
    ('app/main.py', 'app.services.prediction_tracker_service.start_evaluator'),
    # The prediction tracker routes call service functions that do not exist.
    ('app/routes/prediction_tracker.py', 'app.services.prediction_tracker_service.list_predictions'),
    ('app/routes/prediction_tracker.py', 'app.services.prediction_tracker_service.delete_prediction'),
    ('app/routes/prediction_tracker.py', 'app.services.prediction_tracker_service.evaluate_expired_predictions'),
}

_MODULE_ATTRS = {
    '__name__', '__file__', '__doc__', '__dict__', '__spec__', '__path__',
    '__package__', '__loader__', '__all__', '__builtins__', '__cached__',
}


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(REPO_ROOT).with_suffix('').parts)
    if parts[-1] == '__init__':
        parts = parts[:-1]
    return '.'.join(parts)


@functools.lru_cache(maxsize=None)
def _index() -> dict[str, Path | None]:
    """Dotted module name -> file (None for a package without __init__)."""
    index: dict[str, Path | None] = {}
    for directory in [APP_DIR, *[p for p in APP_DIR.rglob('*') if p.is_dir()]]:
        if '__pycache__' in directory.parts:
            continue
        name = _module_name(directory / '__init__.py')
        init = directory / '__init__.py'
        index.setdefault(name, init if init.exists() else None)
    for path in APP_DIR.rglob('*.py'):
        if '__pycache__' in path.parts:
            continue
        index[_module_name(path)] = path
    return index


@functools.lru_cache(maxsize=None)
def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))


def _add_target(target: ast.AST, names: set[str]) -> None:
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            _add_target(elt, names)
    elif isinstance(target, ast.Starred):
        _add_target(target.value, names)


def _collect(stmts, names: set[str], stars: list[ast.ImportFrom], flags: dict) -> None:
    for node in stmts:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            if node.name == '__getattr__' and not isinstance(node, ast.ClassDef):
                flags['dynamic'] = True
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                _add_target(target, names)
            if any(isinstance(t, ast.Name) and t.id == '__all__' for t in node.targets):
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            names.add(elt.value)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            _add_target(node.target, names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == '*':
                    stars.append(node)
                else:
                    names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            _add_target(node.target, names)
            _collect(node.body, names, stars, flags)
            _collect(node.orelse, names, stars, flags)
        elif isinstance(node, ast.While):
            _collect(node.body, names, stars, flags)
            _collect(node.orelse, names, stars, flags)
        elif isinstance(node, ast.If):
            _collect(node.body, names, stars, flags)
            _collect(node.orelse, names, stars, flags)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    _add_target(item.optional_vars, names)
            _collect(node.body, names, stars, flags)
        elif isinstance(node, ast.Try) or type(node).__name__ == 'TryStar':
            _collect(node.body, names, stars, flags)
            for handler in node.handlers:
                if handler.name:
                    names.add(handler.name)
                _collect(handler.body, names, stars, flags)
            _collect(node.orelse, names, stars, flags)
            _collect(node.finalbody, names, stars, flags)


def _resolve_from(current: str, is_package: bool, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module
    package = current if is_package else current.rpartition('.')[0]
    parts = package.split('.') if package else []
    if node.level - 1 > len(parts):
        return None
    if node.level > 1:
        parts = parts[: len(parts) - (node.level - 1)]
    if node.module:
        parts.append(node.module)
    return '.'.join(parts)


@functools.lru_cache(maxsize=None)
def _top_level(module: str) -> tuple[frozenset[str], bool]:
    """(names bound at the module's top level, accepts-any-name flag)."""
    path = _index().get(module)
    if path is None:
        return frozenset(), False
    tree = _parse(path)
    names: set[str] = set()
    stars: list[ast.ImportFrom] = []
    flags = {'dynamic': False}
    _collect(tree.body, names, stars, flags)
    # Names assigned through `global` inside functions are module-level too.
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            declared = {n for stmt in ast.walk(node) if isinstance(stmt, ast.Global) for n in stmt.names}
            names.update(declared)
    dynamic = flags['dynamic']
    for star in stars:
        target = _resolve_from(module, path.name == '__init__.py', star)
        if target and (target == 'app' or target.startswith('app.')) and target in _index():
            star_names, star_dynamic = _top_level(target)
            names.update(star_names)
            dynamic = dynamic or star_dynamic
        else:
            dynamic = True  # star import from outside app/: cannot enumerate
    return frozenset(names), dynamic


def _name_exists(module: str, name: str) -> bool:
    if f'{module}.{name}' in _index():
        return True
    names, dynamic = _top_level(module)
    return dynamic or name in names


def _is_app(module: str | None) -> bool:
    return bool(module) and (module == 'app' or module.startswith('app.'))


def _check_file(path: Path) -> dict[tuple[str, str], int]:
    """Return {(file, unresolved target): first line number}."""
    rel = _rel(path)
    failures: dict[tuple[str, str], int] = {}

    def fail(target: str, lineno: int) -> None:
        failures.setdefault((rel, target), lineno)

    try:
        tree = _parse(path)
    except SyntaxError as exc:  # compileall in CI reports the details
        fail('<syntax error>', exc.lineno or 0)
        return failures

    current = _module_name(path)
    is_package = path.name == '__init__.py'
    index = _index()
    aliases: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _is_app(alias.name):
                    continue
                if alias.name not in index:
                    fail(alias.name, node.lineno)
                elif alias.asname:
                    aliases[alias.asname] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = _resolve_from(current, is_package, node)
            if not _is_app(module):
                continue
            if module not in index:
                fail(module, node.lineno)
                continue
            for alias in node.names:
                if alias.name == '*':
                    continue
                if not _name_exists(module, alias.name):
                    fail(f'{module}.{alias.name}', node.lineno)
                elif f'{module}.{alias.name}' in index:
                    aliases[alias.asname or alias.name] = f'{module}.{alias.name}'

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name)
            and node.value.id in aliases
            and node.attr not in _MODULE_ATTRS
        ):
            module = aliases[node.value.id]
            if not _name_exists(module, node.attr):
                fail(f'{module}.{node.attr}', node.lineno)
    return failures


@functools.lru_cache(maxsize=None)
def _all_failures() -> dict[tuple[str, str], int]:
    failures: dict[tuple[str, str], int] = {}
    for path in sorted(APP_DIR.rglob('*.py')):
        if '__pycache__' in path.parts:
            continue
        for key, lineno in _check_file(path).items():
            failures.setdefault(key, lineno)
    return failures


def _format(entries) -> str:
    return '\n'.join(f'  {file}:{line}: {target}' for (file, target), line in entries)


def test_checker_sees_the_app_package():
    assert 'app.main' in _index()
    assert 'app.services.snapshot_store' in _index()


def test_no_new_imports_of_missing_modules_or_names():
    failures = _all_failures()
    new = sorted((k, v) for k, v in failures.items() if k not in KNOWN_BROKEN)
    assert not new, (
        'These imports or module attributes do not resolve to anything under app/. '
        'Fix them (or, only if the failure predates this check, add them to KNOWN_BROKEN):\n'
        + _format(new)
    )


def test_known_broken_list_only_shrinks():
    failures = _all_failures()
    fixed = sorted(k for k in KNOWN_BROKEN if k not in failures)
    assert not fixed, (
        'These KNOWN_BROKEN entries resolve now; remove them from KNOWN_BROKEN:\n'
        + '\n'.join(f'  {file}: {target}' for file, target in fixed)
    )
