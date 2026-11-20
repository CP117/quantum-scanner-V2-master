"""Phase 3: router mount failures are recorded and shown in /system/status.

Before Phase 3 a router that failed to import left a log line (or, for
threads_debug, nothing) and its pages returned 404.  The prediction
tracker had been unmounted this way since the repo was imported.

These tests never import app.main, because importing it starts the
scanner; main.py is checked with ast instead.
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import mount_errors as me
from app.routes import status as status_route

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN = REPO_ROOT / 'app' / 'main.py'


@pytest.fixture
def errors(monkeypatch):
    monkeypatch.delenv('MRD_STRICT_MOUNT', raising=False)
    saved = dict(me.MOUNT_ERRORS)
    me.MOUNT_ERRORS.clear()
    yield me.MOUNT_ERRORS
    me.MOUNT_ERRORS.clear()
    me.MOUNT_ERRORS.update(saved)


def _raised(exc: BaseException) -> BaseException:
    try:
        raise exc
    except BaseException as caught:  # noqa: BLE001
        return caught


def test_failure_is_recorded_with_type_and_message_and_logged(errors, caplog):
    caplog.set_level(logging.ERROR)
    exc = _raised(ImportError(
        f"cannot import name 'start_evaluator' from 'app.services.x' ({REPO_ROOT}/app/services/x.py)"
    ))
    me.record_mount_error('prediction_tracker', exc, logger='app.prediction_tracker')

    assert errors == {
        'prediction_tracker':
            "ImportError: cannot import name 'start_evaluator' from 'app.services.x' "
            "(<repo>/app/services/x.py)",
    }
    records = [r for r in caplog.records if r.name == 'app.prediction_tracker']
    assert len(records) == 1
    assert records[0].levelno == logging.ERROR
    assert records[0].exc_info and records[0].exc_info[1] is exc
    assert 'prediction_tracker failed to mount' in records[0].getMessage()


def test_home_directory_is_shortened(errors):
    home = str(Path.home())
    if len(home) <= 3:
        pytest.skip('home directory too short to replace safely')
    me.record_mount_error('demo', ValueError(f'cannot read {home}/settings.json'))
    assert errors['demo'] == 'ValueError: cannot read ~/settings.json'


def test_long_messages_are_cut(errors):
    me.record_mount_error('demo', ValueError('x' * 5000))
    assert len(errors['demo']) == 500


def test_mount_errors_returns_a_copy(errors):
    me.record_mount_error('demo', ValueError('boom'))
    copy = me.mount_errors()
    copy['other'] = 'x'
    assert 'other' not in errors


@pytest.mark.parametrize('value, strict', [('1', True), (' 1 ', True), ('0', False), ('', False), ('true', False)])
def test_strict_flag(monkeypatch, value, strict):
    monkeypatch.setenv('MRD_STRICT_MOUNT', value)
    assert me.strict_mount() is strict


def test_strict_mode_stops_startup_and_still_records(errors, monkeypatch):
    monkeypatch.setenv('MRD_STRICT_MOUNT', '1')
    cause = _raised(KeyError('router'))
    with pytest.raises(RuntimeError, match='demo failed to mount') as err:
        me.record_mount_error('demo', cause)
    assert err.value.__cause__ is cause
    assert errors['demo'] == "KeyError: 'router'"


def test_system_status_reports_mount_errors(errors):
    app = FastAPI()
    app.include_router(status_route.router)
    client = TestClient(app)

    body = client.get('/system/status').json()
    assert body['mount_errors'] == {}

    errors['prediction_tracker'] = 'ImportError: demo'
    body = client.get('/system/status').json()
    assert body['mount_errors'] == {'prediction_tracker': 'ImportError: demo'}


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        c for c in ast.walk(node)
        if isinstance(c, ast.Call)
        and ((isinstance(c.func, ast.Name) and c.func.id == name)
             or (isinstance(c.func, ast.Attribute) and c.func.attr == name))
    ]


def test_every_guarded_router_mount_in_main_records_its_failure():
    tree = ast.parse(MAIN.read_text(encoding='utf-8'))
    names: list[str] = []
    guarded = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        body = ast.Module(body=node.body, type_ignores=[])
        if not _calls(body, 'include_router'):
            continue
        guarded += 1
        for handler in node.handlers:
            recorded = _calls(handler, '_record_mount_error')
            assert recorded, (
                f'app/main.py line {node.lineno}: this try mounts a router but its '
                'except does not call _record_mount_error'
            )
            first = recorded[0].args[0]
            assert isinstance(first, ast.Constant) and isinstance(first.value, str)
            names.append(first.value)
    assert guarded >= 18
    assert len(names) == len(set(names)), names
    assert {'prediction_tracker', 'threads_debug', 'snapshot', 'tiered_scanner'} <= set(names)


def test_main_reexports_the_same_dict():
    tree = ast.parse(MAIN.read_text(encoding='utf-8'))
    imported = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == 'app.mount_errors'
        for alias in node.names
    }
    assert {'MOUNT_ERRORS', '_record_mount_error'} <= imported
