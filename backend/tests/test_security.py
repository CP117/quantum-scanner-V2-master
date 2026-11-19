"""Tests for app/security.py: the admin guard, CORS order and error handler.

Builds a small FastAPI app with the same middleware order as app/main.py
instead of importing app.main (which starts the whole scanner).
"""
from __future__ import annotations

import logging
import os
import stat

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from app import security
from app.security import AdminGuardMiddleware, requires_admin, unhandled_exception_handler

TOKEN = 'unit-test-token-0123456789abcdef'
ORIGINS = ['http://127.0.0.1:8001', 'http://localhost:8001']


def make_app() -> FastAPI:
    app = FastAPI()
    # Same order as app/main.py: the guard is added first so CORS wraps it.
    app.add_middleware(AdminGuardMiddleware, token=TOKEN, allowed_origins=ORIGINS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ORIGINS,
        allow_credentials=False,
        allow_methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
        allow_headers=['Content-Type', 'Authorization', 'X-MRD-Admin-Token', 'If-None-Match'],
    )
    app.add_exception_handler(Exception, unhandled_exception_handler)

    @app.post('/api/universes/toggle')
    def toggle():
        return {'ok': True}

    @app.delete('/api/api-keys/{provider}')
    def delete_key(provider: str):
        return {'deleted': provider}

    @app.get('/api/api-keys')
    def list_keys():
        return {'configured': {}}

    @app.get('/api/scan/snapshot')
    def snapshot():
        return {'rows': []}

    @app.get('/system/threads')
    def threads():
        return {'threads': []}

    @app.get('/boom')
    def boom():
        raise ValueError('secret-detail')

    return app


@pytest.fixture(scope='module')
def app() -> FastAPI:
    return make_app()


def client_for(app, client_host='127.0.0.1', base_url='http://127.0.0.1:8001') -> TestClient:
    return TestClient(app, base_url=base_url, client=(client_host, 50000), raise_server_exceptions=False)


@pytest.fixture
def local(app):
    return client_for(app)


@pytest.fixture
def remote(app):
    # Another device on the LAN, after starting with --lan.
    return client_for(app, client_host='192.168.0.50', base_url='http://192.168.0.39:8001')


# --- trusted local -------------------------------------------------------

def test_local_post_without_origin_is_allowed(local):
    resp = local.post('/api/universes/toggle', json={})
    assert resp.status_code == 200
    assert resp.json() == {'ok': True}


def test_local_same_origin_browser_post_is_allowed(local):
    headers = {'Origin': 'http://127.0.0.1:8001', 'Sec-Fetch-Site': 'same-origin'}
    assert local.post('/api/universes/toggle', json={}, headers=headers).status_code == 200


def test_local_same_origin_on_fallback_port_is_allowed(app):
    # The launchers fall back to port 8011 when 8001 is busy.
    client = client_for(app, base_url='http://localhost:8011')
    headers = {'Origin': 'http://localhost:8011', 'Sec-Fetch-Site': 'same-origin'}
    assert client.post('/api/universes/toggle', json={}, headers=headers).status_code == 200


def test_ipv4_mapped_loopback_client_is_trusted(app):
    client = client_for(app, client_host='::ffff:127.0.0.1')
    assert client.post('/api/universes/toggle', json={}).status_code == 200


def test_local_sensitive_get_is_allowed(local):
    assert local.get('/system/threads').status_code == 200
    assert local.get('/api/api-keys').status_code == 200


# --- relayed requests (tunnel, proxies) ----------------------------------

@pytest.mark.parametrize('header', ['cf-connecting-ip', 'x-forwarded-for', 'cf-ray', 'forwarded', 'x-real-ip'])
def test_forwarded_post_needs_token(local, header):
    relayed = {header: '1.2.3.4'}
    assert local.post('/api/universes/toggle', json={}, headers=relayed).status_code == 401
    good = {**relayed, 'X-MRD-Admin-Token': TOKEN}
    assert local.post('/api/universes/toggle', json={}, headers=good).status_code == 200
    bad = {**relayed, 'X-MRD-Admin-Token': TOKEN + 'x'}
    assert local.post('/api/universes/toggle', json={}, headers=bad).status_code == 401


def test_bearer_token_is_accepted(local):
    headers = {'cf-connecting-ip': '1.2.3.4', 'Authorization': f'Bearer {TOKEN}'}
    assert local.post('/api/universes/toggle', json={}, headers=headers).status_code == 200
    headers['Authorization'] = 'Bearer wrong'
    assert local.post('/api/universes/toggle', json={}, headers=headers).status_code == 401


def test_rejection_body_and_header(local):
    resp = local.post('/api/universes/toggle', json={}, headers={'x-forwarded-for': '1.2.3.4'})
    assert resp.status_code == 401
    assert resp.json() == {'error': 'admin_token_required'}
    assert resp.headers.get('www-authenticate') == 'Bearer'


# --- cross-site requests from the owner's browser ------------------------

@pytest.mark.parametrize('headers', [
    {'Origin': 'https://evil.example'},
    {'Origin': 'null'},
    {'Sec-Fetch-Site': 'cross-site'},
    {'Sec-Fetch-Site': 'same-site'},
    {'Host': 'evil.example'},
    {'Host': 'evil.example:8001'},
    {'Host': '127.0.0.1:9999'},
])
def test_cross_site_post_needs_token(local, headers):
    assert local.post('/api/universes/toggle', json={}, headers=headers).status_code == 401


def test_cross_site_post_with_token_is_allowed(local):
    headers = {'Origin': 'https://evil.example', 'X-MRD-Admin-Token': TOKEN}
    assert local.post('/api/universes/toggle', json={}, headers=headers).status_code == 200


# --- other devices -------------------------------------------------------

def test_remote_post_needs_token(remote):
    assert remote.post('/api/universes/toggle', json={}).status_code == 401
    assert remote.post('/api/universes/toggle', json={}, headers={'X-MRD-Admin-Token': TOKEN}).status_code == 200


def test_remote_delete_api_key_needs_token(remote):
    assert remote.delete('/api/api-keys/finnhub').status_code == 401


def test_remote_ordinary_get_is_public(remote):
    assert remote.get('/api/scan/snapshot').status_code == 200


def test_remote_sensitive_get_needs_token(remote):
    assert remote.get('/system/threads').status_code == 401
    assert remote.get('/api/api-keys').status_code == 401
    assert remote.get('/system/threads', headers={'X-MRD-Admin-Token': TOKEN}).status_code == 200


def test_remote_spoofed_loopback_host_is_not_trusted(remote):
    # A device on the LAN controls its own Host header.
    assert remote.post('/api/universes/toggle', json={}, headers={'Host': '127.0.0.1:8001'}).status_code == 401


# --- CORS ----------------------------------------------------------------

def test_preflight_from_allowed_origin_passes(remote):
    resp = remote.options('/api/universes/toggle', headers={
        'Origin': 'http://127.0.0.1:8001',
        'Access-Control-Request-Method': 'POST',
        'Access-Control-Request-Headers': 'X-MRD-Admin-Token, Content-Type',
    })
    assert resp.status_code == 200
    assert resp.headers.get('access-control-allow-origin') == 'http://127.0.0.1:8001'


def test_preflight_from_unknown_origin_is_refused(remote):
    resp = remote.options('/api/universes/toggle', headers={
        'Origin': 'https://evil.example',
        'Access-Control-Request-Method': 'POST',
    })
    assert 'access-control-allow-origin' not in resp.headers


# --- rules ---------------------------------------------------------------

@pytest.mark.parametrize('method,path,expected', [
    ('GET', '/api/scan/snapshot', False),
    ('HEAD', '/api/scan/snapshot', False),
    ('OPTIONS', '/api/api-keys/finnhub', False),
    ('GET', '/system/threads/summary', True),
    ('HEAD', '/download/source.zip', True),
    ('GET', '/api/download/source/info', True),
    ('GET', '/api/api-keys/preview', True),
    ('GET', '/api/admin/anything', True),
    ('POST', '/api/scan/snapshot', True),
    ('put', '/api/watchlists/1', True),
    ('PATCH', '/x', True),
    ('DELETE', '/api/api-keys/finnhub', True),
])
def test_requires_admin(method, path, expected):
    assert requires_admin(method, path) is expected


def test_exempt_paths_start_empty():
    assert security.EXEMPT_PATHS == frozenset()


# --- error handler -------------------------------------------------------

def test_exception_handler_hides_details(local, caplog):
    with caplog.at_level(logging.ERROR, logger='app.errors'):
        resp = local.get('/boom')
    assert resp.status_code == 500
    body = resp.json()
    assert 'secret-detail' not in resp.text
    assert body['error'] == 'internal_server_error'
    assert len(body['request_id']) == 32
    # The traceback and request id go to the log instead.
    assert body['request_id'] in caplog.text
    assert 'secret-detail' in caplog.text


# --- token storage -------------------------------------------------------

@pytest.fixture
def token_file(tmp_path, monkeypatch):
    path = tmp_path / 'data' / 'admin_token.txt'
    monkeypatch.setattr(security, 'ADMIN_TOKEN_FILE', path)
    monkeypatch.setattr(security, '_cached_token', None)
    monkeypatch.setattr(security, '_token_source', None)
    monkeypatch.delenv('MRD_ADMIN_TOKEN', raising=False)
    return path


def test_token_is_generated_once_and_stored(token_file):
    token = security.get_admin_token()
    assert len(token) >= 40
    assert token_file.read_text(encoding='utf-8').strip() == token
    if os.name == 'posix':
        assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
    assert security.get_admin_token() == token


def test_existing_token_file_is_reused(token_file):
    token_file.parent.mkdir(parents=True)
    token_file.write_text('kept-token\n', encoding='utf-8')
    assert security.get_admin_token() == 'kept-token'


def test_environment_token_wins(token_file, monkeypatch):
    monkeypatch.setenv('MRD_ADMIN_TOKEN', 'from-env')
    assert security.get_admin_token() == 'from-env'
    assert not token_file.exists()


def test_startup_log_never_contains_token(token_file, caplog):
    with caplog.at_level(logging.INFO, logger='app.security'):
        security.announce_security_settings()
    token = security.get_admin_token()
    assert str(token_file) in caplog.text
    assert token not in caplog.text


def test_cors_origins_from_environment(monkeypatch):
    monkeypatch.delenv('MRD_CORS_ORIGINS', raising=False)
    assert security.cors_origins() == ORIGINS
    monkeypatch.setenv('MRD_CORS_ORIGINS', ' http://a.test:1/ , http://b.test:2 ,')
    assert security.cors_origins() == ['http://a.test:1', 'http://b.test:2']
