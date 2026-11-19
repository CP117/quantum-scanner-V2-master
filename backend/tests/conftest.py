"""Shared pytest configuration for backend/tests.

* Puts the repository root on ``sys.path`` once, so test modules can
  ``import app...`` no matter where pytest is started from.  Existing
  per-file ``sys.path`` inserts are left alone.
* Sets safe environment defaults before any ``app`` module is imported:
  no automatic Cloudflare tunnel, and no background threads (the second
  flag is honored once the lifespan/registry work lands).
* Blocks outbound network access for unit tests.  Tests that need a
  running server are marked ``integration``; tests that need the real
  internet are marked ``network``.  Both are deselected by default
  (see pytest.ini).
* Provides the ``backend_url`` fixture for integration tests.
"""
from __future__ import annotations

import ipaddress
import os
import socket
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault('MRD_AUTO_TUNNEL_ON_STARTUP', '0')
os.environ.setdefault('MRD_START_BACKGROUND', '0')

_LEGACY_URL_VARS = ('REACT_APP_BACKEND_URL', 'BACKEND_TEST_BASE_URL', 'BACKEND_URL_OVERRIDE')
_DEFAULT_BACKEND_URL = 'http://127.0.0.1:8001'

_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_create_connection = socket.create_connection


def _is_loopback_host(host) -> bool:
    if host is None:
        return False
    if isinstance(host, bytes):
        host = host.decode('ascii', 'ignore')
    host = str(host).strip('[]')
    if host in ('localhost', 'localhost.localdomain', ''):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _address_allowed(family, address) -> bool:
    if family == getattr(socket, 'AF_UNIX', object()):
        return True
    if isinstance(address, (str, bytes)):
        # AF_UNIX path passed without a family hint.
        return True
    if isinstance(address, tuple) and address:
        return _is_loopback_host(address[0])
    return False


class NetworkBlockedError(RuntimeError):
    """Raised when an unmarked unit test tries to reach the network."""


def _blocked(address):
    raise NetworkBlockedError(
        f'network access in a unit test (attempted {address!r}); '
        'mark the test network or integration if this is intended'
    )


def _guarded_connect(self, address):
    if not _address_allowed(self.family, address):
        _blocked(address)
    return _real_connect(self, address)


def _guarded_connect_ex(self, address):
    if not _address_allowed(self.family, address):
        _blocked(address)
    return _real_connect_ex(self, address)


def _guarded_create_connection(address, *args, **kwargs):
    if not _address_allowed(None, address):
        _blocked(address)
    return _real_create_connection(address, *args, **kwargs)


@pytest.fixture(autouse=True)
def _block_network(request, monkeypatch):
    """Fail fast if a unit test opens a non-loopback connection."""
    if request.node.get_closest_marker('network') or request.node.get_closest_marker('integration'):
        yield
        return
    monkeypatch.setattr(socket.socket, 'connect', _guarded_connect)
    monkeypatch.setattr(socket.socket, 'connect_ex', _guarded_connect_ex)
    monkeypatch.setattr(socket, 'create_connection', _guarded_create_connection)
    yield


def resolve_backend_url() -> str:
    """BACKEND_URL, else the legacy variables, else http://127.0.0.1:8001."""
    for name in ('BACKEND_URL',) + _LEGACY_URL_VARS:
        value = (os.environ.get(name) or '').strip()
        if value:
            return value.rstrip('/')
    return _DEFAULT_BACKEND_URL


@pytest.fixture(scope='session')
def backend_url() -> str:
    """Base URL of the server under test (integration tests only)."""
    return resolve_backend_url()
