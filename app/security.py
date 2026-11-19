"""Request guard for writes and sensitive reads.

The dashboard is meant to run on the owner's machine.  When it is opened
up with ``--lan`` or ``--tunnel``, other people can reach it, so every
state-changing request and every sensitive read must prove it comes from
the owner.  A request is allowed when either:

* it is a *trusted local* request: a same-origin browser request (or a
  plain client such as curl) sent directly from this machine, or
* it carries the admin token in ``X-MRD-Admin-Token`` or
  ``Authorization: Bearer``.

The trusted-local check resists two attacks:

* Cross-site requests from the owner's own browser.  Any web page can make
  the browser send a POST to 127.0.0.1, so the Origin, Host and
  Sec-Fetch-Site headers must all point at this app.
* Requests that arrive through the Cloudflare tunnel or another proxy.
  They reach the app over loopback too, but carry forwarding headers
  (cf-connecting-ip, x-forwarded-for, ...), and any of those disqualifies
  the request.

The admin token comes from ``MRD_ADMIN_TOKEN`` or, when that is unset, from
``data/admin_token.txt``, which is generated on first start.  The token is
never written to the log.  See docs/SECURITY.md.
"""
from __future__ import annotations

import hmac
import ipaddress
import logging
import os
import secrets
import threading
import uuid
from pathlib import Path
from typing import Iterable, Optional

from starlette.requests import Request
from starlette.responses import JSONResponse

log = logging.getLogger('app.security')

_REPO_ROOT = Path(__file__).resolve().parent.parent
ADMIN_TOKEN_FILE = _REPO_ROOT / 'data' / 'admin_token.txt'

DEFAULT_CORS_ORIGINS = ('http://127.0.0.1:8001', 'http://localhost:8001')

# Methods that never change state.  Everything else needs the guard.
SAFE_METHODS = frozenset({'GET', 'HEAD', 'OPTIONS'})

# GET/HEAD paths that are guarded anyway because they expose secrets or
# internals: live thread stacks, the full source bundle, API key status
# and admin views.  Matched with str.startswith on the request path.
SENSITIVE_PREFIXES = (
    '/system/threads',
    '/download/source',
    '/api/download/source',
    '/api/api-keys',
    '/api/admin',
)

# Paths that skip the guard entirely.  Keep this empty unless there is a
# written reason next to the entry.
EXEMPT_PATHS: frozenset[str] = frozenset()

# Any of these means the request was relayed by a proxy or the tunnel,
# so it did not come directly from this machine.
FORWARDING_HEADERS = (
    'cf-connecting-ip',
    'cf-ray',
    'cf-ipcountry',
    'x-forwarded-for',
    'x-forwarded-host',
    'x-real-ip',
    'forwarded',
)

LOOPBACK_HOSTNAMES = frozenset({'127.0.0.1', 'localhost', '[::1]'})

_token_lock = threading.Lock()
_cached_token: Optional[str] = None
_token_source: Optional[str] = None


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def cors_origins() -> list[str]:
    """Allowed browser origins, from MRD_CORS_ORIGINS (comma-separated)."""
    raw = os.environ.get('MRD_CORS_ORIGINS', '')
    origins = [o.strip().rstrip('/') for o in raw.split(',') if o.strip()]
    return origins or list(DEFAULT_CORS_ORIGINS)


def get_admin_token() -> str:
    """Return the admin token, creating data/admin_token.txt if needed."""
    global _cached_token, _token_source
    env_token = os.environ.get('MRD_ADMIN_TOKEN', '').strip()
    if env_token:
        _token_source = 'environment variable MRD_ADMIN_TOKEN'
        return env_token
    with _token_lock:
        if _cached_token:
            return _cached_token
        path = ADMIN_TOKEN_FILE
        try:
            existing = path.read_text(encoding='utf-8').strip()
        except FileNotFoundError:
            existing = ''
        if existing:
            _cached_token = existing
        else:
            token = secrets.token_urlsafe(32)
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                fh.write(token + '\n')
            try:
                os.chmod(path, 0o600)
            except OSError:
                # Windows ignores most mode bits; the file still lives in
                # the git-ignored data/ directory.
                pass
            _cached_token = token
        _token_source = str(path)
        return _cached_token


def admin_token_location() -> str:
    """Human-readable description of where the token comes from."""
    get_admin_token()
    return _token_source or str(ADMIN_TOKEN_FILE)


def announce_security_settings() -> None:
    """Log the guard settings once at startup.  Never logs the token."""
    try:
        location = admin_token_location()
    except OSError as exc:
        log.error('security: could not create the admin token file %s: %s', ADMIN_TOKEN_FILE, exc)
        return
    tunnel = 'on' if os.environ.get('MRD_AUTO_TUNNEL_ON_STARTUP', '0') == '1' else 'off'
    log.info(
        'security: writes need the admin token unless sent from this machine; '
        'token stored in %s; startup tunnel %s; CORS origins %s',
        location, tunnel, ', '.join(cors_origins()),
    )


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _is_loopback_ip(host: Optional[str]) -> bool:
    if not host:
        return False
    try:
        ip = ipaddress.ip_address(host.strip('[]'))
    except ValueError:
        return False
    mapped = getattr(ip, 'ipv4_mapped', None)
    if mapped is not None:
        ip = mapped
    return ip.is_loopback


def _split_host_header(value: str) -> Optional[tuple[str, Optional[int]]]:
    """Split a Host header into (hostname, port).  None when malformed."""
    value = value.strip().lower()
    if not value:
        return None
    if value.startswith('['):
        end = value.find(']')
        if end < 0:
            return None
        hostname, rest = value[:end + 1], value[end + 1:]
    else:
        hostname, sep, port_text = value.partition(':')
        rest = (':' + port_text) if sep else ''
    if not rest:
        return hostname, None
    if not rest.startswith(':') or not rest[1:].isdigit():
        return None
    return hostname, int(rest[1:])


def is_trusted_local(request: Request, allowed_origins: Optional[Iterable[str]] = None) -> bool:
    """True only for a request sent directly from this machine to this app."""
    client = request.client
    if client is None or not _is_loopback_ip(client.host):
        return False
    headers = request.headers
    if any(name in headers for name in FORWARDING_HEADERS):
        return False
    host_value = headers.get('host', '')
    parsed = _split_host_header(host_value)
    if parsed is None:
        return False
    hostname, port = parsed
    if hostname not in LOOPBACK_HOSTNAMES:
        return False
    server = request.scope.get('server')
    if server and len(server) > 1 and server[1] is not None:
        if (port if port is not None else 80) != server[1]:
            return False
    origin = headers.get('origin')
    if origin is not None:
        allowed = {o.rstrip('/').lower() for o in (allowed_origins if allowed_origins is not None else cors_origins())}
        # The page's own origin is also accepted, so the dashboard keeps
        # working when the launcher falls back to another port (8011).
        # The Host header was validated above, so this is still loopback.
        allowed.add('http://' + host_value.strip().lower())
        if origin.strip().rstrip('/').lower() not in allowed:
            return False
    fetch_site = headers.get('sec-fetch-site')
    if fetch_site is not None and fetch_site.strip().lower() not in ('same-origin', 'none'):
        return False
    return True


def has_admin_token(request: Request, expected: Optional[str] = None) -> bool:
    """True when the request carries the admin token."""
    supplied = request.headers.get('x-mrd-admin-token', '').strip()
    if not supplied:
        scheme, _, value = request.headers.get('authorization', '').partition(' ')
        if scheme.lower() == 'bearer':
            supplied = value.strip()
    if not supplied:
        return False
    token = expected if expected is not None else get_admin_token()
    if not token:
        return False
    return hmac.compare_digest(supplied.encode('utf-8'), token.encode('utf-8'))


def requires_admin(method: str, path: str) -> bool:
    """True when the guard applies to this method and path."""
    if path in EXEMPT_PATHS:
        return False
    method = method.upper()
    if method not in SAFE_METHODS:
        return True
    if method in ('GET', 'HEAD'):
        return any(path.startswith(prefix) for prefix in SENSITIVE_PREFIXES)
    return False


# ---------------------------------------------------------------------------
# Middleware and error handler
# ---------------------------------------------------------------------------

class AdminGuardMiddleware:
    """Pure ASGI middleware that enforces the rules above before routing."""

    def __init__(self, app, token: Optional[str] = None, allowed_origins: Optional[Iterable[str]] = None):
        self.app = app
        self._token = token
        self._allowed_origins = list(allowed_origins) if allowed_origins is not None else None

    async def __call__(self, scope, receive, send):
        if scope.get('type') != 'http':
            await self.app(scope, receive, send)
            return
        if not requires_admin(scope.get('method', 'GET'), scope.get('path', '')):
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        origins = self._allowed_origins if self._allowed_origins is not None else cors_origins()
        if is_trusted_local(request, origins) or has_admin_token(request, self._token):
            await self.app(scope, receive, send)
            return
        response = JSONResponse(
            {'error': 'admin_token_required'},
            status_code=401,
            headers={'WWW-Authenticate': 'Bearer'},
        )
        await response(scope, receive, send)


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Answer 500 without internals; the traceback goes to the log only."""
    request_id = uuid.uuid4().hex
    logging.getLogger('app.errors').exception(
        'unhandled error request_id=%s method=%s path=%s',
        request_id, request.method, request.url.path,
        exc_info=exc,
    )
    return JSONResponse(
        status_code=500,
        content={'error': 'internal_server_error', 'request_id': request_id},
    )
