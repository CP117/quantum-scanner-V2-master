"""Phase 2A: CBOE host move and explicit redirect handling.

cdn.cboe.com now answers 307 to cdn-api.cboe.com. httpx does not follow
redirects by default, so every CBOE fetch used to end as `http_307` and
count toward the cboe_options circuit breaker. These tests pin:

* the new default host and the MRD_CBOE_BASE_URL override,
* `resilient_get(follow_redirects=...)` for both clients, with an
  in-process fake transport (real httpx / requests machinery, no sockets),
* the provider's `redirected` outcome and hourly warning,
* the MRD_CBOE_MAX_FETCHES_PER_MINUTE cap, and that results still go
  through the options-chain cache.
"""
from __future__ import annotations

import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
import requests
from requests.adapters import BaseAdapter
from requests.structures import CaseInsensitiveDict

from app.services import http_client
from app.services.http_client import ResilientGetConfig, resilient_get
from app.services.providers import base as providers_base
from app.services.providers import cboe_options_provider as cboe

FIXTURE = Path(__file__).parent / 'fixtures' / 'cboe_aapl_trimmed.json'
OLD_TEMPLATE = 'https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json'
NEW_TEMPLATE = 'https://cdn-api.cboe.com/api/global/delayed_quotes/options/{symbol}.json'
OLD_AAPL = OLD_TEMPLATE.format(symbol='AAPL')
NEW_AAPL = NEW_TEMPLATE.format(symbol='AAPL')
FAST = ResilientGetConfig(max_attempts=1, connect_timeout=1.0, read_timeout=1.0,
                          backoff_initial=0.0, jitter_max=0.0)
# The fixture's contracts expire 2026-12-18; the parser drops expirations
# before "today", so the provider's clock is pinned to the capture date.
FROZEN_TODAY = datetime(2026, 11, 19, 15, 0, tzinfo=timezone.utc)


class _FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):  # noqa: D102 - test double
        return FROZEN_TODAY if tz is not None else FROZEN_TODAY.replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Fake transport
# ---------------------------------------------------------------------------
class FakeWeb:
    """URL -> (status, headers, body) table served to both HTTP clients.

    `httpx_get` / `requests_get` replace the module-level `httpx.get` /
    `requests.get` and hand the request to a real httpx.Client or
    requests.Session whose transport is this table, so each client's own
    redirect logic runs exactly as it would on the wire.
    """

    def __init__(self) -> None:
        self.routes: dict[str, tuple[int, dict, bytes]] = {}
        self.seen: list[tuple[str, dict]] = []
        self.httpx_kwargs: list[dict] = []
        self.requests_kwargs: list[dict] = []

    def add(self, url: str, status: int, *, location: str | None = None,
            body: bytes = b'', content_type: str = 'text/plain') -> None:
        headers = {'Content-Type': content_type}
        if location is not None:
            headers['Location'] = location
        self.routes[url] = (status, headers, body)

    def urls(self) -> list[str]:
        return [u for u, _ in self.seen]

    def _respond(self, url: str, headers: dict) -> tuple[int, dict, bytes]:
        self.seen.append((url, {k.lower(): v for k, v in headers.items()}))
        return self.routes.get(url, (404, {'Content-Type': 'text/plain'}, b'not found'))

    # httpx -----------------------------------------------------------------
    def _httpx_handler(self, request: httpx.Request) -> httpx.Response:
        status, headers, body = self._respond(str(request.url), dict(request.headers))
        return httpx.Response(status, headers=headers, content=body)

    def httpx_get(self, url, **kwargs):
        self.httpx_kwargs.append(dict(kwargs))
        follow = kwargs.get('follow_redirects', False)  # httpx.get's default
        with httpx.Client(transport=httpx.MockTransport(self._httpx_handler),
                          follow_redirects=follow) as c:
            return c.get(url, params=kwargs.get('params'), headers=kwargs.get('headers'))

    # requests --------------------------------------------------------------
    def requests_get(self, url, **kwargs):
        self.requests_kwargs.append(dict(kwargs))
        session = requests.Session()
        adapter = _TableAdapter(self)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session.get(url, **kwargs)


class _TableAdapter(BaseAdapter):
    def __init__(self, web: FakeWeb) -> None:
        super().__init__()
        self.web = web

    def send(self, request, **kwargs):  # noqa: D102 - test double
        status, headers, body = self.web._respond(request.url, dict(request.headers))
        resp = requests.Response()
        resp.status_code = status
        resp.headers = CaseInsensitiveDict(headers)
        resp._content = body
        resp._content_consumed = True
        resp.raw = io.BytesIO(body)
        resp.url = request.url
        resp.request = request
        resp.encoding = 'utf-8'
        resp.reason = 'fake'
        resp.connection = self
        return resp

    def close(self) -> None:  # noqa: D102 - test double
        pass


@pytest.fixture
def web(monkeypatch):
    fake = FakeWeb()
    monkeypatch.setattr(http_client.httpx, 'get', fake.httpx_get)
    monkeypatch.setattr(http_client.requests, 'get', fake.requests_get)
    return fake


class _Clock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(cboe, '_now', c)
    return c


@pytest.fixture(autouse=True)
def isolated_cboe(monkeypatch):
    """Fresh CBOE module state and a fresh cboe_options stats row."""
    monkeypatch.delenv('MRD_CBOE_BASE_URL', raising=False)
    monkeypatch.delenv('MRD_CBOE_MAX_FETCHES_PER_MINUTE', raising=False)
    monkeypatch.setattr(cboe, 'datetime', _FrozenDateTime)
    monkeypatch.setattr(cboe, '_MIN_GAP_SECONDS', 0.0)
    monkeypatch.setattr(cboe, '_cache', {})
    monkeypatch.setattr(cboe, '_PREFIX_KNOWN', {})
    monkeypatch.setattr(cboe, '_HTTP_FAIL_UNTIL', {})
    monkeypatch.setattr(cboe, '_fetch_times', cboe.deque())
    monkeypatch.setattr(cboe, '_last_warning_at', {})
    monkeypatch.setattr(cboe, '_last_request_ts', 0.0)
    monkeypatch.setattr(cboe, '_rl_stats', {k: 0 for k in cboe._rl_stats})
    saved = providers_base._stats.pop('cboe_options', None)
    providers_base.configure_circuit('cboe_options', threshold=8, cooldown_seconds=120.0)
    yield
    providers_base._stats.pop('cboe_options', None)
    if saved is not None:
        providers_base._stats['cboe_options'] = saved


def _fixture_bytes() -> bytes:
    return FIXTURE.read_bytes()


def _cboe_row() -> dict:
    return providers_base.provider_stats_snapshot().get('cboe_options') or {}


# ---------------------------------------------------------------------------
# Fixture sanity
# ---------------------------------------------------------------------------
def test_fixture_is_trimmed_to_parser_fields():
    rows = json.loads(_fixture_bytes())['data']['options']
    assert len(rows) == 3
    expected = {'option', 'open_interest', 'volume', 'iv', 'delta', 'gamma',
                'bid', 'ask', 'last_trade_price'}
    assert all(set(r) == expected for r in rows)


# ---------------------------------------------------------------------------
# Base URL
# ---------------------------------------------------------------------------
def test_default_base_url_is_the_new_cdn_api_host():
    assert cboe._base_url() == NEW_TEMPLATE
    assert 'cdn-api.cboe.com' in cboe._BASE_URL


def test_base_url_override_and_bad_override(monkeypatch, caplog):
    monkeypatch.setenv('MRD_CBOE_BASE_URL', OLD_TEMPLATE)
    assert cboe._base_url() == OLD_TEMPLATE
    monkeypatch.setenv('MRD_CBOE_BASE_URL', 'https://cdn-api.cboe.com/no-placeholder.json')
    with caplog.at_level(logging.WARNING, logger='app.options_chain.cboe'):
        assert cboe._base_url() == NEW_TEMPLATE
        assert cboe._base_url() == NEW_TEMPLATE
    assert sum('no {symbol} placeholder' in r.getMessage() for r in caplog.records) == 1


# ---------------------------------------------------------------------------
# Provider: the 307 that used to fail
# ---------------------------------------------------------------------------
def test_old_host_307_is_followed_and_records_no_failure(web, monkeypatch, caplog):
    monkeypatch.setenv('MRD_CBOE_BASE_URL', OLD_TEMPLATE)
    web.add(OLD_AAPL, 307, location=NEW_AAPL, content_type='text/html')
    web.add(NEW_AAPL, 200, body=_fixture_bytes(), content_type='application/json')

    with caplog.at_level(logging.WARNING, logger='app.options_chain.cboe'):
        result = cboe.get_options_positioning('AAPL', 352.0)

    assert isinstance(result, dict) and result
    assert result['expiration_dates'] == ['2026-12-18']
    assert result['bias'] == 'bullish'
    composite = result['composite']
    assert composite['target_price'] == 350.0
    assert composite['call_wall'] == pytest.approx(355.49)
    assert composite['put_wall'] == 350.0
    assert web.urls() == [OLD_AAPL, NEW_AAPL]
    # Followed manually: httpx itself was told not to follow.
    assert all(kw.get('follow_redirects') is False for kw in web.httpx_kwargs)

    row = _cboe_row()
    assert row['hits'] == 1
    assert row['errors'] == 0
    assert row['consecutive_failures'] == 0
    assert row['last_error'] is None
    assert cboe._rl_stats['redirects_followed'] == 1

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert OLD_AAPL in warnings[0] and NEW_AAPL in warnings[0]


def test_redirect_warning_is_at_most_hourly(web, monkeypatch, clock, caplog):
    monkeypatch.setenv('MRD_CBOE_BASE_URL', OLD_TEMPLATE)
    for sym in ('AAPL', 'MSFT', 'NVDA'):
        web.add(OLD_TEMPLATE.format(symbol=sym), 307, location=NEW_TEMPLATE.format(symbol=sym))
        web.add(NEW_TEMPLATE.format(symbol=sym), 200, body=_fixture_bytes())

    with caplog.at_level(logging.WARNING, logger='app.options_chain.cboe'):
        assert cboe.get_options_positioning('AAPL', 352.0)
        clock.t += 61
        assert cboe.get_options_positioning('MSFT', 352.0)
        n_before_hour = sum(r.levelno == logging.WARNING for r in caplog.records)
        clock.t += 3600
        assert cboe.get_options_positioning('NVDA', 352.0)
    n_after_hour = sum(r.levelno == logging.WARNING for r in caplog.records)
    assert n_before_hour == 1
    assert n_after_hour == 2


def test_default_host_needs_no_redirect(web):
    web.add(NEW_AAPL, 200, body=_fixture_bytes(), content_type='application/json')
    assert cboe.get_options_positioning('AAPL', 352.0)
    assert web.urls() == [NEW_AAPL]
    assert cboe._rl_stats['redirects_followed'] == 0


@pytest.mark.parametrize('target', [
    'https://cdn.example.net/api/global/delayed_quotes/options/AAPL.json',
    'https://evilcboe.com/AAPL.json',
    'https://cboe.com.example.net/AAPL.json',
])
def test_redirect_outside_cboe_is_not_followed(web, target, caplog):
    web.add(NEW_AAPL, 307, location=target)
    with caplog.at_level(logging.WARNING, logger='app.options_chain.cboe'):
        assert cboe.get_options_positioning('AAPL', 352.0) is None
    assert target not in web.urls()
    # One request only: a 3xx is never retried, and the '_' prefix probe
    # (only for 404s) does not run.
    assert web.urls() == [NEW_AAPL]
    row = _cboe_row()
    assert row['errors'] == 1
    assert row['last_error'].startswith('redirected: 307 -> ')
    assert target in row['last_error']
    assert 'host_not_allowed' in row['last_error']
    assert 'http_307' not in row['last_error']
    assert row['timeouts'] == 0 and row['rate_limits'] == 0
    assert cboe._rl_stats['redirects_not_followed'] == 1
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and NEW_AAPL in warnings[0] and target in warnings[0]


def test_provider_redirect_loop_stops_and_reports_redirected(web):
    a = NEW_AAPL
    b = 'https://cdn-api2.cboe.com/api/global/delayed_quotes/options/AAPL.json'
    web.add(a, 307, location=b)
    web.add(b, 307, location=a)
    assert cboe.get_options_positioning('AAPL', 352.0) is None
    assert len(web.urls()) == cboe._MAX_REDIRECTS + 1
    assert 'too_many_redirects' in _cboe_row()['last_error']


def test_404_still_probes_the_underscore_prefix(web):
    web.add(NEW_TEMPLATE.format(symbol='SPX'), 404)
    web.add(NEW_TEMPLATE.format(symbol='_SPX'), 200, body=_fixture_bytes())
    assert cboe.get_options_positioning('SPX', 352.0)
    assert web.urls() == [NEW_TEMPLATE.format(symbol='SPX'), NEW_TEMPLATE.format(symbol='_SPX')]


# ---------------------------------------------------------------------------
# resilient_get redirect semantics
# ---------------------------------------------------------------------------
A = 'https://a.cboe.com/start.json'
B = 'https://b.cboe.com/next.json'


@pytest.mark.parametrize('client', ['httpx', 'requests'])
def test_redirect_loop_stops_at_max_redirects(web, client):
    web.add(A, 307, location=B)
    web.add(B, 302, location=A)
    r = resilient_get(A, cfg=FAST, client=client, follow_redirects=True, max_redirects=3,
                      redirect_allow_suffixes=('.cboe.com',))
    assert r.status_code in (302, 307)
    assert r.error == 'redirect_not_followed: too_many_redirects'
    assert len(r.redirects) == 3
    assert len(web.urls()) == 4
    assert r.url == A


@pytest.mark.parametrize('client', ['httpx', 'requests'])
def test_follow_true_reports_final_url_and_hops(web, client):
    web.add(A, 301, location='/relative.json')
    web.add('https://a.cboe.com/relative.json', 308, location=B)
    web.add(B, 200, body=b'{"ok": true}', content_type='application/json')
    r = resilient_get(A, cfg=FAST, client=client, follow_redirects=True,
                      redirect_allow_suffixes=('.cboe.com',))
    assert r.status_code == 200 and r.json() == {'ok': True}
    assert r.url == A
    assert r.final_url == B
    assert r.redirects == [(301, 'https://a.cboe.com/relative.json'), (308, B)]
    assert r.attempts == 3
    assert r.error is None


def test_follow_none_keeps_requests_following_natively(web):
    web.add(A, 307, location=B)
    web.add(B, 200, body=b'done')
    r = resilient_get(A, cfg=FAST)  # client='requests', follow_redirects=None
    assert 'allow_redirects' not in web.requests_kwargs[0]
    assert r.status_code == 200 and r.text == 'done'
    assert r.final_url == B
    assert r.redirects == [(307, B)]


def test_follow_none_keeps_httpx_returning_the_3xx(web):
    web.add(A, 307, location=B)
    r = resilient_get(A, cfg=FAST, client='httpx')
    assert 'follow_redirects' not in web.httpx_kwargs[0]
    assert r.status_code == 307
    assert r.location == B
    assert r.final_url == A
    assert r.redirects == []
    assert r.error is None
    assert web.urls() == [A]


@pytest.mark.parametrize('client', ['httpx', 'requests'])
def test_follow_false_returns_the_3xx_for_both_clients(web, client):
    web.add(A, 307, location=B)
    r = resilient_get(A, cfg=FAST, client=client, follow_redirects=False)
    assert r.status_code == 307 and r.error is None
    assert web.urls() == [A]


@pytest.mark.parametrize('client', ['httpx', 'requests'])
def test_3xx_is_never_retried(web, client):
    web.add(A, 307, location='https://elsewhere.example/x')
    cfg = ResilientGetConfig(max_attempts=3, backoff_initial=0.0, jitter_max=0.0)
    r = resilient_get(A, cfg=cfg, client=client, follow_redirects=True,
                      redirect_allow_suffixes=('.cboe.com',))
    assert r.status_code == 307
    assert r.error == 'redirect_not_followed: host_not_allowed'
    assert r.attempts == 1
    assert web.urls() == [A]


@pytest.mark.parametrize('location, reason', [
    ('http://a.cboe.com/plain.json', 'insecure_downgrade'),
    ('ftp://a.cboe.com/x', 'unsupported_scheme'),
    ('', 'missing_location'),
])
def test_unsafe_or_missing_locations_are_not_followed(web, location, reason):
    web.add(A, 302, location=location)
    r = resilient_get(A, cfg=FAST, client='httpx', follow_redirects=True)
    assert r.status_code == 302
    assert r.error == f'redirect_not_followed: {reason}'
    assert web.urls() == [A]


def test_no_allowlist_follows_any_https_host(web):
    web.add(A, 307, location='https://mirror.example.org/x.json')
    web.add('https://mirror.example.org/x.json', 200, body=b'ok')
    r = resilient_get(A, cfg=FAST, client='httpx', follow_redirects=True)
    assert r.status_code == 200 and r.final_url == 'https://mirror.example.org/x.json'


def test_credentials_are_dropped_when_the_host_changes(web):
    web.add(A, 307, location='https://a.cboe.com/same-host.json')
    web.add('https://a.cboe.com/same-host.json', 307, location=B)
    web.add(B, 200, body=b'ok')
    r = resilient_get(A, cfg=FAST, client='httpx', follow_redirects=True,
                      headers={'Authorization': 'Bearer t', 'Cookie': 'c=1', 'Accept': 'x/y'},
                      redirect_allow_suffixes=('.cboe.com',))
    assert r.status_code == 200
    (_, h1), (_, h2), (_, h3) = web.seen
    assert h1['authorization'] == 'Bearer t' and h2['authorization'] == 'Bearer t'
    assert 'authorization' not in h3 and 'cookie' not in h3
    assert h3['accept'] == 'x/y'
    # One User-Agent for the whole chain, like a browser.
    assert h1['user-agent'] == h2['user-agent'] == h3['user-agent']


def test_params_apply_to_the_first_hop_only(web):
    web.add('https://a.cboe.com/start.json?s=aapl', 307, location=B + '?moved=1')
    web.add(B + '?moved=1', 200, body=b'ok')
    r = resilient_get(A, params={'s': 'aapl'}, cfg=FAST, client='httpx', follow_redirects=True)
    assert r.status_code == 200
    assert web.urls() == ['https://a.cboe.com/start.json?s=aapl', B + '?moved=1']


@pytest.mark.parametrize('host, ok', [
    ('cboe.com', True), ('cdn-api.cboe.com', True), ('CDN.CBOE.COM', True),
    ('evilcboe.com', False), ('cboe.com.example.net', False), ('', False),
])
def test_host_suffix_matching(host, ok):
    assert http_client._host_allowed(host, ('.cboe.com',)) is ok


# ---------------------------------------------------------------------------
# Per-minute fetch cap
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('raw, expected', [
    (None, 20), ('', 20), ('5', 5), ('0', 20), ('-3', 20), ('abc', 20),
])
def test_fetch_cap_parsing(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv('MRD_CBOE_MAX_FETCHES_PER_MINUTE', raising=False)
    else:
        monkeypatch.setenv('MRD_CBOE_MAX_FETCHES_PER_MINUTE', raw)
    assert cboe._max_fetches_per_minute() == expected


def test_fetch_cap_limits_network_fetches_per_rolling_minute(web, monkeypatch, clock):
    monkeypatch.setenv('MRD_CBOE_MAX_FETCHES_PER_MINUTE', '2')
    for sym in ('AAPL', 'MSFT', 'NVDA'):
        web.add(NEW_TEMPLATE.format(symbol=sym), 200, body=_fixture_bytes())

    assert cboe.get_options_positioning('AAPL', 352.0)
    assert cboe.get_options_positioning('MSFT', 352.0)
    assert cboe.get_options_positioning('NVDA', 352.0) is None  # over the cap
    assert len(web.urls()) == 2
    assert cboe._rl_stats['throttled_per_minute'] == 1
    # A throttle is our own choice, not a provider failure.
    assert _cboe_row()['errors'] == 0

    # Cache hits do not spend the budget.
    assert cboe.get_options_positioning('AAPL', 352.0)
    assert len(web.urls()) == 2

    snap = cboe.fetch_budget_snapshot()
    assert snap['max_fetches_per_minute'] == 2
    assert snap['fetches_last_minute'] == 2
    assert snap['throttled_per_minute'] == 1

    clock.t += 60.5
    assert cboe.get_options_positioning('NVDA', 352.0)
    assert len(web.urls()) == 3


def test_results_still_go_through_the_options_chain_cache(web, monkeypatch):
    from app.services import options_chain_service as ocs
    from app.services.memory_store import MemoryStore

    monkeypatch.setattr(ocs, 'memory_store', MemoryStore())
    monkeypatch.setattr(ocs, '_cache', {})
    web.add(NEW_AAPL, 200, body=_fixture_bytes(), content_type='application/json')

    first = ocs.get_real_options_positioning('AAPL', 352.0)
    cboe.cache_clear()  # force the second call to rely on options_chain_service
    second = ocs.get_real_options_positioning('AAPL', 352.0)
    assert first and second
    assert first['composite'] == second['composite']
    assert web.urls() == [NEW_AAPL]


def test_fetch_budget_snapshot_with_bad_settings_does_not_deadlock(monkeypatch):
    import threading

    monkeypatch.setenv('MRD_CBOE_BASE_URL', 'https://cdn-api.cboe.com/no-placeholder.json')
    monkeypatch.setenv('MRD_CBOE_MAX_FETCHES_PER_MINUTE', 'zero')
    out: dict = {}
    worker = threading.Thread(target=lambda: out.update(cboe.fetch_budget_snapshot()), daemon=True)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive(), 'fetch_budget_snapshot deadlocked on the provider lock'
    assert out['base_url'] == NEW_TEMPLATE
    assert out['max_fetches_per_minute'] == 20


def test_status_row_exposes_the_fetch_limit():
    from app.routes.providers import _options_chain_provider_rows

    rows = _options_chain_provider_rows({'cboe_attempts': 1, 'cboe_hits': 1}, {})
    limit = rows[0]['fetch_limit']
    assert limit['max_fetches_per_minute'] == 20
    assert limit['base_url'] == NEW_TEMPLATE
