"""Phase 2B: Stooq is off by default and opt-in with a key.

Since March 2026 Stooq needs an API key for CSV downloads. Keyless (or
rejected) requests get a plain-text "Get your apikey" page, scripted
requests are often refused with "Access denied", and a spent daily quota
comes back as HTTP 200 with "Exceeded the daily hits limit". These tests
pin:

* the gate: download_quotes() and fetch() leave Stooq alone unless
  MRD_STOOQ_ENABLED=1 and a key exists and the circuit is closed,
* reply classification before parsing, and that refusals open the circuit
  (quota until the next UTC midnight, others on the cooldown ladder)
  without counting against the symbol,
* that the key is sent as `apikey` and masked in errors and logs,
* the lazily created two-worker pool and the batch budget.

The HTTP layer is a requests.Session with an in-process adapter, so the
real URL building, `params` encoding and exception text all run; no socket
is opened.
"""
from __future__ import annotations

import io
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
import requests
from requests.adapters import BaseAdapter
from requests.structures import CaseInsensitiveDict

from app.services import api_keys
from app.services import http_client
from app.services import scoring_service
from app.services.providers import base as providers_base
from app.services.providers import cryptocompare_provider, coinpaprika_provider
from app.services.providers import finnhub_provider, yahoo_chart_provider, yfinance_provider
from app.services.providers import stooq_provider as stooq

FIXTURE = Path(__file__).parent / 'fixtures' / 'stooq_aapl_daily.csv'
KEY = 'k3yABCDEF0123456789abcdef0123456'

# Stooq's reply to a CSV request without a valid key (plain text).
KEY_PAGE = (
    'Get your apikey:\n'
    '1. Open https://stooq.com/q/d/?s=aapl.us&get_apikey\n'
    '2. Enter the captcha code.\n'
    '3. Copy the CSV download link at the bottom of the page - it will contain the <apikey> variable.\n'
    '4. Append the <apikey> variable with its value to your requests, e.g.\n'
    '   https://stooq.com/q/d/l/?s=aapl.us&i=d&apikey=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n'
)
QUOTA = 'Exceeded the daily hits limit'
HTML = '<!DOCTYPE html><html><head><title>Stooq</title></head><body>...</body></html>'


# ---------------------------------------------------------------------------
# Fake web
# ---------------------------------------------------------------------------
class FakeStooq:
    """Serves one reply (or raises) for every request and records the URLs."""

    def __init__(self) -> None:
        self.status = 200
        self.content_type = 'text/csv'
        self.body = ''
        self.raise_exc: type[Exception] | None = None
        self.delay = 0.0
        self.seen: list[str] = []
        self._lock = threading.Lock()

    def reply(self, status: int, body: str, content_type: str = 'text/plain') -> None:
        self.status, self.body, self.content_type = status, body, content_type

    def query(self, i: int = 0) -> dict[str, list[str]]:
        return parse_qs(urlsplit(self.seen[i]).query)

    def requests_get(self, url, **kwargs):
        session = requests.Session()
        adapter = _Adapter(self)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session.get(url, **kwargs)


class _Adapter(BaseAdapter):
    def __init__(self, web: FakeStooq) -> None:
        super().__init__()
        self.web = web

    def send(self, request, **kwargs):  # noqa: D102 - test double
        with self.web._lock:
            self.web.seen.append(request.url)
        if self.web.delay:
            time.sleep(self.web.delay)
        if self.web.raise_exc is not None:
            # Same shape as urllib3's message: the full path, query included.
            raise self.web.raise_exc(
                "HTTPSConnectionPool(host='stooq.com', port=443): Max retries exceeded "
                f"with url: {request.path_url} (Caused by ConnectTimeoutError('timed out'))"
            )
        body = self.web.body.encode('utf-8')
        resp = requests.Response()
        resp.status_code = self.web.status
        resp.headers = CaseInsensitiveDict({'Content-Type': self.web.content_type})
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
    fake = FakeStooq()
    monkeypatch.setattr(http_client.requests, 'get', fake.requests_get)
    return fake


class _Clock:
    def __init__(self, start: float = 10_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def mono(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(stooq, '_mono', c)
    return c


@pytest.fixture(autouse=True)
def isolated_stooq(monkeypatch, tmp_path):
    """Fresh Stooq state, no pool, no key, defaults for the env."""
    monkeypatch.delenv('MRD_STOOQ_ENABLED', raising=False)
    monkeypatch.delenv('STOOQ_API_KEY', raising=False)
    monkeypatch.delenv('FINNHUB_API_KEY', raising=False)
    monkeypatch.setattr(api_keys, '_KEYS_FILE', tmp_path / 'user_api_keys.json')
    monkeypatch.setattr(api_keys, '_CACHE', {})
    fresh = {k: (None if k in ('last_block_reason', 'last_block_utc') else
                 False if k == 'half_open' else
                 0.0 if isinstance(v, float) else 0)
             for k, v in stooq._diag.items()}
    monkeypatch.setattr(stooq, '_diag', fresh)
    monkeypatch.setattr(stooq, '_pool', None)
    saved = providers_base._stats.pop('stooq', None)
    yield
    pool = stooq._pool
    if pool is not None:
        pool.shutdown(wait=True, cancel_futures=True)
    providers_base._stats.pop('stooq', None)
    if saved is not None:
        providers_base._stats['stooq'] = saved


def _enable(monkeypatch, key: str | None = KEY) -> None:
    monkeypatch.setenv('MRD_STOOQ_ENABLED', '1')
    if key:
        monkeypatch.setenv('STOOQ_API_KEY', key)


def _stub_other_providers(monkeypatch) -> list[str]:
    """Upstream layers resolve nothing; returns the list of layers called."""
    called: list[str] = []

    def empty(name):
        def _fetch(symbols, market):
            called.append(name)
            return {}
        return _fetch

    for mod, name in ((yfinance_provider, 'yfinance'), (yahoo_chart_provider, 'yahoo-chart'),
                      (finnhub_provider, 'finnhub'), (cryptocompare_provider, 'cryptocompare'),
                      (coinpaprika_provider, 'coinpaprika')):
        monkeypatch.setattr(mod, 'fetch', empty(name))
    return called


def _stooq_threads() -> list[str]:
    return [t.name for t in threading.enumerate() if t.name.startswith('stooq')]


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------
def test_default_download_quotes_never_calls_stooq(monkeypatch, web):
    _stub_other_providers(monkeypatch)
    stooq_calls: list[list[str]] = []
    monkeypatch.setattr(stooq, 'fetch', lambda symbols, market: stooq_calls.append(list(symbols)) or {})

    assert scoring_service.download_quotes(['AAPL', 'MSFT'], 'stocks') == {}

    assert stooq_calls == []
    assert web.seen == []
    assert (providers_base.provider_stats_snapshot().get('stooq') or {}).get('calls', 0) == 0
    assert stooq.availability() == (False, 'disabled')
    snap = stooq.stats_snapshot()
    assert (snap['enabled'], snap['has_key'], snap['idle_reason']) == (False, False, 'disabled')
    assert stooq._pool is None and snap['pool_started'] is False
    assert _stooq_threads() == []


def test_direct_fetch_while_disabled_sends_nothing(web):
    assert stooq.fetch(['AAPL'], 'stocks') == {}
    assert web.seen == [] and stooq._pool is None


def test_enabled_without_key_is_still_skipped(monkeypatch, web):
    _enable(monkeypatch, key=None)
    _stub_other_providers(monkeypatch)
    stooq_calls: list[list[str]] = []
    monkeypatch.setattr(stooq, 'fetch', lambda symbols, market: stooq_calls.append(list(symbols)) or {})

    scoring_service.download_quotes(['AAPL'], 'stocks')

    assert stooq_calls == []
    assert stooq.availability() == (False, 'no_api_key')
    snap = stooq.stats_snapshot()
    assert (snap['enabled'], snap['has_key'], snap['idle_reason']) == (True, False, 'no_api_key')


def test_enabled_without_key_direct_fetch_sends_nothing(monkeypatch, web):
    _enable(monkeypatch, key=None)
    assert stooq.fetch(['AAPL'], 'stocks') == {}
    assert web.seen == [] and stooq._pool is None


def test_enabled_with_key_joins_the_cascade_last(monkeypatch, web):
    _enable(monkeypatch)
    called = _stub_other_providers(monkeypatch)
    stooq_calls: list[list[str]] = []
    monkeypatch.setattr(stooq, 'fetch', lambda symbols, market: stooq_calls.append(list(symbols)) or {})

    scoring_service.download_quotes(['AAPL', 'MSFT'], 'stocks')

    assert called == ['yfinance', 'yahoo-chart']  # no finnhub key in this test
    assert stooq_calls == [['AAPL', 'MSFT']]
    assert stooq.availability() == (True, None)
    assert stooq.stats_snapshot()['idle_reason'] is None


def test_open_circuit_keeps_stooq_out_of_the_cascade(monkeypatch, mono, web):
    _enable(monkeypatch)
    _stub_other_providers(monkeypatch)
    stooq_calls: list[list[str]] = []
    monkeypatch.setattr(stooq, 'fetch', lambda symbols, market: stooq_calls.append(list(symbols)) or {})
    stooq._diag['circuit_open_until'] = mono.t + 60

    scoring_service.download_quotes(['AAPL'], 'stocks')

    assert stooq_calls == []
    assert stooq.availability() == (False, 'circuit_open')
    assert stooq.stats_snapshot()['idle_reason'] == 'circuit_open'


def test_crypto_cascade_never_includes_stooq(monkeypatch, web):
    _enable(monkeypatch)
    _stub_other_providers(monkeypatch)
    stooq_calls: list[list[str]] = []
    monkeypatch.setattr(stooq, 'fetch', lambda symbols, market: stooq_calls.append(list(symbols)) or {})
    scoring_service.download_quotes(['BTC'], 'crypto')
    assert stooq_calls == []


def test_key_can_be_stored_from_the_dashboard(monkeypatch):
    monkeypatch.setattr(api_keys, '_CACHE', None)
    assert 'stooq' in api_keys.SUPPORTED_PROVIDERS
    assert api_keys.list_configured()['stooq'] is False
    assert api_keys.set_key('stooq', KEY) is True
    assert api_keys.get('stooq') == KEY
    assert api_keys.list_configured()['stooq'] is True
    monkeypatch.setenv('MRD_STOOQ_ENABLED', '1')
    assert stooq.availability() == (True, None)
    assert api_keys.delete('stooq') is True
    assert stooq.availability() == (False, 'no_api_key')


def test_env_key_fallback(monkeypatch):
    monkeypatch.setenv('STOOQ_API_KEY', KEY)
    assert api_keys.get('stooq') == KEY
    assert stooq.stats_snapshot()['has_key'] is True


@pytest.mark.parametrize('raw', ['', '0', 'false', 'no', 'off', 'maybe'])
def test_only_truthy_values_enable_stooq(monkeypatch, raw):
    monkeypatch.setenv('MRD_STOOQ_ENABLED', raw)
    assert stooq.is_enabled() is False


@pytest.mark.parametrize('raw', ['1', 'true', 'YES', ' on '])
def test_truthy_values_enable_stooq(monkeypatch, raw):
    monkeypatch.setenv('MRD_STOOQ_ENABLED', raw)
    assert stooq.is_enabled() is True


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
CSV = FIXTURE.read_text(encoding='utf-8')


@pytest.mark.parametrize(('status', 'ctype', 'body', 'expected'), [
    (200, 'text/csv', CSV, ('ok', None)),
    (200, 'text/csv', '\ufeff' + CSV, ('ok', None)),
    (200, 'text/plain', KEY_PAGE, ('provider_down', 'key_rejected')),
    (200, 'text/html; charset=utf-8', KEY_PAGE, ('provider_down', 'key_rejected')),
    (200, 'text/html; charset=utf-8', '<html><body>Stooq</body></html>', ('provider_down', 'blocked')),
    (200, 'text/plain', HTML, ('provider_down', 'blocked')),
    (200, 'text/plain', 'Access denied', ('provider_down', 'blocked')),
    (403, 'text/plain', 'Forbidden', ('provider_down', 'blocked')),
    (200, 'text/plain', QUOTA, ('rate_limited', 'quota')),
    (429, 'text/plain', QUOTA, ('rate_limited', 'quota')),
    (429, 'text/plain', '', ('rate_limited', 'http_429')),
    (503, 'text/html', HTML, ('http_error', 'http_503')),
    (404, 'text/plain', 'nope', ('http_error', 'http_404')),
    (200, 'text/csv', 'No data', ('not_found', 'no_data')),
    (200, 'text/csv', '', ('bad_payload', 'empty_body')),
    (200, 'text/csv', 'Symbol,Date,Time,Open,High,Low,Close\nAAPL.US,...', ('bad_payload', 'no_csv_header')),
    (200, 'application/json', '{"error": "x"}', ('bad_payload', 'no_csv_header')),
])
def test_classify_response(status, ctype, body, expected):
    assert stooq.classify_response(status, ctype, body) == expected


def test_csv_fixture_parses_into_last_and_previous_close(monkeypatch, web):
    _enable(monkeypatch)
    web.reply(200, CSV, 'text/csv')

    out = stooq.fetch(['AAPL'], 'stocks')

    row = out['AAPL']
    assert row['last_price'] == pytest.approx(192.53)
    assert row['previous_close'] == pytest.approx(193.58)
    assert (row['open'], row['day_high'], row['day_low']) == pytest.approx((192.90, 193.15, 191.72))
    assert row['volume'] == pytest.approx(42628800)
    assert (row['source'], row['provider_outcome']) == ('stooq', 'live_success')
    q = web.query()
    assert q == {'s': ['aapl.us'], 'i': ['d'], 'apikey': [KEY]}
    assert stooq._diag['successes'] == 1 and stooq._diag['consecutive_failures'] == 0


def test_parser_does_not_depend_on_row_order():
    header, *rows = CSV.strip().splitlines()
    parsed = stooq._parse_csv('\n'.join([header, *reversed(rows)]))
    assert [r['date'] for r in parsed] == ['2023-12-27', '2023-12-28', '2023-12-29']


# ---------------------------------------------------------------------------
# Refusals open the circuit and are not symbol failures
# ---------------------------------------------------------------------------
def _no_symbol_blame() -> None:
    d = stooq._diag
    assert d['no_data'] == 0 and d['parse_errors'] == 0 and d['successes'] == 0
    assert d['consecutive_failures'] == 0


def test_quota_opens_the_circuit_until_next_utc_midnight(monkeypatch, mono, web):
    _enable(monkeypatch)
    monkeypatch.setattr(stooq, '_utcnow', lambda: datetime(2026, 11, 19, 22, 30, tzinfo=timezone.utc))
    web.reply(200, QUOTA)

    assert stooq.fetch(['AAPL', 'MSFT', 'NVDA', 'AMD'], 'stocks') == {}

    # Two workers: at most two requests were in flight; the rest were dropped unsent.
    assert 1 <= len(web.seen) <= 2
    assert stooq._diag['skipped_circuit_open'] == 4 - len(web.seen)
    assert stooq._diag['quota_hits'] == len(web.seen)
    assert stooq._diag['circuit_trip_count'] == 0  # quota does not climb the ladder
    _no_symbol_blame()
    snap = stooq.stats_snapshot()
    assert snap['circuit_open'] is True
    assert snap['circuit_remaining_seconds'] == pytest.approx(90 * 60)
    assert snap['circuit_reopens_at_utc'] == '2026-11-20T00:00:00+00:00'
    assert (snap['last_block_reason'], snap['idle_reason']) == ('quota', 'circuit_open')
    assert stooq.availability() == (False, 'circuit_open')
    row = providers_base.provider_stats_snapshot()['stooq']
    assert row['rate_limits'] >= 1 and 'quota' in row['last_error']

    # Still closed one second before midnight; open again after it.
    mono.t += 90 * 60 - 1
    assert stooq.fetch(['AAPL'], 'stocks') == {}
    assert stooq.availability() == (False, 'circuit_open')
    sent = len(web.seen)
    mono.t += 2
    assert stooq.availability() == (True, None)
    web.reply(200, CSV, 'text/csv')
    assert stooq.fetch(['AAPL'], 'stocks')['AAPL']['last_price'] == pytest.approx(192.53)
    assert len(web.seen) == sent + 1


def test_quota_just_after_midnight_waits_a_full_day(monkeypatch, mono, web):
    _enable(monkeypatch)
    monkeypatch.setattr(stooq, '_utcnow', lambda: datetime(2026, 11, 20, 0, 0, 30, tzinfo=timezone.utc))
    web.reply(200, QUOTA)
    stooq.fetch(['AAPL'], 'stocks')
    assert stooq.stats_snapshot()['circuit_remaining_seconds'] == pytest.approx(24 * 3600 - 30)


@pytest.mark.parametrize(('status', 'ctype', 'body', 'reason'), [
    (200, 'text/html', HTML, 'blocked'),
    (200, 'text/plain', 'Access denied', 'blocked'),
    (403, 'text/plain', 'Access denied', 'blocked'),
    (200, 'text/plain', KEY_PAGE, 'key_rejected'),
])
def test_refusal_trips_the_ladder_once_and_stops_the_batch(monkeypatch, mono, web, status, ctype, body, reason):
    _enable(monkeypatch)
    web.reply(status, body, ctype)

    assert stooq.fetch(['AAPL', 'MSFT', 'NVDA', 'AMD'], 'stocks') == {}

    assert 1 <= len(web.seen) <= 2
    assert stooq._diag['skipped_circuit_open'] == 4 - len(web.seen)
    assert stooq._diag[reason] == len(web.seen)
    assert stooq._diag['circuit_trip_count'] == 1  # two refusals, one trip
    _no_symbol_blame()
    snap = stooq.stats_snapshot()
    assert snap['circuit_remaining_seconds'] == pytest.approx(30 * 60)
    assert (snap['last_block_reason'], snap['idle_reason']) == (reason, 'circuit_open')
    assert stooq.availability() == (False, 'circuit_open')


def test_second_refusal_after_reopen_climbs_the_ladder(monkeypatch, mono, web):
    _enable(monkeypatch)
    web.reply(200, HTML, 'text/html')
    stooq.fetch(['AAPL'], 'stocks')
    mono.t += 30 * 60 + 1
    stooq.fetch(['AAPL'], 'stocks')  # half-open probe, refused again
    assert stooq._diag['circuit_trip_count'] == 2
    assert stooq.stats_snapshot()['circuit_remaining_seconds'] == pytest.approx(60 * 60)


def test_bad_payload_counts_toward_the_threshold_not_the_symbol(monkeypatch, mono, web):
    _enable(monkeypatch)
    web.reply(200, '{"unexpected": true}', 'application/json')
    stooq.fetch(['AAPL'], 'stocks')
    d = stooq._diag
    assert d['bad_payload'] == 1 and d['consecutive_failures'] == 1
    assert d['no_data'] == 0 and d['parse_errors'] == 0
    assert stooq.stats_snapshot()['circuit_open'] is False
    assert providers_base.provider_stats_snapshot()['stooq']['last_error'] == 'bad_payload: no_csv_header'


# ---------------------------------------------------------------------------
# Key handling
# ---------------------------------------------------------------------------
def test_key_is_masked_in_errors_and_logs(monkeypatch, web, caplog):
    _enable(monkeypatch)
    web.raise_exc = requests.exceptions.ConnectTimeout
    caplog.set_level(logging.DEBUG)

    assert stooq.fetch(['AAPL'], 'stocks') == {}

    assert len(web.seen) == 1  # one attempt, no retry of a connect timeout
    assert KEY in web.seen[0]  # it was sent...
    row = providers_base.provider_stats_snapshot()['stooq']
    assert KEY not in row['last_error']  # ...but never recorded
    assert 'apikey=***' in row['last_error']
    assert row['timeouts'] == 1 and stooq._diag['timeouts'] == 1
    assert all(KEY not in r.getMessage() for r in caplog.records)


def test_urllib3_request_log_lines_are_masked(caplog):
    caplog.set_level(logging.DEBUG, logger='urllib3.connectionpool')
    logging.getLogger('urllib3.connectionpool').debug(
        '%s://%s:%s "%s %s %s" %s %s', 'https', 'stooq.com', 443, 'GET',
        f'/q/d/l/?s=aapl.us&i=d&apikey={KEY}', 'HTTP/1.1', 200, None,
    )
    msgs = [r.getMessage() for r in caplog.records if r.name == 'urllib3.connectionpool']
    assert msgs and all(KEY not in m for m in msgs)
    assert 'apikey=***' in msgs[-1]


@pytest.mark.parametrize(('text', 'expected'), [
    (f'url: /q/d/l/?s=aapl.us&i=d&apikey={KEY} (Caused by x)', 'url: /q/d/l/?s=aapl.us&i=d&apikey=*** (Caused by x)'),
    (f"url='https://stooq.com/q/d/l/?apikey={KEY}&s=aapl.us'", "url='https://stooq.com/q/d/l/?apikey=***&s=aapl.us'"),
    (f'ApiKey={KEY[:7]}', 'ApiKey=***'),  # cut short by a [:200] truncation
    (f'key alone: {KEY}', 'key alone: ***'),
])
def test_redact(text, expected):
    assert stooq.redact(text, KEY) == expected


# ---------------------------------------------------------------------------
# Pool and batch budget
# ---------------------------------------------------------------------------
def test_pool_is_created_on_first_use_with_two_workers(monkeypatch, web):
    assert stooq._pool is None
    _enable(monkeypatch)
    web.reply(200, CSV, 'text/csv')
    stooq.fetch(['AAPL', 'MSFT', 'NVDA', 'AMD', 'TSLA'], 'stocks')
    assert stooq._pool is not None and stooq._pool._max_workers == 2
    assert 1 <= len(_stooq_threads()) <= 2
    assert stooq.stats_snapshot()['pool_started'] is True


def test_budget_cancels_symbols_that_have_not_started(monkeypatch, web):
    _enable(monkeypatch)
    monkeypatch.setattr(stooq, '_BATCH_BUDGET_SECONDS', 0.2)
    web.reply(200, CSV, 'text/csv')
    web.delay = 0.6
    started = time.monotonic()

    assert stooq.fetch([f'S{i}' for i in range(10)], 'stocks') == {}

    assert time.monotonic() - started < 0.55
    stooq._pool.shutdown(wait=True)
    assert len(web.seen) == 2  # the two in flight; the other eight were cancelled
    assert stooq._diag['attempts'] == 2


# ---------------------------------------------------------------------------
# Status page: the dedicated circuit row
# ---------------------------------------------------------------------------
def test_circuit_row_shows_reopen_time_only_while_open():
    from app.routes.providers import _circuit_breaker_rows

    diag = {'attempts': 20, 'circuit_open': True, 'consecutive_failures': 20,
            'fail_threshold': 20, 'circuit_trip_count': 1,
            'circuit_reopens_at_utc': '2026-11-20T00:21:58+00:00'}
    row = next(r for r in _circuit_breaker_rows([], diag) if r['kind'] == 'dedicated-exponential')
    assert row['state'] == 'open'
    assert row['open_until_utc'] == '2026-11-20T00:21:58+00:00'

    diag['circuit_open'] = False
    row = next(r for r in _circuit_breaker_rows([], diag) if r['kind'] == 'dedicated-exponential')
    assert row['state'] == 'closed' and row['open_until_utc'] is None

    assert _circuit_breaker_rows([], {'attempts': 0}) == []
