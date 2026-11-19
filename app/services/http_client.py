"""
Shared HTTP resilience helper for outbound provider calls.

Every external provider in the Ultra Scanner makes single-attempt HTTP
calls today. That works fine on a quiet day but loses data on:
  * Transient 5xx from yfinance / CBOE CDN / Stooq edge nodes.
  * Connection resets during peak market-open traffic.
  * Brief 429 windows that would clear in a second or two.

This module centralizes resilient HTTP so we maximize successful intake
without each provider re-implementing the same retry + backoff + UA-rotation
logic. Behaviorally, it's a near-drop-in replacement for the bare
`requests.get(...)` / `httpx.get(...)` call sites.

Feature set:
  * **Retries** on transient failures (5xx, ConnectionError, ReadTimeout,
    httpx.ReadError, RemoteProtocolError) with exponential backoff +
    bounded jitter. Permanent failures (4xx other than 408/429) are NOT
    retried.
  * **Retry-After** header honored on 429 / 503 — capped at the
    configured max backoff so we never hang a worker indefinitely.
  * **User-Agent rotation** from a small pool of plausible real-browser
    UAs to reduce per-UA throttling on shared providers.
  * **Separate connect/read timeouts**.
  * **Telemetry** stamped into the existing `providers_base.*` counters
    so the providers-health page shows retries / per-attempt status.
  * **Redirects** are explicit. By default each client keeps its own
    behaviour (requests follows, httpx does not). With
    `follow_redirects=True` both clients follow manually, hop by hop,
    with a hop cap and an optional host-suffix allowlist. A 3xx is
    never retried.

Phase 26.18.c
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx
import requests

log = logging.getLogger('app.http_client')


# ---------------------------------------------------------------------------
# User-Agent pool
# ---------------------------------------------------------------------------
# A small, current pool of real browser/version UA strings. Providers that
# rate-limit per-UA (e.g. yfinance's chart endpoint historically) see less
# 429-pressure when callers cycle through several. The pool is intentionally
# small (4) so we don't look like a scraper farm.
_UA_POOL = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 '
    '(KHTML, like Gecko) Version/17.6 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0',
)


def random_user_agent() -> str:
    return random.choice(_UA_POOL)


# ---------------------------------------------------------------------------
# Resilient GET configuration
# ---------------------------------------------------------------------------
@dataclass
class ResilientGetConfig:
    """Single source of truth for resilient outbound GET behaviour."""
    # Total attempts (1 initial + N retries). 3 = initial + 2 retries.
    max_attempts: int = 3
    # Connect timeout in seconds — short because connect failures are
    # almost always immediate.
    connect_timeout: float = 4.0
    # Read timeout in seconds — generous because the bytes-on-wire latency
    # can spike under load.
    read_timeout: float = 12.0
    # Initial backoff between retries (seconds). Doubles on each retry.
    backoff_initial: float = 0.4
    # Cap the exponential backoff so we never wait forever.
    backoff_cap: float = 4.0
    # Bounded random jitter added to each backoff (0..jitter_max). Avoids
    # synchronized retry storms when N providers fail at the same time.
    jitter_max: float = 0.25
    # Cap how long we'll wait on a Retry-After header before just bailing.
    # Providers sometimes return Retry-After: 60 which would block a
    # scoring batch for a minute — better to fall through to the next
    # provider in the cascade.
    retry_after_cap_seconds: float = 5.0


DEFAULT_GET = ResilientGetConfig()


# Status codes that warrant a retry (transient on the provider side).
# Permanent 4xx (400, 401, 403, 404, 410) are NOT retried — those won't
# recover by waiting and we'd just waste budget.
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

# Statuses that carry a Location to follow. Other 3xx codes (300, 304, ...)
# are returned to the caller unchanged.
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})

# Headers that must not be forwarded when a redirect changes host.
_CREDENTIAL_HEADERS = frozenset({'authorization', 'cookie', 'proxy-authorization'})


@dataclass
class ResilientResponse:
    """Mirror of requests.Response just enough for callers to use."""
    status_code: int
    text: str
    headers: dict
    url: str  # the URL the caller asked for (unchanged by redirects)
    attempts: int
    error: str | None = None
    # URL of the request that produced this response. Differs from `url`
    # when a redirect was followed.
    final_url: str = ''
    # Redirect hops that were followed, oldest first: (status, absolute
    # target URL). Empty when no redirect was followed.
    redirects: list[tuple[int, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    @property
    def location(self) -> str | None:
        """The Location header, looked up case-insensitively."""
        return _header(self.headers, 'location')

    def json(self) -> Any:
        import json as _json
        return _json.loads(self.text)


def _sleep_for_attempt(attempt_index: int, cfg: ResilientGetConfig,
                       retry_after: float | None = None) -> None:
    """Compute and sleep for the appropriate backoff before the next try."""
    if retry_after is not None and retry_after > 0:
        wait = min(retry_after, cfg.retry_after_cap_seconds)
    else:
        # Exponential: 0.4s, 0.8s, 1.6s, ... capped at backoff_cap.
        wait = min(cfg.backoff_cap, cfg.backoff_initial * (2 ** attempt_index))
    wait += random.uniform(0.0, cfg.jitter_max)
    time.sleep(wait)


def _retry_after_seconds(headers: dict) -> float | None:
    """Parse the Retry-After header (delta-seconds form only — HTTP-date
    form is exceedingly rare in practice for these providers).
    """
    val = headers.get('Retry-After') or headers.get('retry-after')
    if not val:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _header(headers: dict | None, name: str) -> str | None:
    """Case-insensitive lookup on a plain header dict.

    `dict(httpx.Headers)` lower-cases its keys while
    `dict(requests.structures.CaseInsensitiveDict)` keeps the server's
    casing, so callers must not index the dict directly.
    """
    if not headers:
        return None
    want = name.lower()
    for key, value in headers.items():
        if str(key).lower() == want:
            return value
    return None


def _host_allowed(host: str, suffixes: tuple[str, ...]) -> bool:
    """True when `host` is a suffix's bare domain or a subdomain of it.

    '.cboe.com' allows 'cboe.com' and 'cdn-api.cboe.com' but not
    'evilcboe.com' or 'cboe.com.example.net'.
    """
    host = (host or '').lower().rstrip('.')
    if not host:
        return False
    for suffix in suffixes:
        domain = str(suffix).strip().lower().strip('.')
        if domain and (host == domain or host.endswith('.' + domain)):
            return True
    return False


def _redirect_block_reason(current_url: str, target_url: str,
                           allow_suffixes: tuple[str, ...]) -> str | None:
    """Why a redirect from `current_url` to `target_url` must not be
    followed, or None when it may be followed."""
    current = urlsplit(current_url)
    target = urlsplit(target_url)
    if target.scheme not in ('http', 'https'):
        return 'unsupported_scheme'
    if current.scheme == 'https' and target.scheme != 'https':
        return 'insecure_downgrade'
    if allow_suffixes and not _host_allowed(target.hostname or '', allow_suffixes):
        return 'host_not_allowed'
    return None


def _history_hops(resp: Any) -> list[tuple[int, str]]:
    """Redirect hops a client followed natively, as (status, target URL)."""
    hops: list[tuple[int, str]] = []
    for hop in getattr(resp, 'history', None) or []:
        hop_url = str(getattr(hop, 'url', '') or '')
        location = _header(dict(getattr(hop, 'headers', {}) or {}), 'location') or ''
        hops.append((int(hop.status_code), urljoin(hop_url, location) if location else hop_url))
    return hops


def _get_with_retries(
    url: str,
    *,
    params: dict | None,
    hdrs: dict,
    cfg: ResilientGetConfig,
    rotate_ua: bool,
    client: str,
    native_follow: bool | None,
) -> ResilientResponse:
    """One logical GET with bounded retries + backoff + UA rotation.

    `native_follow` is handed to the client only when it is not None, so
    with None the call is exactly the one this module has always made:
    requests follows redirects itself, httpx does not.
    """
    last_status = 0
    last_error: str | None = None
    last_headers: dict = {}
    last_text: str = ''
    last_final_url = url
    last_hops: list[tuple[int, str]] = []
    attempt = 0

    while attempt < cfg.max_attempts:
        if rotate_ua and 'User-Agent' not in hdrs and 'user-agent' not in hdrs:
            hdrs['User-Agent'] = random_user_agent()
        elif rotate_ua and attempt > 0:
            # On retries, freshen the UA — sometimes provider rate limits
            # are keyed per-UA. We replace, not append.
            hdrs['User-Agent'] = random_user_agent()
        try:
            if client == 'httpx':
                timeout = httpx.Timeout(
                    cfg.read_timeout, connect=cfg.connect_timeout
                )
                extra: dict[str, Any] = {}
                if native_follow is not None:
                    extra['follow_redirects'] = native_follow
                resp = httpx.get(url, params=params, headers=hdrs, timeout=timeout, **extra)
            else:
                extra = {}
                if native_follow is not None:
                    extra['allow_redirects'] = native_follow
                resp = requests.get(
                    url, params=params, headers=hdrs,
                    timeout=(cfg.connect_timeout, cfg.read_timeout),
                    **extra,
                )
            last_status = resp.status_code
            last_text = resp.text
            last_headers = dict(resp.headers)
            last_final_url = str(getattr(resp, 'url', '') or url)
            last_hops = _history_hops(resp)
            if last_status not in _RETRYABLE_STATUS:
                # 2xx success, 3xx redirect or 4xx permanent failure —
                # return immediately. A 3xx is never retried.
                return ResilientResponse(
                    status_code=last_status, text=last_text,
                    headers=last_headers, url=url, attempts=attempt + 1,
                    error=None, final_url=last_final_url, redirects=last_hops,
                )
            # Retryable status — fall through to retry sleep below.
            last_error = f'http_{last_status}'
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectTimeout,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.Timeout,
        ) as exc:
            last_error = f'{type(exc).__name__}: {exc}'[:200]
        except (
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.ConnectTimeout,
            httpx.RemoteProtocolError,
            httpx.ReadError,
            httpx.WriteError,
        ) as exc:
            last_error = f'{type(exc).__name__}: {exc}'[:200]
        except Exception as exc:  # noqa: BLE001 - any other unexpected error
            # Don't retry unexpected exceptions — they're likely programmer
            # errors (bad URL, malformed params, etc.) that won't fix
            # themselves on a retry.
            return ResilientResponse(
                status_code=0, text='', headers={}, url=url,
                attempts=attempt + 1, error=f'{type(exc).__name__}: {exc}'[:200],
                final_url=url,
            )

        attempt += 1
        if attempt >= cfg.max_attempts:
            break
        # Sleep before next attempt — honor Retry-After if it was set.
        _sleep_for_attempt(attempt - 1, cfg, _retry_after_seconds(last_headers))

    # Ran out of attempts.
    return ResilientResponse(
        status_code=last_status, text=last_text, headers=last_headers,
        url=url, attempts=attempt, error=last_error,
        final_url=last_final_url, redirects=last_hops,
    )


def resilient_get(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    cfg: ResilientGetConfig = DEFAULT_GET,
    rotate_ua: bool = True,
    client: str = 'requests',  # 'requests' | 'httpx'
    follow_redirects: bool | None = None,
    max_redirects: int = 3,
    redirect_allow_suffixes: tuple[str, ...] = (),
) -> ResilientResponse:
    """Run an outbound GET with bounded retries + backoff + UA rotation.

    Returns a `ResilientResponse` carrying the final status, body, and
    error string (if any). Callers should treat `status_code == 0` plus
    a non-None `error` as "all retries exhausted on a transient failure";
    any explicit HTTP status (200/404/500/...) reflects what the server
    actually returned on the last attempt.

    Redirects:
      * `follow_redirects=None` (default) keeps each client's own
        behaviour: requests follows redirects, httpx returns the 3xx.
      * `follow_redirects=False` returns the first 3xx for both clients.
      * `follow_redirects=True` follows manually for both clients, one
        hop at a time, at most `max_redirects` hops. When
        `redirect_allow_suffixes` is non-empty, every hop's host must
        match one of the suffixes (e.g. '.cboe.com'). An https -> http
        downgrade is never followed. Credential headers are dropped when
        a hop changes host. A redirect that is not followed comes back
        as the 3xx response with `error='redirect_not_followed: <reason>'`
        and the Location still in `headers`.

    `final_url` is the URL that produced the returned response and
    `redirects` lists the hops that were followed. A 3xx is never retried.

    The helper itself never raises for HTTP errors. It WILL raise on
    obvious programmer errors (None URL, etc.) so they're caught in dev.
    """
    if not url:
        raise ValueError('resilient_get: url is required')

    hdrs = dict(headers or {})
    if follow_redirects is not True:
        return _get_with_retries(
            url, params=params, hdrs=hdrs, cfg=cfg, rotate_ua=rotate_ua,
            client=client, native_follow=follow_redirects,
        )

    allow = tuple(redirect_allow_suffixes or ())
    hop_limit = max(0, int(max_redirects))
    hops: list[tuple[int, str]] = []
    current = url
    hop_params = params
    attempts = 0
    while True:
        resp = _get_with_retries(
            current, params=hop_params, hdrs=hdrs, cfg=cfg, rotate_ua=rotate_ua,
            client=client, native_follow=False,
        )
        attempts += resp.attempts
        resp.url = url
        resp.attempts = attempts
        resp.redirects = list(hops)
        if resp.status_code not in REDIRECT_STATUSES:
            return resp
        here = resp.final_url or current
        location = (resp.location or '').strip()
        if not location:
            resp.error = 'redirect_not_followed: missing_location'
            return resp
        target = urljoin(here, location)
        reason = _redirect_block_reason(here, target, allow)
        if reason is None and len(hops) >= hop_limit:
            reason = 'too_many_redirects'
        if reason is not None:
            resp.error = f'redirect_not_followed: {reason}'
            return resp
        hops.append((resp.status_code, target))
        if (urlsplit(target).hostname or '').lower() != (urlsplit(here).hostname or '').lower():
            for key in [k for k in hdrs if str(k).lower() in _CREDENTIAL_HEADERS]:
                hdrs.pop(key, None)
        log.debug('resilient_get: following %d %s -> %s', resp.status_code, here, target)
        current = target
        # The Location carries its own query string.
        hop_params = None
