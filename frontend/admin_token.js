/* Admin token support for every dashboard page.
 *
 * Writes (POST, PUT, PATCH, DELETE) and a few sensitive reads need the
 * admin token unless the page is opened on the computer running the
 * scanner (http://127.0.0.1:8001 or http://localhost:8001), where
 * same-origin requests are trusted and nothing changes.
 *
 * Load this file first, without defer, so it wraps window.fetch before
 * any other script runs:
 *   - same-origin requests get the X-MRD-Admin-Token header when a token
 *     is stored in localStorage under "mrdAdminToken";
 *   - a write answered with 401 {"error": "admin_token_required"} shows a
 *     prompt, stores the pasted token and retries once;
 *   - reads are never prompted; they fail as they did before.
 * Other origins never see the token.  See docs/SECURITY.md.
 */
(function () {
  'use strict';
  if (window.__mrdAdminFetchInstalled || typeof window.fetch !== 'function') return;
  window.__mrdAdminFetchInstalled = true;

  var STORAGE_KEY = 'mrdAdminToken';
  var HEADER = 'X-MRD-Admin-Token';
  var WRITE_METHODS = { POST: true, PUT: true, PATCH: true, DELETE: true };
  var nativeFetch = window.fetch.bind(window);
  var hasRequest = typeof Request !== 'undefined';

  function readToken() {
    try { return window.localStorage.getItem(STORAGE_KEY) || ''; } catch (_) { return ''; }
  }

  function storeToken(token) {
    try { window.localStorage.setItem(STORAGE_KEY, token); } catch (_) { /* private mode */ }
  }

  function clearToken() {
    try { window.localStorage.removeItem(STORAGE_KEY); } catch (_) { /* private mode */ }
  }

  function isSameOrigin(url) {
    try { return new URL(String(url), window.location.href).origin === window.location.origin; } catch (_) { return false; }
  }

  function requestUrl(input) {
    return hasRequest && input instanceof Request ? input.url : String(input);
  }

  function requestMethod(input, init) {
    var method = (init && init.method) || (hasRequest && input instanceof Request ? input.method : 'GET');
    return String(method || 'GET').toUpperCase();
  }

  function withToken(input, init, token) {
    var opts = Object.assign({}, init || {});
    var base = opts.headers || (hasRequest && input instanceof Request ? input.headers : undefined);
    var headers = new Headers(base || {});
    headers.set(HEADER, token);
    opts.headers = headers;
    return opts;
  }

  function isTokenRejection(response) {
    if (!response || response.status !== 401) return Promise.resolve(false);
    return response.clone().json().then(
      function (body) { return !!body && body.error === 'admin_token_required'; },
      function () { return false; }
    );
  }

  function askForToken() {
    var answer = window.prompt(
      'This action needs the admin token.\n\n' +
      'Paste the token from data/admin_token.txt on the computer running the scanner ' +
      '(or the value of MRD_ADMIN_TOKEN). It is saved in this browser.',
      ''
    );
    answer = answer == null ? '' : String(answer).trim();
    if (answer) storeToken(answer);
    return answer;
  }

  function adminFetch(input, init) {
    if (!isSameOrigin(requestUrl(input))) return nativeFetch(input, init);
    var method = requestMethod(input, init);
    var used = readToken();
    var first = used ? nativeFetch(input, withToken(input, init, used)) : nativeFetch(input, init);
    // Request objects with a body cannot be replayed, so only plain URLs retry.
    if (!WRITE_METHODS[method] || (hasRequest && input instanceof Request)) return first;
    return first.then(function (response) {
      return isTokenRejection(response).then(function (rejected) {
        if (!rejected) return response;
        // Another request may have stored a fresh token while this one ran.
        var current = readToken();
        var token = current && current !== used ? current : askForToken();
        if (!token) return response;
        return nativeFetch(input, withToken(input, init, token));
      });
    });
  }

  window.fetch = adminFetch;

  // Beacons cannot carry custom headers.  With a stored token, report
  // "not queued" so callers fall back to fetch(), which adds the header.
  if (navigator && typeof navigator.sendBeacon === 'function') {
    var nativeBeacon = navigator.sendBeacon.bind(navigator);
    try {
      navigator.sendBeacon = function (url, data) {
        if (readToken() && isSameOrigin(url)) return false;
        return nativeBeacon(url, data);
      };
    } catch (_) { /* read-only in some browsers; beacons then go without the token */ }
  }

  window.MRDAdminToken = { get: readToken, set: storeToken, clear: clearToken };
})();
