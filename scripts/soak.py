#!/usr/bin/env python3
"""Soak monitor for a running scanner server (standard library only).

Polls the server every --interval seconds for --minutes minutes and writes
one CSV row per sample:

* total threads and the 10 most common thread-name prefixes
  (/system/threads/summary, mounted only when the server runs with
  MRD_ENABLE_DEBUG_ROUTES=1; it is a sensitive route, so run the soak from
  the same machine or pass --token);
* per-provider calls, errors and rate-limit counts (/api/providers/status);
* rows, universe size, sweeps completed, current batch and evaluations for
  stocks and crypto (/api/scan/snapshot/meta);
* process RSS, if /system/status exposes it, or from /proc/<pid>/status when
  --pid is given (Linux).

A missing endpoint or field becomes a blank cell, never an error.  Rows are
flushed as they are written, so an interrupted run keeps its data.  At the
end (or on Ctrl+C) it prints max threads, thread growth per hour (least-
squares slope), rate-limit deltas per provider and symbols scored per second.

Example:
    python scripts/soak.py --minutes 15 --interval 30
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone

MARKETS = ('stocks', 'crypto')
META_FIELDS = ('rows_scored', 'universe_size', 'sweeps_completed', 'current_batch_index',
               'total_batches', 'evaluations_ever')
COLUMNS = (
    ['utc', 'elapsed_s', 'threads', 'top_thread_prefixes',
     'provider_calls_total', 'provider_errors_total', 'provider_rate_limits_total', 'providers']
    + [f'{m}_{f}' for m in MARKETS for f in META_FIELDS]
    + ['rss_mb']
)


def fetch_json(base_url: str, path: str, token: str | None, timeout: float):
    """GET base_url+path and decode JSON; None on any failure."""
    req = urllib.request.Request(base_url.rstrip('/') + path, headers={'Accept': 'application/json'})
    if token:
        req.add_header('X-MRD-Admin-Token', token)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8', 'replace'))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


def thread_prefix(name: str) -> str:
    """'snap-worker-3' -> 'snap-worker'; 'ThreadPoolExecutor-0_12' -> 'ThreadPoolExecutor'."""
    stripped = re.sub(r'[-_ ]?\d+', '', name or '').strip(' -_')
    return stripped or '<unnamed>'


def find_rss_mb(payload) -> float | None:
    """Look for an rss_mb / rss_bytes field anywhere in a JSON payload."""
    stack = [payload]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key, value in item.items():
                if key == 'rss_mb' and isinstance(value, (int, float)):
                    return float(value)
                if key == 'rss_bytes' and isinstance(value, (int, float)):
                    return float(value) / (1024 * 1024)
                if isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(item, list):
            stack.extend(item)
    return None


def proc_rss_mb(pid: int | None) -> float | None:
    if not pid:
        return None
    try:
        with open(f'/proc/{pid}/status', encoding='ascii') as fh:
            for line in fh:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def take_sample(args, started: float) -> dict:
    now = time.time()
    row: dict = {col: '' for col in COLUMNS}
    row['utc'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    row['elapsed_s'] = round(now - started, 1)
    providers: dict[str, dict] = {}

    threads = fetch_json(args.base_url, '/system/threads/summary', args.token, args.timeout)
    if isinstance(threads, dict):
        if isinstance(threads.get('thread_count'), int):
            row['threads'] = threads['thread_count']
        names = [t.get('name', '') for t in threads.get('threads') or [] if isinstance(t, dict)]
        top = Counter(thread_prefix(n) for n in names).most_common(10)
        row['top_thread_prefixes'] = ';'.join(f'{p}:{c}' for p, c in top)

    status = fetch_json(args.base_url, '/api/providers/status', args.token, args.timeout)
    if isinstance(status, dict):
        for prov in status.get('quote_providers') or []:
            if not isinstance(prov, dict) or not prov.get('name'):
                continue
            providers[prov['name']] = {
                k: int(prov.get(k) or 0) for k in ('calls', 'errors', 'rate_limits')
            }
        if providers:
            row['provider_calls_total'] = sum(p['calls'] for p in providers.values())
            row['provider_errors_total'] = sum(p['errors'] for p in providers.values())
            row['provider_rate_limits_total'] = sum(p['rate_limits'] for p in providers.values())
            row['providers'] = ';'.join(
                f"{name}:{p['calls']}/{p['errors']}/{p['rate_limits']}" for name, p in sorted(providers.items())
            )

    meta = fetch_json(args.base_url, '/api/scan/snapshot/meta', args.token, args.timeout)
    if isinstance(meta, dict):
        for market in MARKETS:
            m = meta.get(market)
            if isinstance(m, dict):
                for field in META_FIELDS:
                    if isinstance(m.get(field), (int, float)):
                        row[f'{market}_{field}'] = m[field]

    rss = proc_rss_mb(args.pid)
    if rss is None:
        rss = find_rss_mb(fetch_json(args.base_url, '/system/status', args.token, args.timeout))
    if rss is not None:
        row['rss_mb'] = round(rss, 1)

    row['_providers'] = providers
    row['_t'] = now
    return row


def _numeric(samples, key):
    return [(s['_t'], float(s[key])) for s in samples if s.get(key) not in ('', None)]


def summarize(samples: list[dict]) -> str:
    if not samples:
        return 'No samples collected.'
    lines = [f'Samples: {len(samples)} over {samples[-1]["elapsed_s"]} s']
    threads = _numeric(samples, 'threads')
    if threads:
        lines.append(f'Max threads: {int(max(v for _, v in threads))}')
        if len(threads) >= 2 and threads[-1][0] > threads[0][0]:
            xs = [(t - threads[0][0]) / 3600.0 for t, _ in threads]
            ys = [v for _, v in threads]
            slope = statistics.linear_regression(xs, ys).slope
            lines.append(f'Thread growth: {slope:+.1f} threads/hour (least-squares slope)')
    else:
        lines.append('Threads: no data (/system/threads/summary unavailable)')

    first = next((s['_providers'] for s in samples if s['_providers']), {})
    last = next((s['_providers'] for s in reversed(samples) if s['_providers']), {})
    if last:
        lines.append('Rate-limit deltas per provider:')
        for name in sorted(last):
            delta = last[name]['rate_limits'] - first.get(name, {}).get('rate_limits', 0)
            calls = last[name]['calls'] - first.get(name, {}).get('calls', 0)
            lines.append(f'  {name}: +{delta} rate limits, +{calls} calls')
    else:
        lines.append('Providers: no data (/api/providers/status unavailable)')

    for market in MARKETS:
        evals = _numeric(samples, f'{market}_evaluations_ever')
        if len(evals) >= 2 and evals[-1][0] > evals[0][0]:
            rate = (evals[-1][1] - evals[0][1]) / (evals[-1][0] - evals[0][0])
            lines.append(f'{market}: {rate:.2f} symbols scored per second')
    rss = _numeric(samples, 'rss_mb')
    if rss:
        lines.append(f'RSS: first {rss[0][1]:.1f} MB, last {rss[-1][1]:.1f} MB, max {max(v for _, v in rss):.1f} MB')
    return '\n'.join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--base-url', default='http://127.0.0.1:8001')
    parser.add_argument('--minutes', type=float, default=15.0)
    parser.add_argument('--interval', type=float, default=30.0)
    parser.add_argument('--out', default='soak.csv')
    parser.add_argument('--token', default=None, help='admin token, sent as X-MRD-Admin-Token')
    parser.add_argument('--pid', type=int, default=None, help='server PID, to read RSS from /proc (Linux)')
    parser.add_argument('--timeout', type=float, default=10.0, help='per-request timeout in seconds')
    args = parser.parse_args(argv)

    started = time.time()
    deadline = started + args.minutes * 60.0
    samples: list[dict] = []
    print(f'Soak: {args.base_url} for {args.minutes:g} min every {args.interval:g} s -> {args.out}')
    try:
        with open(args.out, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction='ignore')
            writer.writeheader()
            while True:
                sample = take_sample(args, started)
                samples.append(sample)
                writer.writerow(sample)
                fh.flush()
                shown = {k: ('-' if sample[k] == '' else sample[k])
                         for k in ('threads', 'provider_rate_limits_total', 'stocks_rows_scored')}
                print(f"[{sample['utc']}] threads={shown['threads']} "
                      f"rate_limits={shown['provider_rate_limits_total']} "
                      f"stocks_rows={shown['stocks_rows_scored']}", flush=True)
                if time.time() + args.interval > deadline:
                    break
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print('Interrupted; summarizing what was collected.')
    print(summarize(samples))
    return 0


if __name__ == '__main__':
    sys.exit(main())
