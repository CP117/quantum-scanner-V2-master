# Copilot instructions for quantum-scanner-V2-master

## What this repository is
A local-first market scanner. Backend: Python 3.12, FastAPI and uvicorn, assembled in `app/main.py`, served on port 8001 by the launchers (`start.bat`, `start.sh`). Frontend: static files in `frontend/`, vanilla JavaScript with no build step (`frontend/app.js` is about 6,300 lines). State lives in process memory plus JSON and SQLite files under `data/`. Market data comes from Yahoo (yfinance plus the chart and spark JSON endpoints), Finnhub (keyed), CBOE delayed options, CoinGecko, CoinPaprika and CryptoCompare.

## Layout
- `app/main.py`: app assembly, router mounting, background-service startup.
- `app/routes/`: HTTP routes. `app/services/`: domain logic. `app/services/providers/`: one module per data provider. `app/utils/`: shared helpers (`app/utils/time.py` has `utcnowiso`).
- Scan pipeline: `snapshot_store` (sweep loop and ranked snapshot) -> `batch_service` -> `result_store.get_results_batch` -> `scoring_service.score_symbol_rows` -> `scoring_service.download_quotes` (provider cascade).
- Tiers: `tier_manager`, `tier_1_active_scanner`, `tier_2_monitor_scanner`, `tier_3_background_scanner`.
- Tests: `backend/tests/`.

## Rules
- Make the smallest change that satisfies the task. Do not reformat or reorganize code the task does not mention.
- Every new tunable is an environment variable prefixed `MRD_`, read in one place, with a safe default, and listed in `.env.example` with a one-line description. API keys keep the existing `{PROVIDER}_API_KEY` convention.
- Pin every new dependency to an exact version in `requirements.txt` or `requirements-dev.txt`, and explain it in the PR.
- Once `app/services/net/` exists, all outbound HTTP to market-data hosts goes through it. Do not add direct `requests.get`, `httpx.get` or `yf.*` calls that bypass it.
- Never retry HTTP 429. Honor `Retry-After`. Never rotate or randomize User-Agent strings.
- No thread may start at import time. Background threads have a name prefix and a stop event, and (once Phase 5 lands) are registered with the thread registry.
- Do not swallow exceptions. A broad `except Exception` must log at WARNING or above with context, or increment a counter that an endpoint exposes.
- Do not import modules or names that do not exist; `backend/tests/test_static_imports.py` enforces this.
- Never log secrets (API keys, the admin token). Never commit user state or secrets under `data/`.
- Keep existing endpoint responses backward compatible: add fields; do not rename or remove them unless the task says so.
- Unit tests run offline. Mark server-dependent tests `@pytest.mark.integration` and internet-dependent tests `@pytest.mark.network`.

## Commands
- Install: `python -m pip install -r requirements.txt -r requirements-dev.txt`
- Unit tests: `python -m pytest`
- Integration tests (server running): `python -m pytest -m integration`
- Run locally: `MRD_AUTO_TUNNEL_ON_STARTUP=0 python -m uvicorn app.main:app --host 127.0.0.1 --port 8001`
- Soak check (server running): `python scripts/soak.py --minutes 15`

## Pull requests
Each PR covers one phase of the fix plan. The description lists: summary, changes by file, tests added, verification output, deviations from the phase prompt, and follow-ups. If a phase grows beyond about 800 changed lines (excluding tests and fixtures), stop and propose a split.
