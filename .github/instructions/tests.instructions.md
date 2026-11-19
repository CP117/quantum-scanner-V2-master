---
applyTo: "backend/tests/**"
---
- Unit tests are offline and fast. The socket guard in `conftest.py` fails any unmarked test that opens a network connection.
- Before Phase 5 lands, do not import `app.main` in unit tests. After it lands, rely on `MRD_START_BACKGROUND=0`, which `conftest.py` sets.
- Server-dependent tests use the `backend_url` fixture and the `integration` marker. Never hard-code hosts, ports, or absolute paths such as `/app/...`.
- Recorded HTTP payloads live in `backend/tests/fixtures/`, trimmed to the fields the code reads.
- Prefer testing pure functions, and inject time (`now=` or a clock object) instead of sleeping.
