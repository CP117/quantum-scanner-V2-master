---
applyTo: "app/services/providers/**,app/services/net/**,app/services/http_client.py,app/services/scoring_service.py,app/services/provider_session.py"
---
- Every provider returns the canonical quote dict from `app/services/quote_math.build_quote` (after Phase 2C) and sets `provider_outcome` to one of: `live_success`, `rate_limited`, `provider_down`, `not_found`, `bad_payload`, `disabled`.
- Rate limits and outages are provider problems, not symbol problems. They must never reach `symbol_blacklist_service.record_failure`.
- Pacing and concurrency come from the shared host limiter, not from sleeps or private thread pools. No provider owns an executor larger than 4 workers.
- Redirects are explicit: follow them with a host allowlist, or treat 3xx as a distinct outcome.
- Validate content type and shape before parsing. An HTML body where JSON or CSV is expected is `bad_payload`.
- Unit tests use recorded fixtures under `backend/tests/fixtures/`; no live calls.
