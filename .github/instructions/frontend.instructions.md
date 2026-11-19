---
applyTo: "frontend/**"
---
- No build step and no new frameworks. Plain modern JavaScript in the existing static files.
- All requests go through the shared `fetchJson` helper, so the admin token, ETag handling and error reporting stay in one place.
- Row order is deterministic: score rounded to 2 decimals descending, then `scored_at_utc` descending, then symbol ascending.
- Never remove a row from the table because of age alone; show staleness instead.
- Re-render only rows whose data changed, and keep scroll position and selection stable across polls.
