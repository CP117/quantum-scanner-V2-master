# Security

The scanner is built to run on your own computer. By default nobody else can
reach it. You can open it to your network or the internet on purpose, and
then everything that changes state needs an admin token.

## Default: this machine only

- `start.sh` and `start.bat` bind the server to `127.0.0.1`. Other devices
  cannot connect.
- No Cloudflare tunnel starts, and cloudflared is not downloaded.
- The thread-dump and source-download routes are not mounted.
- On this machine the dashboard works as before. Pages opened at
  `http://127.0.0.1:8001` or `http://localhost:8001` never ask for a token.

## Opening it up

| Option | Effect |
|---|---|
| `--lan` (or `MRD_BIND_HOST=0.0.0.0`) | Other devices on your network can connect. |
| `--tunnel` (or `MRD_TUNNEL=1`) | A public `https://*.trycloudflare.com` URL is opened. Anyone with the URL can connect. |

Examples: `./start.sh --lan`, `start.bat --lan --tunnel`.

With either option:

- **Reads are public.** Anyone who can reach the server can see the
  dashboard, scan results, watchlists and provider status.
- **Writes need the admin token.** Every `POST`, `PUT`, `PATCH` and `DELETE`
  is refused with `401 {"error": "admin_token_required"}` unless it carries
  the token or comes directly from this machine.
- **Sensitive reads need the token too.** `GET` and `HEAD` on paths starting
  with `/system/threads`, `/download/source`, `/api/download/source`,
  `/api/api-keys` or `/api/admin`.

The rules live in `app/security.py` (`SENSITIVE_PREFIXES`, `EXEMPT_PATHS`,
`AdminGuardMiddleware`). `EXEMPT_PATHS` is empty; any entry added there needs
a written reason next to it.

## What counts as "from this machine"

A request skips the token only when all of these hold:

1. It arrives from a loopback address (`127.0.0.0/8`, `::1`, or
   `::ffff:127.0.0.1`).
2. It carries none of `cf-connecting-ip`, `cf-ray`, `cf-ipcountry`,
   `x-forwarded-for`, `x-forwarded-host`, `x-real-ip` or `forwarded`.
   Tunnel traffic also arrives over loopback, but Cloudflare always adds
   these headers. Zone rules cannot remove `cf-ray`
   ([Cloudflare HTTP headers](https://developers.cloudflare.com/fundamentals/reference/http-headers/),
   [request header rules](https://developers.cloudflare.com/rules/transform/request-header-modification/)).
3. The `Host` header is `127.0.0.1`, `localhost` or `[::1]` with the port the
   server listens on. This blocks DNS-rebinding pages.
4. `Origin`, if present, is in `MRD_CORS_ORIGINS` or is the page's own
   origin (`http://` + the validated `Host`). The own-origin rule keeps the
   dashboard working when the launcher falls back to port 8011.
5. `Sec-Fetch-Site`, if present, is `same-origin` or `none`.

Rules 3 to 5 stop other websites you visit from sending writes to your local
scanner through your browser.

Opening the dashboard on the same computer through its LAN address
(`http://192.168.x.x:8001`) does not count as local. Use `127.0.0.1` or
`localhost` there.

`frontend/proxy-server.js` (a development proxy) adds `X-Forwarded-For`, so
requests through it always need the token.

## The admin token

- **Where it lives:** `data/admin_token.txt`, created on first start with
  permissions `0600` where the OS supports it. The file is git-ignored. If
  `MRD_ADMIN_TOKEN` is set, that value is used instead and no file is
  written.
- **How to send it:** `X-MRD-Admin-Token: <token>` or
  `Authorization: Bearer <token>`.
- **In the browser:** the first write from another device shows a prompt.
  Paste the token; it is saved in that browser's `localStorage` under
  `mrdAdminToken` and sent with later requests to the same origin only. To
  forget it, run `MRDAdminToken.clear()` in the browser console or clear the
  site's data.
- **Rotating it:** stop the server, delete `data/admin_token.txt` (or change
  `MRD_ADMIN_TOKEN`) and start again. Browsers that saved the old token are
  asked for the new one on their next write.
- **Logging:** the startup log says where the token is stored. The token
  itself is never logged.

Example from another device, with the token copied from
`data/admin_token.txt` on the scanner computer:

```bash
curl -X POST -H "X-MRD-Admin-Token: <token>" \
     -H "Content-Type: application/json" -d '{}' \
     http://192.168.1.20:8001/api/universes/toggle
```

## Errors

Unhandled errors return
`500 {"error": "internal_server_error", "request_id": "<32 hex chars>"}`.
The traceback, method and path are logged under the same `request_id`.
Exception messages are not sent to the client.

## Debug and source-download routes

| Variable | Mounts |
|---|---|
| `MRD_ENABLE_DEBUG_ROUTES=1` | `/system/threads`, `/system/threads/summary` (live thread stacks). `scripts/soak.py` uses the summary for thread counts when it is mounted, and the `threads` block of `/system/status` (counts and name prefixes, no stacks) when it is not. |
| `MRD_ENABLE_SOURCE_DOWNLOAD=1` | `/download/source.zip`, `/api/download/source.zip`, `/api/download/source/info`. |

Both default to `0`. When mounted, these routes still need the token from
other devices.

## Variables

| Variable | Default | Meaning |
|---|---|---|
| `MRD_ADMIN_TOKEN` | unset | Admin token. Unset means generate one into `data/admin_token.txt`. |
| `MRD_CORS_ORIGINS` | `http://127.0.0.1:8001,http://localhost:8001` | Browser origins allowed by CORS, comma-separated. |
| `MRD_BIND_HOST` | `127.0.0.1` | Interface the launchers bind to. `--lan` sets `0.0.0.0`. |
| `MRD_TUNNEL` | `0` | `1` makes the launchers open a Cloudflare Quick Tunnel (same as `--tunnel`). |
| `MRD_AUTO_TUNNEL_ON_STARTUP` | `0` | `1` lets the backend open its own tunnel at startup if the launcher has not. The launchers set it from `--tunnel`. |
| `MRD_ENABLE_DEBUG_ROUTES` | `0` | `1` mounts the thread-dump routes. |
| `MRD_ENABLE_SOURCE_DOWNLOAD` | `0` | `1` mounts the source-download routes. |

## Limits to know about

- On the LAN, traffic is plain HTTP, so someone who can watch your network
  traffic could read the token. Tunnel traffic is HTTPS between the browser
  and Cloudflare.
- Reads are not protected once you open the app up. Do not use `--lan` or
  `--tunnel` on networks you do not trust.
- `app/data/public_url.txt` is written at runtime with the current URLs and
  is not tracked by git.
