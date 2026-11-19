// Tiny reverse proxy: port 3000 -> backend 8001 (preview ingress sends non-/api traffic here).
const http = require('http');

const TARGET_PORT = 8001;

const server = http.createServer((req, res) => {
  // Mark every proxied request with X-Forwarded-For.  The backend treats
  // only direct loopback requests as trusted-local, so without this header
  // anyone who can reach this proxy would pass that check (app/security.py).
  const prior = req.headers['x-forwarded-for'];
  const client = req.socket.remoteAddress || 'unknown';
  const opts = {
    hostname: '127.0.0.1',
    port: TARGET_PORT,
    path: req.url,
    method: req.method,
    headers: {
      ...req.headers,
      host: `127.0.0.1:${TARGET_PORT}`,
      'x-forwarded-for': prior ? `${prior}, ${client}` : client,
    },
  };
  const upstream = http.request(opts, (up) => {
    res.writeHead(up.statusCode, up.headers);
    up.pipe(res);
  });
  upstream.on('error', () => {
    res.writeHead(502, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'backend_unavailable' }));
  });
  req.pipe(upstream);
});

server.listen(process.env.PORT || 3000, '0.0.0.0', () => {
  console.log(`proxy listening on ${process.env.PORT || 3000} -> ${TARGET_PORT}`);
});
