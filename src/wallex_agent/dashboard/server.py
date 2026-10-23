"""Tiny dependency-free web dashboard (stdlib http.server) for the running agent.

Reads state/status_<mode>.json (written every loop) and the decision log; serves:
  GET /                 HTML dashboard (auto-refresh, RTL Persian)
  GET /api/status       the status JSON
  GET /api/decisions    last N decision-log records   (?n=100&kind=entry,exit)
  GET /healthz          200 if the agent looped within the last 5 minutes
Protect with DASHBOARD_TOKEN (query ?token=... or header X-Token); open when unset.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List, Optional
from urllib.parse import parse_qs, urlparse

log = logging.getLogger(__name__)
HTML_PATH = Path(__file__).with_name("index.html")


class DashboardState:
    def __init__(self, state_dir: str, decision_dir: str, mode: str):
        self.state_dir = Path(state_dir)
        self.decision_dir = Path(decision_dir)
        self.mode = mode

    def status(self) -> dict:
        p = self.state_dir / f"status_{self.mode}.json"
        if not p.exists():
            return {"mode": self.mode, "updated_at": None, "summary": None, "message": "agent has not completed a loop yet"}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"mode": self.mode, "updated_at": None, "message": "status file unreadable (being written)"}

    def decisions(self, n: int = 100, kinds: Optional[List[str]] = None) -> List[dict]:
        files = sorted(self.decision_dir.glob(f"{self.mode}_*.jsonl"))
        out: List[dict] = []
        for f in reversed(files):
            try:
                lines = f.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in reversed(lines):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if kinds and r.get("kind") not in kinds:
                    continue
                out.append(_compact(r))
                if len(out) >= n:
                    return out
        return out


def _compact(r: dict) -> dict:
    """Keep the dashboard payload small: drop bulky prompt copies, keep the human-readable parts."""
    keep = {k: r.get(k) for k in ("id", "ts", "kind", "symbol", "module", "stage", "reason", "reasons", "why", "pnl_tmn", "pnl_pct",
                                  "held_minutes", "entry_price", "exit_price", "qty", "score", "vetoes", "equity_before", "equity")}
    if r.get("ai_decision"):
        d = r["ai_decision"]; keep["ai"] = {k: d.get(k) for k in ("action", "confidence", "size_factor", "rationale", "key_risks")}
    if r.get("decision"):
        d = r["decision"]; keep["ai"] = {k: d.get(k) for k in ("action", "confidence", "size_factor", "rationale", "key_risks")}
    if r.get("regime"):
        g = r["regime"]; keep["regime"] = {k: g.get(k) for k in ("regime", "confidence", "allow_entries", "size_factor", "summary")} if isinstance(g, dict) else g
    if r.get("risk_plan"):
        keep["risk_plan"] = {k: r["risk_plan"].get(k) for k in ("notional_tmn", "stop_loss_pct", "take_profit_pct", "size_factor")}
    if r.get("fill"):
        f = r["fill"]; keep["fill"] = {k: f.get(k) for k in ("qty", "price", "notional_tmn", "fee_tmn", "status")} if isinstance(f, dict) else f
    if r.get("signal") and isinstance(r["signal"], dict):
        keep["signal"] = {"score": r["signal"].get("score"), "reasons": r["signal"].get("reasons"), "vetoes": r["signal"].get("vetoes")}
    return {k: v for k, v in keep.items() if v not in (None, [], {})}


def make_handler(state: DashboardState, token: str):
    from wallex_agent.dashboard.mcp import MCPHandler
    mcp = MCPHandler(state)

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet
            log.debug("dash %s", fmt % args)

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self) -> bytes:
            """Support both Content-Length and chunked transfer encoding (some MCP clients stream the JSON body)."""
            te = (self.headers.get("Transfer-Encoding") or "").lower()
            if "chunked" in te:
                chunks = []
                while True:
                    line = self.rfile.readline().strip()
                    if not line:
                        break
                    size = int(line.split(b";")[0], 16)
                    if size == 0:
                        self.rfile.readline()          # trailing CRLF
                        break
                    chunks.append(self.rfile.read(size))
                    self.rfile.readline()              # CRLF after chunk
                return b"".join(chunks)
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(n) if n else b""

        def _authed(self, q) -> bool:
            if not token:
                return True
            return q.get("token", [None])[0] == token or self.headers.get("X-Token") == token

        def do_POST(self):
            u = urlparse(self.path); q = parse_qs(u.query)
            if u.path != "/mcp":
                return self._send(404, b"not found", "text/plain")
            if not self._authed(q):
                return self._send(401, json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32001, "message": "unauthorized"}}).encode(), "application/json")
            body = self._read_body()
            code, out = mcp.handle(body)
            if code == 202:
                self.send_response(202); self.send_header("Content-Length", "0"); self.end_headers(); return
            return self._send(code, out, "application/json; charset=utf-8")

        def do_DELETE(self):
            u = urlparse(self.path)
            return self._send(200 if u.path == "/mcp" else 404, b"", "text/plain")

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Token, Mcp-Session-Id, Mcp-Protocol-Version, Authorization")
            self.end_headers()

        def do_GET(self):
            u = urlparse(self.path); q = parse_qs(u.query)
            if u.path == "/mcp":
                return self._send(405, b"MCP: use POST (stateless Streamable HTTP, no SSE stream)", "text/plain")
            if u.path == "/healthz":
                st = state.status(); fresh = st.get("updated_at") and time.time() - float(st["updated_at"]) < 300
                return self._send(200 if fresh else 503, (b"ok" if fresh else b"stale"), "text/plain")
            if not self._authed(q):
                return self._send(401, b"unauthorized: add ?token=...", "text/plain")
            if u.path == "/api/status":
                return self._send(200, json.dumps(state.status(), ensure_ascii=False, default=str).encode(), "application/json; charset=utf-8")
            if u.path == "/api/decisions":
                n = int(q.get("n", ["100"])[0]); kinds = q.get("kind", [None])[0]
                data = state.decisions(min(n, 1000), kinds.split(",") if kinds else None)
                return self._send(200, json.dumps(data, ensure_ascii=False, default=str).encode(), "application/json; charset=utf-8")
            if u.path in ("/", "/index.html"):
                html = HTML_PATH.read_text(encoding="utf-8").replace("__TOKEN__", token or "")
                return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            return self._send(404, b"not found", "text/plain")
    return H


def start_dashboard(state_dir: str, decision_dir: str, mode: str, port: int, token: str = "") -> threading.Thread:
    srv = ThreadingHTTPServer(("0.0.0.0", port), make_handler(DashboardState(state_dir, decision_dir, mode), token))
    t = threading.Thread(target=srv.serve_forever, name="dashboard", daemon=True)
    t.start()
    log.info("dashboard listening on :%d (mode=%s, token=%s)", port, mode, "set" if token else "none")
    return t
