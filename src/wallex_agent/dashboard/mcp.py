"""Minimal MCP server (Streamable HTTP, stateless JSON responses) exposing the agent's state to an LLM client.

Protocol: JSON-RPC 2.0 over POST /mcp (spec 2025-03-26). No SSE stream is needed for request/response tools,
so GET /mcp returns 405 as the spec allows. Auth: the same DASHBOARD_TOKEN (query ?token= or header X-Token).

Connect from Claude Code:
    claude mcp add --transport http wallex-agent "https://<app>.liara.run/mcp?token=<TOKEN>"
Claude Desktop / claude.ai custom connector: URL  https://<app>.liara.run/mcp?token=<TOKEN>

All tools are READ-ONLY by design: the model can inspect and explain, never trade.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

PROTOCOL = "2025-03-26"


def _tool(name, desc, props=None, required=None):
    return {"name": name, "description": desc,
            "inputSchema": {"type": "object", "properties": props or {}, "required": required or []}}


TOOLS = [
    _tool("get_status", "Live summary of the Wallex trading agent: equity, return %, cash/exposure, drawdowns, risk flags, "
                        "AI usage, challenge window and time until automatic liquidation to TMN."),
    _tool("get_positions", "Open positions with entry, current price, unrealised PnL %, stop-loss and age."),
    _tool("get_markets", "Latest prices, spreads, 24h change and indicator snapshot (RSI, ATR%, Bollinger position) for every observed market."),
    _tool("get_regimes", "Latest AI regime classification per (module, symbol): regime, confidence, allow_entries, size_factor, summary."),
    _tool("get_decisions", "Most recent decision-log records (entries, exits, rejects, regime calls, scans). Each record explains WHY.",
          {"n": {"type": "integer", "minimum": 1, "maximum": 500, "default": 50},
           "kind": {"type": "string", "description": "comma-separated kinds, e.g. 'entry,exit' or 'reject'"}}),
    _tool("get_trades", "Closed trades (exit records) with PnL, holding time and exit reason, plus aggregate stats.",
          {"n": {"type": "integer", "default": 100}}),
    _tool("get_risk_limits", "The risk configuration the agent enforces (caps, stops, drawdown halts, fees)."),
    _tool("get_equity_curve", "Equity curve statistics and a downsampled series [[unix_ts, equity_tmn], ...].",
          {"points": {"type": "integer", "default": 120}}),
    _tool("get_strategy_description", "Human-readable description of the strategy modules and the research behind them (from README/docs)."),
]


class MCPHandler:
    def __init__(self, state, repo_root: Path | None = None):
        self.state = state          # DashboardState
        self.root = repo_root or Path(__file__).resolve().parents[3]

    # ------------------------------------------------------------ JSON-RPC
    def handle(self, body: bytes) -> tuple[int, bytes]:
        try:
            msg = json.loads(body or b"{}")
        except ValueError:
            return 400, json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}).encode()
        if isinstance(msg, list):                                   # batch
            out = [r for r in (self._dispatch(m) for m in msg) if r is not None]
            return (200, json.dumps(out).encode()) if out else (202, b"")
        res = self._dispatch(msg)
        return (200, json.dumps(res, ensure_ascii=False, default=str).encode()) if res is not None else (202, b"")

    def _dispatch(self, m: Dict[str, Any]):
        method, mid, params = m.get("method", ""), m.get("id"), m.get("params") or {}
        if method.startswith("notifications/"):
            return None
        try:
            if method == "initialize":
                req = str(params.get("protocolVersion") or PROTOCOL)
                result = {"protocolVersion": req if req in ("2024-11-05", "2025-03-26", "2025-06-18") else PROTOCOL,
                          "capabilities": {"tools": {"listChanged": False}},
                          "serverInfo": {"name": "wallex-agent", "version": "0.1.0"},
                          "instructions": "Read-only view of an autonomous Wallex trading agent (StarCoach hackathon). "
                                          "Use get_status first; get_decisions explains every trade and every rejection."}
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                result = self._call(params.get("name", ""), params.get("arguments") or {})
            else:
                return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
            return {"jsonrpc": "2.0", "id": mid, "result": result}
        except Exception as e:  # noqa: BLE001
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32000, "message": f"{type(e).__name__}: {e}"}}

    # ------------------------------------------------------------ tools
    def _text(self, obj) -> Dict[str, Any]:
        return {"content": [{"type": "text", "text": json.dumps(obj, ensure_ascii=False, indent=1, default=str)}], "isError": False}

    def _call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        st = self.state.status()
        if name == "get_status":
            c = st.get("challenge") or {}
            now = time.time()
            out = {k: st.get(k) for k in ("mode", "updated_at", "git_commit", "summary", "risk", "ai", "last_step")}
            out["agent_alive"] = bool(st.get("updated_at")) and now - float(st["updated_at"]) < 300
            out["seconds_since_last_loop"] = round(now - float(st["updated_at"])) if st.get("updated_at") else None
            out["challenge"] = {**c, "minutes_until_liquidation": round((float(c["liquidate_at"]) - now) / 60) if c.get("liquidate_at") else None}
            if out.get("risk"):
                out["risk"].pop("limits", None)
            return self._text(out)
        if name == "get_positions":
            return self._text({"positions": st.get("positions", []), "count": len(st.get("positions", []))})
        if name == "get_markets":
            return self._text({"prices": st.get("prices"), "markets": st.get("markets"), "snapshots": st.get("snapshots")})
        if name == "get_regimes":
            return self._text(st.get("regimes", {}))
        if name == "get_decisions":
            kinds = args.get("kind"); n = int(args.get("n", 50))
            return self._text(self.state.decisions(min(n, 500), kinds.split(",") if kinds else None))
        if name == "get_trades":
            ex = self.state.decisions(int(args.get("n", 100)), ["exit"])
            pnls = [e.get("pnl_tmn", 0) for e in ex]
            agg = {"trades": len(ex), "net_pnl_tmn": round(sum(pnls)), "wins": sum(1 for p in pnls if p > 0),
                   "avg_holding_minutes": round(sum(e.get("held_minutes", 0) for e in ex) / len(ex)) if ex else 0}
            return self._text({"aggregate": agg, "trades": ex})
        if name == "get_risk_limits":
            return self._text((st.get("risk") or {}).get("limits") or {})
        if name == "get_equity_curve":
            curve: List[List[float]] = st.get("equity_curve") or []
            pts = max(2, int(args.get("points", 120)))
            step = max(1, len(curve) // pts)
            ds = curve[::step] + ([curve[-1]] if curve and curve[-1] not in curve[::step] else [])
            ys = [p[1] for p in curve] or [0]
            peak = ys[0]; mdd = 0.0
            for y in ys:
                peak = max(peak, y); mdd = max(mdd, (peak - y) / peak * 100 if peak else 0)
            return self._text({"points": len(curve), "first": curve[0] if curve else None, "last": curve[-1] if curve else None,
                               "min": min(ys), "max": max(ys), "max_drawdown_pct": round(mdd, 3), "series": ds})
        if name == "get_strategy_description":
            txt = ""
            for rel in ("README.md", "docs/research.md"):
                p = self.root / rel
                if p.exists():
                    txt += f"\n\n===== {rel} =====\n" + p.read_text(encoding="utf-8")[:12000]
            return {"content": [{"type": "text", "text": txt.strip() or "documentation not bundled"}], "isError": False}
        return {"content": [{"type": "text", "text": f"unknown tool {name}"}], "isError": True}
