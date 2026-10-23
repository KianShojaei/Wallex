import json, threading, time, urllib.request
from pathlib import Path

from wallex_agent.dashboard.server import DashboardState, make_handler
from http.server import ThreadingHTTPServer


def test_dashboard_endpoints(tmp_path: Path):
    st = tmp_path / "state"; st.mkdir(); dl = tmp_path / "dec"; dl.mkdir()
    (st / "status_shadow.json").write_text(json.dumps({"mode": "shadow", "updated_at": time.time(), "summary": {"equity_tmn": 5_000_000}}), encoding="utf-8")
    (dl / "shadow_x.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"id": "1", "ts": "t1", "kind": "scan", "symbol": "XAUTTMN", "score": 0.1, "vetoes": ["x"]},
        {"id": "2", "ts": "t2", "kind": "entry", "symbol": "XAUTTMN", "why": "ok", "risk_plan": {"notional_tmn": 1, "stop_loss_pct": 2.5}, "ai_decision": {"action": "approve", "confidence": 0.8, "rationale": "r"}, "signal": {"score": 0.7, "reasons": ["a"]}, "regime": {"regime": "trend_up", "confidence": 0.7, "allow_entries": True, "size_factor": 1, "summary": "s"}},
    ]), encoding="utf-8")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(DashboardState(str(st), str(dl), "shadow"), "secret"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert urllib.request.urlopen(base + "/healthz").status == 200
        try:
            urllib.request.urlopen(base + "/api/status"); assert False, "token required"
        except urllib.error.HTTPError as e:
            assert e.code == 401
        s = json.loads(urllib.request.urlopen(base + "/api/status?token=secret").read())
        assert s["summary"]["equity_tmn"] == 5_000_000
        d = json.loads(urllib.request.urlopen(base + "/api/decisions?token=secret&n=10&kind=entry").read())
        assert len(d) == 1 and d[0]["kind"] == "entry" and d[0]["ai"]["action"] == "approve" and "signal" in d[0]
        html = urllib.request.urlopen(base + "/?token=secret").read().decode()
        assert "ایجنت" in html and "__TOKEN__" not in html
    finally:
        srv.shutdown()


def test_mcp_endpoint(tmp_path: Path):
    st = tmp_path / "state"; st.mkdir(); dl = tmp_path / "dec"; dl.mkdir()
    (st / "status_live.json").write_text(json.dumps({"mode": "live", "updated_at": time.time(), "summary": {"equity_tmn": 5_050_000, "return_pct": 1.0},
                                                      "positions": [], "equity_curve": [[1, 5_000_000], [2, 5_050_000]], "risk": {"limits": {"x": 1}},
                                                      "challenge": {"liquidate_at": time.time() + 3600}}), encoding="utf-8")
    (dl / "live_a.jsonl").write_text(json.dumps({"id": "1", "ts": "t", "kind": "exit", "symbol": "XAUTTMN", "reason": "r", "pnl_tmn": 100, "held_minutes": 10}) + "\n", encoding="utf-8")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(DashboardState(str(st), str(dl), "live"), "tok"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}/mcp?token=tok"

    def rpc(payload):
        req = urllib.request.Request(base, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as r:
            return r.status, (json.loads(r.read()) if r.status == 200 else None)
    try:
        code, init = rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}})
        assert code == 200 and init["result"]["serverInfo"]["name"] == "wallex-agent"
        code, _ = rpc({"jsonrpc": "2.0", "method": "notifications/initialized"})
        assert code == 202
        _, tools = rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = {t["name"] for t in tools["result"]["tools"]}
        assert {"get_status", "get_decisions", "get_trades", "get_equity_curve"} <= names
        _, res = rpc({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "get_status", "arguments": {}}})
        body = json.loads(res["result"]["content"][0]["text"])
        assert body["agent_alive"] is True and body["summary"]["equity_tmn"] == 5_050_000
        _, res = rpc({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "get_trades", "arguments": {}}})
        assert json.loads(res["result"]["content"][0]["text"])["aggregate"]["trades"] == 1
        _, res = rpc({"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "get_equity_curve", "arguments": {"points": 10}}})
        assert json.loads(res["result"]["content"][0]["text"])["max"] == 5_050_000
        # unauthorized
        try:
            urllib.request.urlopen(urllib.request.Request(base.replace("tok", "bad"), data=b"{}", headers={"Content-Type": "application/json"}, method="POST"))
            assert False
        except urllib.error.HTTPError as e:
            assert e.code == 401
    finally:
        srv.shutdown()
