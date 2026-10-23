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
