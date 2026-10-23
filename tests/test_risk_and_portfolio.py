import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from wallex_agent.portfolio.state import Portfolio, Position
from wallex_agent.risk.manager import RiskManager


def _risk(settings, end_in_hours=48):
    return RiskManager(settings.risk, datetime.now(timezone.utc) + timedelta(hours=end_in_hours), settings.challenge.liquidate_before_end_minutes)


def test_position_sizing_respects_caps(settings):
    pf = Portfolio(cash_tmn=5_000_000, initial_capital=5_000_000)
    r = _risk(settings)
    eq = pf.mark({})
    plan, notes = r.plan_entry("BTCTMN", pf, eq, {}, ai_size_factor=1.0, ai_stop=None, ai_tp=None, atr_pct=0.5)
    assert plan is not None
    assert plan.notional_tmn <= 5_000_000 * settings.risk.max_position_pct
    # AI can only tighten the stop, never loosen
    plan2, _ = r.plan_entry("BTCTMN", pf, eq, {}, 1.0, ai_stop=2.5, ai_tp=None, atr_pct=0.5)
    assert plan2.stop_loss_pct == settings.risk.stop_loss_pct
    plan3, _ = r.plan_entry("BTCTMN", pf, eq, {}, 1.0, ai_stop=0.5, ai_tp=None, atr_pct=0.5)
    assert plan3.stop_loss_pct == 0.5
    # high volatility shrinks size
    plan4, _ = r.plan_entry("BTCTMN", pf, eq, {}, 1.0, None, None, atr_pct=2.0)
    assert r.plan_entry("BTCTMN", pf, eq, {}, 1.0, None, None, atr_pct=4.0)[0] is None  # too small -> no dust order
    assert plan4.notional_tmn < plan.notional_tmn


def test_exposure_cap_and_max_positions(settings):
    pf = Portfolio(cash_tmn=5_000_000, initial_capital=5_000_000)
    r = _risk(settings)
    now = time.time()
    syms = [f"S{i}TMN" for i in range(settings.risk.max_open_positions)]
    for sym in syms:
        pf.positions[sym] = Position(sym, 1, 1_000_000, now, 990_000, 1_016_000, 0.8, 1_000_000)
    pf.cash_tmn = 5_000_000 - 1_000_000 * len(syms)
    prices = {sym: 1_000_000 for sym in syms}
    eq = pf.mark(prices)
    ok, blocks = r.global_checks(pf, eq)
    assert not ok and any("max open positions" in b for b in blocks)
    for sym in syms[1:]:
        del pf.positions[sym]
    pf.cash_tmn = 4_000_000
    plan, notes = r.plan_entry("SOLTMN", pf, 5_000_000, prices, 1.0, None, None, 0.5)
    assert plan.notional_tmn <= 5_000_000 * settings.risk.max_total_exposure_pct - 1_000_000


def test_exits_sl_tp_trailing_time_liquidation(settings):
    r = _risk(settings)
    now = time.time()
    pf = Portfolio(cash_tmn=0, initial_capital=5_000_000)
    pf.positions["X"] = Position("X", 1, 100.0, now, 99.0, 101.6, 0.8, 100.0, max_holding_minutes=720)
    assert r.check_exits(pf, {"X": 98.9}, {}, now)[0].reason.startswith("stop-loss")
    assert r.check_exits(pf, {"X": 101.7}, {}, now)[0].reason.startswith("take-profit")
    pf.positions["X"].highest = 101.0
    assert r.check_exits(pf, {"X": 100.15}, {}, now)[0].reason.startswith("trailing")
    pf.positions["X"].highest = 100.0
    assert r.check_exits(pf, {"X": 100.3}, {}, now + 720 * 60 + 1)[0].reason.startswith("time stop")
    assert r.check_exits(pf, {"X": 100.7}, {"X": "mean reverted"}, now)[0].reason == "mean reverted"
    assert not r.check_exits(pf, {"X": 100.3}, {}, now)
    r2 = RiskManager(settings.risk, datetime.now(timezone.utc) + timedelta(minutes=10), 45)
    assert "liquidation" in r2.check_exits(pf, {"X": 100.3}, {}, now)[0].reason
    assert not r2.global_checks(pf, 5_000_000, now)[0]


def test_drawdown_halts(settings):
    r = _risk(settings)
    pf = Portfolio(cash_tmn=5_000_000, initial_capital=5_000_000)
    pf.mark({})
    pf.cash_tmn = 5_000_000 * (1 - settings.risk.daily_drawdown_halt_pct / 100 - 0.001)
    eq = pf.mark({})
    ok, blocks = r.global_checks(pf, eq)
    assert not ok and any("daily drawdown" in b for b in blocks)
    pf.cash_tmn = 5_000_000 * (1 - settings.risk.total_drawdown_kill_pct / 100 - 0.001)
    assert r.should_kill(pf, pf.mark({}))


def test_portfolio_persistence(tmp_path: Path):
    pf = Portfolio(cash_tmn=1, initial_capital=5)
    pf.positions["A"] = Position("A", 2, 3, 4, 5, 6, 7, 8)
    pf.save(tmp_path / "p.json")
    pf2 = Portfolio.load(tmp_path / "p.json")
    assert pf2.positions["A"].qty == 2 and pf2.cash_tmn == 1


def test_blocked_markets_loaded_and_enforced(settings):
    assert settings.markets.blocked, "organiser block list should load from config/blocked_markets.yaml"
    assert settings.markets.is_blocked("APTTMN") and settings.markets.is_blocked("APT/TMN")
    assert not settings.markets.is_blocked("USDTTMN") and not settings.markets.is_blocked("BTCUSDT")
    for sym in settings.markets.allowed:
        assert not settings.markets.is_blocked(sym)


def test_data_dir_relocates_writable_paths(tmp_path, monkeypatch):
    from wallex_agent.config import load_settings
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "disk"))
    st = load_settings("config/settings.yaml", env_file="/nonexistent.env")
    assert st.execution.state_dir.startswith(str(tmp_path / "disk"))
    assert (tmp_path / "disk" / "logs" / "decisions").is_dir() and (tmp_path / "disk" / "candles").is_dir()


def test_reconcile_rebuilds_missing_position(settings):
    """State lost on redeploy: coins on the exchange must become positions again (exchange is the source of truth)."""
    from wallex_agent.engine.loop import reconcile_positions
    from wallex_agent.wallex.client import MarketInfo

    class FakeClient:
        def markets(self):
            return {"XAUTTMN": MarketInfo("XAUTTMN", "XAUT", "TMN", 0, 5, 1e-5, 50000, 1_100_000_000, 1_101_000_000, 1_100_500_000, 1e12, 0.0, {})}
        def my_trades(self, sym):
            return [{"side": "BUY", "quantity": "0.00113", "price": "1095000000"}, {"side": "SELL", "quantity": "0.5", "price": "1"}]

    pf = Portfolio(cash_tmn=1_344_443, initial_capital=5_000_000)
    reconcile_positions(settings, FakeClient(), pf, {"TMN": {"value": 1_344_443}, "XAUT": {"value": 0.00113}, "USDC": {"value": 0.00673}})
    assert "XAUTTMN" in pf.positions and "USDCTMN" not in pf.positions          # dust ignored
    p = pf.positions["XAUTTMN"]
    assert p.qty == 0.00113 and abs(p.entry_price - 1_095_000_000) < 1 and p.module == "carry"
    assert abs(p.stop_loss - 1_095_000_000 * 0.975) < 1
    assert abs(pf.equity({"XAUTTMN": 1_100_500_000}) - (1_344_443 + 0.00113 * 1_100_500_000)) < 1
    # recorded position with no coins is dropped; mismatched qty is corrected
    pf.positions["BTCTMN"] = Position("BTCTMN", 1, 1, 0, 0, 0, 0, 1)
    pf.positions["XAUTTMN"].qty = 0.002
    reconcile_positions(settings, FakeClient(), pf, {"XAUT": {"value": 0.00113}})
    assert "BTCTMN" not in pf.positions and pf.positions["XAUTTMN"].qty == 0.00113
