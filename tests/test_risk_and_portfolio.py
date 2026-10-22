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
    pf.positions["BTCTMN"] = Position("BTCTMN", 1, 1_000_000, now, 990_000, 1_016_000, 0.8, 1_000_000)
    pf.positions["ETHTMN"] = Position("ETHTMN", 1, 1_000_000, now, 990_000, 1_016_000, 0.8, 1_000_000)
    pf.cash_tmn = 3_000_000
    pf.cash_tmn = 3_000_000
    prices = {"BTCTMN": 1_000_000, "ETHTMN": 1_000_000}
    eq = pf.mark(prices)
    ok, blocks = r.global_checks(pf, eq)
    assert not ok and any("max open positions" in b for b in blocks)
    del pf.positions["ETHTMN"]
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
