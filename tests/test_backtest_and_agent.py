import json

from wallex_agent.backtest.runner import run_backtest
from wallex_agent.data.features import compute_indicators, snapshot_from
from wallex_agent.ai.analyst import Analyst, Decision, Regime
from wallex_agent.ai.llm import LLMClient
from wallex_agent.strategy.signals import Signal
from tests.conftest import synthetic_candles


MR_ONLY = {"mean_reversion": {"enabled": True, "exclude": ["USDTTMN"], "max_position_pct": 0.2, "stop_loss_pct": 1.0,
                              "take_profit_pct": 1.6, "trailing_pct": 0.8, "max_holding_minutes": 720}}


def test_backtest_end_to_end_rule_only(settings, tmp_path):
    settings.strategy.modules = MR_ONLY
    settings.markets.allowed = ["BTCTMN", "ETHTMN"]
    frames = {"BTCTMN": synthetic_candles(seed=3), "ETHTMN": synthetic_candles(seed=5, start_price=50_000.0)}
    settings.data.candle_cache_dir = str(tmp_path / "c")
    r = run_backtest(settings, list(frames), None, None, ai_mode="off", out_root=str(tmp_path / "bt"), frames=frames)
    m = r.metrics
    assert m["trades"] >= 1, m
    assert (r.out_dir / "metrics.json").exists() and (r.out_dir / "equity.csv").exists()
    # realized-in-TMN rule: no open positions at the end
    recs = [json.loads(l) for l in open(r.out_dir / "backtest_decisions.jsonl", encoding="utf-8")]
    entries = [x for x in recs if x["kind"] == "entry"]
    exits = [x for x in recs if x["kind"] == "exit"]
    assert len(entries) == len(exits)
    for e in entries:                       # every trade carries its reason & risk plan
        assert e["why"] and e["signal"]["reasons"] and e["risk_plan"]["stop_loss_pct"] > 0
        assert e["risk_plan"]["notional_tmn"] <= e["equity_before"] * 0.2 + 1
    assert abs(m["final_equity_tmn"] - (5_000_000 + sum(x["pnl_tmn"] for x in exits))) < 5  # accounting closes


def test_backtest_is_reproducible(settings, tmp_path):
    settings.strategy.modules = MR_ONLY
    settings.markets.allowed = ["BTCTMN"]
    frames = {"BTCTMN": synthetic_candles(seed=3)}
    a = run_backtest(settings, ["BTCTMN"], None, None, "off", out_root=str(tmp_path / "a"), frames=frames).metrics
    b = run_backtest(settings, ["BTCTMN"], None, None, "off", out_root=str(tmp_path / "b"), frames=frames).metrics
    assert a["final_equity_tmn"] == b["final_equity_tmn"] and a["trades"] == b["trades"]


def test_analyst_threshold_and_fallback(settings, tmp_path):
    llm = LLMClient("http://x", "", "f", "d", cache_dir=str(tmp_path), mode="off")
    an = Analyst(llm, 0.6)
    sig = Signal("BTCTMN", "BUY", 0.8, reasons=["RSI low"], vetoes=[], snapshot={})
    d = an.decide(sig, None, {}, {})
    assert d.action == "approve_reduced" and d.size_factor == 0.5 and "ai_unavailable" in d.key_risks
    weak = Signal("BTCTMN", "BUY", 0.6, reasons=["RSI low"], vetoes=[], snapshot={})
    assert an.decide(weak, None, {}, {}).action == "reject"
    rg = Regime(regime="trend_down", confidence=0.9, allow_entries=False, summary="down")
    assert an.decide(sig, rg, {}, {}).action == "reject"
    # replay mode with no cache -> falls back deterministically
    llm2 = LLMClient("http://x", "", "f", "d", cache_dir=str(tmp_path), mode="replay")
    assert Analyst(llm2, 0.6).decide(sig, None, {}, {}).action == "approve_reduced"


def test_carry_module_on_trending_usdt(settings, tmp_path):
    """Synthetic USDTTMN with a steady upward drift (regime symbol, read-only) and a gold carrier XAUTTMN that
    follows it: the carry core must go long the CARRIER, respect its 50% cap, hold no fixed take-profit, and end flat."""
    import numpy as np, pandas as pd
    n = 500
    idx = pd.date_range("2026-09-01", periods=n, freq="60min", tz="UTC")
    rng = np.random.default_rng(0)
    def frame(start, drift, vol, seed):
        r = np.random.default_rng(seed)
        c = start * np.cumprod(1 + r.normal(drift, vol, n))
        return pd.DataFrame({"open": np.r_[c[0], c[:-1]], "high": c * 1.001, "low": c * 0.999, "close": c, "volume": 1.0}, index=idx)
    frames = {"USDTTMN": frame(100000, 0.0004, 0.002, 0), "XAUTTMN": frame(1_000_000_000, 0.0004, 0.004, 1)}
    r = run_backtest(settings, ["XAUTTMN"], None, None, "off", out_root=str(tmp_path / "bt"), frames=frames)
    recs = [json.loads(l) for l in open(r.out_dir / "backtest_decisions.jsonl", encoding="utf-8")]
    entries = [x for x in recs if x["kind"] == "entry"]
    assert entries and all(e["module"] == "carry" and e["symbol"] == "XAUTTMN" for e in entries)
    for e in entries:
        assert e["risk_plan"]["notional_tmn"] <= e["equity_before"] * settings.risk.max_total_exposure_pct + 1
        assert e["risk_plan"]["take_profit_pct"] == 0 and e["risk_plan"]["stop_loss_pct"] == 2.5
    assert len([x for x in recs if x["kind"] == "exit"]) == len(entries)
    # behaviour, not luck: the rule floor must put at least 40% of equity to work on a strong regime
    assert max(e["risk_plan"]["notional_tmn"] / e["equity_before"] for e in entries) >= 0.39
    assert r.metrics["trades"] == len(entries) and r.metrics["max_drawdown_pct"] < 6.0


def test_carry_basket_and_rebalance(settings, tmp_path):
    """Three carriers share the module cap by weight; an oversized position gets trimmed to its basket cap."""
    import numpy as np, pandas as pd
    from wallex_agent.strategy.carry import CarryStrategy
    from wallex_agent.strategy.base import Context
    from wallex_agent.data.features import compute_indicators, snapshot_from
    settings.strategy.modules["carry"]["dynamic_weights"] = False       # static weights for this test
    cs = CarryStrategy(settings.strategy.modules["carry"])
    assert abs(cs.position_cap_pct("XAUTTMN") - 0.40) < 1e-9 and abs(cs.position_cap_pct("BTCTMN") - 0.24) < 1e-9
    n = 300; idx = pd.date_range("2026-09-01", periods=n, freq="60min", tz="UTC")
    def frame(start, drift, vol, seed):
        r = np.random.default_rng(seed); c = start * np.cumprod(1 + r.normal(drift, vol, n))
        return compute_indicators(pd.DataFrame({"open": np.r_[c[0], c[:-1]], "high": c * 1.001, "low": c * 0.999, "close": c, "volume": 1.0}, index=idx), settings.strategy)
    frames = {"USDTTMN": frame(100000, 0.0006, 0.001, 0), "XAUTTMN": frame(1e9, 0.0005, 0.003, 1), "BTCTMN": frame(2e10, 0.0005, 0.004, 2), "ETHTMN": frame(6e8, 0.0005, 0.004, 3)}
    snaps = {k: snapshot_from(v, k, spread_pct=0.1) for k, v in frames.items() if k != "USDTTMN"}
    ctx = Context(now=idx[-1].timestamp(), snapshots=snaps, frames=frames, prices={k: float(v.close.iloc[-1]) for k, v in frames.items()})
    sigs = cs.candidates(ctx)
    assert {s.symbol for s in sigs} == {"XAUTTMN", "BTCTMN", "ETHTMN"}
    caps = {s.symbol: s.max_position_pct for s in sigs}
    assert abs(caps["XAUTTMN"] - 0.4) < 1e-9 and abs(caps["ETHTMN"] - 0.16) < 1e-9
    # rebalance: run one engine step with an oversized gold position -> partial sell down to the cap
    from wallex_agent.engine.core import Agent
    from wallex_agent.execution.broker import SimulatedBroker
    from wallex_agent.decision_log.logger import DecisionLog
    from wallex_agent.portfolio.state import Portfolio, Position
    from wallex_agent.risk.manager import RiskManager
    from wallex_agent.backtest.runner import _fake_market
    from datetime import datetime, timedelta, timezone
    px = ctx.prices["XAUTTMN"]
    pf = Portfolio(cash_tmn=1_350_000, initial_capital=5_000_000)
    pf.positions["XAUTTMN"] = Position("XAUTTMN", 3_650_000 / px, px, ctx.now - 3600, px * 0.975, 0.0, 0.0, px, module="carry", last_add_time=ctx.now)
    risk = RiskManager(settings.risk, datetime.now(timezone.utc) + timedelta(days=2), 45)
    agent = Agent(settings, pf, risk, None, SimulatedBroker(0.0, 0.0), DecisionLog(str(tmp_path), mode="test"))
    agent.step(ctx.now, snaps, ctx.prices, {k: _fake_market(k, v) for k, v in ctx.prices.items()}, frames=frames)
    eq = pf.equity(ctx.prices)
    gold_share = pf.positions["XAUTTMN"].qty * px / eq
    assert 0.38 <= gold_share <= 0.41, gold_share            # trimmed from 73% to the 40% basket cap
    recs = [json.loads(l) for l in open(agent.log.path, encoding="utf-8")]
    assert any(r["kind"] == "rebalance" for r in recs)


def test_dynamic_weights_tilt_toward_strong_carrier(settings):
    import numpy as np, pandas as pd
    from wallex_agent.strategy.carry import CarryStrategy
    from wallex_agent.strategy.base import Context
    cs = CarryStrategy(settings.strategy.modules["carry"])
    n = 200; idx = pd.date_range("2026-09-01", periods=n, freq="60min", tz="UTC")
    def frame(start, drift):
        c = start * np.cumprod(np.full(n, 1 + drift))
        return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=idx)
    frames = {"USDTTMN": frame(1e5, 0.0), "XAUTTMN": frame(1e9, 0.0), "BTCTMN": frame(2e10, 0.0006), "ETHTMN": frame(6e8, -0.0006)}
    ctx = Context(now=idx[-1].timestamp(), snapshots={}, frames=frames, prices={})
    cs.update_weights(ctx)
    w = cs.weights
    assert abs(sum(w.values()) - 1) < 1e-9
    assert w["BTCTMN"] > cs.carriers["BTCTMN"] and w["ETHTMN"] < cs.carriers["ETHTMN"]      # strong up, weak down
    assert w["BTCTMN"] <= cs.carriers["BTCTMN"] * 1.5 / sum(w.values()) + 1e-9
    assert all(v > 0 for v in w.values())
