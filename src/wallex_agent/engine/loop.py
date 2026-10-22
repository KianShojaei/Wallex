"""Live / shadow runner: wires data -> features -> Agent.step on a fixed interval."""
from __future__ import annotations

import logging
import signal as os_signal
import time
from pathlib import Path
from typing import Dict, List, Optional

from wallex_agent.ai.analyst import Analyst
from wallex_agent.ai.llm import LLMClient, LLMUnavailable
from wallex_agent.data.features import Snapshot, compute_indicators, orderbook_imbalance, snapshot_from
from wallex_agent.data.market_data import CandleStore, MarketData
from wallex_agent.decision_log.logger import DecisionLog
from wallex_agent.engine.core import Agent
from wallex_agent.execution.broker import LiveBroker, SimulatedBroker
from wallex_agent.portfolio.state import Portfolio
from wallex_agent.risk.manager import RiskManager
from wallex_agent.wallex.client import WallexClient, WallexError
from wallex_agent.wallex.mcp_tools import WallexMCP

log = logging.getLogger(__name__)


def build_agent(s, mode: str, ai_mode: str = "live", mcp: bool = False):
    client = WallexClient(s.wallex_base_url, s.wallex_api_key)
    state_path = Path(s.execution.state_dir) / f"portfolio_{mode}.json"
    if state_path.exists():
        pf = Portfolio.load(state_path)
        log.info("resumed portfolio state from %s (equity curve points: %d)", state_path, len(pf.equity_curve))
    else:
        pf = Portfolio(cash_tmn=s.challenge.initial_capital_tmn, initial_capital=s.challenge.initial_capital_tmn)
    fee = s.risk.maker_fee_pct if s.execution.order_style == "maker" else s.risk.fee_pct
    if mode == "live":
        try:
            bal = client.balances()
            pf.cash_tmn = bal.get(s.challenge.quote_asset, {}).get("free", pf.cash_tmn)
            fees = client.fees()
            tmn = {k: v for k, v in fees.items() if k.endswith(s.challenge.quote_asset) and v["taker"] > 0}
            if tmn:
                key = "maker" if s.execution.order_style == "maker" else "taker"
                fee = max(v[key] for v in tmn.values()) or fee
            log.info("live account: %s=%.0f, taker fee=%.3f%%", s.challenge.quote_asset, pf.cash_tmn, fee)
        except WallexError as e:
            log.error("cannot read live account: %s", e); raise
        broker = LiveBroker(client, fee, s.execution.order_type, s.execution.limit_offset_pct, s.execution.order_timeout_seconds,
                            style=s.execution.order_style)
    else:
        broker = SimulatedBroker(fee, s.risk.slippage_pct)
    analyst = None
    if s.strategy.ai_enabled and ai_mode != "off":
        try:
            llm = LLMClient(s.ai_base_url, s.ai_api_key, s.ai_model_fast, s.ai_model_deep, s.execution.ai_cache_dir, mode=ai_mode)
            analyst = Analyst(llm, s.strategy.ai_min_confidence)
        except LLMUnavailable as e:
            log.warning("AI disabled: %s", e)
    risk = RiskManager(s.risk, s.challenge_end_utc, s.challenge.liquidate_before_end_minutes)
    dlog = DecisionLog(s.execution.decision_log_dir, mode=mode)
    agent = Agent(s, pf, risk, analyst, broker, dlog)
    md = MarketData(client, CandleStore(s.data.candle_cache_dir), s.data.resolution, s.data.fallback_resolution, s.data.lookback_bars)
    mcp_bridge = WallexMCP(api_key=s.wallex_api_key) if mcp else None
    return client, agent, md, state_path, mcp_bridge


def _git_commit() -> str:
    """Record which code version produced the trades (judging: delivered code == running code)."""
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def select_universe(markets: Dict, s) -> List[str]:
    out = []
    for sym in s.markets.allowed:
        m = markets.get(sym)
        if not m:
            log.warning("allowed market %s not found on exchange", sym); continue
        if m.quote != s.challenge.quote_asset and m.quote:
            log.warning("skip %s: quote %s != %s", sym, m.quote, s.challenge.quote_asset); continue
        if m.quote_volume_24h and m.quote_volume_24h < s.markets.min_24h_quote_volume_tmn:
            continue
        out.append(sym)
    return out


def run(s, mode: str, ai_mode: str = "live", once: bool = False, mcp: bool = False) -> None:
    client, agent, md, state_path, mcp_bridge = build_agent(s, mode, ai_mode, mcp)
    stop = {"flag": False}

    def _sig(*_):
        stop["flag"] = True
        log.info("stop requested; finishing current loop")
    os_signal.signal(os_signal.SIGINT, _sig)
    os_signal.signal(os_signal.SIGTERM, _sig)

    agent.log.write("run_start", mode=mode, ai_mode=ai_mode, git_commit=_git_commit(),
                    config=s.model_dump(exclude={"wallex_api_key", "ai_api_key"}))
    mcp_tools = mcp_bridge.list_tools_sync() if mcp_bridge else []
    if mcp_tools:
        log.info("Wallex MCP tools: %s", [t["name"] for t in mcp_tools])

    while not stop["flag"]:
        t0 = time.time()
        try:
            markets = client.markets()
            universe = select_universe(markets, s)
            snaps: Dict[str, Snapshot] = {}
            prices: Dict[str, float] = {}
            recent: Dict[str, List[Dict]] = {}
            frames: Dict = {}
            for sym in set(universe) | set(agent.pf.positions):
                m = markets.get(sym)
                if not m:
                    continue
                df = md.fetch(sym)
                if df.empty:
                    continue
                df = compute_indicators(df, s.strategy)
                frames[sym] = df
                prices[sym] = m.mid if m.mid > 0 else float(df["close"].iloc[-1])
                imb = None
                try:
                    imb = orderbook_imbalance(client.depth(sym))
                except WallexError:
                    pass
                snap = snapshot_from(df, sym, spread_pct=m.spread_pct, change_24h_pct=m.change_24h_pct,
                                     quote_volume_24h=m.quote_volume_24h, orderbook_imbalance=imb)
                if snap:
                    snap.close = prices[sym]          # use live mid, indicators from closed candles
                    snaps[sym] = snap
                    recent[sym] = [{"t": str(i)[:16], "o": float(r.open), "h": float(r.high), "l": float(r.low), "c": float(r.close), "v": float(r.volume)}
                                   for i, r in df.tail(24).iterrows()]
            res = agent.step(t0, snaps, prices, markets, recent, frames=frames)
            agent.pf.save(state_path)
            summ = agent.pf.summary(prices)
            log.info("loop ok | equity=%s TMN (%.2f%%) | pos=%d | entries=%s exits=%s | ai_calls=%d cost=$%.4f",
                     f"{summ['equity_tmn']:,}", summ["return_pct"], len(agent.pf.positions), res["entries"], res["exits"],
                     agent.analyst.llm.calls if agent.analyst else 0, agent.analyst.llm.total_cost_usd if agent.analyst else 0)
            if agent.pf.killed and not agent.pf.positions:
                log.error("kill switch engaged and flat: stopping agent"); break
            if agent.risk.in_liquidation_window(t0) and not agent.pf.positions and t0 >= agent.risk.end_ts:
                log.info("challenge ended and flat: stopping"); break
        except WallexError as e:
            log.error("exchange error: %s", e)
            agent.log.write("error", where="loop", error=str(e))
        except Exception as e:  # noqa: BLE001  never die silently during the challenge
            log.exception("unexpected error in loop: %s", e)
            agent.log.write("error", where="loop", error=repr(e))
        if once:
            break
        time.sleep(max(1.0, s.execution.loop_interval_seconds - (time.time() - t0)))
    agent.log.write("run_stop", mode=mode)
