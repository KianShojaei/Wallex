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
            reconcile_notes = reconcile_positions(s, client, pf, bal)
        except WallexError as e:
            log.error("cannot read live account: %s", e); raise
        broker = LiveBroker(client, fee, s.execution.order_type, s.execution.limit_offset_pct, s.execution.order_timeout_seconds,
                            style=s.execution.order_style, fee_fn=lambda sym: s.risk.fee_for(sym, s.execution.order_style))
    else:
        reconcile_notes = []
        broker = SimulatedBroker(fee, s.risk.slippage_pct, lambda sym: s.risk.fee_for(sym, s.execution.order_style))
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
    for note in reconcile_notes:
        dlog.write("reconcile", **note)
    md = MarketData(client, CandleStore(s.data.candle_cache_dir), s.data.resolution, s.data.fallback_resolution, s.data.lookback_bars)
    mcp_bridge = WallexMCP(api_key=s.wallex_api_key) if mcp else None
    return client, agent, md, state_path, mcp_bridge


def reconcile_positions(s, client: WallexClient, pf: Portfolio, balances: Dict) -> List[Dict]:
    """Make the portfolio match the EXCHANGE, which is the source of truth.

    If the state file was lost (redeploy without a persistent disk) or an order was filled outside a loop,
    coins held on the exchange would otherwise be invisible to the agent and missing from equity. For every
    held base asset of an allowed market we (re)build the position: quantity from the balance, entry price
    from the average of our recent BUY fills on that market (falls back to the current mid), stop/take-profit
    from the module config. Recorded positions whose coins are gone are dropped.
    """
    from wallex_agent.portfolio.state import Position
    quote = s.challenge.quote_asset
    notes: List[Dict] = []
    try:
        markets = client.markets()
    except WallexError as e:
        log.warning("reconcile: cannot load markets (%s)", e); return notes
    carry_cfg = (s.strategy.modules or {}).get("carry") or {}
    carry_syms = set((carry_cfg.get("carriers") or {carry_cfg.get("symbol", "XAUTTMN"): 1.0}).keys())
    now = time.time()
    seen = set()
    for asset, b in balances.items():
        if asset == quote:
            continue
        qty = float(b.get("value") or 0.0)
        sym = f"{asset}{quote}"
        m = markets.get(sym)
        if not m or qty <= 0:
            continue
        value = qty * (m.mid or m.last)
        if value < s.risk.min_order_notional_tmn * 0.5:
            continue                                     # dust (e.g. leftovers below the exchange minimum)
        seen.add(sym)
        if sym in pf.positions:
            p = pf.positions[sym]
            if abs(p.qty - qty) / max(qty, 1e-12) > 0.001:
                log.warning("reconcile %s: recorded qty %.8f != exchange %.8f -> using exchange", sym, p.qty, qty)
                p.qty = qty
            continue
        # unknown holding -> rebuild from our own fills
        entry = m.mid or m.last
        try:
            buys = [t for t in client.my_trades(sym) if str(t.get("side", t.get("isBuyer", ""))).upper() in ("BUY", "TRUE")]
            q_acc = 0.0; cost = 0.0
            for t in buys[:50]:
                tq = float(t.get("quantity") or t.get("qty") or 0); tp = float(t.get("price") or 0)
                if tq <= 0 or tp <= 0:
                    continue
                take = min(tq, qty - q_acc)
                q_acc += take; cost += take * tp
                if q_acc >= qty * 0.999:
                    break
            if q_acc > 0:
                entry = cost / q_acc
        except WallexError as e:
            log.warning("reconcile %s: cannot read trades (%s); using mid price as entry", sym, e)
        is_carry = sym in carry_syms
        sl_pct = float(carry_cfg.get("stop_loss_pct", s.risk.stop_loss_pct)) if is_carry else s.risk.stop_loss_pct
        tp_pct = float(carry_cfg.get("take_profit_pct", 0) or 0) if is_carry else (s.risk.take_profit_pct or 0)
        pf.positions[sym] = Position(symbol=sym, qty=qty, entry_price=entry, entry_time=now,
                                     stop_loss=entry * (1 - sl_pct / 100), take_profit=entry * (1 + tp_pct / 100) if tp_pct else 0.0,
                                     trailing_pct=0.0, highest=max(entry, m.mid or entry), entry_fee_tmn=0.0,
                                     entry_order_id="reconciled", decision_id="reconciled", module="carry" if is_carry else "",
                                     last_add_time=now)
        log.warning("reconcile: rebuilt position %s qty=%.8f entry=%.0f value=%.0f TMN (state was missing it)", sym, qty, entry, value)
        notes.append({"symbol": sym, "qty": qty, "entry_price": round(entry), "value_tmn": round(value),
                      "reason": "position rebuilt from exchange balance and our recent BUY fills (state file was missing it)"})
    for sym in list(pf.positions):
        if sym not in seen:
            log.warning("reconcile: %s recorded but no coins on the exchange -> dropping record", sym)
            notes.append({"symbol": sym, "reason": "recorded position dropped: no coins on the exchange"})
            del pf.positions[sym]
    return notes


def write_status(s, mode, agent, prices, markets, snaps, summ, res, t0, state_dir: Path) -> None:
    """Snapshot for the web dashboard (state/status_<mode>.json)."""
    import json
    regimes = {k: {**v.model_dump(), "age_min": round((t0 - agent.regime_ts.get(k, t0)) / 60)} for k, v in agent.regimes.items()}
    status = {
        "mode": mode, "updated_at": t0, "git_commit": _git_commit()[:10],
        "challenge": {"start": s.challenge.start.isoformat(), "end": s.challenge.end.isoformat(),
                      "liquidate_at": agent.risk.liq_ts, "end_ts": agent.risk.end_ts,
                      "initial_capital": s.challenge.initial_capital_tmn},
        "summary": summ,
        "prices": {k: round(v, 2) for k, v in prices.items()},
        "markets": {k: {"bid": m.bid, "ask": m.ask, "spread_pct": round(m.spread_pct, 4), "vol24_tmn": m.quote_volume_24h,
                        "change_24h_pct": m.change_24h_pct} for k, m in markets.items() if k in prices},
        "positions": [{"symbol": p.symbol, "module": p.module, "qty": p.qty, "entry_price": p.entry_price, "entry_time": p.entry_time,
                       "stop_loss": p.stop_loss, "take_profit": p.take_profit, "highest": p.highest, "adds": p.adds,
                       "price": prices.get(p.symbol), "upnl_pct": round(p.unrealized_pct(prices.get(p.symbol, p.entry_price)), 3),
                       "value_tmn": round(p.qty * prices.get(p.symbol, p.entry_price))} for p in agent.pf.positions.values()],
        "snapshots": {k: v.to_dict() for k, v in snaps.items()},
        "regimes": regimes,
        "last_step": res,
        "ai": {"enabled": agent.analyst is not None, "calls": agent.analyst.llm.calls if agent.analyst else 0,
               "cost_usd": round(agent.analyst.llm.total_cost_usd, 4) if agent.analyst else 0.0,
               "model_fast": s.ai_model_fast, "model_deep": s.ai_model_deep},
        "equity_curve": agent.pf.equity_curve[-2000:],
        "risk": {"killed": agent.pf.killed, "halted_today": agent.pf.halted_until_day == agent.pf.day_key,
                 "peak_equity": agent.pf.peak_equity, "day_start_equity": agent.pf.day_start_equity,
                 "drawdown_pct": round(agent.pf.drawdown_pct(summ["equity_tmn"]), 3),
                 "daily_drawdown_pct": round(agent.pf.daily_drawdown_pct(summ["equity_tmn"]), 3),
                 "limits": s.risk.model_dump()},
        "decision_log": str(agent.log.path),
    }
    state_dir.mkdir(parents=True, exist_ok=True)
    tmp = state_dir / f"status_{mode}.tmp"
    tmp.write_text(json.dumps(status, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(state_dir / f"status_{mode}.json")


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
        if s.markets.is_blocked(sym):
            log.error("market %s is disabled in the challenge account; skipping", sym); continue
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
            for sym in set(universe) | set(agent.pf.positions) | set(s.markets.observe):
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
            write_status(s, mode, agent, prices, markets, snaps, summ, res, t0, state_path.parent)
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
