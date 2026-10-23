"""The decision core shared by live, shadow and backtest runs.

One `step()` = one evaluation of all markets at one point in time:
  1. mark portfolio, run global risk gates (kill switch, drawdown halts, liquidation window)
  2. exits: risk engine checks SL/TP/trailing/time/rule-exit for every open position -> SELL
  3. entries: rule layer scores each market -> candidates -> AI analyst approves/rejects/resizes
     -> risk engine sizes & clamps -> broker executes -> position recorded
  4. every decision (including "do nothing" with its reasons) goes to the JSONL decision log
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from wallex_agent.ai.analyst import Analyst, Regime
from wallex_agent.data.features import Snapshot
from wallex_agent.decision_log.logger import DecisionLog
from wallex_agent.execution.broker import Broker
from wallex_agent.portfolio.state import Portfolio
from wallex_agent.risk.manager import RiskManager
from wallex_agent.strategy.base import Context
from wallex_agent.strategy.registry import build_modules
from wallex_agent.wallex.client import MarketInfo

log = logging.getLogger(__name__)


class Agent:
    def __init__(self, settings, portfolio: Portfolio, risk: RiskManager, analyst: Optional[Analyst],
                 broker: Broker, dlog: DecisionLog):
        self.s = settings
        self.pf = portfolio
        self.risk = risk
        self.analyst = analyst
        self.broker = broker
        self.log = dlog
        self.regimes: Dict[str, Regime] = {}
        self.regime_ts: Dict[str, float] = {}
        self.modules = build_modules(settings.strategy.modules)

    # ----------------------------------------------------------------- helpers
    def _risk_defaults(self) -> Dict:
        r = self.s.risk
        return {"max_position_pct": r.max_position_pct, "stop_loss_pct": r.stop_loss_pct,
                "take_profit_pct": r.take_profit_pct, "fee_pct_per_side": r.fee_pct}

    def _refresh_regime(self, sym: str, snap: Snapshot, recent: List[Dict], now: float, mcp_ctx: Optional[Dict],
                        module: str = "carry", extra: Optional[Dict] = None) -> Optional[Regime]:
        if not self.analyst or not self.s.strategy.ai_enabled:
            return None
        key = f"{module}:{sym}"
        if key in self.regimes and now - self.regime_ts.get(key, 0) < self.s.strategy.regime_refresh_minutes * 60:
            return self.regimes[key]
        snapd = {**snap.to_dict(), **(extra or {})}
        rg = self.analyst.regime(sym, snapd, recent, mcp_ctx, module=module)
        if rg is not None:
            self.regimes[key] = rg
            self.regime_ts[key] = now
            self.log.write("regime", now=now, module=module, symbol=sym, regime=rg.model_dump(), snapshot=snapd)
        return rg

    # ----------------------------------------------------------------- step
    def step(self, now: float, snapshots: Dict[str, Snapshot], prices: Dict[str, float], markets: Dict[str, MarketInfo],
             recent_bars: Optional[Dict[str, List[Dict]]] = None, mcp_ctx: Optional[Dict[str, Dict]] = None,
             frames: Optional[Dict] = None) -> Dict:
        recent_bars = recent_bars or {}
        mcp_ctx = mcp_ctx or {}
        ctx = Context(now=now, snapshots=snapshots, frames=frames or {}, prices=prices, quote_asset=self.s.challenge.quote_asset)
        equity = self.pf.mark(prices, now)
        summary = {"equity": equity, "exits": [], "entries": [], "skipped": []}

        # kill switch
        if not self.pf.killed and self.risk.should_kill(self.pf, equity):
            self.pf.killed = True
            self.log.write("kill_switch", now=now, equity=equity, drawdown_pct=self.pf.drawdown_pct(equity))

        # 2) exits -----------------------------------------------------------
        hints: Dict[str, str] = {}
        for sym, p in self.pf.positions.items():
            for m in self.modules:
                if p.module and m.name != p.module:
                    continue
                h = m.exit_hint(sym, ctx)
                if h:
                    hints[sym] = h; break
        for ex in self.risk.check_exits(self.pf, prices, hints, now):
            self._close(ex.symbol, ex.reason, prices, markets, now)
            summary["exits"].append(ex.symbol)

        # 3) entries ---------------------------------------------------------
        ok, blocks = self.risk.global_checks(self.pf, equity, now)
        if not ok:
            self.log.write("no_entry", now=now, reasons=blocks, equity=equity)
            summary["skipped"] = blocks
            return summary

        candidates = []
        for m in self.modules:
            for sig in m.candidates(ctx):
                if sig.symbol not in markets:
                    continue
                if sig.symbol in self.pf.positions and not sig.allow_top_up:
                    continue
                if sig.is_candidate and sig.score >= self.s.strategy.min_signal_score:
                    candidates.append(sig)
                else:
                    self.log.write("scan", now=now, module=m.name, symbol=sig.symbol, score=sig.score, vetoes=sig.vetoes,
                                   reasons=sig.reasons, close=prices.get(sig.symbol))
        candidates.sort(key=lambda x: x.score, reverse=True)

        for sig in candidates:
            ok, blocks = self.risk.global_checks(self.pf, equity, now)
            if not ok and not (sig.symbol in self.pf.positions and all("max open positions" in b for b in blocks)):
                self.log.write("no_entry", now=now, symbol=sig.symbol, reasons=blocks); break
            snap = snapshots[sig.symbol]
            extra = {k: v for k, v in (sig.snapshot or {}).items() if k not in snap.to_dict()}   # module-specific numbers (e.g. carry_r24)
            regime = self._refresh_regime(sig.symbol, snap, recent_bars.get(sig.symbol, []), now, mcp_ctx.get(sig.symbol), sig.module, extra)
            if regime and not regime.allow_entries:
                self.log.write("reject", now=now, symbol=sig.symbol, stage="regime", signal=sig.to_dict(),
                               regime=regime.model_dump(), reason="AI regime disallows entries")
                continue
            if self.analyst and self.s.strategy.ai_enabled:
                rd = self._risk_defaults()
                for k in ("max_position_pct", "stop_loss_pct", "take_profit_pct"):
                    if getattr(sig, k) is not None:
                        rd[k] = getattr(sig, k)
                decision = self.analyst.decide(sig, regime, self.pf.summary(prices), rd)
                ai_used = True
            else:
                decision = self.analyst._fallback(sig, regime) if self.analyst else None
                ai_used = False
            if decision is None:
                size_f, ai_sl, ai_tp, rationale, ddump = 1.0, None, None, "[rule-only] AI disabled", None
            else:
                ddump = decision.model_dump()
                if decision.action == "reject":
                    self.log.write("reject", now=now, symbol=sig.symbol, stage="ai", signal=sig.to_dict(),
                                   regime=regime.model_dump() if regime else None, decision=ddump, ai_used=ai_used)
                    continue
                size_f, ai_sl, ai_tp, rationale = decision.size_factor, decision.stop_loss_pct, decision.take_profit_pct, decision.rationale
            overrides = {k: getattr(sig, k) for k in ("max_position_pct", "stop_loss_pct", "take_profit_pct", "trailing_pct", "max_holding_minutes", "allow_top_up")}
            plan, notes = self.risk.plan_entry(sig.symbol, self.pf, equity, prices, size_f, ai_sl, ai_tp, snap.atr_pct,
                                               regime.size_factor if regime else 1.0, now, overrides)
            if plan is None:
                self.log.write("reject", now=now, symbol=sig.symbol, stage="risk", signal=sig.to_dict(), decision=ddump, reasons=notes)
                continue
            self._open(sig, plan, notes, ddump, regime, rationale, prices, markets, now)
            summary["entries"].append(sig.symbol)
            equity = self.pf.equity(prices)
        return summary

    # ----------------------------------------------------------------- execution
    def _tradable(self, symbol: str) -> bool:
        """Challenge compliance: only whitelisted, non-blocked spot markets may ever receive an order."""
        return symbol in self.s.markets.allowed and not self.s.markets.is_blocked(symbol)

    def _open(self, sig, plan, notes, decision, regime, rationale, prices, markets, now):
        if not self._tradable(sig.symbol):
            self.log.write("reject", now=now, symbol=sig.symbol, stage="compliance", reason="market not allowed in the challenge account")
            return
        m = markets[sig.symbol]
        ref = m.ask if m.ask > 0 else prices[sig.symbol]
        fill = self.broker.buy(sig.symbol, plan.notional_tmn, m, ref)
        did = self.log.write(("top_up" if sig.symbol in self.pf.positions else "entry") if fill.ok else "entry_failed", now=now,
                             symbol=sig.symbol, module=sig.module, broker=self.broker.name,
                             signal=sig.to_dict(), regime=regime.model_dump() if regime else None, ai_decision=decision,
                             why=rationale, risk_plan=plan.__dict__, risk_notes=notes, equity_before=round(self.pf.equity(prices)),
                             fill=fill.__dict__ if fill.ok else {"error": fill.error, "status": fill.status})
        if not fill.ok:
            log.warning("entry %s failed: %s", sig.symbol, fill.error)
            return
        self.pf.cash_tmn -= fill.notional_tmn            # buy fee is charged in the base coin (already netted from fill.qty)
        self.pf.fees_paid_tmn += fill.fee_tmn
        old = self.pf.positions.get(sig.symbol)
        if old is not None:                       # top-up: average in, keep stop/tp percentages from the new plan
            tot_qty = old.qty + fill.qty
            avg = (old.qty * old.entry_price + fill.qty * fill.price) / tot_qty
            old.qty, old.entry_price = tot_qty, avg
            old.entry_fee_tmn += fill.fee_tmn
            old.stop_loss = avg * (1 - plan.stop_loss_pct / 100)
            old.take_profit = avg * (1 + plan.take_profit_pct / 100) if plan.take_profit_pct else 0.0
            old.highest = max(old.highest, fill.price)
            old.last_add_time = now; old.adds += 1
            return
        pos = self.risk.build_position(sig.symbol, fill.qty, fill.price, fill.fee_tmn, plan, fill.order_id, did, now)
        pos.module = sig.module
        pos.last_add_time = now
        self.pf.positions[sig.symbol] = pos
        log.info("BUY %s qty=%s @%s notional=%.0f (%s)", sig.symbol, fill.qty, fill.price, fill.notional_tmn, rationale[:80])

    def _close(self, sym: str, reason: str, prices, markets, now):
        p = self.pf.positions.get(sym)
        if not p or sym not in markets:
            return
        m = markets[sym]
        ref = m.bid if m.bid > 0 else prices[sym]
        fill = self.broker.sell(sym, p.qty, m, ref)
        if not fill.ok:
            self.log.write("exit_failed", now=now, symbol=sym, reason=reason, error=fill.error)
            log.warning("exit %s failed: %s", sym, fill.error)
            return
        pnl = fill.notional_tmn - fill.fee_tmn - p.paid_tmn
        self.pf.cash_tmn += fill.notional_tmn - fill.fee_tmn
        self.pf.fees_paid_tmn += fill.fee_tmn
        self.pf.realized_pnl_tmn += pnl
        self.pf.trades += 1
        if pnl > 0:
            self.pf.wins += 1
        else:
            self.pf.last_loss_time[sym] = now
        held_min = (now - p.entry_time) / 60
        del self.pf.positions[sym]
        self.log.write("exit", now=now, symbol=sym, broker=self.broker.name, reason=reason, entry_price=p.entry_price,
                       exit_price=fill.price, qty=fill.qty, pnl_tmn=round(pnl), pnl_pct=round((fill.price / p.entry_price - 1) * 100, 3),
                       held_minutes=round(held_min), entry_decision_id=p.decision_id, fill=fill.__dict__)
        log.info("SELL %s @%s pnl=%.0f TMN (%s)", sym, fill.price, pnl, reason)

    def liquidate_all(self, prices, markets, now: Optional[float] = None, reason: str = "manual liquidation"):
        now = now or time.time()
        self.broker.cancel_all()
        for sym in list(self.pf.positions):
            self._close(sym, reason, prices, markets, now)
