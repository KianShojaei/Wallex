"""Risk & capital management. The only component allowed to size orders or open/close positions.

Hard rules (all configurable in settings.yaml, all logged):
  * max_position_pct of equity per position, max_total_exposure_pct overall, max_open_positions
  * never deploy the whole account (exposure cap) and never below min_order_notional
  * fixed stop-loss / take-profit / trailing stop / time stop per position
  * daily drawdown halt (no new entries today) and total drawdown kill switch (flatten + stop)
  * per-symbol cooldown after a losing trade
  * challenge-end liquidation: everything back to TMN before the official end
  * AI may only shrink size or tighten stops; the risk engine clamps every AI suggestion
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from wallex_agent.portfolio.state import Portfolio, Position


@dataclass
class EntryPlan:
    symbol: str
    notional_tmn: float
    stop_loss_pct: float
    take_profit_pct: float
    trailing_pct: float
    size_factor: float
    notes: List[str]


@dataclass
class ExitOrder:
    symbol: str
    reason: str


class RiskManager:
    def __init__(self, cfg, challenge_end_utc: datetime, liquidate_before_min: int):
        self.cfg = cfg
        self.end_ts = challenge_end_utc.timestamp()
        self.liq_ts = self.end_ts - liquidate_before_min * 60

    # ------------------------------------------------------------ gates
    def in_liquidation_window(self, now: Optional[float] = None) -> bool:
        return (now or time.time()) >= self.liq_ts

    def global_checks(self, pf: Portfolio, equity: float, now: Optional[float] = None) -> Tuple[bool, List[str]]:
        """Can we open NEW positions right now?"""
        now = now or time.time()
        blocks: List[str] = []
        if pf.killed:
            blocks.append("kill switch active")
        if self.in_liquidation_window(now):
            blocks.append("challenge-end liquidation window")
        dd = pf.drawdown_pct(equity)
        if dd >= self.cfg.total_drawdown_kill_pct:
            blocks.append(f"total drawdown {dd:.2f}% >= kill {self.cfg.total_drawdown_kill_pct}%")
        ddd = pf.daily_drawdown_pct(equity)
        if ddd >= self.cfg.daily_drawdown_halt_pct or pf.halted_until_day == pf.day_key:
            pf.halted_until_day = pf.day_key
            blocks.append(f"daily drawdown {ddd:.2f}% >= halt {self.cfg.daily_drawdown_halt_pct}%")
        if len(pf.positions) >= self.cfg.max_open_positions:
            blocks.append(f"max open positions ({self.cfg.max_open_positions}) reached")
        return (not blocks), blocks

    def should_kill(self, pf: Portfolio, equity: float) -> bool:
        return pf.drawdown_pct(equity) >= self.cfg.total_drawdown_kill_pct

    # ------------------------------------------------------------ sizing
    def plan_entry(self, symbol: str, pf: Portfolio, equity: float, prices: Dict[str, float], ai_size_factor: float,
                   ai_stop: Optional[float], ai_tp: Optional[float], atr_pct: float, regime_size_factor: float = 1.0,
                   now: Optional[float] = None) -> Tuple[Optional[EntryPlan], List[str]]:
        now = now or time.time()
        notes: List[str] = []
        if symbol in pf.positions:
            return None, ["already in position"]
        lt = pf.last_loss_time.get(symbol)
        if lt and now - lt < self.cfg.cooldown_after_loss_minutes * 60:
            return None, [f"cooldown after loss ({int((now - lt) / 60)}m < {self.cfg.cooldown_after_loss_minutes}m)"]
        # volatility scaling: target ~1% move per ATR unit; shrink when ATR% is high
        vol_factor = min(1.0, 1.0 / max(atr_pct, 0.5))
        size_factor = max(0.0, min(1.0, ai_size_factor)) * max(0.0, min(1.0, regime_size_factor)) * vol_factor
        notes.append(f"size_factor = ai {ai_size_factor:.2f} x regime {regime_size_factor:.2f} x vol {vol_factor:.2f} = {size_factor:.2f}")
        max_pos = equity * self.cfg.max_position_pct
        room = equity * self.cfg.max_total_exposure_pct - pf.exposure_tmn(prices)
        notional = min(max_pos * size_factor, room, pf.cash_tmn * 0.98)
        if notional < self.cfg.min_order_notional_tmn:
            return None, notes + [f"notional {notional:,.0f} < min {self.cfg.min_order_notional_tmn:,.0f} (room {room:,.0f}, cash {pf.cash_tmn:,.0f})"]
        sl = self.cfg.stop_loss_pct
        if ai_stop is not None and ai_stop < sl:
            sl = max(0.3, ai_stop); notes.append(f"AI tightened stop to {sl}%")
        tp = self.cfg.take_profit_pct
        if ai_tp is not None and 0.5 <= ai_tp <= self.cfg.take_profit_pct * 1.5:
            tp = ai_tp; notes.append(f"AI set take-profit {tp}%")
        return EntryPlan(symbol, round(notional), sl, tp, self.cfg.trailing_stop_pct, size_factor, notes), notes

    # ------------------------------------------------------------ exits
    def check_exits(self, pf: Portfolio, prices: Dict[str, float], rule_exit_hints: Dict[str, str],
                    now: Optional[float] = None) -> List[ExitOrder]:
        now = now or time.time()
        out: List[ExitOrder] = []
        liq = self.in_liquidation_window(now)
        for sym, p in list(pf.positions.items()):
            px = prices.get(sym)
            if px is None:
                continue
            p.highest = max(p.highest, px)
            if liq:
                out.append(ExitOrder(sym, "challenge-end liquidation to TMN")); continue
            if pf.killed:
                out.append(ExitOrder(sym, "kill switch: flatten")); continue
            if px <= p.stop_loss:
                out.append(ExitOrder(sym, f"stop-loss hit ({px:,.0f} <= {p.stop_loss:,.0f})")); continue
            if px >= p.take_profit:
                out.append(ExitOrder(sym, f"take-profit hit ({px:,.0f} >= {p.take_profit:,.0f})")); continue
            gain_pct = (p.highest / p.entry_price - 1) * 100
            tp_pct = (p.take_profit / p.entry_price - 1) * 100
            if gain_pct >= tp_pct / 2 and px <= p.highest * (1 - p.trailing_pct / 100):
                out.append(ExitOrder(sym, f"trailing stop ({p.trailing_pct}% off high {p.highest:,.0f})")); continue
            if now - p.entry_time >= self.cfg.max_holding_minutes * 60:
                out.append(ExitOrder(sym, f"time stop ({self.cfg.max_holding_minutes}m)")); continue
            if sym in rule_exit_hints and px > p.entry_price * (1 + 2 * self.cfg.fee_pct / 100):
                out.append(ExitOrder(sym, rule_exit_hints[sym]))
        return out

    @staticmethod
    def build_position(symbol: str, qty: float, fill_price: float, fee_tmn: float, plan: EntryPlan,
                       order_id: str, decision_id: str, now: Optional[float] = None) -> Position:
        now = now or time.time()
        return Position(symbol=symbol, qty=qty, entry_price=fill_price, entry_time=now,
                        stop_loss=fill_price * (1 - plan.stop_loss_pct / 100),
                        take_profit=fill_price * (1 + plan.take_profit_pct / 100),
                        trailing_pct=plan.trailing_pct, highest=fill_price, entry_fee_tmn=fee_tmn,
                        entry_order_id=order_id, decision_id=decision_id, size_factor=plan.size_factor)
