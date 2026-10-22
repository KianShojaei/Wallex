"""Mean-reversion module (legacy). Kept for completeness and comparison; DISABLED by default because
on Sep-Oct 2026 Wallex data it was negative in every timeframe after 0.6% round-trip fees
(0 of 648 parameter sets positive in both walk-forward halves, see docs/research.md)."""
from __future__ import annotations

from types import SimpleNamespace
from typing import List, Optional

from wallex_agent.strategy.base import Context, Signal, StrategyModule
from wallex_agent.strategy.signals import evaluate_entry, evaluate_exit


class MeanReversionStrategy(StrategyModule):
    name = "mean_reversion"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.p = SimpleNamespace(rsi_period=14, rsi_oversold=cfg.get("rsi_oversold", 32), rsi_exit=cfg.get("rsi_exit", 55),
                                 ema_fast=20, ema_slow=50, trend_ema=100, volume_zscore_min=cfg.get("volume_zscore_min", -1.0),
                                 max_spread_pct=cfg.get("max_spread_pct", 0.35))
        self.exclude = set(cfg.get("exclude", ["USDTTMN"]))

    def symbols(self, universe):
        return [s for s in universe if s not in self.exclude]

    def candidates(self, ctx: Context) -> List[Signal]:
        out = []
        for sym in self.symbols(list(ctx.snapshots)):
            s = evaluate_entry(ctx.snapshots[sym], self.p)
            out.append(Signal(symbol=sym, side="BUY", score=s.score, module=self.name, reasons=s.reasons, vetoes=s.vetoes,
                              snapshot=s.snapshot, max_position_pct=self.cfg.get("max_position_pct"),
                              stop_loss_pct=self.cfg.get("stop_loss_pct", 1.0), take_profit_pct=self.cfg.get("take_profit_pct", 1.6),
                              trailing_pct=self.cfg.get("trailing_pct", 0.8), max_holding_minutes=self.cfg.get("max_holding_minutes", 720)))
        return out

    def exit_hint(self, symbol: str, ctx: Context) -> Optional[str]:
        if symbol in self.exclude or symbol not in ctx.snapshots:
            return None
        return evaluate_exit(ctx.snapshots[symbol], self.p)
