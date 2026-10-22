"""Satellite: hourly trend-following on crypto/TMN markets, measured in USD terms.

Why USD terms: a TMN chart of BTC mixes two drivers (BTC/USD and USDT/TMN). Dividing by the
USDTTMN close isolates the crypto's own trend; the carry module already handles the FX driver.
Entry: EMA(fast) crosses above EMA(slow) on the USD series, or a 24-bar breakout with fast>slow,
and the TMN regime is not hostile. Exit hint: fast < slow. Fixed SL/TP come from module cfg.
DISABLED by default: on Sep-Oct 2026 data this family was positive in-sample but only ~35% of its
parameter neighbourhood stayed positive out-of-sample after 0.6% fees (docs/research.md).
"""
from __future__ import annotations

from typing import List, Optional

from wallex_agent.data.features import ema
from wallex_agent.strategy.base import Context, Signal, StrategyModule


class TrendStrategy(StrategyModule):
    name = "trend"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.fast = int(cfg.get("ema_fast", 20)); self.slow = int(cfg.get("ema_slow", 100)); self.brk = int(cfg.get("breakout_bars", 24))
        self.exclude = set(cfg.get("exclude", ["USDTTMN"]))

    def symbols(self, universe):
        return [s for s in universe if s not in self.exclude]

    def _usd(self, symbol: str, ctx: Context):
        df = ctx.frames.get(symbol); u = ctx.frames.get(ctx.usdt_symbol)
        if df is None or len(df) < self.slow + 5:
            return None, None
        c = df["close"]
        if u is not None:
            uc = u["close"].reindex(c.index).ffill()
            c = (c / uc).dropna()
        return df, c

    def candidates(self, ctx: Context) -> List[Signal]:
        out = []
        for sym in self.symbols(list(ctx.snapshots)):
            df, c = self._usd(sym, ctx)
            if c is None or len(c) < self.slow + 5:
                continue
            ef, es = ema(c, self.fast), ema(c, self.slow)
            cross_up = ef.iloc[-1] > es.iloc[-1] and ef.iloc[-2] <= es.iloc[-2]
            brk = c.iloc[-1] > c.iloc[-self.brk - 1:-1].max() and ef.iloc[-1] > es.iloc[-1]
            reasons, vetoes = [], []
            if cross_up:
                reasons.append(f"EMA{self.fast} crossed above EMA{self.slow} (USD terms)")
            if brk:
                reasons.append(f"{self.brk}-bar breakout in USD terms with EMA{self.fast}>EMA{self.slow}")
            if not (cross_up or brk):
                vetoes.append("no trend trigger")
            snap = ctx.snapshots[sym]
            if snap.spread_pct is not None and snap.spread_pct > float(self.cfg.get("max_spread_pct", 0.35)):
                vetoes.append(f"spread {snap.spread_pct:.2f}% too wide")
            strength = float((ef.iloc[-1] / es.iloc[-1] - 1) * 100)
            score = max(0.0, min(1.0, 0.5 + strength / 2)) if not vetoes else 0.0
            out.append(Signal(symbol=sym, side="BUY", score=round(score, 4), module=self.name, reasons=reasons, vetoes=vetoes,
                              snapshot={**snap.to_dict(), "usd_ema_fast": float(ef.iloc[-1]), "usd_ema_slow": float(es.iloc[-1])},
                              max_position_pct=self.cfg.get("max_position_pct"), stop_loss_pct=self.cfg.get("stop_loss_pct", 3.0),
                              take_profit_pct=self.cfg.get("take_profit_pct", 4.0), trailing_pct=self.cfg.get("trailing_pct", 0),
                              max_holding_minutes=self.cfg.get("max_holding_minutes", 72 * 60)))
        return out

    def exit_hint(self, symbol: str, ctx: Context) -> Optional[str]:
        if symbol in self.exclude:
            return None
        df, c = self._usd(symbol, ctx)
        if c is None:
            return None
        ef, es = ema(c, self.fast), ema(c, self.slow)
        if ef.iloc[-1] < es.iloc[-1]:
            return f"trend exit: EMA{self.fast} < EMA{self.slow} in USD terms"
        return None
