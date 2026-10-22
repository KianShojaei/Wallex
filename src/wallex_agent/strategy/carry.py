"""Currency-regime core ("Toman carry").

Empirical basis (30 days of Wallex 1-min data, Sep 22 - Oct 22 2026, see docs/research.md):
  * every crypto fell in USD terms while USDT/TMN rose 18%; TMN-market gains were FX, not crypto;
  * after 0.6% round-trip fees no technical timing rule on any TMN market beat holding USDT over
    3-day windows (median +1.0%, P(>0)=0.60, p10=-1.7%, worst -5.3%).
So the core decision is a *currency allocation*: hold a bounded share of equity in USDT while the
Toman-weakening regime holds, sit in TMN otherwise. Rules (hourly closed bars of USDTTMN):
  regime_up  = close > EMA(72h)  OR  24h return > +0.3%
  regime_off = close < EMA(72h)  AND 24h return < -0.3%          (hysteresis: no flip-flopping)
  score      = blend of 24h/72h momentum and distance to EMA72, scaled by realised volatility
The AI analyst sees the same numbers plus recent bars and may veto / shrink (never enlarge).
Risk: dedicated cap (`max_position_pct`), tail stop (default 2.5%), no fixed take-profit,
re-entry cooldown after a stop, and the global challenge-end liquidation to TMN.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from wallex_agent.data.features import ema
from wallex_agent.strategy.base import Context, Signal, StrategyModule


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


class CarryStrategy(StrategyModule):
    name = "carry"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.symbol = cfg.get("symbol", "USDTTMN")
        self.ema_n = int(cfg.get("ema_hours", 72))
        self.mom_h = int(cfg.get("momentum_hours", 24))
        self.mom_on = float(cfg.get("momentum_on_pct", 0.3))
        self.mom_off = float(cfg.get("momentum_off_pct", -0.3))
        self.min_score = float(cfg.get("min_score", 0.5))

    def symbols(self, universe):
        return [self.symbol] if self.symbol in universe else []

    def _state(self, ctx: Context):
        df = ctx.frames.get(self.symbol)
        if df is None or len(df) < self.ema_n + 5:
            return None
        c = df["close"]
        e = ema(c, self.ema_n)
        px = float(c.iloc[-1]); ev = float(e.iloc[-1])
        r24 = float((c.iloc[-1] / c.iloc[-1 - self.mom_h] - 1) * 100)
        r72 = float((c.iloc[-1] / c.iloc[-1 - 3 * self.mom_h] - 1) * 100) if len(c) > 3 * self.mom_h else r24
        vol24 = float((c.pct_change() * 100).tail(24).std(ddof=0) or 0.0)
        dist = (px / ev - 1) * 100
        return dict(px=px, ema=ev, dist_pct=dist, r24=r24, r72=r72, vol24=vol24,
                    regime_up=(px > ev) or (r24 > self.mom_on), regime_off=(px < ev) and (r24 < self.mom_off))

    def candidates(self, ctx: Context) -> List[Signal]:
        st = self._state(ctx)
        if st is None or self.symbol not in ctx.snapshots:
            return []
        reasons, vetoes = [], []
        if not st["regime_up"]:
            vetoes.append(f"TMN not weakening: close {st['px']:,.0f} <= EMA{self.ema_n} {st['ema']:,.0f} and 24h {st['r24']:+.2f}% <= {self.mom_on}%")
        else:
            if st["px"] > st["ema"]:
                reasons.append(f"USDT/TMN above EMA{self.ema_n} by {st['dist_pct']:+.2f}%")
            if st["r24"] > self.mom_on:
                reasons.append(f"24h momentum {st['r24']:+.2f}% > {self.mom_on}%")
            if st["r72"] > 0:
                reasons.append(f"72h momentum {st['r72']:+.2f}%")
        # score: momentum strength, penalised by abnormal volatility (spikes mean-revert)
        mom = _clip(st["r24"] / 1.5) * 0.5 + _clip(st["r72"] / 3.0) * 0.3 + _clip(st["dist_pct"] / 1.0) * 0.2
        vol_pen = 0.0 if st["vol24"] < 0.5 else _clip((st["vol24"] - 0.5) / 1.0) * 0.3
        score = _clip(mom - vol_pen) if not vetoes else 0.0
        if not vetoes and score < self.min_score:
            vetoes.append(f"carry score {score:.2f} < {self.min_score}")
        snap = ctx.snapshots[self.symbol].to_dict()
        snap.update({f"carry_{k}": (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()})
        return [Signal(symbol=self.symbol, side="BUY", score=round(score, 4), module=self.name, reasons=reasons,
                       vetoes=vetoes, snapshot=snap,
                       max_position_pct=self.cfg.get("max_position_pct"), stop_loss_pct=self.cfg.get("stop_loss_pct", 2.5),
                       take_profit_pct=self.cfg.get("take_profit_pct", 0), trailing_pct=self.cfg.get("trailing_pct", 0),
                       max_holding_minutes=self.cfg.get("max_holding_minutes", 0), allow_top_up=bool(self.cfg.get("allow_top_up", False)))]

    def exit_hint(self, symbol: str, ctx: Context) -> Optional[str]:
        if symbol != self.symbol:
            return None
        st = self._state(ctx)
        if st and st["regime_off"]:
            return f"carry regime off: close {st['px']:,.0f} < EMA{self.ema_n} {st['ema']:,.0f} and 24h {st['r24']:+.2f}% < {self.mom_off}%"
        return None
