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
        # `symbol` is what we TRADE; `regime_symbol` is where the Toman-weakening regime is MEASURED.
        # The challenge account cannot place orders on USDTTMN (verified live 2026-10-23), but USDTTMN prices are
        # still readable, so the regime is measured on USDTTMN and expressed through a tradable carrier such as
        # XAUTTMN (Tether Gold: gold/USD x USDT/TMN, 0% maker fee) whose own trend must not be hostile.
        self.symbol = cfg.get("symbol", "XAUTTMN")
        self.regime_symbol = cfg.get("regime_symbol", "USDTTMN")
        self.carrier_ema = int(cfg.get("carrier_ema_hours", 48))
        self.carrier_max_drop_pct = float(cfg.get("carrier_max_24h_drop_pct", 1.5))
        self.ema_n = int(cfg.get("ema_hours", 72))
        self.mom_h = int(cfg.get("momentum_hours", 24))
        self.mom_on = float(cfg.get("momentum_on_pct", 0.3))
        self.mom_off = float(cfg.get("momentum_off_pct", -0.3))
        self.min_score = float(cfg.get("min_score", 0.5))
        # Rule-layer floor: when the FX regime is clearly on and the carrier guard passes, always hold at least
        # base_size_factor x cap (research: passive exposure beat every timing filter). The AI sizes the rest.
        self.base_size_factor = float(cfg.get("base_size_factor", 0.5))
        self.floor_min_score = float(cfg.get("floor_min_score", 0.6))
        self.floor_max_size_factor = float(cfg.get("floor_max_size_factor", 0.8))   # floor at score 1.0 (AI still owns the rest up to 1.0)

    def symbols(self, universe):
        return [self.symbol] if self.symbol in universe else []

    def _carrier_state(self, ctx: Context):
        """Own-trend check of the traded carrier (skip when the carrier is the regime symbol itself)."""
        if self.symbol == self.regime_symbol:
            return {"ok": True, "note": "carrier = regime symbol"}
        df = ctx.frames.get(self.symbol)
        if df is None or len(df) < self.carrier_ema + 25:
            return {"ok": False, "note": f"not enough {self.symbol} history"}
        c = df["close"]
        e = float(ema(c, self.carrier_ema).iloc[-1]); px = float(c.iloc[-1])
        r24 = float((c.iloc[-1] / c.iloc[-25] - 1) * 100)
        ok = px >= e * 0.995 and r24 > -self.carrier_max_drop_pct
        return {"ok": ok, "px": px, "ema": e, "r24": r24,
                "note": f"{self.symbol} close {px:,.0f} vs EMA{self.carrier_ema} {e:,.0f}, 24h {r24:+.2f}%"}

    def _state(self, ctx: Context):
        df = ctx.frames.get(self.regime_symbol)
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
        car = self._carrier_state(ctx)
        if not car["ok"]:
            vetoes.append(f"carrier trend hostile: {car['note']}")
        elif self.symbol != self.regime_symbol:
            reasons.append(f"carrier ok: {car['note']}")
        if not st["regime_up"]:
            vetoes.append(f"TMN not weakening ({self.regime_symbol}): close {st['px']:,.0f} <= EMA{self.ema_n} {st['ema']:,.0f} and 24h {st['r24']:+.2f}% <= {self.mom_on}%")
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
        snap.update({f"regime_{k}": (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()})
        snap.update({f"carrier_{k}": (round(v, 4) if isinstance(v, float) else v) for k, v in car.items() if k != "note"})
        snap["regime_symbol"] = self.regime_symbol
        floor = 0.0
        if not vetoes and score >= self.floor_min_score:
            # linear ramp: score=floor_min_score -> base_size_factor, score=1.0 -> floor_max_size_factor
            t = (score - self.floor_min_score) / max(1e-9, 1.0 - self.floor_min_score)
            floor = round(self.base_size_factor + (self.floor_max_size_factor - self.base_size_factor) * _clip(t), 3)
            cap = float(self.cfg.get("max_position_pct") or 0.5)
            reasons.append(f"rule floor: score {score:.2f} -> at least {floor:.0%} of module cap (~{floor * cap:.0%} of equity); AI sizes the rest")
        return [Signal(symbol=self.symbol, side="BUY", score=round(score, 4), module=self.name, reasons=reasons,
                       vetoes=vetoes, snapshot=snap, floor_size_factor=floor,
                       max_position_pct=self.cfg.get("max_position_pct"), stop_loss_pct=self.cfg.get("stop_loss_pct", 2.5),
                       take_profit_pct=self.cfg.get("take_profit_pct", 0), trailing_pct=self.cfg.get("trailing_pct", 0),
                       max_holding_minutes=self.cfg.get("max_holding_minutes", 0), allow_top_up=bool(self.cfg.get("allow_top_up", False)))]

    def exit_hint(self, symbol: str, ctx: Context) -> Optional[str]:
        if symbol != self.symbol:
            return None
        st = self._state(ctx)
        if st and st["regime_off"]:
            return f"carry regime off ({self.regime_symbol}): close {st['px']:,.0f} < EMA{self.ema_n} {st['ema']:,.0f} and 24h {st['r24']:+.2f}% < {self.mom_off}%"
        if self.symbol != self.regime_symbol:
            car = self._carrier_state(ctx)
            if car.get("px") and car["r24"] < -2 * self.carrier_max_drop_pct:
                return f"carrier crash exit: {car['note']}"
        return None
