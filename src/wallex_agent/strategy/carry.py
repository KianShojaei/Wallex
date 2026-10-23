"""Currency-regime core ("Toman carry") expressed through a diversified basket of carriers.

Empirical basis (30 days of Wallex 1-min data, Sep 22 - Oct 22 2026, see docs/research.md):
  * every crypto fell in USD terms while USDT/TMN rose 18%; TMN-market gains were FX, not crypto;
  * after fees no technical timing rule on any TMN market beat holding USD exposure over 3-day windows.
So the core decision is a *currency allocation*. The regime is measured on USDTTMN (readable, but the
challenge account cannot buy it) and expressed through a weighted BASKET of tradable carriers, each of
which is "USD asset x USDT/TMN": Tether Gold (0% maker fee), BTC, ETH ... Diversifying the USD leg across
gold and crypto removes single-asset risk while keeping the FX thesis.

Rules (hourly closed bars):
  regime_up  = USDTTMN close > EMA(72h)  OR  24h return > +0.3%
  regime_off = USDTTMN close < EMA(72h)  AND 24h return < -0.3%          (hysteresis)
  score      = blend of 24h/72h momentum and distance to EMA72, penalised by abnormal volatility
  carrier ok = carrier close >= 0.995 x its EMA(48h) and its 24h return > -carrier_max_24h_drop_pct
  floor      = linear in score between base_size_factor (score=floor_min_score) and floor_max_size_factor (score=1)
Each carrier gets cap = module max_position_pct x weight; the AI sizes between floor and cap, never below
the floor, never above the cap. Exits: regime off (all carriers), carrier crash, tail stop, challenge end.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from wallex_agent.data.features import ema
from wallex_agent.strategy.base import Context, Signal, StrategyModule


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


class CarryStrategy(StrategyModule):
    name = "carry"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.regime_symbol = cfg.get("regime_symbol", "USDTTMN")
        carriers = cfg.get("carriers")
        if not carriers:                                   # backward compatible single-carrier config
            carriers = {cfg.get("symbol", "XAUTTMN"): 1.0}
        tot = sum(float(w) for w in carriers.values()) or 1.0
        self.carriers: Dict[str, float] = {k: float(w) / tot for k, w in carriers.items()}
        self.module_cap = float(cfg.get("max_position_pct") or 0.5)
        self.carrier_ema = int(cfg.get("carrier_ema_hours", 48))
        self.carrier_max_drop_pct = float(cfg.get("carrier_max_24h_drop_pct", 1.5))
        self.ema_n = int(cfg.get("ema_hours", 72))
        self.mom_h = int(cfg.get("momentum_hours", 24))
        self.mom_on = float(cfg.get("momentum_on_pct", 0.3))
        self.mom_off = float(cfg.get("momentum_off_pct", -0.3))
        self.min_score = float(cfg.get("min_score", 0.5))
        self.base_size_factor = float(cfg.get("base_size_factor", 0.5))
        self.floor_min_score = float(cfg.get("floor_min_score", 0.6))
        self.floor_max_size_factor = float(cfg.get("floor_max_size_factor", 0.8))

    # ------------------------------------------------------------------ helpers
    def symbols(self, universe):
        return [s for s in self.carriers if s in universe]

    def position_cap_pct(self, symbol: str) -> Optional[float]:
        w = self.carriers.get(symbol)
        return self.module_cap * w if w else None

    def _carrier_state(self, symbol: str, ctx: Context):
        if symbol == self.regime_symbol:
            return {"ok": True, "note": "carrier = regime symbol"}
        df = ctx.frames.get(symbol)
        if df is None or len(df) < self.carrier_ema + 25:
            return {"ok": False, "note": f"not enough {symbol} history"}
        c = df["close"]
        e = float(ema(c, self.carrier_ema).iloc[-1]); px = float(c.iloc[-1])
        r24 = float((c.iloc[-1] / c.iloc[-25] - 1) * 100)
        ok = px >= e * 0.995 and r24 > -self.carrier_max_drop_pct
        return {"ok": ok, "px": px, "ema": e, "r24": r24,
                "note": f"{symbol} close {px:,.0f} vs EMA{self.carrier_ema} {e:,.0f}, 24h {r24:+.2f}%"}

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

    def _score(self, st) -> float:
        mom = _clip(st["r24"] / 1.5) * 0.5 + _clip(st["r72"] / 3.0) * 0.3 + _clip(st["dist_pct"] / 1.0) * 0.2
        vol_pen = 0.0 if st["vol24"] < 0.5 else _clip((st["vol24"] - 0.5) / 1.0) * 0.3
        return _clip(mom - vol_pen)

    # ------------------------------------------------------------------ signals
    def candidates(self, ctx: Context) -> List[Signal]:
        st = self._state(ctx)
        if st is None:
            return []
        out: List[Signal] = []
        regime_reasons, regime_vetoes = [], []
        if not st["regime_up"]:
            regime_vetoes.append(f"TMN not weakening ({self.regime_symbol}): close {st['px']:,.0f} <= EMA{self.ema_n} {st['ema']:,.0f} and 24h {st['r24']:+.2f}% <= {self.mom_on}%")
        else:
            if st["px"] > st["ema"]:
                regime_reasons.append(f"USDT/TMN above EMA{self.ema_n} by {st['dist_pct']:+.2f}%")
            if st["r24"] > self.mom_on:
                regime_reasons.append(f"24h momentum {st['r24']:+.2f}% > {self.mom_on}%")
            if st["r72"] > 0:
                regime_reasons.append(f"72h momentum {st['r72']:+.2f}%")
        base_score = self._score(st) if not regime_vetoes else 0.0
        for sym, w in self.carriers.items():
            if sym not in ctx.snapshots:
                continue
            reasons, vetoes = list(regime_reasons), list(regime_vetoes)
            car = self._carrier_state(sym, ctx)
            if not car["ok"]:
                vetoes.append(f"carrier trend hostile: {car['note']}")
            else:
                reasons.append(f"carrier ok: {car['note']}")
            score = base_score if not vetoes else 0.0
            if not vetoes and score < self.min_score:
                vetoes.append(f"carry score {score:.2f} < {self.min_score}")
            floor = 0.0
            cap = self.module_cap * w
            if not vetoes and score >= self.floor_min_score:
                t = (score - self.floor_min_score) / max(1e-9, 1.0 - self.floor_min_score)
                floor = round(self.base_size_factor + (self.floor_max_size_factor - self.base_size_factor) * _clip(t), 3)
                reasons.append(f"rule floor: score {score:.2f} -> at least {floor:.0%} of this carrier's cap {cap:.0%} (~{floor * cap:.0%} of equity); AI sizes the rest")
            snap = ctx.snapshots[sym].to_dict()
            snap.update({f"regime_{k}": (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()})
            snap.update({f"carrier_{k}": (round(v, 4) if isinstance(v, float) else v) for k, v in car.items() if k != "note"})
            snap["regime_symbol"] = self.regime_symbol
            snap["basket_weight"] = w
            out.append(Signal(symbol=sym, side="BUY", score=round(score, 4), module=self.name, reasons=reasons, vetoes=vetoes,
                              snapshot=snap, floor_size_factor=floor, max_position_pct=cap,
                              stop_loss_pct=self.cfg.get("stop_loss_pct", 2.5), take_profit_pct=self.cfg.get("take_profit_pct", 0),
                              trailing_pct=self.cfg.get("trailing_pct", 0), max_holding_minutes=self.cfg.get("max_holding_minutes", 0),
                              allow_top_up=bool(self.cfg.get("allow_top_up", False))))
        return out

    def exit_hint(self, symbol: str, ctx: Context) -> Optional[str]:
        if symbol not in self.carriers:
            return None
        st = self._state(ctx)
        if st and st["regime_off"]:
            return f"carry regime off ({self.regime_symbol}): close {st['px']:,.0f} < EMA{self.ema_n} {st['ema']:,.0f} and 24h {st['r24']:+.2f}% < {self.mom_off}%"
        car = self._carrier_state(symbol, ctx)
        if car.get("px") and car["r24"] < -2 * self.carrier_max_drop_pct:
            return f"carrier crash exit: {car['note']}"
        return None
