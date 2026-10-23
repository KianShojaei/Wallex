"""The AI analyst: two real decision roles, both bounded by the risk engine.

1. `regime(symbol, snapshot, recent_bars)`  (deep model, hourly)
   Classifies the market regime and sets a *bias* the rule layer must respect:
   trend_up / range / trend_down / high_volatility, with allowed actions.
2. `decide(signal, regime, portfolio)`      (fast model, per candidate)
   Approves / rejects / resizes a rule-layer candidate and writes the human-readable
   rationale that is stored in the decision log. It can only *shrink* risk:
   size_factor in [0,1], stop tightened not loosened, take-profit within bounds.

Both outputs are strict JSON validated by pydantic. If the LLM is off or fails,
`decide` falls back to a conservative deterministic policy and the log says so.
"""
from __future__ import annotations

import json
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, confloat

from wallex_agent.ai.llm import LLMClient
from wallex_agent.strategy.signals import Signal

SYSTEM = (
    "You are the risk-aware analyst inside an automated spot-trading agent on the Iranian exchange Wallex "
    "(quote currency Toman, TMN). Long-only. Fees ~0.25%/side, so a position must plausibly earn >0.6% net. "
    "Be conservative: capital preservation beats activity, but do not reject a thesis that the numbers support. "
    "Judge the candidate against ITS OWN strategy brief (given in the request), not against a generic dip-buying "
    "template. Wallex reports zero volume for some overnight hours on USDTTMN; treat that as low activity, not as "
    "stale or broken data. The `time` field in the data IS the present moment for your decision (this may be a historical "
    "replay); never compare it with your own notion of today's date and never call the data stale because of it. "
    "You never see news; reason only from the numbers given. Answer with ONE JSON object only."
)

# What each strategy module is trying to do: the analyst judges the candidate against this thesis.
MODULE_BRIEFS = {
    "carry": (
        "CURRENCY-REGIME CORE. Thesis: the Toman is weakening (USDT/TMN rising), so holding a USD-linked carrier earns "
        "the FX drift. The regime is measured on USDTTMN (fields regime_*) but the TRADED asset may be a carrier such "
        "as XAUTTMN (Tether Gold = gold/USD x USDT/TMN; fields carrier_*), because the account cannot buy USDT directly. "
        "Judge BOTH: is the Toman regime intact, and is the carrier's own trend (gold) not hostile. "
        "Holding horizon 6h-3 days. This is a trend/continuation position, NOT a dip buy: being near the upper "
        "Bollinger band or a high RSI is EXPECTED in this regime and is not by itself a reason to reject. Reject or "
        "shrink when the regime looks exhausted or reversing: sharp parabolic spike likely to retrace, momentum fading "
        "(24h/72h returns rolling over), price breaking below the 72h EMA, or volatility far above its norm. "
        "Approve when the uptrend is orderly and intact. Stop-loss is a 2.5% tail guard; there is no take-profit."
    ),
    "trend": (
        "CRYPTO TREND SATELLITE. Thesis: this coin is in an uptrend in USD terms AND the Toman is not strengthening. "
        "Entry on EMA crossover / breakout. Judge trend quality and whether a 4% target vs 3% stop is realistic."
    ),
    "mean_reversion": (
        "MEAN REVERSION. Thesis: oversold bounce (RSI low, below lower Bollinger band) in a non-collapsing market. "
        "Reject if momentum is strongly negative or the bounce cannot clear fees."
    ),
}


class Regime(BaseModel):
    regime: Literal["trend_up", "range", "trend_down", "high_volatility"]
    confidence: confloat(ge=0, le=1)
    allow_entries: bool
    size_factor: confloat(ge=0, le=1) = 1.0
    summary: str = Field(max_length=400)


class Decision(BaseModel):
    action: Literal["approve", "reject", "approve_reduced"]
    confidence: confloat(ge=0, le=1)
    size_factor: confloat(ge=0, le=1) = 1.0
    stop_loss_pct: Optional[confloat(ge=0.3, le=3.0)] = None     # may only tighten (checked by risk engine)
    take_profit_pct: Optional[confloat(ge=0.5, le=4.0)] = None
    rationale: str = Field(max_length=600)
    key_risks: List[str] = Field(default_factory=list, max_length=5)


REGIME_SCHEMA = '{"regime":"trend_up|range|trend_down|high_volatility","confidence":0..1,"allow_entries":true|false,"size_factor":0..1,"summary":"<=400 chars"}'
DECISION_SCHEMA = '{"action":"approve|approve_reduced|reject","confidence":0..1,"size_factor":0..1,"stop_loss_pct":null|0.3..3.0,"take_profit_pct":null|0.5..4.0,"rationale":"<=600 chars, cite the numbers","key_risks":["..."]}'


class Analyst:
    def __init__(self, llm: LLMClient, min_confidence: float = 0.6):
        self.llm = llm
        self.min_confidence = min_confidence

    # ------------------------------------------------------------ regime
    def regime(self, symbol: str, snapshot: Dict, recent: List[Dict], mcp_context: Optional[Dict] = None,
               module: str = "carry") -> Optional[Regime]:
        user = {
            "task": "Classify the current market regime for the next 6-72 hours and decide whether NEW entries for the "
                    "strategy described in `strategy_brief` are sensible now (allow_entries) and at what size_factor (0..1).",
            "strategy_brief": MODULE_BRIEFS.get(module, "generic long-only strategy"),
            "symbol": symbol,
            "latest": snapshot,
            "recent_hourly_bars_oldest_first": recent[-24:],
            "extra_exchange_context": mcp_context or {},
            "output_schema": REGIME_SCHEMA,
        }
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]
        return self.llm.complete_json(msgs, Regime, deep=True, tag=f"regime:{module}:{symbol}:{snapshot.get('time','')[:13]}")

    # ----------------------------------------------------------- decide
    def decide(self, signal: Signal, regime: Optional[Regime], portfolio: Dict, risk_defaults: Dict) -> Decision:
        user = {
            "task": "A rule-based scanner produced a LONG entry candidate for the strategy in `strategy_brief`. Decide "
                    "approve / approve_reduced / reject. You may shrink size or tighten the stop, never increase risk. "
                    "Explain with the numbers.",
            "strategy_brief": MODULE_BRIEFS.get(signal.module, "generic long-only strategy"),
            "candidate": signal.to_dict(),
            "regime": regime.model_dump() if regime else "unknown",
            "portfolio": portfolio,
            "risk_defaults": risk_defaults,
            "output_schema": DECISION_SCHEMA,
        }
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]
        d = self.llm.complete_json(msgs, Decision, deep=False, tag=f"decide:{signal.symbol}")
        if d is None:
            return self._fallback(signal, regime)
        if d.action != "reject" and d.confidence < self.min_confidence:
            d = Decision(action="reject", confidence=d.confidence, size_factor=0.0,
                         rationale=f"[auto] AI confidence {d.confidence:.2f} below threshold {self.min_confidence}. AI said: {d.rationale}",
                         key_risks=d.key_risks)
        return d

    def _fallback(self, signal: Signal, regime: Optional[Regime]) -> Decision:
        """Deterministic conservative policy used when the LLM is off/unavailable."""
        if regime and not regime.allow_entries:
            return Decision(action="reject", confidence=0.5, size_factor=0.0,
                            rationale="[fallback] regime disallows entries", key_risks=["ai_unavailable"])
        if signal.score >= 0.75:
            return Decision(action="approve_reduced", confidence=0.6, size_factor=0.5,
                            rationale=f"[fallback] strong rule score {signal.score:.2f}; AI unavailable so half size",
                            key_risks=["ai_unavailable"])
        return Decision(action="reject", confidence=0.5, size_factor=0.0,
                        rationale=f"[fallback] rule score {signal.score:.2f} < 0.75 and AI unavailable",
                        key_risks=["ai_unavailable"])
