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
    "You are the risk-aware trading analyst inside an automated spot-trading agent on the Iranian "
    "exchange Wallex (quote currency Toman, TMN). Horizon: intraday to 12h. Long-only. Fees ~0.25%/side, "
    "so a trade must plausibly earn >0.8%. Be conservative: capital preservation beats activity. "
    "You never see news; reason only from the numbers given. Always answer with ONE JSON object only."
)


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
    def regime(self, symbol: str, snapshot: Dict, recent: List[Dict], mcp_context: Optional[Dict] = None) -> Optional[Regime]:
        user = {
            "task": "Classify the current market regime for the next 1-12 hours and say whether mean-reversion LONG entries are sensible.",
            "symbol": symbol,
            "latest": snapshot,
            "recent_hourly_bars_oldest_first": recent[-24:],
            "extra_exchange_context": mcp_context or {},
            "output_schema": REGIME_SCHEMA,
        }
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]
        return self.llm.complete_json(msgs, Regime, deep=True, tag=f"regime:{symbol}:{snapshot.get('time','')[:13]}")

    # ----------------------------------------------------------- decide
    def decide(self, signal: Signal, regime: Optional[Regime], portfolio: Dict, risk_defaults: Dict) -> Decision:
        user = {
            "task": "A rule-based scanner produced a LONG entry candidate. Decide approve / approve_reduced / reject. "
                    "You may shrink size or tighten the stop, never increase risk. Explain with the numbers.",
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
