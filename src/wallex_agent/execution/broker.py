"""Brokers share one interface so the SAME engine code runs in live, shadow and backtest.

  LiveBroker     -> real Wallex orders (REST). Marketable LIMIT by default, MARKET optional.
  ShadowBroker   -> no orders sent; fills simulated at the live best ask/bid (+slippage, fees).
  BacktestBroker -> fills at candle open/close of the next bar (+slippage, fees).
"""
from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional

from wallex_agent.wallex.client import MarketInfo, WallexClient, WallexError

log = logging.getLogger(__name__)


@dataclass
class Fill:
    ok: bool
    order_id: str
    symbol: str
    side: str
    qty: float
    price: float
    notional_tmn: float
    fee_tmn: float
    status: str
    raw: Optional[dict] = None
    error: str = ""


def round_down(x: float, precision: int) -> float:
    q = 10 ** precision
    return math.floor(x * q) / q


def round_price(x: float, precision: int) -> float:
    return round(x, precision)


class Broker:
    name = "base"

    def __init__(self, fee_pct: float):
        self.fee_pct = fee_pct

    def buy(self, symbol: str, notional_tmn: float, market: MarketInfo, ref_price: float) -> Fill: ...
    def sell(self, symbol: str, qty: float, market: MarketInfo, ref_price: float) -> Fill: ...
    def cancel_all(self, symbol: Optional[str] = None) -> None: ...


class SimulatedBroker(Broker):
    """Shared by shadow & backtest: instant fills with slippage. `fee_fn(symbol)` gives the %-fee per side."""
    name = "shadow"

    def __init__(self, fee_pct: float, slippage_pct: float, fee_fn=None):
        super().__init__(fee_pct)
        self.slip = slippage_pct
        self.fee_fn = fee_fn or (lambda sym: fee_pct)

    def buy(self, symbol, notional_tmn, market, ref_price):
        px = ref_price * (1 + self.slip / 100)
        qty = round_down(notional_tmn / px, market.qty_precision)
        if qty <= 0 or qty < market.min_qty:
            return Fill(False, "", symbol, "BUY", 0, px, 0, 0, "REJECTED", error="qty below minimum")
        fee = self.fee_fn(symbol)
        notional = qty * px
        fee_tmn = notional * fee / 100
        net_qty = qty * (1 - fee / 100)          # Wallex charges buy fees in the base coin
        return Fill(True, f"sim-{uuid.uuid4().hex[:10]}", symbol, "BUY", net_qty, px, notional, fee_tmn, "FILLED")

    def sell(self, symbol, qty, market, ref_price):
        px = ref_price * (1 - self.slip / 100)
        qty = round_down(qty, market.qty_precision)
        notional = qty * px
        return Fill(True, f"sim-{uuid.uuid4().hex[:10]}", symbol, "SELL", qty, px, notional, notional * self.fee_fn(symbol) / 100, "FILLED")

    def cancel_all(self, symbol=None):
        return None


class LiveBroker(Broker):
    name = "live"

    def __init__(self, client: WallexClient, fee_pct: float, order_type: str = "LIMIT", limit_offset_pct: float = 0.1,
                 timeout_s: int = 45, poll_s: float = 2.0, style: str = "maker", fee_fn=None):
        super().__init__(fee_pct)
        self.fee_fn = fee_fn or (lambda sym: fee_pct)
        self.c = client
        self.order_type = order_type
        self.style = style            # maker: rest at our side of the book; taker: cross the spread by `offset`
        self.offset = limit_offset_pct
        self.timeout = timeout_s
        self.poll = poll_s

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _parse(o: dict) -> dict:
        """Wallex order object (verified live 2026-10-23): clientOrderId, status, origQty, executedQty, executedSum,
        executedPrice (= limit price, NOT the average fill), fills[{price, quantity, fee, feeAsset, feeCoefficient}].
        Fees are charged in the RECEIVED asset: buys pay fee in the base coin, sells pay fee in TMN."""
        def f(k, d=0.0, src=None):
            try:
                return float((src or o).get(k, d) or d)
            except (TypeError, ValueError):
                return d
        fills = o.get("fills") or []
        fee_base = sum(f("fee", 0.0, x) for x in fills if str(x.get("feeAsset", "")).upper() not in ("TMN", "IRT", "TOMAN"))
        fee_quote = sum(f("fee", 0.0, x) for x in fills if str(x.get("feeAsset", "")).upper() in ("TMN", "IRT", "TOMAN"))
        side = str(o.get("side", "")).upper()
        if not fills and f("executedQty") > 0 and f("fee") > 0:
            # no fill details: assume fee in received asset
            if side == "BUY":
                fee_base = f("fee")
            else:
                fee_quote = f("fee")
        return {"id": str(o.get("clientOrderId") or o.get("client_id") or o.get("id") or ""),
                "status": str(o.get("status", "")).upper(), "executed_qty": f("executedQty"),
                "executed_sum": f("executedSum"), "executed_price": f("executedPrice"),
                "orig_qty": f("origQty"), "price": f("price"), "fee_base": fee_base, "fee_quote": fee_quote}

    def _wait(self, order_id: str) -> dict:
        t0 = time.time()
        last = {}
        while time.time() - t0 < self.timeout:
            try:
                last = self._parse(self.c.order(order_id))
            except WallexError as e:
                log.warning("order poll %s: %s", order_id, e)
            if last.get("status") in ("FILLED", "CANCELED", "CANCELLED", "REJECTED", "EXPIRED"):
                return last
            time.sleep(self.poll)
        try:
            self.c.cancel_order(order_id)
            time.sleep(1.0)
            last = self._parse(self.c.order(order_id))
            last["status"] = last.get("status") or "CANCELED"
        except WallexError as e:
            log.warning("cancel %s failed: %s", order_id, e)
        return last

    def _submit(self, symbol: str, side: str, qty: float, price: Optional[float], market: MarketInfo) -> Fill:
        cid = f"wa-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        qty_s = f"{qty:.{market.qty_precision}f}"
        price_s = None if price is None else f"{round_price(price, market.price_precision):.{market.price_precision}f}"
        log.info("placing %s %s qty=%s price=%s client_id=%s", side, symbol, qty_s, price_s, cid)
        try:
            raw = self.c.place_order(symbol, side, self.order_type if price is not None else "MARKET", qty_s, price_s, cid)
        except WallexError as e:
            detail = f"{e} payload={e.payload!r}" if e.payload is not None else str(e)
            return Fill(False, cid, symbol, side, 0, price or 0, 0, 0, "REJECTED", error=detail)
        p = self._parse(raw if isinstance(raw, dict) else {})
        oid = p["id"] or cid
        if p["status"] != "FILLED":
            p = self._wait(oid)
        eq, es = p.get("executed_qty", 0), p.get("executed_sum", 0)
        avg = es / eq if eq else (p.get("executed_price") or price or 0)
        ok = eq > 0
        notional = es or eq * avg
        # fee accounting in TMN terms: base-asset fee valued at the fill price; the qty we actually RECEIVE on a buy
        # is executedQty minus the base fee, so the position size reflects what can later be sold.
        fee_tmn = p.get("fee_quote", 0.0) + p.get("fee_base", 0.0) * avg
        if fee_tmn <= 0 and ok:
            fee_tmn = notional * self.fee_fn(symbol) / 100
        net_qty = eq - p.get("fee_base", 0.0) if side == "BUY" else eq
        return Fill(ok, oid, symbol, side, net_qty, avg, notional, fee_tmn,
                    p.get("status", ""), raw=raw if isinstance(raw, dict) else None,
                    error="" if ok else f"no fill (status {p.get('status')})")

    # -------------------------------------------------------------- api
    def buy(self, symbol, notional_tmn, market, ref_price):
        """ref_price = best ask. maker: rest at best bid (fills when someone sells to us); taker: ask + offset."""
        if self.order_type == "MARKET":
            px = None
        elif self.style == "maker" and market.bid > 0:
            px = market.bid
        else:
            px = ref_price * (1 + self.offset / 100)
        qty = round_down(notional_tmn / (px or ref_price), market.qty_precision)
        if qty <= 0 or qty < market.min_qty or qty * (px or ref_price) < market.min_notional:
            return Fill(False, "", symbol, "BUY", 0, ref_price, 0, 0, "REJECTED", error="below exchange minimum")
        fill = self._submit(symbol, "BUY", qty, px, market)
        if not fill.ok and px is not None and self.style == "maker":
            # maker order did not fill in time -> one taker attempt so the strategy is not starved
            log.info("maker buy %s unfilled, retrying as taker", symbol)
            fill = self._submit(symbol, "BUY", qty, ref_price * (1 + self.offset / 100), market)
        return fill

    def sell(self, symbol, qty, market, ref_price):
        """ref_price = best bid. Exits must happen: maker first, then taker fallback.
        The quantity is capped at the free exchange balance of the base asset (fees were taken in-kind on the buy)."""
        try:
            free = self.c.balances().get(market.base, {}).get("free")
            if free is not None and free < qty:
                log.info("sell %s: capping qty %.8f to free balance %.8f", symbol, qty, free)
                qty = free
        except WallexError as e:
            log.warning("balance check before sell failed: %s", e)
        qty = round_down(qty, market.qty_precision)
        if self.order_type == "MARKET":
            return self._submit(symbol, "SELL", qty, None, market)
        if self.style == "maker" and market.ask > 0:
            fill = self._submit(symbol, "SELL", qty, market.ask, market)
            if fill.ok:
                return fill
            log.info("maker sell %s unfilled, retrying as taker", symbol)
            qty = round_down(qty - fill.qty, market.qty_precision) if fill.qty else qty
        return self._submit(symbol, "SELL", qty, ref_price * (1 - self.offset / 100), market)

    def cancel_all(self, symbol=None):
        try:
            for o in self.c.open_orders(symbol):
                oid = o.get("clientOrderId") or o.get("id")
                if oid:
                    self.c.cancel_order(str(oid))
        except WallexError as e:
            log.warning("cancel_all failed: %s", e)
