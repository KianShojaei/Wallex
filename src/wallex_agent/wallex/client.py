"""Thin, well-logged REST client for the Wallex exchange API.

Reference (api-docs.wallex.ir):
  Base URL  https://api.wallex.ir   auth header  X-API-Key
  Public    GET /v1/markets | GET /v1/depth?symbol= | GET /v1/trades?symbol=
            GET /v1/udf/history?symbol&resolution&from&to
  Private   GET /v1/account/balances | GET /v1/account/fee | GET /v1/account/openOrders
            GET /v1/account/trades | GET /v1/account/orders/{clientOrderId}
            POST /v1/account/orders {symbol,type,side,price,quantity,client_id}
            DELETE /v1/account/orders?clientOrderId=
Responses are wrapped as {success, message, result}.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

log = logging.getLogger(__name__)


class WallexError(RuntimeError):
    def __init__(self, status: int, message: str, payload: Any = None):
        super().__init__(f"Wallex HTTP {status}: {message}")
        self.status = status
        self.payload = payload


@dataclass
class MarketInfo:
    symbol: str
    base: str
    quote: str
    price_precision: int
    qty_precision: int
    min_qty: float
    min_notional: float
    bid: float
    ask: float
    last: float
    quote_volume_24h: float
    change_24h_pct: float
    raw: Dict[str, Any]

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last

    @property
    def spread_pct(self) -> float:
        m = self.mid
        return (self.ask - self.bid) / m * 100 if m and self.bid and self.ask else 999.0


def _f(x, default=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


class WallexClient:
    def __init__(self, base_url: str = "https://api.wallex.ir", api_key: str = "", timeout: float = 15.0,
                 max_retries: int = 3, min_interval_s: float = 0.2):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_retries = max_retries
        self.min_interval_s = min_interval_s
        self._last_call = 0.0
        headers = {"Accept": "application/json", "Content-Type": "application/json",
                   "User-Agent": "wallex-agent/0.1"}
        if api_key:
            headers["X-API-Key"] = api_key
        self._http = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout)

    # ------------------------------------------------------------------ core
    def _request(self, method: str, path: str, *, params=None, json=None, signed=False) -> Any:
        if signed and not self.api_key:
            raise WallexError(401, "WALLEX_API_KEY is required for private endpoints")
        backoff = 1.0
        for attempt in range(1, self.max_retries + 1):
            wait = self.min_interval_s - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                r = self._http.request(method, path, params=params, json=json)
                self._last_call = time.monotonic()
            except httpx.HTTPError as e:
                log.warning("wallex %s %s network error (%s), attempt %d", method, path, e, attempt)
                if attempt == self.max_retries:
                    raise WallexError(0, str(e)) from e
                time.sleep(backoff); backoff *= 2
                continue
            if r.status_code == 429 or r.status_code >= 500:
                log.warning("wallex %s %s -> %s, attempt %d", method, path, r.status_code, attempt)
                if attempt == self.max_retries:
                    raise WallexError(r.status_code, r.text[:300])
                time.sleep(backoff); backoff *= 2
                continue
            try:
                data = r.json()
            except ValueError:
                raise WallexError(r.status_code, f"non-JSON response: {r.text[:200]}")
            if r.status_code >= 400 or (isinstance(data, dict) and data.get("success") is False):
                msg = data.get("message") if isinstance(data, dict) else str(data)
                raise WallexError(r.status_code, msg or "request failed", data)
            return data.get("result", data) if isinstance(data, dict) and "result" in data else data
        raise WallexError(0, "unreachable")

    # ---------------------------------------------------------------- public
    def markets(self) -> Dict[str, MarketInfo]:
        res = self._request("GET", "/v1/markets")
        symbols = res.get("symbols", res) if isinstance(res, dict) else {}
        if isinstance(symbols, list):
            symbols = {m.get("symbol"): m for m in symbols}
        out: Dict[str, MarketInfo] = {}
        for sym, m in symbols.items():
            if not isinstance(m, dict):
                continue
            st = m.get("stats", {}) or {}
            out[sym] = MarketInfo(
                symbol=sym,
                base=m.get("baseAsset", ""),
                quote=m.get("quoteAsset", ""),
                # Wallex: tickSize/stepSize are the number of decimals accepted for price/quantity in orders
                # (e.g. USDTTMN stepSize=2, tickSize=0); the *AssetPrecision fields are display precision.
                price_precision=int(m.get("tickSize", m.get("quotePrecision", 0)) or 0),
                qty_precision=int(m.get("stepSize", m.get("baseAssetPrecision", 8)) or 0),
                min_qty=_f(m.get("minQty")),
                min_notional=_f(m.get("minNotional")),
                bid=_f(st.get("bidPrice")),
                ask=_f(st.get("askPrice")),
                last=_f(st.get("lastPrice")),
                quote_volume_24h=_f(st.get("24h_quoteVolume", st.get("quoteVolume"))),
                change_24h_pct=_f(st.get("24h_ch")),
                raw=m,
            )
        return out

    def depth(self, symbol: str) -> Dict[str, List[Dict[str, float]]]:
        res = self._request("GET", "/v1/depth", params={"symbol": symbol})
        def norm(side):
            rows = res.get(side, []) or []
            return [{"price": _f(x.get("price")), "quantity": _f(x.get("quantity")), "sum": _f(x.get("sum"))}
                    for x in rows]
        return {"ask": norm("ask"), "bid": norm("bid")}

    def recent_trades(self, symbol: str) -> List[Dict[str, Any]]:
        res = self._request("GET", "/v1/trades", params={"symbol": symbol})
        return res.get("latestTrades", res) if isinstance(res, dict) else res

    def candles(self, symbol: str, resolution: str, from_ts: int, to_ts: int) -> Dict[str, list]:
        """TradingView-UDF style: {s:'ok', t:[...], o:[], h:[], l:[], c:[], v:[]}"""
        return self._request("GET", "/v1/udf/history",
                             params={"symbol": symbol, "resolution": resolution, "from": int(from_ts), "to": int(to_ts)})

    # --------------------------------------------------------------- private
    def balances(self) -> Dict[str, Dict[str, float]]:
        res = self._request("GET", "/v1/account/balances", signed=True)
        bal = res.get("balances", res) if isinstance(res, dict) else {}
        out = {}
        for asset, b in bal.items():
            if isinstance(b, dict):
                out[asset] = {"value": _f(b.get("value")), "locked": _f(b.get("locked")),
                              "free": _f(b.get("value")) - _f(b.get("locked"))}
        return out

    def fees(self) -> Dict[str, Dict[str, float]]:
        res = self._request("GET", "/v1/account/fee", signed=True)
        out = {}
        if isinstance(res, dict):
            for sym, f in res.items():
                if isinstance(f, dict):
                    out[sym] = {"maker": _f(f.get("makerFeeRate")) * 100, "taker": _f(f.get("takerFeeRate")) * 100}
        return out

    def place_order(self, symbol: str, side: str, type_: str, quantity: str, price: Optional[str] = None,
                    client_id: Optional[str] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"symbol": symbol, "side": side.upper(), "type": type_.upper(), "quantity": str(quantity)}
        if price is not None:
            body["price"] = str(price)
        if client_id:
            body["client_id"] = client_id
        return self._request("POST", "/v1/account/orders", json=body, signed=True)

    def order(self, client_order_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/v1/account/orders/{client_order_id}", signed=True)

    def cancel_order(self, client_order_id: str) -> Any:
        return self._request("DELETE", "/v1/account/orders", params={"clientOrderId": client_order_id}, signed=True)

    def open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        res = self._request("GET", "/v1/account/openOrders", params={"symbol": symbol} if symbol else None, signed=True)
        return res.get("orders", res) if isinstance(res, dict) else res

    def my_trades(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        res = self._request("GET", "/v1/account/trades", params={"symbol": symbol} if symbol else None, signed=True)
        return res.get("AccountLatestTrades", res) if isinstance(res, dict) else res

    def close(self):
        self._http.close()
