"""Command line entry point.

  wallex-agent check                       connectivity: Wallex public API, account (if key), AI gateway
  wallex-agent fetch-data --days 30        download candles for allowed markets into data/candles
  wallex-agent backtest [--ai off|replay|live] [--symbols BTCTMN,ETHTMN] [--start ... --end ...]
  wallex-agent run --mode shadow|live [--ai live|replay|off] [--once] [--mcp]
  wallex-agent status --mode shadow|live   print persisted portfolio state
  wallex-agent liquidate --mode live       flatten everything to TMN now (algorithmic emergency exit)
  wallex-agent report                      summarise decision logs (trades, reasons, AI usage)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from wallex_agent.config import load_settings

console = Console()


def _logging(verbose: bool):
    Path("logs").mkdir(exist_ok=True)
    handlers = [RichHandler(console=console, show_path=False, rich_tracebacks=True),
                logging.FileHandler("logs/agent.log", encoding="utf-8")]
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s %(name)s %(message)s", handlers=handlers)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def cmd_check(s, a):
    from wallex_agent.wallex.client import WallexClient, WallexError
    c = WallexClient(s.wallex_base_url, s.wallex_api_key)
    try:
        mk = c.markets()
        console.print(f"[green]Wallex public API ok[/]: {len(mk)} markets")
        t = Table("symbol", "bid", "ask", "spread%", "24h vol (TMN)", "found")
        for sym in s.markets.allowed:
            m = mk.get(sym)
            t.add_row(sym, *( [f"{m.bid:,.0f}", f"{m.ask:,.0f}", f"{m.spread_pct:.3f}", f"{m.quote_volume_24h:,.0f}", "yes"] if m else ["-", "-", "-", "-", "[red]NO[/]"]))
        console.print(t)
        payload = c.candles(s.markets.allowed[0], s.data.resolution, int(time.time()) - 86400, int(time.time()))
        console.print(f"candles {s.markets.allowed[0]}@{s.data.resolution}: status={payload.get('s')} bars={len(payload.get('t', []))}")
    except WallexError as e:
        console.print(f"[red]Wallex public API failed:[/] {e}")
    if s.wallex_api_key:
        try:
            b = c.balances(); console.print("[green]account ok[/] balances:", {k: v for k, v in b.items() if v["value"] > 0})
            console.print("fees:", c.fees())
        except WallexError as e:
            console.print(f"[red]account failed:[/] {e}")
    else:
        console.print("[yellow]WALLEX_API_KEY not set; skipping private endpoints[/]")
    if s.ai_api_key:
        from openai import OpenAI
        try:
            cl = OpenAI(base_url=s.ai_base_url, api_key=s.ai_api_key, timeout=60)
            r = cl.chat.completions.create(model=s.ai_model_fast, messages=[{"role": "user", "content": "Reply with exactly: OK"}], max_completion_tokens=20)
            console.print(f"[green]AI gateway ok[/] ({s.ai_model_fast}): {r.choices[0].message.content!r}")
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]AI gateway failed:[/] {e}")
    else:
        console.print("[yellow]AI_API_KEY not set[/]")


def cmd_fetch(s, a):
    from wallex_agent.data.market_data import CandleStore, MarketData
    from wallex_agent.wallex.client import WallexClient
    md = MarketData(WallexClient(s.wallex_base_url, s.wallex_api_key), CandleStore(s.data.candle_cache_dir))
    end = int(time.time()); start = end - a.days * 86400
    for sym in (a.symbols.split(",") if a.symbols else s.markets.allowed):
        df = md.history(sym, a.resolution or s.data.resolution, start, end)
        console.print(f"{sym}: {len(df)} bars cached -> {md.store.path(sym, a.resolution or s.data.resolution)}")


def cmd_backtest(s, a):
    from wallex_agent.backtest.runner import run_backtest
    syms = a.symbols.split(",") if a.symbols else s.markets.allowed
    r = run_backtest(s, syms, a.start, a.end, ai_mode=a.ai, resolution=a.resolution)
    t = Table("metric", "value", title=f"Backtest {r.out_dir.name} (ai={a.ai})")
    for k, v in r.metrics.items():
        t.add_row(k, json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v))
    console.print(t)
    console.print(f"artifacts: {r.out_dir}/metrics.json, equity.csv, trades.csv, backtest_decisions.jsonl")


def cmd_run(s, a):
    from wallex_agent.engine.loop import run
    if a.mode == "live" and not s.wallex_api_key:
        sys.exit("live mode needs WALLEX_API_KEY")
    if a.mode == "live" and not a.yes:
        console.print("[bold red]LIVE MODE: real orders will be placed on Wallex with the challenge account.[/]")
        if input("type LIVE to continue: ").strip() != "LIVE":
            sys.exit("aborted")
    run(s, a.mode, ai_mode=a.ai, once=a.once, mcp=a.mcp)


def cmd_status(s, a):
    from wallex_agent.portfolio.state import Portfolio
    p = Path(s.execution.state_dir) / f"portfolio_{a.mode}.json"
    if not p.exists():
        sys.exit(f"no state at {p}")
    pf = Portfolio.load(p)
    console.print_json(json.dumps(pf.summary({}), ensure_ascii=False))


def cmd_liquidate(s, a):
    from wallex_agent.engine.loop import build_agent
    client, agent, md, state_path, _ = build_agent(s, a.mode, ai_mode="off")
    markets = client.markets()
    prices = {sym: (markets[sym].mid if sym in markets else p.entry_price) for sym, p in agent.pf.positions.items()}
    agent.liquidate_all(prices, markets, reason="manual liquidation command")
    agent.pf.save(state_path)
    console.print(agent.pf.summary(prices))


def cmd_smoke_order(s, a):
    """Place ONE small real order and unwind it, printing raw exchange responses, to validate the live order path."""
    import json as _json
    from wallex_agent.execution.broker import LiveBroker
    from wallex_agent.wallex.client import WallexClient, WallexError
    if not s.wallex_api_key:
        sys.exit("needs WALLEX_API_KEY")
    sym = a.symbol
    if s.markets.is_blocked(sym):
        sys.exit(f"{sym} is on the organisers' disabled list")
    if sym not in s.markets.allowed:
        console.print(f"[yellow]{sym} is not in markets.allowed; testing exchange acceptance only[/]")
    c = WallexClient(s.wallex_base_url, s.wallex_api_key)
    m = c.markets()[sym]
    console.print(f"market {sym}: bid={m.bid:,.0f} ask={m.ask:,.0f} qty_precision={m.qty_precision} price_precision={m.price_precision} min_qty={m.min_qty} min_notional={m.min_notional}")
    console.print("raw market info:", _json.dumps({k: v for k, v in m.raw.items() if k != "stats"}, ensure_ascii=False)[:600])
    if not a.yes:
        if input(f"place a REAL {a.notional:,.0f} TMN {sym} buy+sell round trip (style={a.style})? type YES: ").strip() != "YES":
            sys.exit("aborted")
    br = LiveBroker(c, s.risk.fee_pct, "LIMIT", s.execution.limit_offset_pct, a.timeout, style=a.style)
    t0 = time.time()
    buy = br.buy(sym, a.notional, m, m.ask)
    console.print(f"[bold]BUY[/] ok={buy.ok} status={buy.status} qty={buy.qty} price={buy.price} notional={buy.notional_tmn:,.0f} err={buy.error!r} ({time.time()-t0:.1f}s)")
    console.print("raw order response:", _json.dumps(buy.raw, ensure_ascii=False, default=str)[:1200])
    try:
        console.print("order lookup:", _json.dumps(c.order(buy.order_id), ensure_ascii=False, default=str)[:1200])
    except WallexError as e:
        console.print(f"order lookup failed: {e}")
    if buy.ok and buy.qty > 0:
        m = c.markets()[sym]
        t1 = time.time()
        sell = br.sell(sym, buy.qty, m, m.bid)
        console.print(f"[bold]SELL[/] ok={sell.ok} status={sell.status} qty={sell.qty} price={sell.price} notional={sell.notional_tmn:,.0f} err={sell.error!r} ({time.time()-t1:.1f}s)")
        console.print("raw sell response:", _json.dumps(sell.raw, ensure_ascii=False, default=str)[:1200])
        console.print(f"round-trip pnl incl. fees ≈ {sell.notional_tmn - sell.fee_tmn - buy.notional_tmn - buy.fee_tmn:,.0f} TMN")
    console.print("balances:", {k: v for k, v in c.balances().items() if v["value"] > 0})
    console.print("open orders:", c.open_orders(sym))


def cmd_sell_balance(s, a):
    """Sell the free exchange balance of a coin back to TMN (cleanup after smoke tests / manual recovery)."""
    from wallex_agent.execution.broker import LiveBroker
    from wallex_agent.wallex.client import WallexClient
    c = WallexClient(s.wallex_base_url, s.wallex_api_key)
    sym = a.symbol
    m = c.markets()[sym]
    free = c.balances().get(m.base, {}).get("free", 0.0)
    console.print(f"{m.base} free balance: {free} (~{free * m.bid:,.0f} TMN)")
    if free * m.bid < m.min_notional:
        sys.exit("below exchange minimum notional; nothing to do")
    if not a.yes and input(f"sell {free} {m.base} to TMN now? type YES: ").strip() != "YES":
        sys.exit("aborted")
    br = LiveBroker(c, s.risk.fee_pct, "LIMIT", s.execution.limit_offset_pct, a.timeout, style="taker")
    fill = br.sell(sym, free, m, m.bid)
    console.print(f"SELL ok={fill.ok} status={fill.status} qty={fill.qty} price={fill.price} notional={fill.notional_tmn:,.0f} fee_tmn={fill.fee_tmn:,.0f} err={fill.error!r}")
    console.print("balances:", {k: v for k, v in c.balances().items() if v["value"] > 0})


def cmd_serve(s, a):
    """Run the trading loop AND the web dashboard in one process (what the Liara container runs)."""
    import os
    from wallex_agent.dashboard.server import start_dashboard
    from wallex_agent.engine.loop import run
    mode = a.mode or (s.run_mode if s.run_mode in ("shadow", "live") else "shadow")
    port = int(os.getenv("PORT", a.port))
    token = os.getenv("DASHBOARD_TOKEN", "")
    start_dashboard(s.execution.state_dir, s.execution.decision_log_dir, mode, port, token)
    if mode == "live" and not s.wallex_api_key:
        sys.exit("live mode needs WALLEX_API_KEY")
    console.print(f"[bold]{'LIVE' if mode == 'live' else 'SHADOW'}[/] agent + dashboard on port {port}")
    run(s, mode, ai_mode=a.ai, mcp=a.mcp)


def cmd_dashboard(s, a):
    """Dashboard only (read-only view of an agent running elsewhere on the same disk)."""
    import os
    from wallex_agent.dashboard.server import start_dashboard
    port = int(os.getenv("PORT", a.port))
    start_dashboard(s.execution.state_dir, s.execution.decision_log_dir, a.mode, port, os.getenv("DASHBOARD_TOKEN", ""))
    console.print(f"dashboard on http://localhost:{port}")
    while True:
        time.sleep(3600)


def cmd_report(s, a):
    from wallex_agent.backtest.report import summarize_logs
    summarize_logs(Path(a.path or s.execution.decision_log_dir), console)


def main(argv=None):
    p = argparse.ArgumentParser(prog="wallex-agent", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None); p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    f = sub.add_parser("fetch-data"); f.add_argument("--days", type=int, default=30); f.add_argument("--symbols"); f.add_argument("--resolution"); f.set_defaults(fn=cmd_fetch)
    b = sub.add_parser("backtest"); b.add_argument("--ai", choices=["off", "replay", "live"], default="off"); b.add_argument("--symbols"); b.add_argument("--start"); b.add_argument("--end"); b.add_argument("--resolution"); b.set_defaults(fn=cmd_backtest)
    r = sub.add_parser("run"); r.add_argument("--mode", choices=["shadow", "live"], default=None); r.add_argument("--ai", choices=["live", "replay", "off"], default="live"); r.add_argument("--once", action="store_true"); r.add_argument("--mcp", action="store_true"); r.add_argument("-y", "--yes", action="store_true"); r.set_defaults(fn=cmd_run)
    st = sub.add_parser("status"); st.add_argument("--mode", default="shadow"); st.set_defaults(fn=cmd_status)
    lq = sub.add_parser("liquidate"); lq.add_argument("--mode", default="live"); lq.set_defaults(fn=cmd_liquidate)
    rp = sub.add_parser("report"); rp.add_argument("--path"); rp.set_defaults(fn=cmd_report)
    sv = sub.add_parser("serve", help="agent loop + web dashboard (for Liara/Docker)"); sv.add_argument("--mode", choices=["shadow", "live"], default=None)
    sv.add_argument("--ai", choices=["live", "replay", "off"], default="live"); sv.add_argument("--mcp", action="store_true"); sv.add_argument("--port", type=int, default=8000); sv.set_defaults(fn=cmd_serve)
    db = sub.add_parser("dashboard", help="web dashboard only"); db.add_argument("--mode", default="live"); db.add_argument("--port", type=int, default=8000); db.set_defaults(fn=cmd_dashboard)
    so = sub.add_parser("smoke-order", help="one tiny real buy+sell to validate the live order path")
    so.add_argument("--symbol", default="USDTTMN"); so.add_argument("--notional", type=float, default=300000); so.add_argument("--style", choices=["maker", "taker"], default="taker")
    so.add_argument("--timeout", type=int, default=60); so.add_argument("-y", "--yes", action="store_true"); so.set_defaults(fn=cmd_smoke_order)
    sb = sub.add_parser("sell-balance", help="sell a coin's free balance back to TMN"); sb.add_argument("--symbol", required=True)
    sb.add_argument("--timeout", type=int, default=60); sb.add_argument("-y", "--yes", action="store_true"); sb.set_defaults(fn=cmd_sell_balance)
    a = p.parse_args(argv)
    _logging(a.verbose)
    s = load_settings(a.config)
    if a.cmd == "run" and a.mode is None:
        a.mode = s.run_mode if s.run_mode in ("shadow", "live") else "shadow"
    a.fn(s, a)


if __name__ == "__main__":
    main()
