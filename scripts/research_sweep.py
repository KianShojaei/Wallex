"""Fast vectorised strategy research on real Wallex 1-min data (resampled)."""
import itertools, sys, glob, json
import numpy as np, pandas as pd
import pathlib; ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from wallex_agent.data.features import ema, rsi, bollinger, atr

SYMS = ["BTCTMN","ETHTMN","USDTTMN","SOLTMN","XRPTMN","DOGETMN","TRXTMN","BNBTMN"]
RAW = {s: pd.read_csv(ROOT / "data/candles" / f"{s}_1.csv", parse_dates=["time"], index_col="time") for s in SYMS}

def resample(df, minutes):
    o = df.resample(f"{minutes}min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    return o

def simulate(df, entries, exits_rule, sl_pct, tp_pct, max_hold, fee_rt, trail_pct=None):
    """Long-only, one position at a time, fill at next bar open (conservative). Returns list of trade returns (net, %)."""
    o, h, l, c = df.open.values, df.high.values, df.low.values, df.close.values
    n = len(df); i = 0; trades = []; in_pos = False
    ent = entries.values; exr = exits_rule.values if exits_rule is not None else np.zeros(n, bool)
    while i < n - 1:
        if not in_pos:
            if ent[i]:
                ep = o[i+1]; ei = i+1; hi = ep; in_pos = True; i += 1
                continue
            i += 1; continue
        # in position at bar i (i >= ei)
        hi = max(hi, h[i])
        stop = ep*(1-sl_pct/100); tp = ep*(1+tp_pct/100)
        if l[i] <= stop: xp = stop; reason="sl"
        elif h[i] >= tp: xp = tp; reason="tp"
        elif trail_pct and (hi/ep-1)*100 >= tp_pct/2 and l[i] <= hi*(1-trail_pct/100): xp = hi*(1-trail_pct/100); reason="trail"
        elif i - ei >= max_hold: xp = c[i]; reason="time"
        elif exr[i] and c[i] > ep*(1+fee_rt/100): xp = c[i]; reason="rule"
        else: i += 1; continue
        trades.append(((xp/ep-1)*100 - fee_rt, reason, i-ei))
        in_pos = False; i += 1
    return trades

def mr_signals(df, rsi_n, rsi_th, bb_n, bb_pos_th, trend_filter):
    r = rsi(df.close, rsi_n); lo, mid, hi = bollinger(df.close, bb_n, 2.0)
    pos = (df.close-lo)/(hi-lo)
    ent = (r <= rsi_th) & (pos <= bb_pos_th)
    if trend_filter:
        e_t = ema(df.close, 100); ef, es = ema(df.close, 20), ema(df.close, 50)
        strong_down = (df.close < e_t) & (ef < es) & (df.close.pct_change(24)*100 < -4)
        ent &= ~strong_down
    exr = (r >= 55) & (pos >= 0.5)
    return ent.fillna(False), exr.fillna(False)

def trend_signals(df, fast, slow, brk_n):
    ef, es = ema(df.close, fast), ema(df.close, slow)
    cross_up = (ef > es) & (ef.shift(1) <= es.shift(1))
    brk = df.close > df.high.rolling(brk_n).max().shift(1)
    ent = cross_up | (brk & (ef > es))
    exr = ef < es
    return ent.fillna(False), exr.fillna(False)

def summarize(trades):
    if not trades: return dict(n=0, net=0, win=0, avg=0, pf=0)
    r = np.array([t[0] for t in trades]); w = r[r>0]; lo = r[r<=0]
    return dict(n=len(r), net=round(float(((1+r/100).prod()-1)*100),2), win=round(float((r>0).mean()*100),1), avg=round(float(r.mean()),3),
                pf=round(float(w.sum()/abs(lo.sum())),2) if len(lo) and lo.sum()!=0 else 99.0,
                reasons=dict(pd.Series([t[1] for t in trades]).value_counts()))

if __name__ == "__main__":
    FEE_RT = 0.6   # taker both sides
    split = pd.Timestamp("2026-10-12", tz="UTC")   # walk-forward: train < split <= test
    results = []
    for tf in [15, 30, 60, 240]:
        data = {s: resample(RAW[s], tf) for s in SYMS}
        bars_per_12h = int(720/tf)
        # mean reversion grid
        for rsi_th, bb_th, sl, tp, hold_h, tfilt in itertools.product([25,30,35],[0.05,0.15],[1.0,1.5,2.5],[1.5,2.5,4.0],[12,24,48],[True]):
            for part in ["train","test"]:
                tr_all=[]
                for s in SYMS:
                    df = data[s]
                    ent, exr = mr_signals(df, 14, rsi_th, 20, bb_th, tfilt)
                    m = (df.index < split) if part=="train" else (df.index >= split)
                    # need indicator warmup: slice after computing
                    d2 = df[m]; e2 = ent[m]; x2 = exr[m]
                    tr_all += simulate(d2, e2, x2, sl, tp, int(hold_h*60/tf), FEE_RT, trail_pct=0.8)
                sm = summarize(tr_all); results.append(dict(strat="MR", tf=tf, rsi=rsi_th, bb=bb_th, sl=sl, tp=tp, hold=hold_h, part=part, **sm))
        for fast, slow, brk, sl, tp, hold_h in itertools.product([10,20],[50,100],[24,48],[1.5,2.5,4.0],[3.0,5.0,8.0],[24,72]):
            for part in ["train","test"]:
                tr_all=[]
                for s in SYMS:
                    df = data[s]; ent, exr = trend_signals(df, fast, slow, brk)
                    m = (df.index < split) if part=="train" else (df.index >= split)
                    tr_all += simulate(df[m], ent[m], exr[m], sl, tp, int(hold_h*60/tf), FEE_RT, trail_pct=None)
                sm = summarize(tr_all); results.append(dict(strat="TREND", tf=tf, fast=fast, slow=slow, brk=brk, sl=sl, tp=tp, hold=hold_h, part=part, **sm))
    R = pd.DataFrame(results)
    R.to_csv(ROOT / "logs/research_sweep.csv", index=False)
    key = [c for c in ["strat","tf","rsi","bb","fast","slow","brk","sl","tp","hold"] if c in R]
    piv = R.pivot_table(index=key, columns="part", values=["net","n","win","pf"], aggfunc="first")
    piv.columns = [f"{a}_{b}" for a,b in piv.columns]
    piv = piv.reset_index()
    good = piv[(piv.n_train>=8)].sort_values("net_train", ascending=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("=== TOP 25 by TRAIN net, with TEST shown ===")
    print(good.head(25).to_string(index=False))
    print("\n=== robust: positive in BOTH train & test, sorted by test net ===")
    rob = piv[(piv.net_train>0)&(piv.net_test>0)&(piv.n_train>=8)&(piv.n_test>=3)].sort_values("net_test", ascending=False)
    print(rob.head(25).to_string(index=False))
    print("\ncount robust:", len(rob), "of", len(piv))
