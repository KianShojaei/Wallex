"""Same sweep but on USD-denominated series (TMN price / USDTTMN) with USDT-market fees (0.13% taker -> 0.26% RT)."""
import sys, itertools; import pathlib; ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import pandas as pd, numpy as np
from research_sweep import RAW, SYMS, resample, simulate, mr_signals, trend_signals, summarize
U1 = RAW["USDTTMN"]
def usd_frame(sym, tf):
    d = resample(RAW[sym], tf); u = resample(U1, tf)["close"].reindex(d.index).ffill()
    out = d.copy()
    for c in ["open","high","low","close"]: out[c] = d[c]/u
    return out.dropna()
SY = [s for s in SYMS if s!="USDTTMN"]
split = pd.Timestamp("2026-10-12", tz="UTC")
rows=[]
for FEE in [0.26, 0.20]:
  for tf in [15, 30, 60, 240]:
    data = {s: usd_frame(s, tf) for s in SY}
    for rsi_th, bb_th, sl, tp, hold_h in itertools.product([25,30,35],[0.05,0.15],[0.7,1.0,1.5,2.5],[0.8,1.2,1.8,2.5],[6,12,24,48]):
        rec=dict(fee=FEE,strat="MR",tf=tf,rsi=rsi_th,bb=bb_th,sl=sl,tp=tp,hold=hold_h)
        for part in ["train","test"]:
            tr=[]
            for s in SY:
                df=data[s]; ent,exr=mr_signals(df,14,rsi_th,20,bb_th,True); m=(df.index<split) if part=="train" else (df.index>=split)
                tr+=simulate(df[m],ent[m],exr[m],sl,tp,int(hold_h*60/tf),FEE,trail_pct=0.6)
            sm=summarize(tr); rec[f"net_{part}"]=sm["net"]; rec[f"n_{part}"]=sm["n"]; rec[f"win_{part}"]=sm["win"]; rec[f"pf_{part}"]=sm["pf"]
        rows.append(rec)
    for fast, slow, brk, sl, tp, hold_h in itertools.product([10,20],[50,100],[24,48],[1.5,2.5,4.0],[2.0,3.0,5.0,8.0],[24,72]):
        rec=dict(fee=FEE,strat="TREND",tf=tf,fast=fast,slow=slow,brk=brk,sl=sl,tp=tp,hold=hold_h)
        for part in ["train","test"]:
            tr=[]
            for s in SY:
                df=data[s]; ent,exr=trend_signals(df,fast,slow,brk); m=(df.index<split) if part=="train" else (df.index>=split)
                tr+=simulate(df[m],ent[m],exr[m],sl,tp,int(hold_h*60/tf),FEE)
            sm=summarize(tr); rec[f"net_{part}"]=sm["net"]; rec[f"n_{part}"]=sm["n"]; rec[f"win_{part}"]=sm["win"]; rec[f"pf_{part}"]=sm["pf"]
        rows.append(rec)
R=pd.DataFrame(rows); R.to_csv(ROOT / "logs/research_sweep_usd.csv", index=False)
pd.set_option("display.width",250); pd.set_option("display.max_columns",30)
for fee in [0.26,0.20]:
  for strat in ["MR","TREND"]:
    P=R[(R.strat==strat)&(R.fee==fee)]
    rob=P[(P.net_train>0)&(P.net_test>0)&(P.n_train>=8)&(P.n_test>=4)]
    print(f"\n##### {strat} fee {fee}: robust {len(rob)}/{len(P)} ; frac pos test {(P.net_test>0).mean():.2f}")
    print(P.groupby("tf").agg(net_train=("net_train","mean"),net_test=("net_test","mean"),pos_test=("net_test",lambda x:(x>0).mean()),n_test=("n_test","mean")).round(2))
    print(rob.sort_values("net_test",ascending=False).head(12).drop(columns=["fee","strat"]).to_string(index=False))
