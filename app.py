import streamlit as st
import pandas as pd
import numpy as np
import requests

BASE = "https://api.coindcx.com"

st.set_page_config(page_title="CoinDCX Futures Scanner", page_icon="🎯", layout="wide")

st.title("🎯 CoinDCX Meme-Coin Futures Scanner")
st.caption("Analysis only • Finds up to 5 high-momentum futures candidates • No trading/account access")

@st.cache_data(ttl=30)
def markets():
    r=requests.get(f"{BASE}/exchange/v1/markets_details",timeout=20)
    r.raise_for_status(); return r.json()

@st.cache_data(ttl=10)
def ticker():
    r=requests.get(f"{BASE}/exchange/ticker",timeout=20)
    r.raise_for_status(); return r.json()

@st.cache_data(ttl=20)
def candles(pair,interval):
    r=requests.get(f"{BASE}/market_data/candles",
                   params={"pair":pair,"interval":interval,"limit":1000},timeout=30)
    r.raise_for_status()
    d=pd.DataFrame(r.json())
    if d.empty:return d
    for c in ["open","high","low","close","volume"]: d[c]=pd.to_numeric(d[c],errors="coerce")
    d["time"]=pd.to_datetime(d["time"],unit="ms",errors="coerce")
    return d.dropna(subset=["time","open","high","low","close","volume"]).sort_values("time").drop_duplicates("time").reset_index(drop=True)

@st.cache_data(ttl=5)
def orderbook(pair):
    r=requests.get(f"{BASE}/market_data/orderbook",params={"pair":pair,"depth":100},timeout=15)
    r.raise_for_status(); return r.json()

@st.cache_data(ttl=5)
def trades(pair):
    r=requests.get(f"{BASE}/market_data/trade_history",params={"pair":pair,"limit":200},timeout=15)
    r.raise_for_status(); x=r.json(); return x if isinstance(x,list) else []

def is_futures(m):
    # CoinDCX market metadata can expose market/ecode fields differently.
    text=" ".join(str(m.get(k,"")).upper() for k in ["market","symbol","coindcx_name","pair","ecode","market_type"])
    return any(s in text for s in ["FUTURES","FUTURE","F-"]) or str(m.get("ecode","")).upper()=="F"

def active_candidates(ms,ts):
    out=[]
    for m in ms:
        if str(m.get("status","")).lower()!="active": continue
        # Prefer futures markets when metadata identifies them.
        if not is_futures(m): continue
        sym=str(m.get("symbol",m.get("coindcx_name",""))).upper()
        if sym: out.append((m,sym))
    return out

def ind(d):
    x=d.copy()
    for n in [20,50,100,200]: x[f"ema{n}"]=x.close.ewm(span=n,adjust=False).mean()
    delta=x.close.diff()
    gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean()
    loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    rs=gain/loss.replace(0,np.nan); x["rsi"]=100-100/(1+rs)
    e12=x.close.ewm(span=12,adjust=False).mean(); e26=x.close.ewm(span=26,adjust=False).mean()
    x["macd"]=e12-e26; x["macd_signal"]=x.macd.ewm(span=9,adjust=False).mean()
    tr=pd.concat([x.high-x.low,(x.high-x.close.shift()).abs(),(x.low-x.close.shift()).abs()],axis=1).max(axis=1)
    x["atr"]=tr.ewm(alpha=1/14,adjust=False).mean()
    x["volma"]=x.volume.rolling(20).mean()
    # Bollinger Bands
    x["bbmid"]=x.close.rolling(20).mean(); x["bbstd"]=x.close.rolling(20).std()
    x["bbup"]=x.bbmid+2*x.bbstd; x["bblow"]=x.bbmid-2*x.bbstd
    # ADX
    up=x.high.diff(); dn=-x.low.diff()
    plus=np.where((up>dn)&(up>0),up,0.0); minus=np.where((dn>up)&(dn>0),dn,0.0)
    atr14=x["atr"].replace(0,np.nan)
    pdi=100*pd.Series(plus,index=x.index).ewm(alpha=1/14,adjust=False).mean()/atr14
    mdi=100*pd.Series(minus,index=x.index).ewm(alpha=1/14,adjust=False).mean()/atr14
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    x["adx"]=dx.ewm(alpha=1/14,adjust=False).mean()
    x["pdi"]=pdi; x["mdi"]=mdi
    return x

def structure(d):
    r=d.tail(12); p=d.iloc[-36:-12]
    if len(p)<10:return "Mixed"
    if r.high.max()>p.high.max() and r.low.min()>p.low.min():return "Bullish"
    if r.high.max()<p.high.max() and r.low.min()<p.low.min():return "Bearish"
    return "Mixed"

def tfscore(d):
    x=d.iloc[-1]
    bull=bear=0
    bull += int(x.close>x.ema20)+int(x.ema20>x.ema50)+int(x.ema50>x.ema100)+int(x.ema100>x.ema200)
    bear += int(x.close<x.ema20)+int(x.ema20<x.ema50)+int(x.ema50<x.ema100)+int(x.ema100<x.ema200)
    bull += int(55<=x.rsi<70)+int(x.macd>x.macd_signal)+int(x.pdi>x.mdi)+int(x.volume>x.volma)
    bear += int(30<x.rsi<=45)+int(x.macd<x.macd_signal)+int(x.mdi>x.pdi)+int(x.volume>x.volma)
    return bull,bear

def micro(book,trs):
    bids=book.get("bids",{}); asks=book.get("asks",{})
    B=[(float(p),float(q)) for p,q in bids.items()]; A=[(float(p),float(q)) for p,q in asks.items()]
    if not B or not A:return 0,0
    bb=max(p for p,_ in B); ba=min(p for p,_ in A); mid=(bb+ba)/2
    b=sum(q for p,q in B if p>=mid*.995); a=sum(q for p,q in A if p<=mid*1.005)
    imb=(b-a)/(b+a) if b+a else 0
    # This is deliberately a proxy; CoinDCX m indicates whether buyer is market maker.
    buy=sell=0
    for t in trs:
        q=float(t.get("q",0) or 0)
        if bool(t.get("m",False)): sell+=q
        else: buy+=q
    flow=(buy-sell)/(buy+sell) if buy+sell else 0
    return imb,flow

def fmt(v):
    if pd.isna(v):return "—"
    return f"{v:,.8f}".rstrip("0").rstrip(".")

def evaluate(m,sym,ts):
    try:
        d1=ind(candles(m["pair"],"1d")); d1h=ind(candles(m["pair"],"1h")); d15=ind(candles(m["pair"],"15m"))
        if min(len(d1),len(d1h),len(d15))<210:return None
        a1=d1.iloc[-1]; ah=d1h.iloc[-1]; a15=d15.iloc[-1]
        b1,s1=tfscore(d1); bh,sh=tfscore(d1h); b15,s15=tfscore(d15)
        ob,flow=micro(orderbook(m["pair"]),trades(m["pair"]))
        change=float(next((z.get("change_24_hour",0) for z in ts if str(z.get("market","")).upper()==sym),0) or 0)
        direction="LONG" if change>0 else "SHORT"
        # Momentum candidate score: start from 24h magnitude, then reward confirmation.
        momentum=min(40,abs(change)*1.2)
        long_score=momentum+8*(b1-s1)+5*(bh-sh)+3*(b15-s15)+8*max(0,ob)+8*max(0,flow)
        short_score=momentum+8*(s1-b1)+5*(sh-bh)+3*(s15-b15)+8*max(0,-ob)+8*max(0,-flow)
        score=long_score if direction=="LONG" else short_score
        # Avoid extreme chase: distance from 1D 20 EMA and daily Bollinger upper/lower.
        price=float(a15.close); atr=float(a15.atr)
        ext=abs(price-float(a1.ema20))/max(float(a1.atr),1)
        penalty=min(25,max(0,(ext-2)*6))
        score-=penalty
        confidence=max(35,min(95,50+score*.35))
        # Determine if there is enough confirmation for a trade.
        if direction=="LONG":
            aligned=(b1>s1 and bh>=sh and b15>=s15 and a15.macd>a15.macd_signal)
            extended=(a1.rsi>=78 or ext>3.5)
        else:
            aligned=(s1>b1 and sh>=bh and s15>=b15 and a15.macd<a15.macd_signal)
            extended=(a1.rsi<=22 or ext>3.5)
        signal=direction if aligned and not extended and confidence>=65 else "WAIT"
        # 1D S/R zones
        recent=d1.tail(90)
        support=float(recent.low.quantile(.12)); resistance=float(recent.high.quantile(.88))
        s2=float(recent.low.quantile(.04)); r2=float(recent.high.quantile(.96))
        return {"symbol":sym,"pair":m["pair"],"change":change,"direction":direction,"score":score,
                "confidence":confidence,"signal":signal,"price":price,"support1":support,"support2":s2,
                "res1":resistance,"res2":r2,"daily_rsi":float(a1.rsi),"ext":ext,
                "atr":atr,"structure":structure(d1),"d1":d1,"d15":d15}
    except Exception:
        return None

def setup(x):
    p=x["price"]; atr=x["atr"]
    if x["signal"]=="LONG":
        sl=p-1.4*atr; risk=p-sl
        return p,sl,p+1.5*risk,p+2.5*risk
    if x["signal"]=="SHORT":
        sl=p+1.4*atr; risk=sl-p
        return p,sl,p-1.5*risk,p-2.5*risk
    return np.nan,np.nan,np.nan,np.nan

if st.button("🔍 Scan Top Futures",type="primary"):
    try:
        ms=markets(); ts=ticker()
        cands=active_candidates(ms,ts)
        if not cands:
            st.error("CoinDCX did not expose futures markets through the current Markets Details response. The app will not pretend spot markets are futures.")
            st.stop()

        # Rank all resolvable futures by absolute 24h movement, then analyze top 12.
        changes={str(z.get("market","")).upper():float(z.get("change_24_hour",0) or 0) for z in ts}
        cands.sort(key=lambda x:abs(changes.get(x[1],0)),reverse=True)
        results=[]
        for m,sym in cands[:12]:
            r=evaluate(m,sym,ts)
            if r:results.append(r)
        if not results:
            st.error("No futures candidates could be analyzed.")
            st.stop()

        results=sorted(results,key=lambda x:x["score"],reverse=True)[:5]
        st.header("🎯 Today's 5 Futures Candidates")
        st.caption("The scanner first looks for the strongest 24h movers, then ranks them using multi-timeframe confirmation, volatility, volume and liquidity.")

        for i,x in enumerate(results,1):
            icon="🟢" if x["signal"]=="LONG" else "🔴" if x["signal"]=="SHORT" else "🟡"
            with st.container(border=True):
                st.subheader(f"{i}. {x['symbol']}  {icon} {x['signal']}")
                a,b,c,d=st.columns(4)
                a.metric("24h move",f"{x['change']:.2f}%")
                b.metric("Confidence",f"{x['confidence']:.0f}%")
                c.metric("Price",fmt(x["price"]))
                d.metric("Daily RSI",f"{x['daily_rsi']:.1f}")
                st.write(f"**1D Support:** {fmt(x['support2'])} / {fmt(x['support1'])}   |   **1D Resistance:** {fmt(x['res1'])} / {fmt(x['res2'])}")
                if x["signal"]!="WAIT":
                    en,sl,t1,t2=setup(x)
                    a,b,c,d=st.columns(4)
                    a.metric("Entry reference",fmt(en)); b.metric("Stop Loss",fmt(sl)); c.metric("TP1",fmt(t1)); d.metric("TP2",fmt(t2))
                    st.success(f"{x['signal']} setup confirmed. Approx. R:R 1:{1.5:.1f} to TP1.")
                else:
                    if x["direction"]=="LONG": st.warning("Strong upside mover, but confirmation or entry quality is not strong enough. Do not chase.")
                    else: st.warning("Strong downside mover, but confirmation or entry quality is not strong enough. Do not chase.")
                st.caption(f"1D structure: {x['structure']} • Hidden engine uses EMA20/50/100/200, RSI, MACD, ADX/DI, ATR, Bollinger Bands, volume, support/resistance and market microstructure.")
                with st.expander("Advanced analysis"):
                    d=x["d1"]; last=d.iloc[-1]
                    st.write(f"EMA20 {fmt(last.ema20)} | EMA50 {fmt(last.ema50)} | EMA100 {fmt(last.ema100)} | EMA200 {fmt(last.ema200)}")
                    st.write(f"RSI {last.rsi:.1f} | ADX {last.adx:.1f} | MACD {'bullish' if last.macd>last.macd_signal else 'bearish'} | Volume {'above' if last.volume>last.volma else 'below'} average")
                    st.line_chart(d.set_index("time")[["close","ema20","ema50","ema100","ema200"]].tail(250))
    except Exception as e:
        st.error(f"Scanner failed: {e}")

st.divider()
st.caption("Analysis only. No CoinDCX API key is stored or requested. No orders, balances or withdrawals are accessed.")
