import streamlit as st
import pandas as pd
import numpy as np
import requests

BASE = "https://api.coindcx.com"

st.set_page_config(page_title="CoinDCX AI Analyst v5", page_icon="📊", layout="wide")
st.title("📊 CoinDCX AI Market Analyst v5")
st.caption("Analysis only • Public CoinDCX market data • No trading • No account access")

@st.cache_data(ttl=60)
def get_markets():
    r = requests.get(f"{BASE}/exchange/v1/markets_details", timeout=20)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=10)
def get_ticker():
    r = requests.get(f"{BASE}/exchange/ticker", timeout=20)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=30)
def get_candles(pair, interval):
    r = requests.get(f"{BASE}/market_data/candles",
                     params={"pair": pair, "interval": interval, "limit": 1000},
                     timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Candle API HTTP {r.status_code}: {r.text[:500]}")
    data = r.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected candle response: {data}")
    d = pd.DataFrame(data)
    if d.empty:
        return d
    for c in ["open","high","low","close","volume"]:
        if c not in d.columns:
            raise RuntimeError(f"Missing candle field: {c}")
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["time"] = pd.to_datetime(d["time"], unit="ms", errors="coerce")
    return d.dropna(subset=["time","open","high","low","close","volume"]).sort_values("time").drop_duplicates("time").reset_index(drop=True)

@st.cache_data(ttl=5)
def get_orderbook(pair, depth=100):
    r = requests.get(f"{BASE}/market_data/orderbook",
                     params={"pair": pair, "depth": depth}, timeout=15)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=5)
def get_trades(pair, limit=200):
    r = requests.get(f"{BASE}/market_data/trade_history",
                     params={"pair": pair, "limit": limit}, timeout=15)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []

def resolve_market(text, markets):
    q = text.upper().strip().replace("/","").replace("-","").replace("_","")
    exact, coin = [], []
    for m in markets:
        if str(m.get("status","")).lower() != "active":
            continue
        symbol = str(m.get("symbol",m.get("coindcx_name",""))).upper()
        name = str(m.get("coindcx_name","")).upper()
        base = str(m.get("base_currency_short_name","")).upper()
        target = str(m.get("target_currency_short_name","")).upper()
        if q in (symbol,name) or target+base == q:
            exact.append(m)
        elif q == target:
            coin.append(m)
    choices = exact or coin
    if not choices:
        return None
    choices.sort(key=lambda m: (
        str(m.get("base_currency_short_name","")).upper() not in ("INR","USDT"),
        str(m.get("base_currency_short_name","")).upper() != "INR"
    ))
    return choices[0]

def indicators(d):
    x=d.copy()
    x["ema20"]=x.close.ewm(span=20,adjust=False).mean()
    x["ema50"]=x.close.ewm(span=50,adjust=False).mean()
    x["ema200"]=x.close.ewm(span=200,adjust=False).mean()
    delta=x.close.diff()
    gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean()
    loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    rs=gain/loss.replace(0,np.nan)
    x["rsi"]=100-100/(1+rs)
    e12=x.close.ewm(span=12,adjust=False).mean()
    e26=x.close.ewm(span=26,adjust=False).mean()
    x["macd"]=e12-e26
    x["macd_signal"]=x.macd.ewm(span=9,adjust=False).mean()
    tr=pd.concat([x.high-x.low,(x.high-x.close.shift()).abs(),(x.low-x.close.shift()).abs()],axis=1).max(axis=1)
    x["atr"]=tr.ewm(alpha=1/14,adjust=False).mean()
    x["vol_ma20"]=x.volume.rolling(20).mean()
    return x

def structure(d):
    if len(d)<40: return "Insufficient data"
    r=d.tail(10); p=d.iloc[-30:-10]
    hh=r.high.max()>p.high.max(); hl=r.low.min()>p.low.min()
    lh=r.high.max()<p.high.max(); ll=r.low.min()<p.low.min()
    if hh and hl: return "Bullish: higher highs + higher lows"
    if lh and ll: return "Bearish: lower highs + lower lows"
    return "Mixed / range"

def tf(d):
    x=d.iloc[-1]
    trend="Bullish" if x.ema20>x.ema50>x.ema200 else ("Bearish" if x.ema20<x.ema50<x.ema200 else "Mixed")
    if x.rsi>=80: mom="Extreme overbought"
    elif x.rsi>=70: mom="Overbought"
    elif x.rsi<=20: mom="Extreme oversold"
    elif x.rsi<=30: mom="Oversold"
    elif x.rsi>=55: mom="Bullish"
    elif x.rsi<=45: mom="Bearish"
    else: mom="Neutral"
    return {"trend":trend,"structure":structure(d),"momentum":mom,"rsi":float(x.rsi),
            "macd":"Bullish" if x.macd>x.macd_signal else "Bearish",
            "volume":"Above average" if x.volume>x.vol_ma20 else "Below average",
            "close":float(x.close),"atr":float(x.atr)}

def microstructure(book,trades):
    bids=book.get("bids",{}) if isinstance(book,dict) else {}
    asks=book.get("asks",{}) if isinstance(book,dict) else {}
    B=sorted([(float(p),float(q)) for p,q in bids.items()],reverse=True)
    A=sorted([(float(p),float(q)) for p,q in asks.items()])
    if not B or not A: ob=None
    else:
        best_bid=B[0][0]; best_ask=A[0][0]
        mid=(best_bid+best_ask)/2
        # Compare liquidity close to the midpoint, reducing distortion from distant walls.
        near_b=sum(q for p,q in B if p>=mid*0.995)
        near_a=sum(q for p,q in A if p<=mid*1.005)
        total=near_b+near_a
        imb=(near_b-near_a)/total if total else 0
        ob={"bid":near_b,"ask":near_a,"imbalance":imb,"spread":best_ask-best_bid,
            "pressure":"Buy-side liquidity" if imb>0.15 else "Sell-side liquidity" if imb<-0.15 else "Balanced liquidity"}
    buy=sell=0.0
    for t in trades or []:
        q=float(t.get("q",0) or 0)
        # CoinDCX defines m as whether the buyer is market maker.
        # We report this as maker-side proxy, not a guaranteed aggressor field.
        if bool(t.get("m",False)): sell+=q
        else: buy+=q
    total=buy+sell
    ratio=(buy-sell)/total if total else 0
    flow={"buy":buy,"sell":sell,"ratio":ratio,
          "label":"Maker-side proxy favors buying" if ratio>0.15 else "Maker-side proxy favors selling" if ratio<-0.15 else "Flow balanced"}
    return ob,flow

def decision(a15,a1h,a1d,ob,flow):
    score=0
    for a,w in [(a1d,3),(a1h,2),(a15,1)]:
        score += w if a["trend"]=="Bullish" else -w if a["trend"]=="Bearish" else 0
    score += 1 if a1d["macd"]=="Bullish" else -1
    score += 1 if a1h["macd"]=="Bullish" else -1
    if ob: score += 1 if ob["imbalance"]>0.15 else -1 if ob["imbalance"]<-0.15 else 0
    if flow: score += 1 if flow["ratio"]>0.15 else -1 if flow["ratio"]<-0.15 else 0

    long_align=all(a["trend"]=="Bullish" for a in [a1d,a1h,a15])
    short_align=all(a["trend"]=="Bearish" for a in [a1d,a1h,a15])
    if long_align and a1d["rsi"]<78: verdict="LONG SETUP"
    elif short_align and a1d["rsi"]>22: verdict="SHORT SETUP"
    elif score>=5: verdict="BULLISH BIAS — WAIT FOR ENTRY"
    elif score<=-5: verdict="BEARISH BIAS — WAIT FOR ENTRY"
    else: verdict="WAIT — NO CLEAR EDGE"

    confidence=min(95,50+abs(score)*5+(12 if long_align or short_align else 0))
    if a1d["rsi"]>=80 or a1d["rsi"]<=20: confidence=max(35,confidence-10)
    reasons=[]
    if a1d["trend"]!=a1h["trend"]: reasons.append("Daily and 1H trend disagree.")
    if a15["trend"]!=a1h["trend"]: reasons.append("15m is not aligned with the 1H trend.")
    if a1d["rsi"]>=80: reasons.append("Daily RSI is extremely overbought; avoid chasing longs.")
    elif a1d["rsi"]<=20: reasons.append("Daily RSI is extremely oversold; avoid chasing shorts.")
    if ob: reasons.append(ob["pressure"]+".")
    if flow: reasons.append(flow["label"]+".")

    # Explicit triggers: use 20-bar high/low from the 1H structure.
    # These are levels to watch, not automatic entries.
    return verdict,confidence,score,reasons

def fmt(v):
    return "—" if pd.isna(v) else f"{v:,.8f}".rstrip("0").rstrip(".")

coin=st.text_input("CoinDCX coin or pair","BTC").strip()

if st.button("🔎 Analyze",type="primary"):
    try:
        market=resolve_market(coin,get_markets())
        if not market:
            st.error("Market not found. Try BTC, ETH, SOL, or BTCINR.")
            st.stop()
        pair=market["pair"]; symbol=market.get("symbol",market.get("coindcx_name",pair))
        st.success(f"Using **{symbol}** | Internal pair **{pair}** | {market.get('target_currency_short_name','')}/{market.get('base_currency_short_name','')}")

        try:
            ticks=get_ticker()
            t=next((z for z in ticks if str(z.get("market","")).upper()==str(symbol).upper()),None)
            if t:
                a,b,c,d=st.columns(4)
                a.metric("Last price",fmt(float(t.get("last_price",0))))
                b.metric("24h change",f'{float(t.get("change_24_hour",0)):.2f}%')
                c.metric("24h high",fmt(float(t.get("high",0))))
                d.metric("24h volume",fmt(float(t.get("volume",0))))
        except Exception as ex: st.warning(f"Ticker unavailable: {ex}")

        raw={}
        for k in ["15m","1h","1d"]:
            raw[k]=get_candles(pair,k)
            if len(raw[k])<210:
                st.error(f"{k}: only {len(raw[k])} candles returned; 210+ are required.")
                st.stop()
            raw[k]=indicators(raw[k])

        a15,a1h,a1d=tf(raw["15m"]),tf(raw["1h"]),tf(raw["1d"])
        ob,flow=None,None
        try: ob,flow=microstructure(get_orderbook(pair,100),get_trades(pair,200))
        except Exception as ex: st.warning(f"Microstructure data unavailable: {ex}")

        verdict,confidence,score,reasons=decision(a15,a1h,a1d,ob,flow)

        # Trigger levels based on the latest 1H candle and recent range.
        h=raw["1h"]
        trigger_up=float(h.tail(20).high.max())
        trigger_down=float(h.tail(20).low.min())
        atr=float(raw["15m"].iloc[-1].atr)

        st.header("🧠 AI-style Market Decision")
        a,b,c=st.columns(3)
        a.metric("Overall view",verdict)
        b.metric("Confidence",f"{confidence:.0f}%")
        c.metric("Composite score",score)

        if reasons:
            st.write("**Reasoning:**")
            for r in reasons: st.write("- "+r)

        st.subheader("🎯 What we're waiting for")
        a,b=st.columns(2)
        a.metric("Bullish trigger to watch",fmt(trigger_up))
        b.metric("Bearish trigger to watch",fmt(trigger_down))
        st.caption("Triggers are observation levels derived from the recent 1H range; they are not automatic buy/sell orders.")

        st.subheader("Multi-timeframe dashboard")
        rows=[]
        for label,x in [("1D",a1d),("1H",a1h),("15m",a15)]:
            rows.append({"Timeframe":label,"Trend":x["trend"],"Structure":x["structure"],"Momentum":x["momentum"],"RSI":round(x["rsi"],1),"MACD":x["macd"],"Volume":x["volume"]})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

        if ob:
            st.subheader("📚 Order-book intelligence")
            a,b,c,d=st.columns(4)
            a.metric("Near-price bids",fmt(ob["bid"])); b.metric("Near-price asks",fmt(ob["ask"]))
            c.metric("Near-price imbalance",f'{ob["imbalance"]*100:.2f}%'); d.metric("Spread",fmt(ob["spread"]))
            st.info(ob["pressure"]+" (within ±0.5% of mid-price)")

        if flow:
            st.subheader("⚡ Recent trade-flow proxy")
            a,b,c=st.columns(3)
            a.metric("Non-maker-side volume",fmt(flow["buy"]))
            b.metric("Maker-side volume",fmt(flow["sell"]))
            c.metric("Flow proxy",f'{flow["ratio"]*100:.2f}%')
            st.info(flow["label"]+"; this is an interpretation of CoinDCX's `m` field, not a direct aggressor field.")

        st.subheader("Trade plan")
        if verdict=="LONG SETUP":
            entry=float(raw["15m"].iloc[-1].close); sl=entry-1.2*atr; risk=entry-sl
            tp1=entry+1.5*risk; tp2=entry+2.5*risk
        elif verdict=="SHORT SETUP":
            entry=float(raw["15m"].iloc[-1].close); sl=entry+1.2*atr; risk=sl-entry
            tp1=entry-1.5*risk; tp2=entry-2.5*risk
        else: entry=sl=tp1=tp2=np.nan

        if pd.notna(entry):
            a,b,c,d=st.columns(4)
            a.metric("Entry reference",fmt(entry)); b.metric("Invalidation / SL",fmt(sl))
            c.metric("TP1",fmt(tp1)); d.metric("TP2",fmt(tp2))
            st.warning("Analytical setup only. No order is created.")
        else:
            st.info("No high-confidence entry. WAIT until the trigger and timeframe confirmation improve.")

        for title,key in [("15 Minute","15m"),("1 Hour","1h"),("1 Day","1d")]:
            with st.expander(f"{title} details",expanded=(key=="1h")):
                d=raw[key]; x={"15m":a15,"1h":a1h,"1d":a1d}[key]
                a,b,c,e=st.columns(4)
                a.metric("Close",fmt(x["close"])); b.metric("RSI",f'{x["rsi"]:.1f}')
                c.metric("ATR",fmt(x["atr"])); e.metric("Structure",x["structure"])
                st.line_chart(d.set_index("time")[["close","ema20","ema50","ema200"]].tail(250))
    except Exception as ex:
        st.error(f"Analysis failed: {ex}")

st.divider()
st.caption("Educational analysis only. Public CoinDCX data. No API keys, orders, balances, withdrawals, or account access.")
