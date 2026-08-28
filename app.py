import streamlit as st
import pandas as pd
import numpy as np
import requests

BASE = "https://api.coindcx.com"

st.set_page_config(page_title="CoinDCX AI Analyst", page_icon="📊", layout="wide")
st.title("📊 CoinDCX AI Market Analyst")
st.caption("Analysis only • CoinDCX public market data • No trading • No account access")

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
    r = requests.get(
        f"{BASE}/market_data/candles",
        params={"pair": pair, "interval": interval, "limit": 1000},
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"CoinDCX candle API HTTP {r.status_code}: {r.text[:500]}")
    data = r.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected candle response: {data}")
    df = pd.DataFrame(data)
    if df.empty:
        return df
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            raise RuntimeError(f"Missing candle field: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
    return df.dropna(subset=["time","open","high","low","close","volume"]).sort_values("time").drop_duplicates("time").reset_index(drop=True)

@st.cache_data(ttl=5)
def get_orderbook(pair, depth=50):
    r = requests.get(
        f"{BASE}/market_data/orderbook",
        params={"pair": pair, "depth": depth},
        timeout=15,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Order book HTTP {r.status_code}: {r.text[:500]}")
    return r.json()

@st.cache_data(ttl=5)
def get_trades(pair, limit=100):
    r = requests.get(
        f"{BASE}/market_data/trade_history",
        params={"pair": pair, "limit": limit},
        timeout=15,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Trade history HTTP {r.status_code}: {r.text[:500]}")
    data = r.json()
    return data if isinstance(data, list) else []

def resolve_market(text, markets):
    q = text.upper().strip().replace("/", "").replace("-", "").replace("_", "")
    exact, coin = [], []
    for m in markets:
        if str(m.get("status","")).lower() != "active":
            continue
        symbol = str(m.get("symbol", m.get("coindcx_name",""))).upper()
        name = str(m.get("coindcx_name","")).upper()
        base = str(m.get("base_currency_short_name","")).upper()
        target = str(m.get("target_currency_short_name","")).upper()
        if q in (symbol, name) or target + base == q:
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

def add_indicators(df):
    d = df.copy()
    d["ema20"] = d.close.ewm(span=20, adjust=False).mean()
    d["ema50"] = d.close.ewm(span=50, adjust=False).mean()
    d["ema200"] = d.close.ewm(span=200, adjust=False).mean()
    delta = d.close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi"] = 100 - 100/(1+rs)
    e12 = d.close.ewm(span=12, adjust=False).mean()
    e26 = d.close.ewm(span=26, adjust=False).mean()
    d["macd"] = e12-e26
    d["macd_signal"] = d.macd.ewm(span=9, adjust=False).mean()
    tr = pd.concat([d.high-d.low,(d.high-d.close.shift()).abs(),(d.low-d.close.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    d["vol_ma20"] = d.volume.rolling(20).mean()
    return d

def structure(d):
    if len(d) < 40:
        return "Insufficient data"
    recent = d.tail(10)
    prior = d.iloc[-30:-10]
    hh = recent.high.max() > prior.high.max()
    hl = recent.low.min() > prior.low.min()
    lh = recent.high.max() < prior.high.max()
    ll = recent.low.min() < prior.low.min()
    if hh and hl:
        return "Bullish: higher highs + higher lows"
    if lh and ll:
        return "Bearish: lower highs + lower lows"
    return "Mixed / range"

def tf_analysis(d):
    x = d.iloc[-1]
    trend = "Bullish" if x.ema20 > x.ema50 and x.ema50 > x.ema200 else ("Bearish" if x.ema20 < x.ema50 and x.ema50 < x.ema200 else "Mixed")
    if x.rsi >= 80: momentum = "Extreme overbought"
    elif x.rsi >= 70: momentum = "Overbought"
    elif x.rsi <= 20: momentum = "Extreme oversold"
    elif x.rsi <= 30: momentum = "Oversold"
    elif x.rsi >= 55: momentum = "Bullish"
    elif x.rsi <= 45: momentum = "Bearish"
    else: momentum = "Neutral"
    return {
        "trend": trend, "structure": structure(d), "momentum": momentum,
        "rsi": float(x.rsi), "macd": "Bullish" if x.macd > x.macd_signal else "Bearish",
        "volume": "Above average" if x.volume > x.vol_ma20 else "Below average",
        "close": float(x.close), "atr": float(x.atr)
    }

def orderbook_analysis(book):
    bids = book.get("bids", {}) if isinstance(book, dict) else {}
    asks = book.get("asks", {}) if isinstance(book, dict) else {}
    b = [(float(p), float(q)) for p,q in bids.items()]
    a = [(float(p), float(q)) for p,q in asks.items()]
    if not b or not a:
        return None
    bid_qty = sum(q for _,q in b)
    ask_qty = sum(q for _,q in a)
    total = bid_qty + ask_qty
    imbalance = (bid_qty - ask_qty) / total if total else 0
    best_bid = max(p for p,_ in b)
    best_ask = min(p for p,_ in a)
    mid = (best_bid + best_ask) / 2
    spread = best_ask - best_bid
    if imbalance > 0.15: pressure = "Buy-side order-book pressure"
    elif imbalance < -0.15: pressure = "Sell-side order-book pressure"
    else: pressure = "Balanced order book"
    return {"bid_qty": bid_qty, "ask_qty": ask_qty, "imbalance": imbalance, "pressure": pressure, "spread": spread, "mid": mid}

def trade_flow(trades):
    if not trades:
        return None
    buy = sell = 0.0
    for t in trades:
        q = float(t.get("q", 0) or 0)
        # CoinDCX documents m as whether the buyer is market maker.
        # Treat non-maker buys as aggressive buying; maker buys as less aggressive.
        if bool(t.get("m", False)):
            sell += q
        else:
            buy += q
    total = buy + sell
    ratio = (buy-sell)/total if total else 0
    if ratio > 0.15: label = "Recent trades favor aggressive buying"
    elif ratio < -0.15: label = "Recent trades favor aggressive selling"
    else: label = "Recent trades are balanced"
    return {"buy": buy, "sell": sell, "ratio": ratio, "label": label}

def fmt(v):
    return "—" if pd.isna(v) else f"{v:,.8f}".rstrip("0").rstrip(".")

def decision(a15, a1h, a1d, ob, flow):
    score = 0
    reasons = []

    weights = {"Bullish": 1, "Bearish": -1, "Mixed": 0}
    score += 3 * weights[a1d["trend"]]
    score += 2 * weights[a1h["trend"]]
    score += 1 * weights[a15["trend"]]

    if a1d["macd"] == "Bullish": score += 1
    else: score -= 1
    if a1h["macd"] == "Bullish": score += 1
    else: score -= 1

    if ob:
        if ob["imbalance"] > 0.15: score += 1
        elif ob["imbalance"] < -0.15: score -= 1
    if flow:
        if flow["ratio"] > 0.15: score += 1
        elif flow["ratio"] < -0.15: score -= 1

    aligned_long = all(x["trend"] == "Bullish" for x in [a1d,a1h,a15])
    aligned_short = all(x["trend"] == "Bearish" for x in [a1d,a1h,a15])

    if aligned_long and a1d["rsi"] < 78:
        verdict = "LONG SETUP"
    elif aligned_short and a1d["rsi"] > 22:
        verdict = "SHORT SETUP"
    elif score >= 5:
        verdict = "BULLISH BIAS — WAIT FOR ENTRY"
    elif score <= -5:
        verdict = "BEARISH BIAS — WAIT FOR ENTRY"
    else:
        verdict = "WAIT — NO CLEAR EDGE"

    confidence = min(95, 50 + abs(score)*5 + (12 if aligned_long or aligned_short else 0))
    if a1d["rsi"] >= 80 or a1d["rsi"] <= 20:
        confidence = max(35, confidence-10)

    if a1d["trend"] != a1h["trend"]:
        reasons.append("Daily and 1H trend disagree.")
    if a15["trend"] != a1h["trend"]:
        reasons.append("15m is not aligned with the 1H trend.")
    if a1d["rsi"] >= 80:
        reasons.append("Daily RSI is extremely overbought; avoid chasing longs.")
    elif a1d["rsi"] <= 20:
        reasons.append("Daily RSI is extremely oversold; avoid chasing shorts.")
    if ob:
        reasons.append(ob["pressure"] + ".")
    if flow:
        reasons.append(flow["label"] + ".")

    return verdict, confidence, score, reasons

def levels(d, verdict):
    x = d.iloc[-1]
    if "LONG SETUP" in verdict:
        entry = float(x.close)
        sl = entry - 1.2*float(x.atr)
        risk = entry-sl
        return entry, sl, entry+1.5*risk, entry+2.5*risk
    if "SHORT SETUP" in verdict:
        entry = float(x.close)
        sl = entry + 1.2*float(x.atr)
        risk = sl-entry
        return entry, sl, entry-1.5*risk, entry-2.5*risk
    return np.nan,np.nan,np.nan,np.nan

coin = st.text_input("CoinDCX coin or pair", "BTC").strip()

if st.button("🔎 Analyze", type="primary"):
    try:
        markets = get_markets()
        market = resolve_market(coin, markets)
        if not market:
            st.error("Market not found. Try BTC, ETH, SOL, or an exact pair such as BTCINR.")
            st.stop()

        pair = market["pair"]
        symbol = market.get("symbol", market.get("coindcx_name", pair))
        base = market.get("base_currency_short_name", "")
        target = market.get("target_currency_short_name", "")
        st.success(f"Using **{symbol}** | Internal pair **{pair}** | {target}/{base}")

        ticks = get_ticker()
        t = next((z for z in ticks if str(z.get("market","")).upper() == str(symbol).upper()), None)
        if t:
            a,b,c,e = st.columns(4)
            a.metric("Last price", fmt(float(t.get("last_price",0))))
            b.metric("24h change", f'{float(t.get("change_24_hour",0)):.2f}%')
            c.metric("24h high", fmt(float(t.get("high",0))))
            e.metric("24h volume", fmt(float(t.get("volume",0))))

        data = {}
        for key, interval in [("15m","15m"),("1h","1h"),("1d","1d")]:
            df = get_candles(pair, interval)
            if len(df) < 210:
                st.error(f"{key}: only {len(df)} candles returned; 210+ are required.")
                st.stop()
            data[key] = add_indicators(df)

        a15, a1h, a1d = tf_analysis(data["15m"]), tf_analysis(data["1h"]), tf_analysis(data["1d"])

        ob = None
        flow = None
        try:
            ob = orderbook_analysis(get_orderbook(pair, 50))
        except Exception as ex:
            st.warning(f"Order book unavailable: {ex}")
        try:
            flow = trade_flow(get_trades(pair, 100))
        except Exception as ex:
            st.warning(f"Trade history unavailable: {ex}")

        verdict, confidence, score, reasons = decision(a15,a1h,a1d,ob,flow)

        st.header("🧠 Market Decision")
        a,b,c = st.columns(3)
        a.metric("Overall view", verdict)
        b.metric("Confidence", f"{confidence:.0f}%")
        c.metric("Composite score", score)

        if reasons:
            st.write("**Why:**")
            for r in reasons:
                st.write("- " + r)

        st.subheader("Multi-timeframe dashboard")
        rows=[]
        for tf,x in [("1D",a1d),("1H",a1h),("15m",a15)]:
            rows.append({"Timeframe":tf,"Trend":x["trend"],"Structure":x["structure"],"Momentum":x["momentum"],"RSI":round(x["rsi"],1),"MACD":x["macd"],"Volume":x["volume"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if ob:
            st.subheader("📚 Order-book intelligence")
            a,b,c,d = st.columns(4)
            a.metric("Bid quantity", fmt(ob["bid_qty"]))
            b.metric("Ask quantity", fmt(ob["ask_qty"]))
            c.metric("Imbalance", f'{ob["imbalance"]*100:.2f}%')
            d.metric("Spread", fmt(ob["spread"]))
            st.info(ob["pressure"])

        if flow:
            st.subheader("⚡ Recent trade flow")
            a,b,c = st.columns(3)
            a.metric("Aggressive-buy volume", fmt(flow["buy"]))
            b.metric("Aggressive-sell volume", fmt(flow["sell"]))
            c.metric("Flow imbalance", f'{flow["ratio"]*100:.2f}%')
            st.info(flow["label"])

        entry,sl,tp1,tp2 = levels(data["15m"], verdict)
        st.subheader("Trade plan")
        if pd.notna(entry):
            a,b,c,d = st.columns(4)
            a.metric("Entry reference",fmt(entry)); b.metric("Invalidation / SL",fmt(sl))
            c.metric("TP1",fmt(tp1)); d.metric("TP2",fmt(tp2))
            st.warning("Analytical setup only. No order is created by this application.")
        else:
            st.info("No high-confidence entry. The agent prefers WAIT when confirmation is missing.")

        for title,key in [("15 Minute","15m"),("1 Hour","1h"),("1 Day","1d")]:
            with st.expander(f"{title} details", expanded=(key=="1h")):
                d=data[key]; x={"15m":a15,"1h":a1h,"1d":a1d}[key]
                a,b,c,e=st.columns(4)
                a.metric("Close",fmt(x["close"])); b.metric("RSI",f'{x["rsi"]:.1f}')
                c.metric("ATR",fmt(x["atr"])); e.metric("Structure",x["structure"])
                st.line_chart(d.set_index("time")[["close","ema20","ema50","ema200"]].tail(250))

        st.caption("CoinDCX public market-data source. Order book uses the public orderbook endpoint and recent trades use public trade history.")
    except Exception as ex:
        st.error(f"Analysis failed: {ex}")

st.divider()
st.caption("Educational market analysis only. No order, balance, withdrawal, or account-access functionality is included.")
