import streamlit as st
import pandas as pd
import numpy as np
import requests

BASE = "https://api.coindcx.com"
st.set_page_config(page_title="CoinDCX AI Analyst", page_icon="📊", layout="wide")

st.title("📊 CoinDCX AI Market Analyst")
st.caption("Analysis only • Public CoinDCX market data • No trading or account access")

@st.cache_data(ttl=60)
def get_markets_details():
    r = requests.get(f"{BASE}/exchange/v1/markets_details", timeout=20)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=5)
def get_ticker():
    r = requests.get(f"{BASE}/exchange/ticker", timeout=20)
    r.raise_for_status()
    return r.json()

def resolve_market(user_input, markets):
    q = user_input.upper().strip().replace("/", "").replace("-", "").replace("_", "")
    exact, target = [], []
    for m in markets:
        if str(m.get("status","")).lower() != "active":
            continue
        symbol = str(m.get("symbol", m.get("coindcx_name",""))).upper()
        name = str(m.get("coindcx_name","")).upper()
        base = str(m.get("base_currency_short_name","")).upper()
        tgt = str(m.get("target_currency_short_name","")).upper()
        if q in (symbol, name) or tgt + base == q:
            exact.append(m)
        elif q == tgt:
            target.append(m)
    candidates = exact or target
    if not candidates:
        return None
    # If only a coin was entered, prefer INR, then USDT.
    candidates.sort(key=lambda m: (
        str(m.get("base_currency_short_name","")).upper() not in ("INR","USDT"),
        str(m.get("base_currency_short_name","")).upper() != "INR"
    ))
    return candidates[0]

@st.cache_data(ttl=30)
def get_candles(pair, interval):
    r = requests.get(
        f"{BASE}/market_data/candles",
        params={"pair": pair, "interval": interval, "limit": 1000},
        timeout=30
    )
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
    data = r.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected response: {data}")
    df = pd.DataFrame(data)
    if df.empty:
        return df
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
    return df.dropna().sort_values("time").drop_duplicates("time").reset_index(drop=True)

def indicators(df):
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
    tr = pd.concat([d.high-d.low,(d.high-d.close.shift()).abs(),
                    (d.low-d.close.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    d["vol_ma20"] = d.volume.rolling(20).mean()
    return d

def analyze(d):
    x = d.iloc[-1]
    score = 0
    signals = []
    tests = [
        (x.close > x.ema20, "Price above EMA20", "Price below EMA20"),
        (x.ema20 > x.ema50, "EMA20 above EMA50", "EMA20 below EMA50"),
        (x.ema50 > x.ema200, "EMA50 above EMA200", "EMA50 below EMA200"),
        (x.rsi > 55, f"RSI bullish ({x.rsi:.1f})", f"RSI bearish ({x.rsi:.1f})"),
        (x.macd > x.macd_signal, "MACD bullish", "MACD bearish"),
        (x.volume > x.vol_ma20, "Volume above 20-period average", "Volume below 20-period average")
    ]
    for ok, bull, bear in tests:
        score += 1 if ok else -1
        signals.append(("Bullish" if ok else "Bearish", bull if ok else bear))
    verdict = "LONG BIAS" if score >= 4 else "SHORT BIAS" if score <= -4 else "WAIT"
    recent = d.tail(50)
    support, resistance, atr = recent.low.min(), recent.high.max(), float(x.atr)
    entry = sl = tp1 = tp2 = np.nan
    if verdict == "LONG BIAS":
        entry = float(x.close); sl = entry-1.2*atr
        tp1 = entry+1.5*(entry-sl); tp2 = entry+2.5*(entry-sl)
    elif verdict == "SHORT BIAS":
        entry = float(x.close); sl = entry+1.2*atr
        tp1 = entry-1.5*(sl-entry); tp2 = entry-2.5*(sl-entry)
    return verdict, score, signals, support, resistance, entry, sl, tp1, tp2, x

def fmt(v):
    return "—" if pd.isna(v) else f"{v:,.8f}".rstrip("0").rstrip(".")

coin = st.text_input("CoinDCX coin or pair", "BTC").strip()
if st.button("🔎 Analyze", type="primary"):
    try:
        markets = get_markets_details()
        m = resolve_market(coin, markets)
        if not m:
            st.error("Market not found. Try BTC, ETH, SOL, or an exact pair such as BTCINR/BTCUSDT.")
            st.stop()
        pair = m["pair"]
        symbol = m.get("symbol", m.get("coindcx_name", pair))
        st.success(f"Using market **{symbol}** | Internal pair **{pair}**")

        try:
            ticks = get_ticker()
            t = next((x for x in ticks if str(x.get("market","")).upper() == str(symbol).upper()), None)
            if t:
                a,b,c,d = st.columns(4)
                a.metric("Last price", t.get("last_price","—"))
                b.metric("24h change", f'{t.get("change_24_hour","—")}%')
                c.metric("24h high", t.get("high","—"))
                d.metric("24h volume", t.get("volume","—"))
        except Exception as e:
            st.warning(f"Ticker unavailable: {e}")

        for label, interval in [("15 Minute","15m"),("1 Hour","1h"),("1 Day","1d")]:
            st.subheader(label)
            df = get_candles(pair, interval)
            if len(df) < 210:
                st.warning(f"Only {len(df)} candles returned; EMA200 needs at least 200.")
                continue
            d = indicators(df)
            verdict, score, signals, support, resistance, entry, sl, tp1, tp2, x = analyze(d)
            a,b,c,e = st.columns(4)
            a.metric("Signal", verdict); b.metric("RSI", f"{x.rsi:.1f}")
            c.metric("Support", fmt(support)); e.metric("Resistance", fmt(resistance))
            st.write(f"**Score:** {score}/6")
            for direction, text in signals:
                st.write(f"- **{direction}:** {text}")
            if verdict != "WAIT":
                st.write(f"**Illustrative setup:** Entry `{fmt(entry)}` | SL `{fmt(sl)}` | TP1 `{fmt(tp1)}` | TP2 `{fmt(tp2)}`")
            else:
                st.info("No sufficiently strong directional edge from this rule set.")
            st.line_chart(d.set_index("time")[["close","ema20","ema50","ema200"]].tail(250))
    except Exception as e:
        st.error(f"Analysis failed: {e}")

st.divider()
st.caption("Educational analysis only. No order, balance, or withdrawal functionality is included.")
