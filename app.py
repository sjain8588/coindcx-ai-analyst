
import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="CoinDCX AI Analyst", page_icon="📊", layout="wide")

st.title("📊 CoinDCX AI Market Analyst")
st.caption("Analysis only • No trading • No CoinDCX account credentials required")

@st.cache_data(ttl=60)
def get_markets():
    urls = [
        "https://api.coindcx.com/exchange/ticker",
        "https://api.coindcx.com/exchange/v1/markets_details",
    ]
    # Public ticker endpoint
    r = requests.get(urls[0], timeout=15)
    r.raise_for_status()
    return r.json()

def find_market(ticker, symbol):
    symbol = symbol.upper().replace("/", "")
    for x in ticker:
        pair = str(x.get("market", x.get("symbol",""))).upper().replace("/", "")
        if pair == symbol:
            return x
    return None

def fetch_candles(pair, interval="60", limit=250):
    # CoinDCX public candle endpoint may change; keep endpoint isolated here.
    url = "https://public.coindcx.com/market_data/candles"
    params = {"pair": pair.upper(), "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data)
    if df.empty:
        return df
    if "time" in df:
        df["time"] = pd.to_datetime(df["time"], unit="ms", errors="coerce")
    for c in ["open","high","low","close","volume"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("time").dropna()

def indicators(df):
    d = df.copy()
    d["ema9"] = d.close.ewm(span=9, adjust=False).mean()
    d["ema20"] = d.close.ewm(span=20, adjust=False).mean()
    d["ema50"] = d.close.ewm(span=50, adjust=False).mean()
    d["ema100"] = d.close.ewm(span=100, adjust=False).mean()
    d["ema200"] = d.close.ewm(span=200, adjust=False).mean()

    delta = d.close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi"] = 100 - (100 / (1 + rs))

    fast = d.close.ewm(span=12, adjust=False).mean()
    slow = d.close.ewm(span=26, adjust=False).mean()
    d["macd"] = fast - slow
    d["macd_signal"] = d.macd.ewm(span=9, adjust=False).mean()

    tr = pd.concat([
        d.high-d.low,
        (d.high-d.close.shift()).abs(),
        (d.low-d.close.shift()).abs()
    ], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    d["vol_ma20"] = d.volume.rolling(20).mean()
    return d.dropna()

def analyze(d):
    x = d.iloc[-1]
    score = 0
    reasons = []

    if x.close > x.ema20: score += 1; reasons.append("Price above EMA20")
    else: score -= 1; reasons.append("Price below EMA20")
    if x.ema20 > x.ema50: score += 1; reasons.append("EMA20 above EMA50")
    else: score -= 1; reasons.append("EMA20 below EMA50")
    if x.ema50 > x.ema200: score += 1; reasons.append("EMA50 above EMA200")
    else: score -= 1; reasons.append("EMA50 below EMA200")
    if x.rsi >= 55: score += 1; reasons.append(f"RSI bullish ({x.rsi:.1f})")
    elif x.rsi <= 45: score -= 1; reasons.append(f"RSI bearish ({x.rsi:.1f})")
    else: reasons.append(f"RSI neutral ({x.rsi:.1f})")
    if x.macd > x.macd_signal: score += 1; reasons.append("MACD bullish")
    else: score -= 1; reasons.append("MACD bearish")
    if x.volume > x.vol_ma20: score += 1; reasons.append("Volume above 20-period average")

    if score >= 4: verdict = "LONG BIAS"
    elif score <= -4: verdict = "SHORT BIAS"
    else: verdict = "WAIT / NO CLEAR EDGE"

    recent = d.tail(50)
    support = recent.low.min()
    resistance = recent.high.max()
    atr = x.atr
    if "LONG" in verdict:
        sl = x.close - 1.2*atr
        tp1 = x.close + 1.5*(x.close-sl)
        tp2 = x.close + 2.5*(x.close-sl)
    elif "SHORT" in verdict:
        sl = x.close + 1.2*atr
        tp1 = x.close - 1.5*(sl-x.close)
        tp2 = x.close - 2.5*(sl-x.close)
    else:
        sl = tp1 = tp2 = np.nan

    return verdict, score, reasons, support, resistance, sl, tp1, tp2, x

symbol = st.text_input("CoinDCX pair", "BTCINR").strip().upper()
intervals = {"15m":"15","1H":"60","4H":"240","1D":"1D"}

if st.button("Analyze"):
    try:
        ticker = get_markets()
        m = find_market(ticker, symbol)
        if m:
            st.metric("Current price", m.get("last_price", m.get("lastPrice", "N/A")))
        else:
            st.warning("Pair was not found in the public CoinDCX ticker response. Check the exact CoinDCX market symbol.")

        for name, iv in intervals.items():
            st.subheader(name)
            df = fetch_candles(symbol, iv)
            if df.empty or len(df) < 210:
                st.warning(f"Not enough candle data returned for {name}.")
                continue
            d = indicators(df)
            result = analyze(d)
            verdict, score, reasons, support, resistance, sl, tp1, tp2, x = result
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Verdict", verdict)
            c2.metric("RSI", f"{x.rsi:.1f}")
            c3.metric("Support", f"{support:,.4f}")
            c4.metric("Resistance", f"{resistance:,.4f}")
            st.write("**Signals:** " + " • ".join(reasons))
            if pd.notna(sl):
                st.write(f"**Illustrative setup:** Entry {x.close:,.4f} | SL {sl:,.4f} | TP1 {tp1:,.4f} | TP2 {tp2:,.4f}")
            st.line_chart(d.set_index("time")[["close","ema20","ema50","ema200"]].tail(150))
    except Exception as e:
        st.error(f"Data request failed: {e}")
        st.info("CoinDCX can change public endpoint formats. The data adapter is isolated in fetch_candles(), so it can be updated without changing the analysis engine.")

st.divider()
st.caption("Educational market analysis only. Signals are heuristic, not financial advice. Always verify prices and levels before acting.")
