import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone

BASE = "https://api.coindcx.com"

st.set_page_config(page_title="CoinDCX AI Analyst", page_icon="📊", layout="wide")

st.title("📊 CoinDCX AI Market Analyst")
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
    r = requests.get(
        f"{BASE}/market_data/candles",
        params={"pair": pair, "interval": interval, "limit": 1000},
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"CoinDCX HTTP {r.status_code}: {r.text[:500]}")
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
    return (
        df.dropna(subset=["time", "open", "high", "low", "close", "volume"])
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )

def resolve_market(text, markets):
    q = text.upper().strip().replace("/", "").replace("-", "").replace("_", "")
    exact, coin = [], []
    for m in markets:
        if str(m.get("status", "")).lower() != "active":
            continue
        symbol = str(m.get("symbol", m.get("coindcx_name", ""))).upper()
        name = str(m.get("coindcx_name", "")).upper()
        base = str(m.get("base_currency_short_name", "")).upper()
        target = str(m.get("target_currency_short_name", "")).upper()
        if q in (symbol, name) or target + base == q:
            exact.append(m)
        elif q == target:
            coin.append(m)
    choices = exact or coin
    if not choices:
        return None
    # For coin-only input prefer INR, then USDT.
    choices.sort(key=lambda m: (
        str(m.get("base_currency_short_name", "")).upper() not in ("INR", "USDT"),
        str(m.get("base_currency_short_name", "")).upper() != "INR"
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
    d["rsi"] = 100 - (100 / (1 + rs))

    e12 = d.close.ewm(span=12, adjust=False).mean()
    e26 = d.close.ewm(span=26, adjust=False).mean()
    d["macd"] = e12 - e26
    d["macd_signal"] = d.macd.ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d.macd - d.macd_signal

    tr = pd.concat([
        d.high - d.low,
        (d.high - d.close.shift()).abs(),
        (d.low - d.close.shift()).abs()
    ], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    d["vol_ma20"] = d.volume.rolling(20).mean()
    d["high20"] = d.high.rolling(20).max().shift(1)
    d["low20"] = d.low.rolling(20).min().shift(1)
    return d

def structure(d):
    # Simple swing-structure proxy using recent highs/lows.
    if len(d) < 30:
        return "Insufficient data"
    a = d.iloc[-10:]
    b = d.iloc[-30:-10]
    hh = a.high.max() > b.high.max()
    hl = a.low.min() > b.low.min()
    lh = a.high.max() < b.high.max()
    ll = a.low.min() < b.low.min()
    if hh and hl:
        return "Higher highs + higher lows (bullish structure)"
    if lh and ll:
        return "Lower highs + lower lows (bearish structure)"
    return "Mixed / ranging structure"

def timeframe_analysis(d):
    x = d.iloc[-1]
    trend = "Bullish" if x.ema20 > x.ema50 and x.ema50 > x.ema200 else (
        "Bearish" if x.ema20 < x.ema50 and x.ema50 < x.ema200 else "Mixed"
    )
    if x.rsi >= 70:
        momentum = "Overbought"
    elif x.rsi <= 30:
        momentum = "Oversold"
    elif x.rsi >= 55:
        momentum = "Bullish momentum"
    elif x.rsi <= 45:
        momentum = "Bearish momentum"
    else:
        momentum = "Neutral momentum"

    macd = "Bullish" if x.macd > x.macd_signal else "Bearish"
    volume = "Above average" if x.volume > x.vol_ma20 else "Below average"
    struct = structure(d)
    return {
        "trend": trend, "momentum": momentum, "macd": macd,
        "volume": volume, "structure": struct, "rsi": float(x.rsi),
        "atr": float(x.atr), "close": float(x.close)
    }

def build_market_view(analyses):
    day = analyses["1d"]
    hour = analyses["1h"]
    m15 = analyses["15m"]

    bullish_votes = sum([
        day["trend"] == "Bullish",
        hour["trend"] == "Bullish",
        m15["trend"] == "Bullish",
        day["macd"] == "Bullish",
        hour["macd"] == "Bullish",
        m15["macd"] == "Bullish",
    ])
    bearish_votes = 6 - bullish_votes

    # Higher timeframe carries more weight.
    bias_score = (
        (2 if day["trend"] == "Bullish" else -2 if day["trend"] == "Bearish" else 0) +
        (2 if hour["trend"] == "Bullish" else -2 if hour["trend"] == "Bearish" else 0) +
        (1 if m15["trend"] == "Bullish" else -1 if m15["trend"] == "Bearish" else 0) +
        (1 if day["macd"] == "Bullish" else -1) +
        (1 if hour["macd"] == "Bullish" else -1)
    )

    aligned_long = day["trend"] == "Bullish" and hour["trend"] == "Bullish" and m15["trend"] == "Bullish"
    aligned_short = day["trend"] == "Bearish" and hour["trend"] == "Bearish" and m15["trend"] == "Bearish"

    if aligned_long and day["rsi"] < 75:
        verdict = "LONG SETUP"
    elif aligned_short and day["rsi"] > 25:
        verdict = "SHORT SETUP"
    elif bias_score >= 3:
        verdict = "BULLISH BIAS — WAIT FOR ENTRY"
    elif bias_score <= -3:
        verdict = "BEARISH BIAS — WAIT FOR ENTRY"
    else:
        verdict = "WAIT — NO CLEAR EDGE"

    confidence = min(95, 50 + abs(bias_score) * 7 + (10 if aligned_long or aligned_short else 0))
    if day["rsi"] >= 80 or day["rsi"] <= 20:
        confidence = max(35, confidence - 10)

    return verdict, confidence, bullish_votes, bearish_votes

def setup_levels(d, verdict):
    x = d.iloc[-1]
    entry = float(x.close)
    atr = float(x.atr)
    recent_high = float(d.tail(50).high.max())
    recent_low = float(d.tail(50).low.min())
    if "LONG SETUP" in verdict:
        sl = min(entry - 1.2 * atr, recent_low)
        risk = entry - sl
        return entry, sl, entry + risk * 1.5, entry + risk * 2.5
    if "SHORT SETUP" in verdict:
        sl = max(entry + 1.2 * atr, recent_high)
        risk = sl - entry
        return entry, sl, entry - risk * 1.5, entry - risk * 2.5
    return np.nan, np.nan, np.nan, np.nan

def fmt(v):
    if pd.isna(v):
        return "—"
    return f"{v:,.8f}".rstrip("0").rstrip(".")

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
        st.success(f"Using **{symbol}** | CoinDCX internal pair: **{pair}** | {target}/{base}")

        try:
            ticks = get_ticker()
            t = next((z for z in ticks if str(z.get("market", "")).upper() == str(symbol).upper()), None)
            if t:
                a,b,c,e = st.columns(4)
                a.metric("Last price", fmt(float(t.get("last_price", 0))))
                b.metric("24h change", f'{float(t.get("change_24_hour", 0)):.2f}%')
                c.metric("24h high", fmt(float(t.get("high", 0))))
                e.metric("24h volume", fmt(float(t.get("volume", 0))))
        except Exception as ex:
            st.warning(f"Ticker unavailable: {ex}")

        data = {}
        for label, interval in [("15m", "15m"), ("1h", "1h"), ("1d", "1d")]:
            df = get_candles(pair, interval)
            if len(df) < 210:
                st.error(f"{label}: only {len(df)} candles returned; 210+ are required for the long-term indicators.")
                st.stop()
            data[label] = add_indicators(df)

        analyses = {"15m": timeframe_analysis(data["15m"]),
                    "1h": timeframe_analysis(data["1h"]),
                    "1d": timeframe_analysis(data["1d"])}

        verdict, confidence, bv, sv = build_market_view(analyses)

        st.header("🧠 Market Decision")
        a,b,c = st.columns(3)
        a.metric("Overall view", verdict)
        b.metric("Confidence", f"{confidence:.0f}%")
        c.metric("Bullish / Bearish votes", f"{bv} / {sv}")

        day = analyses["1d"]; hour = analyses["1h"]; m15 = analyses["15m"]
        if day["trend"] == "Bullish" and hour["trend"] != "Bullish":
            explanation = "Daily trend is bullish, but the 1H timeframe has not confirmed it. Treat the 15m move as entry timing, not a standalone trend signal."
        elif day["trend"] == "Bearish" and hour["trend"] != "Bearish":
            explanation = "Daily trend is bearish, but the 1H timeframe has not confirmed it. Avoid treating a short 15m move as a high-confidence short."
        elif day["rsi"] >= 80:
            explanation = "Daily RSI is extremely high. Trend may remain bullish, but chasing a new long has elevated pullback risk."
        elif day["rsi"] <= 20:
            explanation = "Daily RSI is extremely low. Downtrend may remain intact, but chasing a new short has elevated rebound risk."
        else:
            explanation = "The model is using higher timeframes for direction and the 15m timeframe for timing."
        st.info(explanation)

        st.subheader("Multi-timeframe dashboard")
        rows = []
        for tf in ["1d", "1h", "15m"]:
            x = analyses[tf]
            rows.append({
                "Timeframe": tf,
                "Trend": x["trend"],
                "Structure": x["structure"],
                "Momentum": x["momentum"],
                "RSI": round(x["rsi"], 1),
                "MACD": x["macd"],
                "Volume": x["volume"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Primary setup only if all timeframes agree.
        entry, sl, tp1, tp2 = setup_levels(data["15m"], verdict)
        st.subheader("Trade plan")
        if pd.notna(entry):
            a,b,c,e = st.columns(4)
            a.metric("Entry reference", fmt(entry))
            b.metric("Invalidation / SL", fmt(sl))
            c.metric("TP1", fmt(tp1))
            e.metric("TP2", fmt(tp2))
            st.warning("This is an analytical setup, not an order instruction. Verify price action and liquidity before acting.")
        else:
            st.info("No high-confidence entry is generated. The agent prefers WAIT over forcing a trade when timeframes disagree.")

        for label, key in [("15 Minute", "15m"), ("1 Hour", "1h"), ("1 Day", "1d")]:
            d = data[key]
            x = analyses[key]
            with st.expander(f"{label} details", expanded=(key == "1h")):
                a,b,c,dcol = st.columns(4)
                a.metric("Close", fmt(x["close"]))
                b.metric("RSI", f'{x["rsi"]:.1f}')
                c.metric("ATR", fmt(x["atr"]))
                dcol.metric("Structure", x["structure"])
                st.line_chart(d.set_index("time")[["close","ema20","ema50","ema200"]].tail(250))

        st.caption("Data source: CoinDCX public market-data APIs. The CoinDCX documentation identifies the market `pair` in Markets Details and uses that pair for Candles.")

    except Exception as ex:
        st.error(f"Analysis failed: {ex}")
        st.caption("If CoinDCX changes a public endpoint or response format, send the exact error and we can update the connector.")

st.divider()
st.caption("Educational market analysis only. This application contains no order, balance, withdrawal, or account-access functionality.")
