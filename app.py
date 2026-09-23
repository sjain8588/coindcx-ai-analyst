import streamlit as st
import pandas as pd
import numpy as np
import requests
import time

st.set_page_config(
    page_title="CoinDCX 4-5 Coin Position Monitor",
    page_icon="🎯",
    layout="wide",
)

API = "https://api.coindcx.com"
PUBLIC = "https://public.coindcx.com"


# ============================================================
# COINDCX DATA
# ============================================================

def candles(pair, resolution, start_ts, end_ts):
    params = {
        "pair": pair,
        "from": int(start_ts),
        "to": int(end_ts),
        "resolution": resolution,
        "pcode": "f",
    }
    r = requests.get(
        f"{PUBLIC}/market_data/candlesticks",
        params=params,
        timeout=30,
    )
    r.raise_for_status()
    payload = r.json()
    rows = payload.get("data", []) if isinstance(payload, dict) else payload

    if not isinstance(rows, list):
        raise RuntimeError(f"Unexpected candle response for {pair}")

    d = pd.DataFrame(rows)
    if d.empty:
        return d

    required = ["open", "high", "low", "close", "volume"]
    for c in required:
        if c not in d.columns:
            raise RuntimeError(f"{pair}: candle response missing {c}")
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d["time"] = pd.to_datetime(
        d["time"], unit="ms", errors="coerce", utc=True
    )

    return (
        d.dropna(subset=["time", "open", "high", "low", "close", "volume"])
         .sort_values("time")
         .drop_duplicates("time")
         .reset_index(drop=True)
    )


def get_tf(pair, tf):
    now = int(time.time())

    if tf == "15m":
        return candles(pair, "15", now - 12 * 86400, now)

    if tf == "4H":
        return candles(pair, "240", now - 120 * 86400, now)

    if tf == "1D":
        return candles(pair, "1D", now - 900 * 86400, now)

    if tf == "1M":
        # CoinDCX does not need a separate monthly endpoint here.
        # Build monthly candles from daily data.
        d = candles(pair, "1D", now - 4 * 365 * 86400, now)
        if d.empty:
            return d

        x = d.set_index("time")
        m = x.resample("ME").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna().reset_index()

        return m

    raise ValueError(tf)


def completed(d):
    if d is None or len(d) < 2:
        return pd.DataFrame()
    return d.iloc[:-1].copy().reset_index(drop=True)


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(d):
    x = d.copy()

    if x.empty:
        return x

    for n in [20, 50, 100, 200]:
        x[f"ema{n}"] = x.close.ewm(span=n, adjust=False).mean()

    delta = x.close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi"] = 100 - (100 / (1 + rs))

    e12 = x.close.ewm(span=12, adjust=False).mean()
    e26 = x.close.ewm(span=26, adjust=False).mean()
    x["macd"] = e12 - e26
    x["macd_signal"] = x.macd.ewm(span=9, adjust=False).mean()

    tr = pd.concat(
        [
            x.high - x.low,
            (x.high - x.close.shift()).abs(),
            (x.low - x.close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    x["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    x["atr_pct"] = x.atr / x.close.replace(0, np.nan) * 100

    x["vol_ma"] = x.volume.rolling(20).mean()
    x["vol_ratio"] = x.volume / x.vol_ma.replace(0, np.nan)

    return x


# ============================================================
# STRUCTURE
# ============================================================

def pivot_points(d, left=3, right=3):
    x = d.copy().reset_index(drop=True)

    if len(x) < left + right + 10:
        return [], []

    highs = []
    lows = []

    for i in range(left, len(x) - right):
        hi = x.high.iloc[i]
        lo = x.low.iloc[i]

        if hi >= x.high.iloc[i-left:i+right+1].max():
            highs.append((i, float(hi)))

        if lo <= x.low.iloc[i-left:i+right+1].min():
            lows.append((i, float(lo)))

    return highs, lows


def structure_state(d):
    x = d.copy()

    if len(x) < 30:
        return "MIXED"

    highs, lows = pivot_points(x, 3, 3)

    if len(highs) < 2 or len(lows) < 2:
        return "MIXED"

    h1, h2 = highs[-2][1], highs[-1][1]
    l1, l2 = lows[-2][1], lows[-1][1]

    if h2 > h1 and l2 > l1:
        return "HH + HL"

    if h2 < h1 and l2 < l1:
        return "LH + LL"

    if h2 > h1 and l2 <= l1:
        return "BULLISH DEVELOPING"

    if h2 <= h1 and l2 > l1:
        return "BEARISH DEVELOPING"

    return "MIXED"


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

def sr_levels(d, current, count=3):
    """
    Support/resistance from confirmed pivot highs/lows.
    Levels are deduplicated into nearby price zones.
    """
    if d is None or len(d) < 25 or not np.isfinite(current):
        return [], []

    highs, lows = pivot_points(d, 3, 3)

    raw_support = [p for _, p in lows if p < current]
    raw_resistance = [p for _, p in highs if p > current]

    # Also include recent swing extremes when pivot supply is sparse.
    recent = d.tail(min(120, len(d)))
    for p in recent.low.nsmallest(8).tolist():
        if p < current:
            raw_support.append(float(p))

    for p in recent.high.nlargest(8).tolist():
        if p > current:
            raw_resistance.append(float(p))

    def cluster(values):
        values = sorted(float(v) for v in values if np.isfinite(v))
        if not values:
            return []

        clusters = []
        for v in values:
            if not clusters:
                clusters.append([v])
                continue

            anchor = np.mean(clusters[-1])

            # Dynamic zone width: 0.6% of price, minimum based on ATR.
            if abs(v - anchor) / max(abs(anchor), 1e-12) <= 0.006:
                clusters[-1].append(v)
            else:
                clusters.append([v])

        return [float(np.mean(c)) for c in clusters]

    supports = cluster(raw_support)
    resistances = cluster(raw_resistance)

    # Nearest supports first, then farther supports.
    supports = sorted(supports, key=lambda p: abs(current - p))[:count]
    resistances = sorted(resistances, key=lambda p: abs(current - p))[:count]

    supports = sorted(supports, reverse=True)
    resistances = sorted(resistances)

    return supports, resistances


def pct_distance(current, level):
    if level is None or not np.isfinite(level) or current == 0:
        return np.nan
    return (level / current - 1) * 100


# ============================================================
# TIMEFRAME ANALYSIS
# ============================================================

def analyze_tf(d):
    x = completed(d)

    if len(x) < 35:
        raise RuntimeError("Not enough completed candles")

    x = add_indicators(x)
    last = x.iloc[-1]

    current = float(last.close)
    ema20 = float(last.ema20)
    ema50 = float(last.ema50)
    rsi = float(last.rsi) if np.isfinite(last.rsi) else np.nan
    macd = float(last.macd)
    macd_signal = float(last.macd_signal)

    structure = structure_state(x)

    # Recent momentum.
    bars = 4 if len(x) >= 5 else 1
    momentum = (current / float(x.close.iloc[-1-bars]) - 1) * 100

    vol_ratio = (
        float(last.vol_ratio)
        if np.isfinite(last.vol_ratio)
        else np.nan
    )

    ema_distance = (current / ema20 - 1) * 100 if ema20 else np.nan

    supports, resistances = sr_levels(x, current, 3)

    # Direction score is descriptive, not a guaranteed forecast.
    score = 0

    if structure == "HH + HL":
        score += 3
    elif structure == "BULLISH DEVELOPING":
        score += 1
    elif structure == "LH + LL":
        score -= 3
    elif structure == "BEARISH DEVELOPING":
        score -= 1

    if current > ema20:
        score += 1
    else:
        score -= 1

    if current > ema50:
        score += 1
    else:
        score -= 1

    if np.isfinite(rsi):
        if 52 <= rsi <= 68:
            score += 1
        elif rsi < 42:
            score -= 1
        elif rsi > 75:
            score -= 1

    if macd > macd_signal:
        score += 1
    else:
        score -= 1

    if np.isfinite(momentum):
        if momentum > 0:
            score += 1
        elif momentum < 0:
            score -= 1

    if score >= 4:
        bias = "BULLISH"
    elif score <= -4:
        bias = "BEARISH"
    else:
        bias = "MIXED"

    return {
        "data": x,
        "current": current,
        "ema20": ema20,
        "ema50": ema50,
        "ema_distance": ema_distance,
        "rsi": rsi,
        "macd": macd,
        "macd_signal": macd_signal,
        "momentum": momentum,
        "volume_ratio": vol_ratio,
        "structure": structure,
        "score": score,
        "bias": bias,
        "supports": supports,
        "resistances": resistances,
    }


# ============================================================
# CROSS-TIMEFRAME DECISION
# ============================================================

def combined_view(results):
    """
    Higher timeframes carry more weight.
    15m = entry/short-term
    4H  = primary trend
    1D  = major trend
    1M  = macro regime
    """
    weights = {
        "15m": 1.0,
        "4H": 2.0,
        "1D": 2.5,
        "1M": 1.5,
    }

    weighted = 0.0
    total = 0.0

    for tf, result in results.items():
        weighted += result["score"] * weights[tf]
        total += 6 * weights[tf]

    normalized = weighted / total * 100 if total else 0

    # Current location relative to nearest levels.
    r4 = results["4H"]
    current = r4["current"]

    s1 = r4["supports"][-1] if r4["supports"] else np.nan
    r1 = r4["resistances"][0] if r4["resistances"] else np.nan

    near_support = (
        np.isfinite(s1)
        and abs(current - s1) / current <= 0.015
    )

    near_resistance = (
        np.isfinite(r1)
        and abs(r1 - current) / current <= 0.015
    )

    # Do not call this a guaranteed future move.
    if normalized >= 22:
        direction = "BULLISH BIAS"
    elif normalized <= -22:
        direction = "BEARISH BIAS"
    else:
        direction = "MIXED / WAIT"

    # Phase classification.
    if (
        r4["structure"] in ("HH + HL", "BULLISH DEVELOPING")
        and r4["ema_distance"] > -1.5
        and results["1D"]["score"] >= 0
    ):
        phase = "BULLISH / RECOVERY PHASE"
    elif (
        r4["structure"] in ("LH + LL", "BEARISH DEVELOPING")
        and r4["ema_distance"] < 1.5
        and results["1D"]["score"] <= 0
    ):
        phase = "BEARISH / DUMP PHASE"
    else:
        phase = "TRANSITION / MIXED"

    # Risk flags for someone already holding the position.
    risks = []

    if near_resistance:
        risks.append("Near 4H resistance")

    if near_support:
        risks.append("Near 4H support")

    if r4["structure"] == "LH + LL":
        risks.append("4H bearish structure")

    if r4["structure"] == "HH + HL":
        risks.append("4H bullish structure")

    if r4["ema_distance"] < -3:
        risks.append("Price extended below 4H EMA20")

    if r4["ema_distance"] > 5:
        risks.append("Price extended above 4H EMA20")

    return {
        "normalized_score": normalized,
        "direction": direction,
        "phase": phase,
        "risks": risks,
    }


# ============================================================
# DISPLAY HELPERS
# ============================================================

def fmt_price(x):
    if not np.isfinite(x):
        return "—"
    if abs(x) >= 1000:
        return f"{x:,.2f}"
    if abs(x) >= 1:
        return f"{x:,.4f}"
    if abs(x) >= 0.01:
        return f"{x:,.6f}"
    return f"{x:.10f}"


def level_table(result):
    rows = []

    for i, p in enumerate(result["supports"], 1):
        rows.append({
            "Level": f"S{i}",
            "Price": fmt_price(p),
            "Distance": f"{pct_distance(result['current'], p):+.2f}%",
            "Type": "Support",
        })

    for i, p in enumerate(result["resistances"], 1):
        rows.append({
            "Level": f"R{i}",
            "Price": fmt_price(p),
            "Distance": f"{pct_distance(result['current'], p):+.2f}%",
            "Type": "Resistance",
        })

    return pd.DataFrame(rows)


def pattern_text(results):
    r15 = results["15m"]
    r4 = results["4H"]
    r1d = results["1D"]
    r1m = results["1M"]

    if r4["structure"] == "HH + HL" and r1d["structure"] in ("HH + HL", "BULLISH DEVELOPING"):
        return "🟢 Bullish structure across 4H/1D"

    if r4["structure"] == "LH + LL" and r1d["structure"] in ("LH + LL", "BEARISH DEVELOPING"):
        return "🔴 Bearish structure across 4H/1D"

    if r4["structure"] in ("HH + HL", "BULLISH DEVELOPING") and r15["structure"] == "LH + LL":
        return "🟡 Higher-timeframe bullish, 15m pullback"

    if r4["structure"] in ("LH + LL", "BEARISH DEVELOPING") and r15["structure"] == "HH + HL":
        return "🟡 Higher-timeframe bearish, 15m bounce"

    if r4["structure"] == "HH + HL" and r4["ema_distance"] < -1:
        return "🔄 Bullish structure but below 4H EMA20"

    if r4["structure"] == "LH + LL" and r4["ema_distance"] > 1:
        return "🔄 Bearish structure but above 4H EMA20"

    if r1m["score"] >= 3 and r4["score"] < 0:
        return "⚠️ Macro bullish, short-term weakness"

    if r1m["score"] <= -3 and r4["score"] > 0:
        return "⚠️ Macro bearish, short-term recovery"

    return "⚪ Mixed / transition"


# ============================================================
# APP
# ============================================================

st.title("🎯 CoinDCX Position Monitor — 4/5 Coins")
st.caption(
    "Enter the 4–5 Futures coins you are already trading. "
    "The app checks MTF support/resistance, structure, momentum and "
    "whether the current evidence is bullish, bearish or mixed."
)

with st.sidebar:
    st.header("Coins in your trades")

    raw = st.text_area(
        "Enter 4–5 CoinDCX Futures pairs",
        value="B-BTC_USDT\nB-ETH_USDT\nB-SOL_USDT\nB-XRP_USDT",
        height=150,
        help="One pair per line. Example: B-BTC_USDT",
    )

    scan = st.button(
        "🔎 SCAN MY POSITIONS",
        type="primary",
        use_container_width=True,
    )

    st.markdown("---")
    st.write("Timeframes")
    st.write("• 15m — entry / immediate structure")
    st.write("• 4H — primary trend")
    st.write("• 1D — major trend")
    st.write("• 1M — macro trend")

pairs = []
for p in raw.replace(",", "\n").splitlines():
    p = p.strip().upper()
    if p and p not in pairs:
        pairs.append(p)

if len(pairs) > 5:
    st.warning("Only the first 5 unique pairs will be scanned.")
    pairs = pairs[:5]

if not pairs:
    st.info("Enter at least one Futures pair in the sidebar.")
    st.stop()

if scan or "position_results" not in st.session_state:
    results_all = {}

    progress = st.progress(0)
    status = st.empty()

    for idx, pair in enumerate(pairs):
        status.write(f"Scanning {pair}...")

        try:
            tf_results = {}

            for tf in ["15m", "4H", "1D", "1M"]:
                d = get_tf(pair, tf)
                tf_results[tf] = analyze_tf(d)

            combined = combined_view(tf_results)

            results_all[pair] = {
                "timeframes": tf_results,
                "combined": combined,
                "pattern": pattern_text(tf_results),
                "error": None,
            }

        except Exception as exc:
            results_all[pair] = {
                "timeframes": {},
                "combined": {},
                "pattern": "",
                "error": str(exc),
            }

        progress.progress((idx + 1) / len(pairs))

    status.empty()
    progress.empty()

    st.session_state["position_results"] = results_all

results_all = st.session_state.get("position_results", {})

# ============================================================
# SUMMARY
# ============================================================

st.subheader("📊 Position Summary")

summary_rows = []

for pair in pairs:
    item = results_all.get(pair)

    if not item or item.get("error"):
        summary_rows.append({
            "Coin": pair,
            "Current": "ERROR",
            "Direction": "—",
            "Phase": "—",
            "Pattern": item.get("error", "No result") if item else "No result",
            "Score": "—",
            "4H Structure": "—",
            "4H RSI": "—",
            "4H vs EMA20": "—",
        })
        continue

    r4 = item["timeframes"]["4H"]
    c = item["combined"]

    summary_rows.append({
        "Coin": pair,
        "Current": fmt_price(r4["current"]),
        "Direction": c["direction"],
        "Phase": c["phase"],
        "Pattern": item["pattern"],
        "Score": f"{c['normalized_score']:+.1f}",
        "4H Structure": r4["structure"],
        "4H RSI": f"{r4['rsi']:.1f}",
        "4H vs EMA20": f"{r4['ema_distance']:+.2f}%",
    })

st.dataframe(
    pd.DataFrame(summary_rows),
    use_container_width=True,
    hide_index=True,
)

st.warning(
    "The Direction column is a technical bias, not a guaranteed prediction. "
    "The model cannot know whether a coin 'will' dump or gain. "
    "Use the structure + support/resistance reaction + confirmation."
)


# ============================================================
# DETAILED POSITION CARDS
# ============================================================

st.subheader("🔬 Detailed Analysis")

for pair in pairs:
    item = results_all.get(pair)

    if not item:
        continue

    with st.expander(f"📌 {pair}", expanded=True):

        if item.get("error"):
            st.error(item["error"])
            continue

        tf_results = item["timeframes"]
        combined = item["combined"]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Current",
            fmt_price(tf_results["4H"]["current"]),
        )

        c2.metric(
            "Overall Bias",
            combined["direction"],
        )

        c3.metric(
            "Phase",
            combined["phase"],
        )

        c4.metric(
            "MTF Score",
            f"{combined['normalized_score']:+.1f}",
        )

        st.markdown(f"### Pattern: {item['pattern']}")

        if combined["risks"]:
            st.info(" | ".join(combined["risks"]))

        # ----------------------------------------------------
        # SUPPORT / RESISTANCE TABLE
        # ----------------------------------------------------

        st.markdown("#### 🧱 Support & Resistance")

        sr_rows = []

        for tf in ["15m", "4H", "1D", "1M"]:
            r = tf_results[tf]

            for i, p in enumerate(r["supports"], 1):
                sr_rows.append({
                    "Timeframe": tf,
                    "Level": f"S{i}",
                    "Price": fmt_price(p),
                    "Distance": f"{pct_distance(r['current'], p):+.2f}%",
                })

            for i, p in enumerate(r["resistances"], 1):
                sr_rows.append({
                    "Timeframe": tf,
                    "Level": f"R{i}",
                    "Price": fmt_price(p),
                    "Distance": f"{pct_distance(r['current'], p):+.2f}%",
                })

        st.dataframe(
            pd.DataFrame(sr_rows),
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # TIMEFRAME ANALYSIS
        # ----------------------------------------------------

        st.markdown("#### 📈 Timeframe Pattern Analysis")

        tf_rows = []

        for tf in ["15m", "4H", "1D", "1M"]:
            r = tf_results[tf]

            tf_rows.append({
                "TF": tf,
                "Structure": r["structure"],
                "Bias": r["bias"],
                "RSI": f"{r['rsi']:.1f}",
                "EMA20": fmt_price(r["ema20"]),
                "Price vs EMA20": f"{r['ema_distance']:+.2f}%",
                "EMA50": fmt_price(r["ema50"]),
                "Momentum": f"{r['momentum']:+.2f}%",
                "Volume": (
                    f"{r['volume_ratio']:.1f}x"
                    if np.isfinite(r["volume_ratio"])
                    else "—"
                ),
                "Score": f"{r['score']:+d}",
            })

        st.dataframe(
            pd.DataFrame(tf_rows),
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # TRADE INTERPRETATION
        # ----------------------------------------------------

        st.markdown("#### 🧭 What the Agent Sees")

        r15 = tf_results["15m"]
        r4 = tf_results["4H"]
        r1d = tf_results["1D"]
        r1m = tf_results["1M"]

        bullets = []

        if r15["structure"] == "HH + HL":
            bullets.append("15m is making higher highs and higher lows.")
        elif r15["structure"] == "LH + LL":
            bullets.append("15m is making lower highs and lower lows.")
        else:
            bullets.append(f"15m structure is {r15['structure'].lower()}.")

        if r4["structure"] == "HH + HL":
            bullets.append("4H structure is bullish.")
        elif r4["structure"] == "LH + LL":
            bullets.append("4H structure is bearish.")
        else:
            bullets.append(f"4H structure is {r4['structure'].lower()}.")

        if r4["ema_distance"] > 0:
            bullets.append(
                f"Price is {r4['ema_distance']:.2f}% above the 4H EMA20."
            )
        else:
            bullets.append(
                f"Price is {abs(r4['ema_distance']):.2f}% below the 4H EMA20."
            )

        if r4["supports"]:
            s1 = r4["supports"][-1]
            bullets.append(
                f"Nearest major 4H support zone: {fmt_price(s1)}."
            )

        if r4["resistances"]:
            r1 = r4["resistances"][0]
            bullets.append(
                f"Nearest major 4H resistance zone: {fmt_price(r1)}."
            )

        if r1d["structure"] == "HH + HL":
            bullets.append("Daily structure supports the bullish side.")
        elif r1d["structure"] == "LH + LL":
            bullets.append("Daily structure supports the bearish side.")

        if r1m["structure"] == "HH + HL":
            bullets.append("Monthly structure is bullish.")
        elif r1m["structure"] == "LH + LL":
            bullets.append("Monthly structure is bearish.")

        for b in bullets:
            st.write("• " + b)

        # ----------------------------------------------------
        # DECISION FRAMEWORK
        # ----------------------------------------------------

        st.markdown("#### 🎯 Entry / Exit Decision Framework")

        if combined["direction"] == "BULLISH BIAS":
            st.success(
                "Bullish technical bias. For an existing LONG, watch whether "
                "support holds and 15m/4H structure remains HH + HL. "
                "Avoid chasing if price is already far above EMA20 or sitting "
                "directly under major resistance."
            )
        elif combined["direction"] == "BEARISH BIAS":
            st.error(
                "Bearish technical bias. For an existing LONG, watch the "
                "nearest 15m/4H support carefully. A confirmed LH + LL sequence "
                "and loss of support increases downside risk."
            )
        else:
            st.warning(
                "Mixed technical picture. Avoid assuming either a dump or a "
                "rally. Wait for structure to resolve around the nearby "
                "support/resistance zones."
            )

        st.caption(
            "Important: support/resistance are reaction zones, not guaranteed "
            "reversal points. A support break can turn S1 into resistance; "
            "a resistance breakout can turn R1 into support."
        )

st.markdown("---")
st.caption(
    "This tool is analysis-only. It does not place orders or access private "
    "CoinDCX account information."
)
