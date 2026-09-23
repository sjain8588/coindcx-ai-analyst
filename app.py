import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import json
import hmac
import hashlib
import os

st.set_page_config(
    page_title="CoinDCX 4-5 Coin Position Monitor",
    page_icon="🎯",
    layout="wide",
)

API = "https://api.coindcx.com"
PUBLIC = "https://public.coindcx.com"



# ============================================================
# COINDCX PRIVATE FUTURES POSITION API
# ============================================================

PRIVATE_POSITIONS_ENDPOINT = "/exchange/v1/derivatives/futures/positions"


def coindcx_signed_post(path, api_key, api_secret, payload=None):
    """
    CoinDCX private API request.

    IMPORTANT:
    - Use a READ-ONLY API key for this app.
    - Do not give the key withdrawal/order permissions.
    - The API secret is used only in memory for the request.
    """
    if not api_key or not api_secret:
        raise RuntimeError("CoinDCX API key/secret not provided.")

    body = dict(payload or {})
    body["timestamp"] = int(time.time() * 1000)

    raw = json.dumps(body, separators=(",", ":"))
    signature = hmac.new(
        api_secret.encode("utf-8"),
        raw.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    headers = {
        "Content-Type": "application/json",
        "X-AUTH-APIKEY": api_key,
        "X-AUTH-SIGNATURE": signature,
    }

    response = requests.post(
        API + path,
        data=raw,
        headers=headers,
        timeout=30,
    )

    if response.status_code >= 400:
        try:
            detail = response.json()
        except Exception:
            detail = response.text[:500]
        raise RuntimeError(
            f"CoinDCX private API HTTP {response.status_code}: {detail}"
        )

    try:
        return response.json()
    except Exception:
        raise RuntimeError(
            f"CoinDCX private API returned non-JSON response: "
            f"{response.text[:500]}"
        )


def _first_number(obj, keys, default=np.nan):
    for key in keys:
        if isinstance(obj, dict) and key in obj:
            try:
                value = obj[key]
                if value is None or value == "":
                    continue
                return float(value)
            except Exception:
                continue
    return default


def _first_text(obj, keys, default=""):
    for key in keys:
        if isinstance(obj, dict) and key in obj:
            value = obj[key]
            if value is not None and str(value).strip():
                return str(value)
    return default


def normalize_position(row):
    """
    Normalize several CoinDCX position field names into one internal schema.
    The endpoint response has changed field naming across API versions,
    so the normalizer deliberately accepts common aliases.
    """
    pair = _first_text(
        row,
        ["pair", "symbol", "instrument", "market", "contract"],
        "",
    ).upper().strip()

    side = _first_text(
        row,
        ["side", "position_side", "direction"],
        "",
    ).upper().strip()

    qty = _first_number(
        row,
        ["quantity", "qty", "size", "position_size", "active_pos"],
    )

    # Some APIs expose signed active_pos instead of side.
    if not side and np.isfinite(qty):
        if qty > 0:
            side = "LONG"
        elif qty < 0:
            side = "SHORT"

    if side in ("BUY", "B"):
        side = "LONG"
    elif side in ("SELL", "S"):
        side = "SHORT"

    entry = _first_number(
        row,
        [
            "avg_price",
            "average_price",
            "entry_price",
            "avg_entry_price",
            "average_entry_price",
        ],
    )

    mark = _first_number(
        row,
        ["mark_price", "markPrice", "last_price", "price"],
    )

    liq = _first_number(
        row,
        ["liquidation_price", "liquidationPrice", "liq_price"],
    )

    leverage = _first_number(
        row,
        ["leverage", "leverage_value"],
    )

    margin = _first_number(
        row,
        ["margin", "initial_margin", "position_margin"],
    )

    unrealized = _first_number(
        row,
        ["unrealized_pnl", "unrealizedProfit", "unrealized_profit", "pnl"],
    )

    realized = _first_number(
        row,
        ["realized_pnl", "realizedProfit", "realized_profit"],
    )

    tp = _first_number(
        row,
        ["take_profit_price", "take_profit", "tp_price"],
    )

    sl = _first_number(
        row,
        ["stop_loss_price", "stop_loss", "sl_price"],
    )

    return {
        "pair": pair,
        "side": side or "UNKNOWN",
        "quantity": qty,
        "entry": entry,
        "mark": mark,
        "liquidation": liq,
        "leverage": leverage,
        "margin": margin,
        "unrealized_pnl": unrealized,
        "realized_pnl": realized,
        "take_profit": tp,
        "stop_loss": sl,
        "raw": row,
    }


def extract_position_rows(payload):
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ["data", "positions", "result", "active_positions"]:
            value = payload.get(key)
            if isinstance(value, list):
                return value

        # Sometimes one position is returned as a dict.
        if any(
            k in payload
            for k in ["pair", "symbol", "avg_price", "entry_price"]
        ):
            return [payload]

    return []


def fetch_open_positions(api_key, api_secret):
    payload = coindcx_signed_post(
        PRIVATE_POSITIONS_ENDPOINT,
        api_key,
        api_secret,
        {},
    )

    rows = extract_position_rows(payload)

    positions = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        p = normalize_position(row)

        # Ignore zero/closed positions where the API exposes a size.
        if np.isfinite(p["quantity"]) and abs(p["quantity"]) < 1e-15:
            continue

        positions.append(p)

    return positions


def current_price_from_15m(pair):
    now = int(time.time())
    d = candles(pair, "15", now - 2 * 86400, now)
    d = completed(d)

    if d.empty:
        return np.nan

    return float(d.close.iloc[-1])


def calculate_position_pnl(position, current_price):
    entry = position["entry"]
    if not np.isfinite(entry) or entry == 0 or not np.isfinite(current_price):
        return np.nan

    if position["side"] == "SHORT":
        return (entry - current_price) / entry * 100

    return (current_price - entry) / entry * 100


def position_status(position, results):
    """
    Position-management assessment based on structure and nearby levels.
    This is a technical status, not a guaranteed prediction.
    """
    side = position["side"]
    r15 = results["15m"]
    r4 = results["4H"]
    r1d = results["1D"]

    current = r4["current"]
    score = 0
    reasons = []

    if side == "LONG":
        if r15["structure"] == "HH + HL":
            score += 2
            reasons.append("15m HH + HL")
        elif r15["structure"] == "LH + LL":
            score -= 3
            reasons.append("15m LH + LL")

        if r4["structure"] == "HH + HL":
            score += 3
            reasons.append("4H HH + HL")
        elif r4["structure"] == "LH + LL":
            score -= 4
            reasons.append("4H LH + LL")

        if r1d["structure"] == "HH + HL":
            score += 2
            reasons.append("1D HH + HL")
        elif r1d["structure"] == "LH + LL":
            score -= 2
            reasons.append("1D LH + LL")

        if current > r4["ema20"]:
            score += 1
            reasons.append("above 4H EMA20")
        else:
            score -= 1
            reasons.append("below 4H EMA20")

        if score >= 5:
            status = "🟢 LONG STRUCTURE INTACT"
        elif score <= -4:
            status = "🔴 LONG THESIS WEAK / REVERSAL RISK"
        else:
            status = "🟡 LONG — MONITOR"

    elif side == "SHORT":
        if r15["structure"] == "LH + LL":
            score += 2
            reasons.append("15m LH + LL")
        elif r15["structure"] == "HH + HL":
            score -= 3
            reasons.append("15m HH + HL")

        if r4["structure"] == "LH + LL":
            score += 3
            reasons.append("4H LH + LL")
        elif r4["structure"] == "HH + HL":
            score -= 4
            reasons.append("4H HH + HL")

        if r1d["structure"] == "LH + LL":
            score += 2
            reasons.append("1D LH + LL")
        elif r1d["structure"] == "HH + HL":
            score -= 2
            reasons.append("1D HH + HL")

        if current < r4["ema20"]:
            score += 1
            reasons.append("below 4H EMA20")
        else:
            score -= 1
            reasons.append("above 4H EMA20")

        if score >= 5:
            status = "🟢 SHORT STRUCTURE INTACT"
        elif score <= -4:
            status = "🔴 SHORT THESIS WEAK / REVERSAL RISK"
        else:
            status = "🟡 SHORT — MONITOR"

    else:
        status = "⚪ SIDE UNKNOWN"
        reasons.append("Could not determine LONG/SHORT")

    # Nearest 4H levels.
    nearest_support = (
        r4["supports"][-1] if r4["supports"] else np.nan
    )
    nearest_resistance = (
        r4["resistances"][0] if r4["resistances"] else np.nan
    )

    return {
        "status": status,
        "score": score,
        "reasons": reasons,
        "support": nearest_support,
        "resistance": nearest_resistance,
    }


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

st.title("🎯 CoinDCX Live Position Monitor")
st.caption(
    "Reads your open CoinDCX Futures positions (read-only), then checks "
    "15m / 4H / 1D / 1M structure, support/resistance and position risk."
)

with st.sidebar:
    st.header("🔐 CoinDCX Live Positions")

    st.caption(
        "Use a CoinDCX API key with read-only permissions. "
        "Never enable withdrawals. This app does not place orders."
    )

    env_key = os.getenv("COINDCX_API_KEY", "")
    env_secret = os.getenv("COINDCX_API_SECRET", "")

    api_key = st.text_input(
        "CoinDCX API Key",
        value=env_key,
        type="password",
    )

    api_secret = st.text_input(
        "CoinDCX API Secret",
        value=env_secret,
        type="password",
    )

    live_mode = st.checkbox(
        "Read my open Futures positions automatically",
        value=True,
    )

    manual_mode = st.checkbox(
        "Also allow manual coin list",
        value=False,
    )

    manual_raw = st.text_area(
        "Manual pairs (optional)",
        value="B-BTC_USDT\nB-ETH_USDT",
        height=100,
    )

    scan = st.button(
        "🔎 READ POSITIONS + SCAN",
        type="primary",
        use_container_width=True,
    )

    st.markdown("---")
    st.write("Analysis")
    st.write("• 15m — immediate structure")
    st.write("• 4H — primary trend")
    st.write("• 1D — major trend")
    st.write("• 1M — macro trend")
    st.write("• S1/S2/S3 + R1/R2/R3")




pairs = []
live_positions = []

if live_mode and scan:
    try:
        if not api_key or not api_secret:
            raise RuntimeError(
                "Enter your CoinDCX API key and secret, or set "
                "COINDCX_API_KEY and COINDCX_API_SECRET environment variables."
            )

        live_positions = fetch_open_positions(api_key, api_secret)

        if not live_positions:
            st.info(
                "CoinDCX returned no open Futures positions. "
                "If you expected positions, verify the API key permissions "
                "and that the positions are open in the Futures account."
            )

        for pos in live_positions:
            if pos["pair"] and pos["pair"] not in pairs:
                pairs.append(pos["pair"])

    except Exception as exc:
        st.error(f"Could not read CoinDCX open positions: {exc}")

if manual_mode:
    for p in manual_raw.replace(",", "\n").splitlines():
        p = p.strip().upper()
        if p and p not in pairs:
            pairs.append(p)

if len(pairs) > 5:
    st.warning("Only the first 5 unique pairs will be scanned.")
    pairs = pairs[:5]

if not pairs:
    st.info(
        "Enter API credentials and click 'READ POSITIONS + SCAN', "
        "or enable manual pairs."
    )
    st.stop()

if scan or "position_results" not in st.session_state:
    results_all = {}

    progress = st.progress(0)
    status = st.empty()

    # Map live positions by pair.
    live_by_pair = {
        p["pair"]: p for p in live_positions if p.get("pair")
    }

    for idx, pair in enumerate(pairs):
        status.write(f"Scanning {pair}...")

        try:
            tf_results = {}

            for tf in ["15m", "4H", "1D", "1M"]:
                d = get_tf(pair, tf)
                tf_results[tf] = analyze_tf(d)

            combined = combined_view(tf_results)

            position = live_by_pair.get(pair)

            # Public market price is used as a fallback/consistent reference.
            market_price = tf_results["15m"]["current"]

            if position is not None:
                api_mark = position.get("mark")
                if np.isfinite(api_mark):
                    market_price = api_mark

                position["market_price"] = market_price
                position["pnl_pct"] = calculate_position_pnl(
                    position, market_price
                )
                position["management"] = position_status(
                    position, tf_results
                )

            results_all[pair] = {
                "timeframes": tf_results,
                "combined": combined,
                "pattern": pattern_text(tf_results),
                "position": position,
                "error": None,
            }

        except Exception as exc:
            results_all[pair] = {
                "timeframes": {},
                "combined": {},
                "pattern": "",
                "position": live_by_pair.get(pair),
                "error": str(exc),
            }

        progress.progress((idx + 1) / len(pairs))

    status.empty()
    progress.empty()

    st.session_state["position_results"] = results_all
    st.session_state["position_pairs"] = pairs


results_all = st.session_state.get("position_results", {})

# ============================================================
# SUMMARY
# ============================================================

st.subheader("📊 Live Position Summary")

summary_rows = []

for pair in pairs:
    item = results_all.get(pair)

    if not item or item.get("error"):
        summary_rows.append({
            "Coin": pair,
            "Side": "—",
            "Entry": "—",
            "Current": "ERROR",
            "P/L": "—",
            "Position Status": "—",
            "4H Structure": "—",
            "Nearest 4H S": "—",
            "Nearest 4H R": "—",
        })
        continue

    r4 = item["timeframes"]["4H"]
    pos = item.get("position")

    if pos:
        entry = fmt_price(pos["entry"])
        current = fmt_price(pos["market_price"])
        pnl = (
            f"{pos['pnl_pct']:+.2f}%"
            if np.isfinite(pos["pnl_pct"])
            else "—"
        )
        side = pos["side"]
        pstatus = pos["management"]["status"]
    else:
        entry = "Manual"
        current = fmt_price(r4["current"])
        pnl = "—"
        side = "—"
        pstatus = "MARKET ANALYSIS ONLY"

    summary_rows.append({
        "Coin": pair,
        "Side": side,
        "Entry": entry,
        "Current": current,
        "P/L": pnl,
        "Position Status": pstatus,
        "4H Structure": r4["structure"],
        "Nearest 4H S": (
            fmt_price(r4["supports"][-1])
            if r4["supports"] else "—"
        ),
        "Nearest 4H R": (
            fmt_price(r4["resistances"][0])
            if r4["resistances"] else "—"
        ),
    })

st.dataframe(
    pd.DataFrame(summary_rows),
    use_container_width=True,
    hide_index=True,
)

st.warning(
    "Position Status is a technical structure assessment. "
    "It is not a guarantee that price will rise or fall. "
    "The agent uses confirmed market data and nearby levels to identify "
    "where the current position is strengthening or weakening."
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
    "This tool is analysis-only. It can read open Futures positions when "
    "you provide a read-only CoinDCX API key. It never places or modifies orders."
)
