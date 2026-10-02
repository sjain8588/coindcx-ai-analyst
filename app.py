import time
import json
import hmac
import hashlib
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# COINDCX EXPERT FUTURES TRADING AGENT V5
# ============================================================
# Architecture:
#   ALL USDT futures -> strongest pump universe -> expert MTF analysis
#   -> trade proposal -> YOUR APPROVAL -> order -> position manager.
#
# The bot supports:
#   - LONG / SHORT proposals
#   - approval-gated live execution
#   - risk-based sizing
#   - protective stop / targets
#   - structure / ATR / EMA trailing
#   - partial exits
#   - close position
#   - re-entry proposal after a genuinely new setup
#   - trade journal
#   - emergency stop for new trades
#
# IMPORTANT:
#   LIVE_TRADING_ENABLED is OFF by default. Turn it on only after testing
#   authentication and market analysis. The exact private order schema can
#   vary by CoinDCX API version, so verify the current CoinDCX Futures API
#   documentation before enabling live execution.
# ============================================================

API = "https://api.coindcx.com"
PUBLIC = "https://public.coindcx.com"
USER_AGENT = "CoinDCX-Expert-Trade-Agent-V5/1.0"

# User-requested hard-coded credentials.
# Never display these values in Streamlit/logs.
COINDCX_API_KEY = "25fd296f3aaf7f788943f02f03d6ed8cf71d3b9d5ccdac5d"
COINDCX_API_SECRET = "51b9c73d7abba3ad0696b590b43101e918348ba76f0dd2f215b074efa04264e3"

# Safety: live trading is explicitly OFF until you enable it.
LIVE_TRADING_ENABLED = False

# Trading universe / risk defaults
PUMP_FOCUS = 25
MAX_WORKERS = 12
DEFAULT_RISK_PCT = 0.50
DEFAULT_MAX_DAILY_LOSS_PCT = 2.0
DEFAULT_MAX_OPEN_POSITIONS = 2
DEFAULT_MAX_LEVERAGE = 5.0
DEFAULT_MIN_RR = 2.0
DEFAULT_ATR_STOP_MULT = 1.25
DEFAULT_TRAIL_ATR_MULT = 1.50
DEFAULT_TRAIL_ACTIVATION_R = 1.0
DEFAULT_PARTIAL_R = 2.0
DEFAULT_PARTIAL_PCT = 50
DEFAULT_COOLDOWN_MIN = 30

# CoinDCX Futures API paths used by this implementation.
POSITIONS_ENDPOINT = "/exchange/v1/derivatives/futures/positions"
ORDER_CREATE_ENDPOINT = "/exchange/v1/derivatives/futures/orders/create"
ORDER_CANCEL_ALL_ENDPOINT = "/exchange/v1/derivatives/futures/orders/cancel_all"


def get_json(url, params=None, timeout=25):
    r = requests.get(url, params=params, timeout=timeout,
                     headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return r.json()


def signed_post(path, payload=None):
    body = dict(payload or {})
    body["timestamp"] = int(time.time() * 1000)
    raw = json.dumps(body, separators=(",", ":"))
    sig = hmac.new(COINDCX_API_SECRET.encode(), raw.encode(), hashlib.sha256).hexdigest()
    headers = {
        "Content-Type": "application/json",
        "X-AUTH-APIKEY": COINDCX_API_KEY,
        "X-AUTH-SIGNATURE": sig,
        "User-Agent": USER_AGENT,
    }
    r = requests.post(API + path, data=raw, headers=headers, timeout=30)
    if r.status_code >= 400:
        try:
            detail = r.json()
        except Exception:
            detail = r.text[:500]
        raise RuntimeError(f"CoinDCX private API HTTP {r.status_code}: {detail}")
    try:
        return r.json()
    except Exception:
        return {"raw": r.text}


def active_instruments():
    url = API + "/exchange/v1/derivatives/futures/data/active_instruments"
    r = requests.get(
        url,
        params=[("margin_currency_short_name[]", "USDT")],
        timeout=25,
        headers={"User-Agent": USER_AGENT},
    )
    r.raise_for_status()
    payload = r.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected active_instruments response: {payload}")
    out = []
    for x in payload:
        if isinstance(x, str):
            pair = x
            row = {"pair": pair}
        elif isinstance(x, dict):
            pair = x.get("pair") or x.get("symbol") or x.get("instrument") or x.get("symbol_id") or x.get("market")
            row = x.copy()
        else:
            continue
        if pair and "USDT" in str(pair).upper():
            row["pair"] = str(pair).upper().strip()
            out.append(row)
    dedup = {}
    for x in out:
        dedup[x["pair"]] = x
    if not dedup:
        raise RuntimeError("CoinDCX returned no active USDT Futures contracts.")
    return list(dedup.values())


def candles(pair, resolution, days):
    now = int(time.time())
    resolution_map = {"15m": "15", "1H": "60", "4H": "240", "1D": "1D"}
    payload = get_json(
        f"{PUBLIC}/market_data/candlesticks",
        {
            "pair": pair,
            "from": now - int(days * 86400),
            "to": now,
            "resolution": resolution_map.get(resolution, resolution),
            "pcode": "f",
        },
    )
    rows = payload.get("data", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    tcol = "time" if "time" in d.columns else "timestamp"
    if tcol not in d.columns:
        return pd.DataFrame()
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in d.columns:
            return pd.DataFrame()
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["time"] = pd.to_datetime(d[tcol], unit="ms", utc=True, errors="coerce")
    d = d.dropna(subset=["time", "open", "high", "low", "close"]).sort_values("time")
    d = d.drop_duplicates("time").reset_index(drop=True)
    if len(d) > 2:
        d = d.iloc[:-1].copy()  # completed candles only
    return d


def indicators(d):
    x = d.copy()
    for n in [20, 50, 100, 200]:
        x[f"ema{n}"] = x.close.ewm(span=n, adjust=False).mean()
    delta = x.close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    x["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    tr = pd.concat([
        x.high - x.low,
        (x.high - x.close.shift()).abs(),
        (x.low - x.close.shift()).abs(),
    ], axis=1).max(axis=1)
    x["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    x["atr_pct"] = x.atr / x.close * 100
    x["vol_ma"] = x.volume.rolling(20).mean()
    x["vol_ratio"] = x.volume / x.vol_ma.replace(0, np.nan)
    return x


def pivots(d, left=3, right=3):
    highs, lows = [], []
    if len(d) < left + right + 10:
        return highs, lows
    for i in range(left, len(d) - right):
        if d.high.iloc[i] >= d.high.iloc[i-left:i+right+1].max():
            highs.append((i, float(d.high.iloc[i])))
        if d.low.iloc[i] <= d.low.iloc[i-left:i+right+1].min():
            lows.append((i, float(d.low.iloc[i])))
    return highs, lows


def structure(d):
    h, l = pivots(d)
    if len(h) < 2 or len(l) < 2:
        return "MIXED"
    h1, h2 = h[-2][1], h[-1][1]
    l1, l2 = l[-2][1], l[-1][1]
    if h2 > h1 and l2 > l1:
        return "HH + HL"
    if h2 < h1 and l2 < l1:
        return "LH + LL"
    if h2 > h1 and l2 <= l1:
        return "BULLISH DEVELOPING"
    if h2 <= h1 and l2 > l1:
        return "BEARISH DEVELOPING"
    return "MIXED"


def pct(d, bars):
    if len(d) <= bars:
        return np.nan
    return (d.close.iloc[-1] / d.close.iloc[-1-bars] - 1) * 100


def levels(d, current):
    h, l = pivots(d)
    s = sorted({round(v, 12) for _, v in l if v < current}, reverse=True)
    r = sorted({round(v, 12) for _, v in h if v > current})
    recent = d.tail(min(160, len(d)))
    s += [float(v) for v in recent.low.nsmallest(8) if v < current]
    r += [float(v) for v in recent.high.nlargest(8) if v > current]
    def cluster(vals):
        vals = sorted(set(vals))
        out = []
        for v in vals:
            if not out or abs(v-out[-1]) / max(abs(out[-1]), 1e-12) > 0.007:
                out.append(v)
            else:
                out[-1] = (out[-1] + v) / 2
        return out
    return sorted(cluster(s), reverse=True)[:3], sorted(cluster(r))[:3]


def quick_row(pair):
    try:
        d = candles(pair, "15m", 8)
        if len(d) < 100:
            return None
        return {
            "pair": pair,
            "m24": pct(d, 96),
            "m3": pct(d, min(288, len(d)-1)),
            "m7": pct(d, min(672, len(d)-1)),
        }
    except Exception:
        return None


def pump_rank(r):
    # Works with both quick-scan rows (m24/m3/m7) and deep-scan rows
    # (move24/move3d/move7d). Multi-day pump remains dominant.
    def get(name, fallback=0.0):
        try:
            if isinstance(r, dict):
                return float(r.get(name, fallback))
            if name in r.index:
                return float(r[name])
        except Exception:
            pass
        return float(fallback)

    m24 = get("m24", get("move24"))
    m3 = get("m3", get("move3d"))
    m7 = get("m7", get("move7d"))
    return max(m24, 0)*0.30 + max(m3, 0)*0.45 + max(m7, 0)*0.25


def deep_scan(pair):
    try:
        d15 = indicators(candles(pair, "15m", 12))
        d1h = indicators(candles(pair, "1H", 30))
        d4 = indicators(candles(pair, "4H", 150))
        d1 = indicators(candles(pair, "1D", 500))
        if len(d15) < 120 or len(d1h) < 100 or len(d4) < 40 or len(d1) < 30:
            return None

        current = float(d15.close.iloc[-1])
        s15, s1h, s4, s1 = structure(d15.tail(180)), structure(d1h.tail(120)), structure(d4.tail(80)), structure(d1.tail(50))
        e15, e1h, e4 = float(d15.ema20.iloc[-1]), float(d1h.ema20.iloc[-1]), float(d4.ema20.iloc[-1])
        r15, r1h, r4 = float(d15.rsi.iloc[-1]), float(d1h.rsi.iloc[-1]), float(d4.rsi.iloc[-1])
        atr15 = float(d15.atr.iloc[-1])
        vol = float(d15.vol_ratio.iloc[-1]) if np.isfinite(d15.vol_ratio.iloc[-1]) else np.nan
        m24, m3, m7 = pct(d15, 96), pct(d15, 288), pct(d15, min(672, len(d15)-1))
        peak = float(d15.tail(96).high.max())
        low = float(d15.tail(96).low.min())
        draw = (current/peak - 1)*100 if peak else np.nan
        recovery = (current/low - 1)*100 if low else np.nan
        h, l = pivots(d15.tail(180))
        local_hi = h[-1][1] if h else np.nan
        local_lo = l[-1][1] if l else np.nan
        s15l, r15l = levels(d15, current)
        s4l, r4l = levels(d4, current)
        s1l, r1l = levels(d1, current)

        # Pump score: the universe is already pump-only, but score intensity.
        pump = 0
        if m24 >= 15: pump += 2
        if m3 >= 30: pump += 3
        if m7 >= 50: pump += 3
        if r4 >= 70: pump += 2
        if current/e15 - 1 >= .06: pump += 2
        if vol >= 1.8: pump += 1

        short_score = pump
        if s15 == "LH + LL": short_score += 5
        elif s15 == "BEARISH DEVELOPING": short_score += 2
        if s1h == "LH + LL": short_score += 2
        if s4 == "LH + LL": short_score += 2
        if current < e15: short_score += 2
        if current < e1h: short_score += 1
        if -12 <= draw <= -2: short_score += 2
        if draw < -20: short_score -= 4
        if r15 < 40: short_score -= 2
        # Do not short directly into nearby support.
        nearest_s = s15l[0] if s15l else np.nan
        support_gap = (current/nearest_s - 1)*100 if np.isfinite(nearest_s) and nearest_s else np.nan
        if np.isfinite(support_gap) and support_gap < 1.5: short_score -= 3

        long_score = 0
        if s15 == "HH + HL": long_score += 5
        elif s15 == "BULLISH DEVELOPING": long_score += 2
        if s1h == "HH + HL": long_score += 2
        if s4 == "HH + HL": long_score += 2
        if current >= e15: long_score += 2
        if r15 > 70: long_score -= 2

        short_ready = (
            pump >= 5 and
            s15 in ("LH + LL", "BEARISH DEVELOPING") and
            current <= e15 * 1.01 and
            draw > -18 and
            (not np.isfinite(support_gap) or support_gap >= 1.5) and
            np.isfinite(local_lo)
        )

        if short_ready:
            setup, side = "SHORT READY", "SHORT"
            trigger = float(local_lo)
            invalid = max(float(e15), float(local_hi)) if np.isfinite(local_hi) else float(e15)
            thesis = "Hard pump is transitioning into bearish structure with a fresh low-break trigger."
        elif s4 == "LH + LL" and s15 in ("LH + LL", "BEARISH DEVELOPING"):
            setup, side = "SHORT WATCH", "SHORT"
            trigger = float(local_lo) if np.isfinite(local_lo) else np.nan
            invalid = float(e15)
            thesis = "Bearish structure exists; wait for a fresh EMA20 rejection and low break."
        else:
            setup, side = "WATCH", "WAIT"
            trigger = float(local_lo) if np.isfinite(local_lo) else np.nan
            invalid = float(e15)
            thesis = "Pump remains in the focused universe but the fresh SHORT transition is not confirmed."

        return dict(
            pair=pair, current=current, move24=m24, move3d=m3, move7d=m7,
            draw24=draw, recovery24=recovery, rsi15=r15, rsi1h=r1h, rsi4=r4,
            ema15dist=(current/e15-1)*100, ema1hdist=(current/e1h-1)*100,
            ema4dist=(current/e4-1)*100, atr15=atr15, atr_pct=(atr15/current*100),
            vol=vol, s15=s15, s1h=s1h, s4=s4, s1=s1,
            support15=s15l, resistance15=r15l, support4=s4l, resistance4=r4l,
            support1=s1l, resistance1=r1l, local_hi=local_hi, local_lo=local_lo,
            pump=pump, short_score=short_score, long_score=long_score,
            score=short_score, setup=setup, side=side, trigger=trigger,
            invalidation=invalid, thesis=thesis,
        )
    except Exception:
        return None


def scan_market():
    instruments = active_instruments()
    pairs = [x["pair"] for x in instruments]
    meta = {x["pair"]: x for x in instruments}
    quick = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fs = [ex.submit(quick_row, p) for p in pairs]
        for f in as_completed(fs):
            x = f.result()
            if x:
                quick.append(x)
    q = pd.DataFrame(quick)
    if q.empty:
        raise RuntimeError("No market data returned from CoinDCX.")
    q["pump_rank"] = q.apply(pump_rank, axis=1)
    candidates = q.sort_values(["pump_rank", "m3", "m24"], ascending=False).head(PUMP_FOCUS).pair.tolist()
    rows = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fs = [ex.submit(deep_scan, p) for p in candidates]
        for f in as_completed(fs):
            x = f.result()
            if x:
                rows.append(x)
    return q, pd.DataFrame(rows), meta


def fp(v):
    try:
        v = float(v)
        if not np.isfinite(v): return "—"
    except Exception:
        return "—"
    if abs(v) >= 1000: return f"{v:,.2f}"
    if abs(v) >= 1: return f"{v:,.4f}"
    if abs(v) >= .01: return f"{v:,.6f}"
    return f"{v:.10f}"


def lvl(vals):
    return " | ".join(fp(v) for v in vals) if vals else "—"


def normalize_position(row):
    def num(keys):
        for k in keys:
            try:
                if k in row and row[k] not in (None, ""):
                    return float(row[k])
            except Exception:
                pass
        return np.nan
    def txt(keys):
        for k in keys:
            if k in row and row[k] not in (None, ""):
                return str(row[k]).upper().strip()
        return ""
    pair = txt(["pair", "symbol", "instrument", "market", "contract"])
    side = txt(["side", "position_side", "direction"])
    qty = num(["quantity", "qty", "size", "position_size", "active_pos", "open_quantity"])
    entry = num(["entry_price", "avg_entry_price", "avgPrice", "average_entry_price"])
    mark = num(["mark_price", "markPrice", "last_price", "price"])
    pnl = num(["pnl", "unrealized_pnl", "unrealizedPnl", "active_pnl"])
    lev = num(["leverage", "lev"])
    return {"pair": pair, "side": side, "qty": qty, "entry": entry, "mark": mark, "pnl": pnl, "leverage": lev, "raw": row}


def fetch_positions():
    # The broad positions endpoint can contain zero-size contract rows.
    # We filter aggressively for actual non-zero positions.
    payload = signed_post(POSITIONS_ENDPOINT, {})
    rows = []
    if isinstance(payload, list): rows = payload
    elif isinstance(payload, dict):
        for k in ["data", "positions", "result", "active_positions"]:
            if isinstance(payload.get(k), list):
                rows = payload[k]
                break
        if not rows and any(k in payload for k in ["pair", "symbol", "instrument"]):
            rows = [payload]
    out = []
    for row in rows:
        if not isinstance(row, dict): continue
        p = normalize_position(row)
        if p["pair"] and np.isfinite(p["qty"]) and abs(p["qty"]) > 0:
            out.append(p)
    return out


def round_step(value, step):
    try:
        step = float(step)
        if step <= 0 or not np.isfinite(step): return float(value)
        return math.floor(float(value) / step) * step
    except Exception:
        return float(value)


def instrument_step(meta_row, names, default):
    for n in names:
        if isinstance(meta_row, dict) and n in meta_row:
            try:
                v = float(meta_row[n])
                if v > 0: return v
            except Exception: pass
    return default


def calc_trade_plan(r, risk_pct, max_leverage, atr_mult, min_rr):
    entry = float(r.trigger)
    if not np.isfinite(entry) or entry <= 0:
        return None
    atr = float(r.atr15)
    if not np.isfinite(atr) or atr <= 0:
        return None
    if r.side == "SHORT":
        structural = float(r.invalidation)
        stop = max(structural, entry + atr * atr_mult)
        supports = [x for x in r.support15 + r.support4 + r.support1 if np.isfinite(x) and x < entry]
        targets = sorted(set(supports), reverse=True)[:3]
        targets = [x for x in targets if (entry-x)/(stop-entry) >= min_rr]
        side = "SELL"
    else:
        structural = float(r.invalidation)
        stop = min(structural, entry - atr * atr_mult)
        resist = [x for x in r.resistance15 + r.resistance4 + r.resistance1 if np.isfinite(x) and x > entry]
        targets = sorted(set(resist))[:3]
        targets = [x for x in targets if (x-entry)/(entry-stop) >= min_rr]
        side = "BUY"
    risk_per_unit = abs(entry-stop)
    if risk_per_unit <= 0: return None
    rr = []
    for t in targets:
        rr.append(abs(entry-t)/risk_per_unit)
    return {
        "side": side, "entry": entry, "stop": stop,
        "targets": targets, "risk_per_unit": risk_per_unit,
        "rr": rr, "risk_pct": risk_pct, "max_leverage": max_leverage,
        "atr_mult": atr_mult,
    }


def create_market_order(pair, side, quantity, leverage, reduce_only=False, stop_loss=None, take_profit=None):
    if not LIVE_TRADING_ENABLED:
        return {"dry_run": True, "reason": "LIVE_TRADING_ENABLED=False", "pair": pair, "side": side, "quantity": quantity}
    order = {
        "side": side.lower(),
        "pair": pair,
        "order_type": "market_order",
        "price": 0,
        "quantity": quantity,
        "leverage": leverage,
        "notification": "no_notification",
        "hidden": False,
        "time_in_force": "good_till_cancel",
    }
    if reduce_only:
        order["reduce_only"] = True
    if stop_loss is not None:
        order["stop_loss_price"] = stop_loss
    if take_profit is not None:
        order["take_profit_price"] = take_profit
    return signed_post(ORDER_CREATE_ENDPOINT, {"order": order})


def close_position(pos):
    side = "sell" if pos["side"] in ("LONG", "BUY") else "buy"
    qty = abs(float(pos["qty"]))
    return create_market_order(pos["pair"], side, qty, max(1, int(pos["leverage"]) if np.isfinite(pos["leverage"]) else 1), reduce_only=True)


def journal_event(event):
    path = "coindcx_trade_journal.csv"
    row = pd.DataFrame([event])
    try:
        old = pd.read_csv(path)
        out = pd.concat([old, row], ignore_index=True)
    except Exception:
        out = row
    out.to_csv(path, index=False)


# ============================== UI ==========================================
st.set_page_config(page_title="CoinDCX Expert Trader V5", page_icon="🎯", layout="wide")
st.title("🎯 CoinDCX Expert Futures Trader V5")
st.caption("Top-pump universe → expert MTF analysis → your approval → execution → active position management.")

with st.sidebar:
    st.header("Bot Controls")
    workers = st.slider("Scan workers", 4, 20, MAX_WORKERS)
    risk_pct = st.number_input("Risk per trade (%)", 0.10, 2.00, DEFAULT_RISK_PCT, 0.05)
    max_daily_loss = st.number_input("Max daily loss (%)", 0.50, 10.0, DEFAULT_MAX_DAILY_LOSS_PCT, 0.25)
    max_positions = st.number_input("Max open positions", 1, 5, DEFAULT_MAX_OPEN_POSITIONS, 1)
    max_lev = st.number_input("Max leverage", 1.0, 20.0, DEFAULT_MAX_LEVERAGE, 0.5)
    min_rr = st.number_input("Minimum R:R", 1.0, 5.0, DEFAULT_MIN_RR, 0.1)
    atr_mult = st.number_input("Initial SL ATR multiplier", 0.5, 4.0, DEFAULT_ATR_STOP_MULT, 0.05)
    trail_atr = st.number_input("Trailing ATR multiplier", 0.5, 5.0, DEFAULT_TRAIL_ATR_MULT, 0.05)
    trail_r = st.number_input("Trailing activation (R)", 0.5, 5.0, DEFAULT_TRAIL_ACTIVATION_R, 0.25)
    partial_r = st.number_input("Partial profit at (R)", 1.0, 8.0, DEFAULT_PARTIAL_R, 0.25)
    partial_pct = st.slider("Partial exit (%)", 10, 90, DEFAULT_PARTIAL_PCT, 5)
    cooldown = st.number_input("Re-entry cooldown (minutes)", 0, 240, DEFAULT_COOLDOWN_MIN, 5)
    st.markdown("---")
    st.warning("Live trading is OFF by default.")
    live_toggle = st.checkbox("ENABLE LIVE TRADING", value=False)
    st.caption("Approval is still required for every new entry.")
    scan = st.button("🔎 SCAN TOP PUMPS", type="primary", use_container_width=True)
    refresh = st.button("🔄 REFRESH POSITIONS", use_container_width=True)
    close_all = st.button("🚨 CLOSE ALL OPEN POSITIONS", use_container_width=True)

LIVE_TRADING_ENABLED = bool(live_toggle)

if close_all:
    if not LIVE_TRADING_ENABLED:
        st.error("Live trading is OFF. No close order was sent.")
    else:
        try:
            positions = fetch_positions()
            results = []
            for p in positions:
                results.append(close_position(p))
            st.warning(f"Close-all requested for {len(positions)} detected position(s).")
            st.json(results)
        except Exception as e:
            st.error(f"Close-all failed: {e}")

if scan:
    with st.spinner("Scanning all active USDT Futures, then deep-scanning only the strongest pumps..."):
        try:
            MAX_WORKERS = workers
            q, df, meta = scan_market()
            st.session_state["q"] = q
            st.session_state["df"] = df
            st.session_state["meta"] = meta
            st.session_state["scan_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            st.error(f"Scan failed: {e}")

q = st.session_state.get("q")
df = st.session_state.get("df")
meta = st.session_state.get("meta", {})

if q is None or df is None or df.empty:
    st.info("Click SCAN TOP PUMPS to build the focused trading universe.")
else:
    st.success(f"Scan: {st.session_state.get('scan_time','—')} | Active movers reviewed: {len(q)} | Deep pump universe: {len(df)}")

    top = df.copy()
    top["pump_rank"] = top.apply(lambda x: pump_rank(x), axis=1)
    top = top.sort_values("pump_rank", ascending=False).head(10)
    st.subheader("🚀 TOP 10 PUMP UNIVERSE")
    st.dataframe(pd.DataFrame([{
        "Coin": r.pair,
        "24H": f"{r.move24:+.1f}%",
        "3D": f"{r.move3d:+.1f}%",
        "7D": f"{r.move7d:+.1f}%",
        "15m": r.s15,
        "1H": r.s1h,
        "4H": r.s4,
        "4H RSI": f"{r.rsi4:.1f}",
        "EMA20": f"{r.ema15dist:+.1f}%",
        "Setup": r.setup,
        "Score": int(r.short_score),
    } for _, r in top.iterrows()]), use_container_width=True, hide_index=True)

    shorts = df[df.setup.str.contains("SHORT", na=False)].copy()
    shorts["rank"] = shorts.short_score + shorts.pump * 0.5
    shorts = shorts.sort_values("rank", ascending=False).head(5)
    st.subheader("🔴 TOP 5 SHORT PROPOSALS")
    if shorts.empty:
        st.info("No fresh SHORT setup in the current pump universe. The agent will keep watching these pump coins.")
    else:
        st.dataframe(pd.DataFrame([{
            "Coin": r.pair,
            "Setup": r.setup,
            "Score": int(r.short_score),
            "24H": f"{r.move24:+.1f}%",
            "3D": f"{r.move3d:+.1f}%",
            "7D": f"{r.move7d:+.1f}%",
            "15m": r.s15,
            "1H": r.s1h,
            "4H": r.s4,
            "4H RSI": f"{r.rsi4:.1f}",
            "Trigger": fp(r.trigger),
        } for _, r in shorts.iterrows()]), use_container_width=True, hide_index=True)

    names = shorts.pair.tolist() if not shorts.empty else top.pair.tolist()
    selected = st.selectbox("Select setup", names)
    r = df[df.pair == selected].iloc[0]

    st.markdown(f"## {r.pair} — {r.setup}")
    c = st.columns(6)
    c[0].metric("Current", fp(r.current))
    c[1].metric("24H", f"{r.move24:+.2f}%")
    c[2].metric("3D", f"{r.move3d:+.2f}%")
    c[3].metric("7D", f"{r.move7d:+.2f}%")
    c[4].metric("4H RSI", f"{r.rsi4:.1f}")
    c[5].metric("ATR %", f"{r.atr_pct:.2f}%")

    st.write(f"**Thesis:** {r.thesis}")
    st.write(f"**Structure:** 15m `{r.s15}` | 1H `{r.s1h}` | 4H `{r.s4}` | 1D `{r.s1}`")
    st.write(f"**Momentum:** 15m RSI `{r.rsi15:.1f}` | 1H RSI `{r.rsi1h:.1f}` | 4H RSI `{r.rsi4:.1f}` | 15m volume `{r.vol:.1f}x`")

    plan = calc_trade_plan(r, risk_pct, max_lev, atr_mult, min_rr) if r.setup == "SHORT READY" else None
    if plan:
        st.markdown("### 🟠 TRADE PROPOSAL — APPROVAL REQUIRED")
        rr_text = " | ".join(f"{x:.2f}R" for x in plan["rr"]) if plan["rr"] else "No target meets minimum R:R"
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Entry Trigger", fp(plan["entry"]))
        p2.metric("Initial SL", fp(plan["stop"]))
        p3.metric("Risk / unit", fp(plan["risk_per_unit"]))
        p4.metric("R:R", rr_text)
        st.write(f"**Targets:** {', '.join(fp(x) for x in plan['targets']) if plan['targets'] else 'None'}")
        st.write(f"**Risk:** {risk_pct:.2f}% account | **Max leverage:** {max_lev:g}x | **Trailing:** structure + {trail_atr:.2f} ATR after {trail_r:.2f}R")
        st.warning("The trigger must actually occur before approval. The approval button is for a live order only after the trigger is confirmed.")

        if st.button(f"🟠 APPROVE {r.pair} SHORT", type="primary", use_container_width=True):
            if not LIVE_TRADING_ENABLED:
                st.error("Approval received, but LIVE TRADING is OFF. No order was sent.")
            else:
                # User sets margin in the UI immediately before execution.
                st.session_state["pending_plan"] = plan
                st.session_state["pending_pair"] = r.pair
                st.rerun()

        if st.session_state.get("pending_pair") == r.pair and st.session_state.get("pending_plan"):
            st.markdown("#### Confirm order sizing")
            margin = st.number_input("Margin to use (quote currency)", min_value=1.0, value=100.0, step=10.0, key=f"margin_{r.pair}")
            lev = st.number_input("Leverage for this trade", 1.0, float(max_lev), min(5.0, float(max_lev)), 0.5, key=f"lev_{r.pair}")
            if st.button("✅ SEND LIVE ORDER", type="primary", key=f"send_{r.pair}"):
                try:
                    qty = (margin * lev) / plan["entry"]
                    mrow = meta.get(r.pair, {})
                    qstep = instrument_step(mrow, ["quantity_increment", "quantity_step", "step_size", "qty_step"], 0.000001)
                    qty = round_step(qty, qstep)
                    if qty <= 0:
                        raise RuntimeError("Calculated quantity is zero after exchange precision rounding.")
                    result = create_market_order(
                        r.pair, "sell", qty, lev,
                        reduce_only=False,
                        stop_loss=plan["stop"],
                        take_profit=plan["targets"][0] if plan["targets"] else None,
                    )
                    journal_event({
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "pair": r.pair, "side": "SHORT", "entry_trigger": plan["entry"],
                        "stop": plan["stop"], "margin": margin, "leverage": lev,
                        "quantity": qty, "event": "ENTRY_ORDER", "response": str(result)[:1000],
                    })
                    st.success("Live order request sent. Check the position panel for the actual fill.")
                    st.json(result)
                    st.session_state.pop("pending_pair", None)
                    st.session_state.pop("pending_plan", None)
                except Exception as e:
                    st.error(f"Live order failed: {e}")
    else:
        st.info("No fresh approval-ready setup for this coin. The agent will not chase the pump.")

    st.markdown("### 📊 Support / Resistance")
    sr = []
    for tf, ss, rr in [("15m", r.support15, r.resistance15), ("4H", r.support4, r.resistance4), ("1D", r.support1, r.resistance1)]:
        for i, x in enumerate(ss, 1): sr.append({"TF": tf, "Level": f"S{i}", "Price": fp(x), "Distance": f"{(x/r.current-1)*100:+.2f}%"})
        for i, x in enumerate(rr, 1): sr.append({"TF": tf, "Level": f"R{i}", "Price": fp(x), "Distance": f"{(x/r.current-1)*100:+.2f}%"})
    st.dataframe(pd.DataFrame(sr), use_container_width=True, hide_index=True)

# ============================== POSITIONS ===================================
st.divider()
st.subheader("📌 LIVE POSITIONS")
if refresh:
    try:
        st.session_state["positions"] = fetch_positions()
    except Exception as e:
        st.error(f"Could not read positions: {e}")

positions = st.session_state.get("positions", [])
if not positions:
    st.info("Click REFRESH POSITIONS to read open CoinDCX Futures positions.")
else:
    st.dataframe(pd.DataFrame([{
        "Coin": p["pair"], "Side": p["side"], "Qty": fp(p["qty"]),
        "Entry": fp(p["entry"]), "Mark": fp(p["mark"]),
        "P&L": fp(p["pnl"]), "Leverage": fp(p["leverage"]),
    } for p in positions]), use_container_width=True, hide_index=True)
    for i, p in enumerate(positions):
        if st.button(f"Close {p['pair']} {p['side']}", key=f"close_{i}"):
            if not LIVE_TRADING_ENABLED:
                st.error("Live trading is OFF. No close order was sent.")
            else:
                try:
                    result = close_position(p)
                    journal_event({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "pair": p["pair"], "side": p["side"], "event": "MANUAL_CLOSE", "response": str(result)[:1000]})
                    st.success(f"Close request sent for {p['pair']}.")
                    st.json(result)
                except Exception as e:
                    st.error(f"Close failed: {e}")

st.divider()
st.caption("V5 is approval-gated. Live trading is OFF by default. Hard pump alone is never an entry; the strategy requires a fresh structure transition and trigger. Trading involves substantial risk, especially with leverage.")
