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
DEFAULT_DEFAULT_MARGIN = 1000.0
DEFAULT_MIN_RR = 2.0
DEFAULT_ATR_STOP_MULT = 1.25
DEFAULT_TRAIL_ATR_MULT = 1.50
DEFAULT_TRAIL_ACTIVATION_R = 1.0
DEFAULT_PARTIAL_R = 2.0
DEFAULT_PARTIAL_PCT = 50
DEFAULT_COOLDOWN_MIN = 30

# CoinDCX Futures API paths used by this implementation.
# CoinDCX has exposed different futures-position response shapes/endpoints over time.
# V5.5 tries the dedicated active-position route first, then the broad positions route.
# It only adopts rows with a genuinely non-zero position quantity.
# CoinDCX futures position APIs can require the margin currency and pagination.
# Your account is INR-M, so INR is tried first.
POSITION_ENDPOINTS = [
    "/exchange/v1/derivatives/futures/positions/active_positions",
    "/exchange/v1/derivatives/futures/positions",
]
POSITIONS_ENDPOINT = POSITION_ENDPOINTS[-1]  # backwards-compatible reference

POSITION_REQUESTS = [
    {"margin_currency_short_name": "INR", "page": 1, "size": 100},
    {"margin_currency_short_name": "INR"},
    {"margin_currency_short_name": "USDT", "page": 1, "size": 100},
    {"margin_currency_short_name": "USDT"},
    {"page": 1, "size": 100},
    {},
]
FUTURES_BALANCE_ENDPOINT = "/exchange/v1/derivatives/futures/wallets"
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

    # RSI
    delta = x.close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    x["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # ATR / volatility
    tr = pd.concat([
        x.high - x.low,
        (x.high - x.close.shift()).abs(),
        (x.low - x.close.shift()).abs(),
    ], axis=1).max(axis=1)
    x["tr"] = tr
    x["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    x["atr_pct"] = x.atr / x.close * 100

    # Volume
    x["vol_ma"] = x.volume.rolling(20).mean()
    x["vol_ratio"] = x.volume / x.vol_ma.replace(0, np.nan)

    # MACD
    ema12 = x.close.ewm(span=12, adjust=False).mean()
    ema26 = x.close.ewm(span=26, adjust=False).mean()
    x["macd"] = ema12 - ema26
    x["macd_signal"] = x.macd.ewm(span=9, adjust=False).mean()
    x["macd_hist"] = x.macd - x.macd_signal

    # Bollinger Bands
    x["bb_mid"] = x.close.rolling(20).mean()
    bb_std = x.close.rolling(20).std()
    x["bb_upper"] = x.bb_mid + 2 * bb_std
    x["bb_lower"] = x.bb_mid - 2 * bb_std
    x["bb_width"] = (x.bb_upper - x.bb_lower) / x.bb_mid.replace(0, np.nan)
    x["bb_pct"] = (x.close - x.bb_lower) / (x.bb_upper - x.bb_lower).replace(0, np.nan)

    # Stochastic
    low14 = x.low.rolling(14).min()
    high14 = x.high.rolling(14).max()
    x["stoch_k"] = 100 * (x.close - low14) / (high14 - low14).replace(0, np.nan)
    x["stoch_d"] = x.stoch_k.rolling(3).mean()

    # ADX / DMI
    up_move = x.high.diff()
    down_move = -x.low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr14 = x.atr.replace(0, np.nan)
    x["pdi"] = 100 * pd.Series(plus_dm, index=x.index).ewm(alpha=1/14, adjust=False).mean() / atr14
    x["mdi"] = 100 * pd.Series(minus_dm, index=x.index).ewm(alpha=1/14, adjust=False).mean() / atr14
    dx = 100 * (x.pdi - x.mdi).abs() / (x.pdi + x.mdi).replace(0, np.nan)
    x["adx"] = dx.ewm(alpha=1/14, adjust=False).mean()

    # Candle anatomy
    x["body"] = (x.close - x.open).abs()
    x["range"] = (x.high - x.low).replace(0, np.nan)
    x["upper_wick"] = x.high - x[["open", "close"]].max(axis=1)
    x["lower_wick"] = x[["open", "close"]].min(axis=1) - x.low
    x["body_pct"] = x.body / x["range"]

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



def _num(row, keys):
    for k in keys:
        try:
            v = row.get(k)
            if v not in (None, ""):
                v = float(v)
                if np.isfinite(v):
                    return v
        except Exception:
            pass
    return np.nan


def _txt(row, keys):
    for k in keys:
        v = row.get(k)
        if v not in (None, ""):
            return str(v).upper().strip()
    return ""


def _normalize_pair(pair):
    p = str(pair or "").upper().strip()
    p = p.replace("/", "-").replace("_", "-")
    if p.startswith("B-"):
        return p
    if p.endswith("_USDT"):
        return "B-" + p.replace("_", "-")
    if p.endswith("-USDT") and not p.startswith("B-"):
        return "B-" + p
    return p


def normalize_position(row):
    """
    Normalize several CoinDCX position schemas.

    We deliberately require a non-zero position quantity before adoption.
    Contract catalogue rows with active_pos=0 are ignored.
    """
    pair = _normalize_pair(_txt(row, [
        "pair", "symbol", "instrument", "market", "contract",
        "instrument_name", "contract_name", "product_symbol"
    ]))

    side = _txt(row, [
        "side", "position_side", "direction", "positionSide",
        "position_type", "trade_side"
    ])

    # Some schemas use signed quantity. Others expose side separately.
    qty = _num(row, [
        "quantity", "qty", "size", "position_size", "active_pos",
        "open_quantity", "positionQty", "position_qty", "current_qty",
        "net_qty", "net_position", "contracts"
    ])

    entry = _num(row, [
        "entry_price", "avg_entry_price", "avgPrice",
        "average_entry_price", "avg_entry", "entryPrice",
        "averagePrice", "open_price"
    ])
    mark = _num(row, [
        "mark_price", "markPrice", "last_price", "price",
        "current_price", "mark", "index_price"
    ])
    pnl = _num(row, [
        "pnl", "unrealized_pnl", "unrealizedPnl", "active_pnl",
        "unrealized_profit", "unrealizedProfit", "profit"
    ])
    lev = _num(row, ["leverage", "lev", "leverage_value"])
    margin = _num(row, [
        "margin", "margin_amount", "position_margin",
        "initial_margin", "isolated_margin"
    ])
    liq = _num(row, [
        "liq_price", "liquidation_price", "liquidationPrice",
        "liquidation_price_usdt"
    ])

    # If side is absent, infer it from a signed quantity.
    if side not in ("LONG", "SHORT"):
        if np.isfinite(qty) and qty != 0:
            side = "LONG" if qty > 0 else "SHORT"
    if np.isfinite(qty):
        qty = abs(qty)

    return {
        "pair": pair, "side": side, "qty": qty,
        "entry": entry, "mark": mark, "pnl": pnl,
        "leverage": lev, "margin": margin, "liq": liq,
        "raw": row
    }


def _walk_dicts(obj):
    """Yield every dictionary nested inside a CoinDCX JSON response."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk_dicts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_dicts(v)


def _extract_candidate_rows(payload):
    """Find position-like dictionaries anywhere in a nested API response."""
    candidates = []
    seen = set()

    for row in _walk_dicts(payload):
        keys = {str(k).lower() for k in row.keys()}
        has_pair = bool(keys.intersection({
            "pair", "symbol", "instrument", "market", "contract",
            "instrument_name", "contract_name", "product_symbol"
        }))
        has_position_field = bool(keys.intersection({
            "quantity", "qty", "size", "position_size", "active_pos",
            "open_quantity", "positionqty", "position_qty",
            "current_qty", "net_qty", "net_position", "contracts"
        }))
        has_entry = bool(keys.intersection({
            "entry_price", "avg_entry_price", "avgprice",
            "average_entry_price", "entryprice", "averageprice"
        }))
        if has_pair and (has_position_field or has_entry):
            ident = id(row)
            if ident not in seen:
                seen.add(ident)
                candidates.append(row)

    return candidates


def fetch_positions_with_diagnostics():
    """
    Try the active-position and broad-position APIs using the request shapes
    used by CoinDCX futures. INR-M is attempted first.

    A row is adopted only when it contains a recognizable pair and a
    genuinely non-zero position quantity.
    """
    diagnostics = []

    for endpoint in POSITION_ENDPOINTS:
        for request_body in POSITION_REQUESTS:
            try:
                payload = signed_post(endpoint, request_body)
                rows = _extract_candidate_rows(payload)
                normalized = [normalize_position(r) for r in rows]

                active = [
                    p for p in normalized
                    if p["pair"]
                    and np.isfinite(p["qty"])
                    and p["qty"] > 0
                    and p["side"] in ("LONG", "SHORT")
                ]

                diagnostics.append({
                    "endpoint": endpoint,
                    "margin": request_body.get("margin_currency_short_name", "default"),
                    "page": request_body.get("page", ""),
                    "http": "OK",
                    "candidate_rows": len(rows),
                    "active_nonzero": len(active),
                    "sample_pairs": ", ".join(sorted({
                        p["pair"] for p in normalized if p["pair"]
                    })[:10]),
                })

                if active:
                    return active, diagnostics

            except Exception as e:
                diagnostics.append({
                    "endpoint": endpoint,
                    "margin": request_body.get("margin_currency_short_name", "default"),
                    "page": request_body.get("page", ""),
                    "http": "ERROR",
                    "candidate_rows": 0,
                    "active_nonzero": 0,
                    "sample_pairs": "",
                    "error": str(e)[:500],
                })

    return [], diagnostics



def fetch_positions():
    positions, _ = fetch_positions_with_diagnostics()
    return positions



def account_balance_usdt():
    """Best-effort futures wallet read. Returns None if endpoint/schema differs."""
    try:
        payload = signed_post(FUTURES_BALANCE_ENDPOINT, {})
        rows = payload if isinstance(payload, list) else payload.get("data", payload.get("wallets", payload.get("result", []))) if isinstance(payload, dict) else []
        if isinstance(rows, dict): rows = [rows]
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            currency = str(row.get("currency_short_name") or row.get("currency") or row.get("asset") or row.get("short_name") or "").upper()
            if currency == "USDT":
                for k in ["available_balance", "available", "balance", "wallet_balance", "equity"]:
                    try:
                        v=float(row[k])
                        if np.isfinite(v): return v
                    except Exception:
                        pass
    except Exception:
        pass
    return None


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
    stop_pct = abs(stop-entry) / entry * 100 if entry else np.nan
    rec_lev = suggested_leverage(stop_pct, max_leverage, getattr(r, "atr_pct", np.nan))
    return {
        "side": side, "entry": entry, "stop": stop,
        "targets": targets, "risk_per_unit": risk_per_unit,
        "rr": rr, "risk_pct": risk_pct, "max_leverage": max_leverage,
        "atr_mult": atr_mult, "stop_pct": stop_pct, "suggested_leverage": rec_lev,
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




def candle_patterns(d):
    """Detect common completed-candle reversal/continuation patterns."""
    if len(d) < 5:
        return []
    x = d.iloc[-5:].copy()
    out = []
    a = x.iloc[-1]
    b = x.iloc[-2]
    c = x.iloc[-3]

    # Current candle anatomy
    body = max(float(a.body), 1e-12)
    rng = max(float(a.range), 1e-12)

    # Doji / indecision
    if float(a.body_pct) <= 0.12:
        out.append("DOJI / INDECISION")

    # Hammer / shooting star
    if float(a.lower_wick) >= body * 2 and float(a.upper_wick) <= body * 0.8:
        out.append("HAMMER")
    if float(a.upper_wick) >= body * 2 and float(a.lower_wick) <= body * 0.8:
        out.append("SHOOTING STAR")

    # Engulfing
    if b.close < b.open and a.close > a.open and a.open <= b.close and a.close >= b.open:
        out.append("BULLISH ENGULFING")
    if b.close > b.open and a.close < a.open and a.open >= b.close and a.close <= b.open:
        out.append("BEARISH ENGULFING")

    # Inside bar / breakout
    if a.high < b.high and a.low > b.low:
        out.append("INSIDE BAR")
    if a.close > b.high:
        out.append("BULLISH BREAKOUT")
    if a.close < b.low:
        out.append("BEARISH BREAKDOWN")

    # Three-candle reversal approximations
    if c.close < c.open and b.close < b.open and a.close > a.open and a.close > b.high:
        out.append("BULLISH REVERSAL SEQUENCE")
    if c.close > c.open and b.close > b.open and a.close < a.open and a.close < b.low:
        out.append("BEARISH REVERSAL SEQUENCE")

    return out


def divergence_signal(d, lookback=40):
    """Simple price/RSI and price/MACD divergence detector."""
    if len(d) < lookback + 5:
        return []
    x = d.tail(lookback)
    out = []
    h, l = pivots(x, left=2, right=2)

    if len(l) >= 2:
        p1, p2 = l[-2][1], l[-1][1]
        i1, i2 = l[-2][0], l[-1][0]
        r1, r2 = float(x.rsi.iloc[i1]), float(x.rsi.iloc[i2])
        if p2 < p1 and r2 > r1:
            out.append("BULLISH RSI DIVERGENCE")
        m1, m2 = float(x.macd.iloc[i1]), float(x.macd.iloc[i2])
        if p2 < p1 and m2 > m1:
            out.append("BULLISH MACD DIVERGENCE")

    if len(h) >= 2:
        p1, p2 = h[-2][1], h[-1][1]
        i1, i2 = h[-2][0], h[-1][0]
        r1, r2 = float(x.rsi.iloc[i1]), float(x.rsi.iloc[i2])
        if p2 > p1 and r2 < r1:
            out.append("BEARISH RSI DIVERGENCE")
        m1, m2 = float(x.macd.iloc[i1]), float(x.macd.iloc[i2])
        if p2 > p1 and m2 < m1:
            out.append("BEARISH MACD DIVERGENCE")
    return out


def analyze_all_patterns(d15, d1h, d4, d1, side):
    """Build a multi-factor pattern score used for position-exit decisions."""
    frames = {"15m": d15, "1H": d1h, "4H": d4, "1D": d1}
    score = 0
    evidence = []
    warnings = []

    for tf, d in frames.items():
        if len(d) < 30:
            continue
        row = d.iloc[-1]
        prev = d.iloc[-2]

        s = structure(d.tail(min(180, len(d))))
        rsi = float(row.rsi)
        macd_h = float(row.macd_hist)
        macd_prev = float(prev.macd_hist)
        adx = float(row.adx) if np.isfinite(row.adx) else np.nan
        pdi = float(row.pdi) if np.isfinite(row.pdi) else np.nan
        mdi = float(row.mdi) if np.isfinite(row.mdi) else np.nan
        vr = float(row.vol_ratio) if np.isfinite(row.vol_ratio) else np.nan
        bbp = float(row.bb_pct) if np.isfinite(row.bb_pct) else np.nan
        sk = float(row.stoch_k) if np.isfinite(row.stoch_k) else np.nan

        if side == "SHORT":
            if s == "HH + HL":
                score += 3 if tf in ("15m", "1H") else 2
                warnings.append(f"{tf}: HH+HL")
            elif s == "LH + LL":
                score -= 3 if tf in ("15m", "1H") else 2
                evidence.append(f"{tf}: LH+LL")

            if float(row.close) > float(row.ema20):
                score += 2 if tf == "15m" else 1
                warnings.append(f"{tf}: above EMA20")
            else:
                evidence.append(f"{tf}: below EMA20")

            if rsi >= 55:
                score -= 1
                warnings.append(f"{tf}: RSI {rsi:.1f} still strong")
            elif rsi <= 42:
                score += 1
                evidence.append(f"{tf}: RSI {rsi:.1f} weakening")

            if macd_h > 0:
                score -= 1
            if macd_prev > 0 and macd_h < macd_prev:
                score -= 1
                evidence.append(f"{tf}: MACD momentum fading")

            if np.isfinite(pdi) and np.isfinite(mdi):
                if pdi > mdi:
                    score += 1
                    warnings.append(f"{tf}: buyers dominate DMI")
                else:
                    score -= 1
                    evidence.append(f"{tf}: sellers dominate DMI")

            if np.isfinite(adx) and adx >= 25 and np.isfinite(mdi) and np.isfinite(pdi) and mdi > pdi:
                score -= 1
                evidence.append(f"{tf}: bearish ADX trend")
        else:
            if s == "LH + LL":
                score += 3 if tf in ("15m", "1H") else 2
                warnings.append(f"{tf}: LH+LL")
            elif s == "HH + HL":
                score -= 3 if tf in ("15m", "1H") else 2
                evidence.append(f"{tf}: HH+HL")

            if float(row.close) < float(row.ema20):
                score += 2 if tf == "15m" else 1
                warnings.append(f"{tf}: below EMA20")
            else:
                evidence.append(f"{tf}: above EMA20")

            if rsi <= 45:
                score -= 1
                warnings.append(f"{tf}: RSI {rsi:.1f} still weak")
            elif rsi >= 58:
                score += 1
                evidence.append(f"{tf}: RSI {rsi:.1f} recovering")

            if macd_h < 0:
                score -= 1
            if macd_prev < 0 and macd_h > macd_prev:
                score -= 1
                evidence.append(f"{tf}: MACD momentum improving")

            if np.isfinite(pdi) and np.isfinite(mdi):
                if mdi > pdi:
                    score += 1
                    warnings.append(f"{tf}: sellers dominate DMI")
                else:
                    score -= 1
                    evidence.append(f"{tf}: buyers dominate DMI")

            if np.isfinite(adx) and adx >= 25 and np.isfinite(mdi) and np.isfinite(pdi) and pdi > mdi:
                score -= 1
                evidence.append(f"{tf}: bullish ADX trend")

        # Volume climax / reversal context
        pats = candle_patterns(d)
        divs = divergence_signal(d)
        if side == "SHORT":
            if "BEARISH ENGULFING" in pats or "SHOOTING STAR" in pats or "BEARISH REVERSAL SEQUENCE" in pats:
                score -= 2
                evidence.append(f"{tf}: bearish candle reversal")
            if "BULLISH ENGULFING" in pats or "HAMMER" in pats or "BULLISH REVERSAL SEQUENCE" in pats:
                score += 2
                warnings.append(f"{tf}: bullish candle reversal")
            if any("BEARISH" in x for x in divs):
                score -= 2
                evidence.append(f"{tf}: bearish divergence")
            if any("BULLISH" in x for x in divs):
                score += 2
                warnings.append(f"{tf}: bullish divergence")
        else:
            if "BULLISH ENGULFING" in pats or "HAMMER" in pats or "BULLISH REVERSAL SEQUENCE" in pats:
                score -= 2
                evidence.append(f"{tf}: bullish candle reversal")
            if "BEARISH ENGULFING" in pats or "SHOOTING STAR" in pats or "BEARISH REVERSAL SEQUENCE" in pats:
                score += 2
                warnings.append(f"{tf}: bearish candle reversal")
            if any("BULLISH" in x for x in divs):
                score -= 2
                evidence.append(f"{tf}: bullish divergence")
            if any("BEARISH" in x for x in divs):
                score += 2
                warnings.append(f"{tf}: bearish divergence")

        if np.isfinite(vr) and vr >= 2.5:
            evidence.append(f"{tf}: high volume {vr:.1f}x")

        # Keep BB/Stoch observations visible without letting them dominate.
        if np.isfinite(bbp):
            if side == "SHORT" and bbp > 0.95:
                warnings.append(f"{tf}: near upper Bollinger Band")
            if side == "LONG" and bbp < 0.05:
                warnings.append(f"{tf}: near lower Bollinger Band")
        if np.isfinite(sk):
            if side == "SHORT" and sk > 80:
                warnings.append(f"{tf}: stochastic overbought")
            if side == "LONG" and sk < 20:
                warnings.append(f"{tf}: stochastic oversold")

    return {
        "pattern_score": int(score),
        "evidence": evidence[-12:],
        "warnings": warnings[-12:],
    }



def adopt_existing_positions():
    """
    Read the exchange now and place every real non-zero position into the
    agent's managed-position state. This does not place, add to, or resize
    any position.
    """
    positions, diagnostics = fetch_positions_with_diagnostics()
    st.session_state["positions"] = positions
    st.session_state["position_diagnostics"] = diagnostics
    st.session_state["positions_last_sync"] = time.strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["positions_adopted"] = bool(positions)
    return positions, diagnostics


def position_manager_signal(pos, auto_manage=False):
    """
    Comprehensive existing-position decision engine.

    It evaluates structure, EMA alignment, RSI, MACD, DMI/ADX, volume,
    Bollinger position, stochastic, candle patterns, divergences, ATR and
    multi-timeframe agreement before deciding HOLD / TRAIL_CANDIDATE /
    EXIT_SIGNAL.

    Negative P&L alone is never an exit reason.
    Liquidation distance is a safety metric, not a directional signal.
    No averaging down or margin addition is performed.
    """
    pair = pos.get("pair", "")
    side = pos.get("side", "")
    entry = float(pos.get("entry", np.nan))
    mark = float(pos.get("mark", np.nan))
    if not pair or side not in ("SHORT", "LONG") or not np.isfinite(entry) or not np.isfinite(mark):
        return {"action": "HOLD", "reason": "Insufficient position data."}

    try:
        d15 = indicators(candles(pair, "15m", 10))
        d1h = indicators(candles(pair, "1H", 30))
        d4 = indicators(candles(pair, "4H", 150))
        d1 = indicators(candles(pair, "1D", 500))
    except Exception as e:
        return {"action": "HOLD", "reason": f"Market-data error: {e}"}

    if min(len(d15), len(d1h), len(d4), len(d1)) < 30:
        return {"action": "HOLD", "reason": "Not enough completed candles for full pattern analysis."}

    s15 = structure(d15.tail(180))
    s1h = structure(d1h.tail(120))
    s4 = structure(d4.tail(80))
    s1 = structure(d1.tail(50))

    e15 = float(d15.ema20.iloc[-1])
    atr15 = float(d15.atr.iloc[-1])
    r15 = float(d15.rsi.iloc[-1])
    row15 = d15.iloc[-1]

    h, l = pivots(d15.tail(180))
    last_hi = h[-1][1] if h else np.nan
    last_lo = l[-1][1] if l else np.nan

    favorable = ((entry - mark) / entry * 100) if side == "SHORT" else ((mark - entry) / entry * 100)
    risk_pct = abs(mark-entry)/entry*100 if entry else np.nan
    patterns15 = candle_patterns(d15)
    div15 = divergence_signal(d15)

    allp = analyze_all_patterns(d15, d1h, d4, d1, side)
    score = int(allp["pattern_score"])

    # A second, explicit exit score makes the decision easier to audit.
    exit_score = 0
    exit_reasons = []
    hold_reasons = []

    if side == "SHORT":
        if s15 in ("HH + HL", "BULLISH DEVELOPING"):
            exit_score += 4
            exit_reasons.append("15m bullish structure")
        if s1h == "HH + HL":
            exit_score += 3
            exit_reasons.append("1H HH+HL")
        if s4 == "HH + HL":
            exit_score += 2
            exit_reasons.append("4H HH+HL")
        if mark > e15:
            exit_score += 3
            exit_reasons.append("price above 15m EMA20")
        if r15 >= 58:
            exit_score += 2
            exit_reasons.append(f"15m RSI strong ({r15:.1f})")
        if float(row15.macd_hist) > float(d15.macd_hist.iloc[-2]):
            exit_score += 1
            exit_reasons.append("MACD momentum improving")
        if "BULLISH ENGULFING" in patterns15 or "HAMMER" in patterns15:
            exit_score += 2
            exit_reasons.append("bullish reversal candle")
        if any("BULLISH" in x for x in div15):
            exit_score += 2
            exit_reasons.append("bullish divergence")
        if np.isfinite(row15.pdi) and np.isfinite(row15.mdi) and row15.pdi > row15.mdi:
            exit_score += 1
            exit_reasons.append("buyers dominate DMI")

        # Bearish evidence keeps the trade alive.
        if s15 == "LH + LL": hold_reasons.append("15m LH+LL")
        if s1h == "LH + LL": hold_reasons.append("1H LH+LL")
        if s4 == "LH + LL": hold_reasons.append("4H LH+LL")
        if mark < e15: hold_reasons.append("below 15m EMA20")
        if r15 < 50: hold_reasons.append(f"15m RSI not strong ({r15:.1f})")
        if np.isfinite(row15.mdi) and np.isfinite(row15.pdi) and row15.mdi > row15.pdi:
            hold_reasons.append("sellers dominate DMI")
    else:
        if s15 in ("LH + LL", "BEARISH DEVELOPING"):
            exit_score += 4
            exit_reasons.append("15m bearish structure")
        if s1h == "LH + LL":
            exit_score += 3
            exit_reasons.append("1H LH+LL")
        if s4 == "LH + LL":
            exit_score += 2
            exit_reasons.append("4H LH+LL")
        if mark < e15:
            exit_score += 3
            exit_reasons.append("price below 15m EMA20")
        if r15 <= 42:
            exit_score += 2
            exit_reasons.append(f"15m RSI weak ({r15:.1f})")
        if float(row15.macd_hist) < float(d15.macd_hist.iloc[-2]):
            exit_score += 1
            exit_reasons.append("MACD momentum weakening")
        if "BEARISH ENGULFING" in patterns15 or "SHOOTING STAR" in patterns15:
            exit_score += 2
            exit_reasons.append("bearish reversal candle")
        if any("BEARISH" in x for x in div15):
            exit_score += 2
            exit_reasons.append("bearish divergence")
        if np.isfinite(row15.mdi) and np.isfinite(row15.pdi) and row15.mdi > row15.pdi:
            exit_score += 1
            exit_reasons.append("sellers dominate DMI")

        if s15 == "HH + HL": hold_reasons.append("15m HH+HL")
        if s1h == "HH + HL": hold_reasons.append("1H HH+HL")
        if s4 == "HH + HL": hold_reasons.append("4H HH+HL")
        if mark > e15: hold_reasons.append("above 15m EMA20")
        if r15 > 50: hold_reasons.append(f"15m RSI healthy ({r15:.1f})")
        if np.isfinite(row15.pdi) and np.isfinite(row15.mdi) and row15.pdi > row15.mdi:
            hold_reasons.append("buyers dominate DMI")

    # Confirmed swing invalidation is stronger than one noisy candle.
    if side == "SHORT" and np.isfinite(last_hi) and mark > last_hi:
        exit_score += 3
        exit_reasons.append("latest 15m swing high broken")
    if side == "LONG" and np.isfinite(last_lo) and mark < last_lo:
        exit_score += 3
        exit_reasons.append("latest 15m swing low broken")

    # Pattern score is supporting evidence, not a standalone trigger.
    if side == "SHORT" and score >= 7:
        exit_score += 2
        exit_reasons.append(f"multi-factor bullish reversal score {score}")
    if side == "LONG" and score >= 7:
        exit_score += 2
        exit_reasons.append(f"multi-factor bearish reversal score {score}")

    # Profit-aware trailing candidate.
    trail = np.nan
    if favorable > 0 and np.isfinite(atr15):
        if side == "SHORT":
            candidates = [e15]
            if np.isfinite(last_hi):
                candidates.append(last_hi)
            trail = max(candidates) + 0.35 * atr15
        else:
            candidates = [e15]
            if np.isfinite(last_lo):
                candidates.append(last_lo)
            trail = min(candidates) - 0.35 * atr15

    # Decision thresholds:
    # >=8 = strong multi-factor invalidation; 5-7 = warning, continue watching.
    if exit_score >= 8:
        action = "EXIT_SIGNAL"
        close_allowed = bool(auto_manage)
        reason = "Strong multi-factor exit: " + "; ".join(exit_reasons[:6])
    elif favorable > 0 and exit_score >= 5:
        action = "TRAIL_CANDIDATE"
        close_allowed = False
        reason = "Trend is weakening while profitable; tighten the trailing protection. " + "; ".join(exit_reasons[:5])
    else:
        action = "HOLD"
        close_allowed = False
        reason = "Thesis still has supporting evidence. " + "; ".join(hold_reasons[:6])

    return {
        "pair": pair, "side": side, "entry": entry, "mark": mark,
        "pnl_pct_from_entry": favorable,
        "move_against_entry_pct": risk_pct,
        "structure15": s15, "structure1h": s1h, "structure4h": s4, "structure1d": s1,
        "rsi15": r15, "ema20": e15, "atr15": atr15,
        "macd_hist15": float(row15.macd_hist),
        "adx15": float(row15.adx) if np.isfinite(row15.adx) else np.nan,
        "vol_ratio15": float(row15.vol_ratio) if np.isfinite(row15.vol_ratio) else np.nan,
        "patterns15": ", ".join(patterns15) if patterns15 else "None",
        "divergence15": ", ".join(div15) if div15 else "None",
        "pattern_score": score, "exit_score": exit_score,
        "last_hi": last_hi, "last_lo": last_lo, "trail_stop": trail,
        "action": action, "close_allowed": close_allowed,
        "reason": reason,
        "exit_reasons": "; ".join(exit_reasons[:8]),
        "hold_reasons": "; ".join(hold_reasons[:8]),
        "evidence": "; ".join(allp["evidence"][-8:]),
        "warnings": "; ".join(allp["warnings"][-8:]),
    }


def manage_existing_positions(positions, auto_manage=False):
    results = []
    for p in positions:
        sig = position_manager_signal(p, auto_manage=auto_manage)
        results.append(sig)
        if sig.get("action") == "EXIT_SIGNAL" and sig.get("close_allowed"):
            if not LIVE_TRADING_ENABLED:
                sig["close_allowed"] = False
                sig["reason"] += " | Live Trading is OFF, so no close order was sent."
            else:
                try:
                    response = close_position(p)
                    sig["close_response"] = str(response)[:1000]
                    journal_event({
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "pair": p["pair"], "side": p["side"],
                        "event": "AUTO_CLOSE",
                        "reason": sig.get("reason", ""),
                        "response": str(response)[:1000],
                    })
                except Exception as e:
                    sig["close_error"] = str(e)
    return results


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
st.caption("Top-pump universe → expert MTF analysis → approval → execution → existing-position takeover → trailing → exit → re-entry.")

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
    st.markdown("### 💰 INR-M Capital")
    st.caption("This account is INR-M. New trades use only the INR margin you explicitly enter.")
    available_inr = st.number_input(
        "Available INR margin (read from CoinDCX)",
        min_value=0.0, value=float(st.session_state.get("available_inr", 0.0)), step=10.0,
        key="available_inr_input"
    )
    investment_inr = st.number_input(
        "How much INR do you want to invest in this trade?",
        min_value=0.0, value=500.0, step=50.0, key="investment_inr"
    )
    if investment_inr > available_inr:
        st.error("Investment exceeds available INR margin. The agent will not use locked margin or other wallet funds.")
    else:
        st.caption(f"New-trade margin: ₹{investment_inr:,.2f}. Remaining wallet funds are untouched.")
    st.markdown("### 🛡️ Existing-position takeover")
    takeover = st.checkbox("TAKE OVER EXISTING POSITIONS", value=True,
                            help="Manage existing positions without adding margin or averaging down.")
    auto_manage = st.checkbox("AUTO-MANAGE ADOPTED POSITIONS", value=False,
                              help="When enabled, the manager may close an adopted position if its predefined exit/invalidation rules trigger. It never adds margin.")
    continuous_manage = st.checkbox(
        "CONTINUOUS POSITION MONITOR (15s)",
        value=False,
        help="Re-sync and re-evaluate adopted positions every 15 seconds. Auto-close still requires LIVE TRADING ON."
    )
    st.caption("Takeover does NOT add margin, average down, or increase position size.")
    scan = st.button("🔎 SCAN TOP PUMPS", type="primary", use_container_width=True)
    refresh = st.button("🔄 REFRESH POSITIONS", use_container_width=True)
    manage_now = st.button("🧠 MANAGE OPEN POSITIONS NOW", use_container_width=True)
    close_all = st.button("🚨 CLOSE ALL OPEN POSITIONS", use_container_width=True)

LIVE_TRADING_ENABLED = bool(live_toggle)

# ------------------------ REAL POSITION TAKEOVER ----------------------------
# On the first page load, synchronize actual non-zero positions automatically.
# This is read-only unless AUTO-MANAGE + LIVE TRADING are both enabled.
if takeover and "positions_last_sync" not in st.session_state:
    try:
        _adopted, _diag = adopt_existing_positions()
    except Exception as _e:
        st.session_state["positions"] = []
        st.session_state["position_diagnostics"] = [{
            "endpoint": "startup_sync",
            "http": "ERROR",
            "candidate_rows": 0,
            "active_nonzero": 0,
            "sample_pairs": "",
            "error": str(_e)[:500],
        }]



if refresh:
    try:
        _adopted, _diag = adopt_existing_positions()
        if _adopted:
            st.success(f"🟢 Position synchronization complete: {len(_adopted)} real position(s) adopted.")
        else:
            st.warning("No non-zero positions were returned. Review Position API Diagnostics.")
    except Exception as _e:
        st.error(f"Position synchronization failed: {_e}")

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
        notional_preview, stop_pct_preview, loss_preview = trade_risk(investment, plan["suggested_leverage"], plan["entry"], plan["stop"])
        st.write(f"**Stop distance:** {plan['stop_pct']:.2f}% | **Suggested leverage:** **{plan['suggested_leverage']:.1f}x** | **Max allowed:** {max_lev:g}x")
        rp1, rp2, rp3, rp4 = st.columns(4)
        rp1.metric("Your investment", f"₹{investment_inr:,.0f}")
        rp2.metric("Suggested leverage", f"{plan['suggested_leverage']:.1f}x")
        rp3.metric("Approx. notional", f"{notional_preview:,.2f} USDT")
        rp4.metric("Approx. SL loss", f"₹{loss_preview*usdtinr:,.0f}")
        st.caption("Leverage is a risk/position-sizing suggestion based on stop distance and volatility; it is not a guarantee. Fees, funding, slippage and liquidation can change actual results.")
        st.warning("Before any live order, you choose the capital to commit. The agent will not use the rest of your wallet. The trigger must be confirmed before execution.")

        lev = st.number_input("Leverage you want to use", 1.0, float(max_lev), float(plan["suggested_leverage"]), 0.5, key=f"lev_{r.pair}")
        if lev != plan["suggested_leverage"]:
            n2, s2, l2 = trade_risk(investment, lev, plan["entry"], plan["stop"])
            st.info(f"With {lev:.1f}x: notional ≈ {n2:,.2f} USDT; approximate stop loss ≈ ₹{l2*usdtinr:,.0f} ({s2:.2f}% price move).")

        if st.button(f"🟠 APPROVE {r.pair} SHORT", type="primary", use_container_width=True):
            if not LIVE_TRADING_ENABLED:
                st.error("Approval received, but LIVE TRADING is OFF. No order was sent.")
            else:
                st.session_state["pending_plan"] = plan
                st.session_state["pending_pair"] = r.pair
                st.session_state["pending_margin"] = investment
                st.session_state["pending_leverage"] = lev
                st.rerun()

        if st.session_state.get("pending_pair") == r.pair and st.session_state.get("pending_plan"):
            st.markdown("#### Final order confirmation")
            margin = float(st.session_state.get("pending_margin", investment))
            lev = float(st.session_state.get("pending_leverage", plan["suggested_leverage"]))
            final_notional, final_stop_pct, final_loss = trade_risk(margin, lev, plan["entry"], plan["stop"])
            st.write(f"**Invest:** ₹{investment_inr:,.0f} ({margin:,.2f} USDT) | **Leverage:** {lev:.1f}x | **Notional:** {final_notional:,.2f} USDT | **Approx. SL loss:** ₹{final_loss*usdtinr:,.0f}")
            if margin > available_inr:
                st.error("Investment amount is greater than available INR margin. Reduce the investment amount.")
            if st.button("✅ CONFIRM & SEND LIVE ORDER", type="primary", key=f"send_{r.pair}"):
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


# ------------------------ CONTINUOUS MANAGER --------------------------------
if continuous_manage and takeover:
    if hasattr(st, "fragment"):
        @st.fragment(run_every="15s")
        def _continuous_position_manager():
            try:
                live_positions, diag = adopt_existing_positions()
                if live_positions:
                    results = manage_existing_positions(live_positions, auto_manage=auto_manage)
                    st.caption(
                        f"🟢 Manager cycle {time.strftime('%H:%M:%S')} | "
                        f"Adopted {len(live_positions)} position(s) | "
                        f"Live trading: {'ON' if LIVE_TRADING_ENABLED else 'OFF'}"
                    )
                    st.dataframe(pd.DataFrame([{
                        "Coin": r.get("pair"),
                        "Side": r.get("side"),
                        "Action": r.get("action"),
                        "Exit score": r.get("exit_score"),
                        "P&L": f"{r.get('pnl_pct_from_entry', float('nan')):+.2f}%",
                        "Reason": r.get("reason"),
                    } for r in results]), use_container_width=True, hide_index=True)
                else:
                    st.caption(f"Manager cycle {time.strftime('%H:%M:%S')}: no verified open positions returned.")
            except Exception as e:
                st.error(f"Continuous position manager error: {e}")
        _continuous_position_manager()
    else:
        st.info("Your Streamlit version does not support st.fragment(run_every=...). Use MANAGE OPEN POSITIONS NOW for each manager cycle.")

# ============================== POSITIONS ===================================
st.divider()
st.subheader("🛡️ Existing Position Takeover")

if takeover:
    st.success("TAKEOVER MODE: ON — INR-M active-position synchronization is enabled.")
else:
    st.info("TAKEOVER MODE: OFF")

if st.session_state.get("positions_last_sync"):
    st.caption(
        f"Last position synchronization: {st.session_state['positions_last_sync']} | "
        f"Adopted positions: {len(st.session_state.get('positions', []))}"
    )

positions = st.session_state.get("positions", [])

if positions:
    st.success("🟢 REAL OPEN POSITION(S) ADOPTED")
    st.dataframe(pd.DataFrame([{
        "Coin": p["pair"],
        "Side": p["side"],
        "Qty": fp(p["qty"]),
        "Entry": fp(p["entry"]),
        "Mark": fp(p["mark"]),
        "P&L": fp(p["pnl"]),
        "Leverage": fp(p["leverage"]),
        "Margin": fp(p.get("margin", np.nan)),
        "Liq. Price": fp(p.get("liq", np.nan)),
        "Status": "ADOPTED",
    } for p in positions]), use_container_width=True, hide_index=True)
else:
    st.warning(
        "The agent currently has NO verified open position to manage. "
        "This does not mean your CoinDCX position is closed; it means the private API response "
        "has not supplied a non-zero position row yet."
    )

    diagnostics = st.session_state.get("position_diagnostics", [])
    if diagnostics:
        st.markdown("#### 🔎 Position API Diagnostics")
        st.dataframe(pd.DataFrame(diagnostics), use_container_width=True, hide_index=True)

        st.caption(
            "INR-M is tried first with CoinDCX's margin_currency_short_name parameter, "
            "followed by USDT and minimal requests. The agent will not guess a position "
            "from a contract catalogue; it adopts only a verified non-zero position."
        )

if takeover and positions:
    # Always refresh the exchange state before evaluating an existing position.
    if manage_now:
        try:
            positions, diagnostics = adopt_existing_positions()
        except Exception as _e:
            st.error(f"Could not synchronize before management: {_e}")
    st.markdown("### 🧠 Position Manager")
    st.caption(
        "The adopted position is analyzed continuously when you run MANAGE OPEN POSITIONS NOW. "
        "No margin is added and quantity is never increased."
    )

    if manage_now or auto_manage:
        try:
            management = manage_existing_positions(positions, auto_manage=auto_manage)
            st.dataframe(pd.DataFrame([{
                "Coin": x.get("pair"),
                "Side": x.get("side"),
                "Action": x.get("action"),
                "P&L from entry": f"{x.get('pnl_pct_from_entry', float('nan')):+.2f}%",
                "15m": x.get("structure15"),
                "1H": x.get("structure1h"),
                "4H": x.get("structure4h"),
                "1D": x.get("structure1d"),
                "RSI": fp(x.get("rsi15")),
                "MACD": fp(x.get("macd_hist15")),
                "ADX": fp(x.get("adx15")),
                "Vol": fp(x.get("vol_ratio15")),
                "Patterns": x.get("patterns15"),
                "Divergence": x.get("divergence15"),
                "Pattern score": x.get("pattern_score"),
                "Exit score": x.get("exit_score"),
                "Trail": fp(x.get("trail_stop", np.nan)),
                "Decision": x.get("reason"),
                "Auto-close allowed": "YES" if x.get("close_allowed") else "NO",
            } for x in management]), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Position manager failed: {e}")

for i, p in enumerate(positions):
    with st.expander(f"{p['pair']} — {p['side']}"):
        st.write({
            "Entry": p.get("entry"),
            "Mark": p.get("mark"),
            "Quantity": p.get("qty"),
            "Leverage": p.get("leverage"),
            "Margin": p.get("margin"),
            "Liquidation": p.get("liq"),
            "P&L": p.get("pnl"),
        })
        if st.button("Close this position", key=f"close_{i}"):
            try:
                resp = close_position(p)
                st.success("Close request sent.")
                st.json(resp)
                st.session_state["positions"] = [x for j, x in enumerate(positions) if j != i]
            except Exception as e:
                st.error(f"Close failed: {e}")

if st.button("🔄 SYNC & ADOPT POSITIONS NOW", use_container_width=True):
    try:
        positions, diagnostics = adopt_existing_positions()
        if positions:
            st.success(f"🟢 Adopted {len(positions)} real open position(s).")
        else:
            st.warning("No verified non-zero positions returned. Review diagnostics.")
        st.rerun()
    except Exception as e:
        st.error(f"Position synchronization failed: {e}")

if st.button("🚨 CLOSE ALL OPEN POSITIONS", use_container_width=True):
    try:
        positions = fetch_positions()
        results = []
        for p in positions:
            results.append(close_position(p))
        st.warning(f"Close requests sent for {len(results)} verified positions.")
    except Exception as e:
        st.error(f"Close-all failed: {e}")

