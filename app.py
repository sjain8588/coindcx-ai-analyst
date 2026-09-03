import streamlit as st
import pandas as pd
import numpy as np
import requests
import time

API = "https://api.coindcx.com"
PUBLIC = "https://public.coindcx.com"

st.set_page_config(
    page_title="CoinDCX Futures Scanner",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 CoinDCX Futures Momentum Scanner")
st.caption("Analysis only • Momentum + ATH/ATL • No trading")

MEME_WORDS = {
    "DOGE","SHIB","PEPE","BONK","FLOKI","WIF","BOME","MEME",
    "BRETT","MOG","TURBO","MEW","NEIRO","BABYDOGE",
    "1000SHIB","1000PEPE","1000BONK","1000FLOKI","1000LUNC",
    "PONKE","MYRO","SLERF","LADYS","DEGEN","MOTHER","MAGA","TRUMP"
}


# =============================================================================
# COINDCX API
# =============================================================================

@st.cache_data(ttl=30)
def active_instruments(margin="USDT"):
    r = requests.get(
        f"{API}/exchange/v1/derivatives/futures/data/active_instruments",
        params=[("margin_currency_short_name[]", margin)],
        timeout=20
    )
    r.raise_for_status()
    x = r.json()

    if not isinstance(x, list):
        raise RuntimeError(f"Unexpected instruments response: {x}")

    return x


@st.cache_data(ttl=5)
def futures_prices():
    r = requests.get(
        f"{PUBLIC}/market_data/v3/current_prices/futures/rt",
        timeout=20
    )
    r.raise_for_status()

    x = r.json()

    return x.get("prices", {}) if isinstance(x, dict) else {}


@st.cache_data(ttl=30)
def candles(pair, resolution, start_ts, end_ts):
    params = {
        "pair": pair,
        "from": int(start_ts),
        "to": int(end_ts),
        "resolution": resolution,
        "pcode": "f"
    }

    r = requests.get(
        f"{PUBLIC}/market_data/candlesticks",
        params=params,
        timeout=30
    )

    r.raise_for_status()

    x = r.json()
    rows = x.get("data", []) if isinstance(x, dict) else x

    if not isinstance(rows, list):
        raise RuntimeError(
            f"Unexpected candle response for {pair}: {x}"
        )

    d = pd.DataFrame(rows)

    if d.empty:
        return d

    for c in ["open", "high", "low", "close", "volume"]:
        if c not in d:
            raise RuntimeError(
                f"{pair} candle response missing {c}"
            )

        d[c] = pd.to_numeric(d[c], errors="coerce")

    d["time"] = pd.to_datetime(
        d["time"],
        unit="ms",
        errors="coerce"
    )

    return (
        d.dropna(
            subset=[
                "time",
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]
        )
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )


@st.cache_data(ttl=5)
def orderbook(pair):
    r = requests.get(
        f"{PUBLIC}/market_data/v3/orderbook/{pair}-futures/50",
        timeout=15
    )
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=5)
def trades(pair):
    r = requests.get(
        f"{API}/exchange/v1/derivatives/futures/data/trades",
        params={"pair": pair},
        timeout=15
    )
    r.raise_for_status()

    x = r.json()
    return x if isinstance(x, list) else []


# =============================================================================
# INDICATORS
# =============================================================================

def indicators(d):
    x = d.copy()

    for n in [20, 50, 100, 200]:
        x[f"ema{n}"] = (
            x.close
            .ewm(span=n, adjust=False)
            .mean()
        )

    delta = x.close.diff()

    gain = (
        delta.clip(lower=0)
        .ewm(alpha=1 / 14, adjust=False)
        .mean()
    )

    loss = (
        -delta.clip(upper=0)
        .ewm(alpha=1 / 14, adjust=False)
        .mean()
    )

    rs = gain / loss.replace(0, np.nan)

    x["rsi"] = 100 - 100 / (1 + rs)

    e12 = (
        x.close
        .ewm(span=12, adjust=False)
        .mean()
    )

    e26 = (
        x.close
        .ewm(span=26, adjust=False)
        .mean()
    )

    x["macd"] = e12 - e26

    x["macd_signal"] = (
        x.macd
        .ewm(span=9, adjust=False)
        .mean()
    )

    tr = pd.concat(
        [
            x.high - x.low,
            (x.high - x.close.shift()).abs(),
            (x.low - x.close.shift()).abs()
        ],
        axis=1
    ).max(axis=1)

    x["atr"] = (
        tr
        .ewm(alpha=1 / 14, adjust=False)
        .mean()
    )

    x["volma"] = (
        x.volume
        .rolling(20)
        .mean()
    )

    x["bbmid"] = (
        x.close
        .rolling(20)
        .mean()
    )

    x["bbstd"] = (
        x.close
        .rolling(20)
        .std()
    )

    x["bbup"] = (
        x.bbmid +
        2 * x.bbstd
    )

    x["bblow"] = (
        x.bbmid -
        2 * x.bbstd
    )

    up = x.high.diff()
    dn = -x.low.diff()

    plus = pd.Series(
        np.where(
            (up > dn) & (up > 0),
            up,
            0.0
        ),
        index=x.index
    )

    minus = pd.Series(
        np.where(
            (dn > up) & (dn > 0),
            dn,
            0.0
        ),
        index=x.index
    )

    atr = x.atr.replace(0, np.nan)

    pdi = (
        100 *
        plus.ewm(alpha=1 / 14, adjust=False).mean()
        /
        atr
    )

    mdi = (
        100 *
        minus.ewm(alpha=1 / 14, adjust=False).mean()
        /
        atr
    )

    dx = (
        100 *
        (pdi - mdi).abs()
        /
        (pdi + mdi).replace(0, np.nan)
    )

    x["adx"] = (
        dx
        .ewm(alpha=1 / 14, adjust=False)
        .mean()
    )

    x["pdi"] = pdi
    x["mdi"] = mdi

    return x


# =============================================================================
# STRUCTURE
# =============================================================================

def structure(d):
    if len(d) < 40:
        return "Mixed"

    recent = d.tail(12)
    prior = d.iloc[-36:-12]

    if (
        recent.high.max() > prior.high.max()
        and
        recent.low.min() > prior.low.min()
    ):
        return "Bullish"

    if (
        recent.high.max() < prior.high.max()
        and
        recent.low.min() < prior.low.min()
    ):
        return "Bearish"

    return "Mixed"


# =============================================================================
# TECHNICAL SCORE
# =============================================================================

def tech(d):
    x = d.iloc[-1]

    long_points = 0
    short_points = 0

    pairs = [
        (x.close, x.ema20),
        (x.ema20, x.ema50),
        (x.ema50, x.ema100),
        (x.ema100, x.ema200)
    ]

    for a, b in pairs:
        if pd.notna(a) and pd.notna(b):
            long_points += int(a > b)
            short_points += int(a < b)

    long_points += (
        int(50 < x.rsi < 70)
        +
        int(x.macd > x.macd_signal)
        +
        int(x.pdi > x.mdi)
        +
        int(
            pd.notna(x.volma)
            and
            x.volume > x.volma
        )
    )

    short_points += (
        int(30 < x.rsi < 50)
        +
        int(x.macd < x.macd_signal)
        +
        int(x.mdi > x.pdi)
        +
        int(
            pd.notna(x.volma)
            and
            x.volume > x.volma
        )
    )

    return int(long_points), int(short_points)


# =============================================================================
# FUTURES MICROSTRUCTURE
# =============================================================================

def micro(pair):
    ob = None
    flow = None

    try:
        b = orderbook(pair)

        B = [
            (float(p), float(q))
            for p, q in b.get("bids", {}).items()
        ]

        A = [
            (float(p), float(q))
            for p, q in b.get("asks", {}).items()
        ]

        if B and A:
            bid = max(p for p, q in B)
            ask = min(p for p, q in A)
            mid = (bid + ask) / 2

            nb = sum(
                q
                for p, q in B
                if p >= mid * 0.995
            )

            na = sum(
                q
                for p, q in A
                if p <= mid * 1.005
            )

            if nb + na:
                ob = (
                    nb - na
                ) / (
                    nb + na
                )

    except Exception:
        pass

    try:
        buy = 0
        sell = 0

        for t in trades(pair):
            q = float(
                t.get("quantity", 0) or 0
            )

            if bool(
                t.get("is_maker", False)
            ):
                sell += q
            else:
                buy += q

        if buy + sell:
            flow = (
                buy - sell
            ) / (
                buy + sell
            )

    except Exception:
        pass

    return ob, flow


# =============================================================================
# SUPPORT / RESISTANCE
# =============================================================================

def sr_levels(d):
    x = (
        d
        .sort_values("time")
        .reset_index(drop=True)
        .copy()
    )

    price = float(x.iloc[-1].close)

    if (
        "atr" in x
        and
        pd.notna(x.iloc[-1].atr)
    ):
        atr = float(x.iloc[-1].atr)
    else:
        atr = price * 0.02

    tolerance = max(
        atr * 0.60,
        price * 0.002
    )

    levels = []

    for i in range(2, len(x) - 2):
        window = x.iloc[i - 2:i + 3]

        if (
            x.iloc[i].high
            >=
            window.high.max()
        ):
            levels.append(
                (
                    float(x.iloc[i].high),
                    "R",
                    x.iloc[i].time,
                    False
                )
            )

        if (
            x.iloc[i].low
            <=
            window.low.min()
        ):
            levels.append(
                (
                    float(x.iloc[i].low),
                    "S",
                    x.iloc[i].time,
                    False
                )
            )

    try:
        monthly = (
            x
            .set_index("time")
            .resample("ME")
            .agg(
                {
                    "high": "max",
                    "low": "min"
                }
            )
            .dropna()
        )
    except Exception:
        monthly = (
            x
            .set_index("time")
            .resample("M")
            .agg(
                {
                    "high": "max",
                    "low": "min"
                }
            )
            .dropna()
        )

    for t, r in monthly.iterrows():
        levels.append(
            (
                float(r.high),
                "R",
                t,
                True
            )
        )

        levels.append(
            (
                float(r.low),
                "S",
                t,
                True
            )
        )

    def cluster(kind):
        raw = sorted(
            [
                z
                for z in levels
                if z[1] == kind
            ],
            key=lambda z: z[0]
        )

        output = []

        for p, k, t, monthly_flag in raw:
            if (
                not output
                or
                abs(
                    p -
                    output[-1]["price"]
                ) > tolerance
            ):
                output.append(
                    {
                        "price": p,
                        "touches": 1,
                        "last": t,
                        "monthly": monthly_flag
                    }
                )
            else:
                c = output[-1]

                c["price"] = (
                    c["price"] *
                    c["touches"] +
                    p
                ) / (
                    c["touches"] + 1
                )

                c["touches"] += 1
                c["last"] = max(
                    c["last"],
                    t
                )

                c["monthly"] = (
                    c["monthly"]
                    or
                    monthly_flag
                )

        return output

    supports = sorted(
        [
            c
            for c in cluster("S")
            if c["price"] < price * 0.9995
        ],
        key=lambda c: c["price"],
        reverse=True
    )

    resistances = sorted(
        [
            c
            for c in cluster("R")
            if c["price"] > price * 1.0005
        ],
        key=lambda c: c["price"]
    )

    below = monthly.low[
        monthly.low < price
    ]

    above = monthly.high[
        monthly.high > price
    ]

    return {
        "support1":
            supports[0]["price"]
            if supports
            else np.nan,

        "support2":
            supports[1]["price"]
            if len(supports) > 1
            else np.nan,

        "resistance1":
            resistances[0]["price"]
            if resistances
            else np.nan,

        "resistance2":
            resistances[1]["price"]
            if len(resistances) > 1
            else np.nan,

        "monthly_support":
            float(below.max())
            if not below.empty
            else np.nan,

        "monthly_resistance":
            float(above.min())
            if not above.empty
            else np.nan,

        "ath":
            float(x.high.max()),

        "atl":
            float(x.low.min())
    }


# =============================================================================
# CANDIDATE SCORING
# =============================================================================

def score(pair, p, d1, h1, m15):
    x1 = d1.iloc[-1]
    xh = h1.iloc[-1]
    x15 = m15.iloc[-1]

    change = float(
        p.get("pc", 0) or 0
    )

    move = abs(change)

    b1, s1 = tech(d1)
    bh, sh = tech(h1)
    b15, s15 = tech(m15)

    ob, flow = micro(pair)

    side = (
        "LONG"
        if change > 0
        else
        "SHORT"
    )

    if side == "LONG":
        raw = (
            move * 1.5
            +
            (b1 - s1) * 7
            +
            (bh - sh) * 5
            +
            (b15 - s15) * 3
        )

        if ob is not None:
            raw += max(0, ob) * 8

        if flow is not None:
            raw += max(0, flow) * 6

    else:
        raw = (
            move * 1.5
            +
            (s1 - b1) * 7
            +
            (sh - bh) * 5
            +
            (s15 - b15) * 3
        )

        if ob is not None:
            raw += max(0, -ob) * 8

        if flow is not None:
            raw += max(0, -flow) * 6

    atr = (
        float(x1.atr)
        if pd.notna(x1.atr)
        else float(x1.close) * 0.01
    )

    atr = max(atr, 1e-12)

    ext = abs(
        float(
            x1.close -
            x1.ema20
        )
    ) / atr

    raw -= min(
        25,
        max(
            0,
            (ext - 2) * 7
        )
    )

    confidence = max(
        35,
        min(
            95,
            50 + raw * 0.45
        )
    )

    history_factor = min(
        1,
        min(
            len(d1) / 120,
            len(h1) / 240,
            len(m15) / 240
        )
    )

    confidence = min(
        confidence,
        50 + 45 * history_factor
    )

    if history_factor < 0.5:
        confidence = min(
            confidence,
            65
        )

    aligned = (
        (
            side == "LONG"
            and
            b1 > s1
            and
            bh >= sh
            and
            b15 >= s15
            and
            x15.macd > x15.macd_signal
        )
        or
        (
            side == "SHORT"
            and
            s1 > b1
            and
            sh >= bh
            and
            s15 >= b15
            and
            x15.macd < x15.macd_signal
        )
    )

    extreme = (
        (
            side == "LONG"
            and
            x1.rsi >= 78
        )
        or
        (
            side == "SHORT"
            and
            x1.rsi <= 22
        )
    )

    signal = (
        side
        if (
            aligned
            and
            not extreme
            and
            confidence >= 68
        )
        else
        "WAIT"
    )

    price = float(x15.close)

    atr15 = (
        float(x15.atr)
        if pd.notna(x15.atr)
        else price * 0.01
    )

    atr15 = max(atr15, 1e-12)

    sl = np.nan
    tp1 = np.nan
    tp2 = np.nan

    if signal == "LONG":
        sl = price - 1.4 * atr15
        tp1 = price + 1.5 * (price - sl)
        tp2 = price + 2.5 * (price - sl)

    elif signal == "SHORT":
        sl = price + 1.4 * atr15
        tp1 = price - 1.5 * (sl - price)
        tp2 = price - 2.5 * (sl - price)

    return {
        "pair": pair,
        "change": change,
        "side": side,
        "signal": signal,
        "confidence": confidence,
        "price": price,
        "rsi": float(x1.rsi),
        "adx": float(xh.adx),
        "structure": structure(d1),
        "ob": ob,
        "flow": flow,
        "score": raw,
        "entry": price,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        **sr_levels(d1),
        "d1": d1,
        "h1": h1,
        "m15": m15
    }


# =============================================================================
# HELPERS
# =============================================================================

def fmt(v):
    try:
        if pd.isna(v):
            return "—"

        return (
            f"{float(v):,.8f}"
            .rstrip("0")
            .rstrip(".")
        )

    except Exception:
        return "—"


def normalize(s):
    q = (
        s
        .strip()
        .upper()
        .replace("/", "")
        .replace("-", "")
        .replace("_", "")
    )

    for quote in (
        "USDT",
        "INR",
        "USDC"
    ):
        if (
            q.endswith(quote)
            and
            len(q) > len(quote)
        ):
            return q[:-len(quote)]

    return q


def coin_matches(
    pair,
    symbol,
    requested,
    quote
):
    req = normalize(requested)

    names = [
        str(pair)
        .upper()
        .replace("-", "")
        .replace("_", ""),

        str(symbol)
        .upper()
        .replace("-", "")
        .replace("_", "")
    ]

    for n in names:
        variants = [n]

        if n.startswith("B"):
            variants.append(n[1:])

        if n.startswith("I"):
            variants.append(n[1:])

        for v in variants:
            if v == req:
                return True

            if v == req + quote:
                return True

            if v.startswith(req + quote):
                return True

    return False


# =============================================================================
# SETTINGS
# =============================================================================

margin = st.selectbox(
    "Futures margin market",
    ["USDT", "INR"],
    index=0
)

meme_only = st.checkbox(
    "Meme-focused scan",
    value=True
)


# =============================================================================
# MY INVESTED COIN
# =============================================================================

st.divider()

st.header(
    "💼 My Invested Coin — Long / Short Check"
)

st.caption(
    "Analyze a coin you already hold using Futures "
    "trend, momentum, structure and liquidity."
)

coin = st.text_input(
    "Coin / Futures pair",
    placeholder="DOGE, SHIB, PEPE, B-DOGE_USDT"
)

avg = st.number_input(
    "Optional: your average entry price",
    min_value=0.0,
    value=0.0,
    step=0.00000001,
    format="%.8f"
)

if st.button("📊 Analyze My Coin"):

    try:
        prices = futures_prices()

        req = normalize(coin)

        markets = [margin] + [
            q
            for q in ("USDT", "INR")
            if q != margin
        ]

        found = []

        for q in markets:

            for pair in active_instruments(q):

                p = prices.get(pair)

                if not p:
                    continue

                symbol = str(
                    p.get("mkt", pair)
                ).upper()

                if coin_matches(
                    pair,
                    symbol,
                    req,
                    q
                ):
                    found.append(
                        (
                            pair,
                            p,
                            symbol,
                            q
                        )
                    )

        if not found:

            st.error(
                f"No active CoinDCX Futures "
                f"contract found for '{coin}'."
            )

            st.stop()

        found.sort(
            key=lambda z: (
                0 if z[3] == margin else 1,
                len(z[0])
            )
        )

        pair, p, symbol, found_margin = found[0]

        now = int(time.time())

        d1 = candles(
            pair,
            "1D",
            now - 400 * 86400,
            now
        )

        h1 = candles(
            pair,
            "60",
            now - 120 * 86400,
            now
        )

        m15 = candles(
            pair,
            "15",
            now - 30 * 86400,
            now
        )

        if (
            len(d1) < 10
            or
            len(h1) < 30
            or
            len(m15) < 30
        ):
            st.error(
                f"Too little history for {symbol}: "
                f"1D={len(d1)}, "
                f"1H={len(h1)}, "
                f"15m={len(m15)}"
            )

            st.stop()

        x = score(
            pair,
            p,
            indicators(d1),
            indicators(h1),
            indicators(m15)
        )

        st.subheader(
            f"{symbol} • {pair}"
        )

        if avg > 0:
            st.metric(
                "Your position vs current price",
                f"{(x['price'] - avg) / avg * 100:+.2f}%"
            )

        if x["signal"] == "LONG":
            st.success("🟢 LONG BIAS")
        elif x["signal"] == "SHORT":
            st.error("🔴 SHORT BIAS")
        else:
            st.warning("🟡 WAIT")

        a, b, c, d = st.columns(4)

        a.metric("Current", fmt(x["price"]))
        b.metric("24h", f"{x['change']:+.2f}%")
        c.metric("Confidence", f"{x['confidence']:.0f}%")
        d.metric("1D RSI", f"{x['rsi']:.1f}")

        st.write(
            f"**1D structure:** {x['structure']} | "
            f"**1H ADX:** {x['adx']:.1f} | "
            f"**S1/S2:** {fmt(x['support1'])} / "
            f"{fmt(x['support2'])} | "
            f"**R1/R2:** {fmt(x['resistance1'])} / "
            f"{fmt(x['resistance2'])}"
        )

        a, b, c, d = st.columns(4)

        a.metric(
            "Monthly Support",
            fmt(x["monthly_support"])
        )

        b.metric(
            "Monthly Resistance",
            fmt(x["monthly_resistance"])
        )

        c.metric("ATH", fmt(x["ath"]))
        d.metric("ATL", fmt(x["atl"]))

        if x["signal"] in ("LONG", "SHORT"):

            a, b, c, d = st.columns(4)

            a.metric("Entry", fmt(x["entry"]))
            b.metric("Stop", fmt(x["sl"]))
            c.metric("TP1", fmt(x["tp1"]))
            d.metric("TP2", fmt(x["tp2"]))

        with st.expander("Advanced analysis"):

            st.write(
                f"EMA20/50/100/200: "
                f"{fmt(x['d1'].iloc[-1].ema20)} / "
                f"{fmt(x['d1'].iloc[-1].ema50)} / "
                f"{fmt(x['d1'].iloc[-1].ema100)} / "
                f"{fmt(x['d1'].iloc[-1].ema200)}"
            )

            ob_text = (
                "N/A"
                if x["ob"] is None
                else f"{x['ob'] * 100:.2f}%"
            )

            flow_text = (
                "N/A"
                if x["flow"] is None
                else f"{x['flow'] * 100:.2f}%"
            )

            st.write(
                f"Order-book imbalance: {ob_text}"
            )

            st.write(
                f"Trade-flow proxy: {flow_text}"
            )

            st.line_chart(
                x["d1"]
                .set_index("time")
                [
                    [
                        "close",
                        "ema20",
                        "ema50",
                        "ema100",
                        "ema200"
                    ]
                ]
                .tail(250)
            )

    except Exception as e:

        st.error(
            f"My Coin analysis failed: "
            f"{type(e).__name__}: {e}"
        )


# =============================================================================
# TOP FUTURES MOMENTUM SCANNER
# =============================================================================

if st.button(
    "🔍 Scan Top Futures",
    type="primary"
):

    try:

        active = active_instruments(margin)
        prices = futures_prices()

        rows = []

        for pair in active:

            p = prices.get(pair)

            if not p:
                continue

            symbol = str(
                p.get("mkt", pair)
            ).upper()

            if (
                meme_only
                and
                not any(
                    word in symbol
                    or
                    word in pair.upper()
                    for word in MEME_WORDS
                )
            ):
                continue

            try:
                pc = float(
                    p.get("pc", 0) or 0
                )
            except Exception:
                continue

            if pc:
                rows.append(
                    (
                        pair,
                        p,
                        symbol,
                        pc
                    )
                )

        rows.sort(
            key=lambda z: abs(z[3]),
            reverse=True
        )

        results = []
        failures = []

        now = int(time.time())

        # Scan up to 80 candidates.
        for (
            pair,
            p,
            symbol,
            pc
        ) in rows[:80]:

            try:

                d1 = candles(
                    pair,
                    "1D",
                    now - 400 * 86400,
                    now
                )

                h1 = candles(
                    pair,
                    "60",
                    now - 120 * 86400,
                    now
                )

                m15 = candles(
                    pair,
                    "15",
                    now - 30 * 86400,
                    now
                )

                # IMPORTANT:
                # Correct variable names are d1, h1 and m15.
                if (
                    len(d1) < 10
                    or
                    len(h1) < 30
                    or
                    len(m15) < 30
                ):

                    failures.append(
                        f"{symbol}: insufficient candles "
                        f"1D={len(d1)}, "
                        f"1H={len(h1)}, "
                        f"15m={len(m15)}"
                    )

                    continue

                results.append(
                    (
                        symbol,
                        score(
                            pair,
                            p,
                            indicators(d1),
                            indicators(h1),
                            indicators(m15)
                        )
                    )
                )

                if len(results) >= 5:
                    break

            except Exception as e:

                failures.append(
                    f"{symbol}: "
                    f"{type(e).__name__}: {e}"
                )

        if not results:

            st.error(
                "No Futures candidate could be analyzed."
            )

            with st.expander(
                "🔧 Scan diagnostics",
                expanded=True
            ):

                st.write(
                    f"Checked up to "
                    f"{min(len(rows), 80)} "
                    f"candidate contracts."
                )

                st.code(
                    "\n".join(
                        failures[:80]
                    )
                    if failures
                    else
                    "No diagnostics returned."
                )

            st.stop()

        results.sort(
            key=lambda z: z[1]["score"],
            reverse=True
        )

        st.header(
            "🎯 Today's Top 5 Futures Candidates"
        )

        for i, (
            symbol,
            x
        ) in enumerate(
            results[:5],
            1
        ):

            icon = (
                "🟢"
                if x["signal"] == "LONG"
                else
                "🔴"
                if x["signal"] == "SHORT"
                else
                "🟡"
            )

            with st.container(border=True):

                st.subheader(
                    f"{i}. {symbol} "
                    f"{icon} {x['signal']}"
                )

                a, b, c, d = st.columns(4)

                a.metric(
                    "24h",
                    f"{x['change']:+.2f}%"
                )

                b.metric(
                    "Confidence",
                    f"{x['confidence']:.0f}%"
                )

                c.metric(
                    "Price",
                    fmt(x["price"])
                )

                d.metric(
                    "RSI",
                    f"{x['rsi']:.1f}"
                )

                st.write(
                    f"**S1/S2:** "
                    f"{fmt(x['support1'])} / "
                    f"{fmt(x['support2'])} | "
                    f"**R1/R2:** "
                    f"{fmt(x['resistance1'])} / "
                    f"{fmt(x['resistance2'])} | "
                    f"**1D:** {x['structure']} | "
                    f"**1H ADX:** {x['adx']:.1f}"
                )

                if x["signal"] in ("LONG", "SHORT"):

                    a, b, c, d = st.columns(4)

                    a.metric("Entry", fmt(x["entry"]))
                    b.metric("Stop", fmt(x["sl"]))
                    c.metric("TP1", fmt(x["tp1"]))
                    d.metric("TP2", fmt(x["tp2"]))

                with st.expander("Advanced analysis"):

                    st.line_chart(
                        x["d1"]
                        .set_index("time")
                        [
                            [
                                "close",
                                "ema20",
                                "ema50",
                                "ema100",
                                "ema200"
                            ]
                        ]
                        .tail(250)
                    )

    except Exception as e:

        st.error(
            f"Scanner failed: "
            f"{type(e).__name__}: {e}"
        )


# =============================================================================
# ATH / ATL HELPERS
# =============================================================================

def current_price(p):

    for key in (
        "ls",
        "lp",
        "last_price",
        "price",
        "mp",
        "mark_price"
    ):

        try:

            value = float(
                p.get(key, 0) or 0
            )

            if value > 0:
                return value

        except Exception:
            pass

    return 0.0


def extreme_status(
    current,
    extreme,
    mode
):

    if (
        extreme is None
        or
        pd.isna(extreme)
        or
        extreme <= 0
    ):
        return None, np.nan

    distance = (
        current -
        extreme
    ) / extreme * 100

    if mode == "ATH":

        if distance > 0:
            return (
                "🔥 ATH BREAKOUT",
                distance
            )

        if distance >= -0.5:
            return (
                "🟢 AT ATH",
                distance
            )

        if distance >= -5:
            return (
                "🟡 NEAR ATH",
                distance
            )

    else:

        if distance < 0:
            return (
                "🔴 ATL BREAKDOWN",
                distance
            )

        if distance <= 0.5:
            return (
                "🟠 AT ATL",
                distance
            )

        if distance <= 5:
            return (
                "🟡 NEAR ATL",
                distance
            )

    return None, distance


# =============================================================================
# IMPROVED ATH / ATL SCANNER
# =============================================================================

def scan_extremes(
    active,
    prices,
    mode,
    limit=150
):

    rows = []

    # =========================================================
    # BUILD COMPLETE FUTURES UNIVERSE
    # =========================================================

    for pair in active:

        p = prices.get(pair)

        if not p:
            continue

        symbol = str(
            p.get("mkt", pair)
        ).upper()

        try:

            pc = float(
                p.get("pc", 0) or 0
            )

            cur = current_price(p)

        except Exception:

            continue

        if cur <= 0:
            continue

        # ATH/ATL scanner deliberately ignores meme_only.
        rows.append(
            (
                pair,
                p,
                symbol,
                pc,
                cur
            )
        )

    if not rows:
        return [], []

    # =========================================================
    # PRE-FILTER
    # =========================================================

    if mode == "ATH":

        rows.sort(
            key=lambda z: z[3],
            reverse=True
        )

    else:

        rows.sort(
            key=lambda z: z[3]
        )

    # Scan up to 150 contracts.
    rows = rows[:limit]

    now = int(time.time())

    output = []
    failures = []

    # =========================================================
    # DAILY HISTORY
    # =========================================================

    for (
        pair,
        p,
        symbol,
        pc,
        cur
    ) in rows:

        try:

            d = candles(
                pair,
                "1D",
                now - 400 * 86400,
                now
            )

            if len(d) < 2:

                failures.append(
                    f"{symbol}: only {len(d)} daily candles"
                )

                continue

            # Exclude latest daily candle.
            history = d.iloc[:-1]

            if history.empty:
                continue

            previous_ath = float(
                history.high.max()
            )

            previous_atl = float(
                history.low.min()
            )

            # =================================================
            # 7-DAY PERFORMANCE
            # =================================================

            if len(d) >= 8:
                close7 = float(
                    d.iloc[-8].close
                )
            else:
                close7 = float(
                    d.iloc[0].close
                )

            if close7 > 0:

                change7 = (
                    (
                        cur -
                        close7
                    )
                    /
                    close7
                    *
                    100
                )

            else:

                change7 = np.nan

            # =================================================
            # ATH
            # =================================================

            if mode == "ATH":

                distance = (
                    (
                        cur -
                        previous_ath
                    )
                    /
                    previous_ath
                    *
                    100
                )

                if distance > 0:

                    status = "🔥 ATH BREAKOUT"

                elif distance >= -0.5:

                    status = "🟢 AT ATH"

                elif distance >= -5:

                    status = "🟡 NEAR ATH"

                else:

                    continue

                extreme = previous_ath

            # =================================================
            # ATL
            # =================================================

            else:

                distance = (
                    (
                        cur -
                        previous_atl
                    )
                    /
                    previous_atl
                    *
                    100
                )

                if distance < 0:

                    status = "🔴 ATL BREAKDOWN"

                elif distance <= 0.5:

                    status = "🟠 AT ATL"

                elif distance <= 5:

                    status = "🟡 NEAR ATL"

                elif pc <= -8:

                    status = "🔻 STRONG DOWN"

                else:

                    continue

                extreme = previous_atl

            # =================================================
            # INDICATORS
            # =================================================

            ind = indicators(d)

            last = ind.iloc[-1]

            if (
                pd.notna(last.volma)
                and
                float(last.volma) > 0
            ):

                volume_ratio = (
                    float(last.volume)
                    /
                    float(last.volma)
                )

            else:

                volume_ratio = np.nan

            output.append(
                {
                    "symbol": symbol,
                    "pair": pair,
                    "price": cur,
                    "extreme": extreme,
                    "distance": distance,
                    "change": pc,
                    "change7": change7,
                    "status": status,
                    "rsi":
                        (
                            float(last.rsi)
                            if pd.notna(last.rsi)
                            else np.nan
                        ),
                    "adx":
                        (
                            float(last.adx)
                            if pd.notna(last.adx)
                            else np.nan
                        ),
                    "volume_ratio":
                        volume_ratio,
                    "candles": len(d)
                }
            )

        except Exception as e:

            failures.append(
                f"{symbol}: "
                f"{type(e).__name__}: {e}"
            )

            continue

    # =========================================================
    # ATH RANKING
    # =========================================================

    if mode == "ATH":

        def ath_rank(z):

            status_priority = {
                "🔥 ATH BREAKOUT": 0,
                "🟢 AT ATH": 1,
                "🟡 NEAR ATH": 2
            }

            return (
                status_priority.get(
                    z["status"],
                    9
                ),
                abs(z["distance"]),
                -z["change"],
                -z["change7"]
            )

        output.sort(key=ath_rank)

    # =========================================================
    # ATL RANKING
    # =========================================================

    else:

        def atl_rank(z):

            status_priority = {
                "🔴 ATL BREAKDOWN": 0,
                "🟠 AT ATL": 1,
                "🟡 NEAR ATL": 2,
                "🔻 STRONG DOWN": 3
            }

            return (
                status_priority.get(
                    z["status"],
                    9
                ),
                abs(z["distance"]),
                z["change"],
                z["change7"]
            )

        output.sort(key=atl_rank)

    return output[:10], failures


# =============================================================================
# ATH / ATL UI
# =============================================================================

st.divider()

st.header(
    "🔥 / 🩸 Historical Extreme Scanner"
)

st.caption(
    "Previous ATH/ATL excludes the latest daily candle. "
    "The scanner checks up to 150 Futures contracts. "
    "ATH finds breakouts and coins within 5% of ATH. "
    "ATL finds breakdowns, coins within 5% of ATL, "
    "and strong downside movers. "
    "History is based on up to 400 days of daily data."
)

a, b = st.columns(2)

do_ath = a.button(
    "🔥 Scan Top 10 Near / Above ATH"
)

do_atl = b.button(
    "🩸 Scan Top 10 Near / Below ATL"
)

if do_ath or do_atl:

    mode = (
        "ATH"
        if do_ath
        else
        "ATL"
    )

    try:

        results, failures = scan_extremes(
            active_instruments(margin),
            futures_prices(),
            mode,
            limit=150
        )

        if not results:

            st.warning(
                f"No {mode} candidates found "
                "in the scanned universe."
            )

            if failures:

                with st.expander(
                    "🔧 Extreme scanner diagnostics",
                    expanded=True
                ):

                    st.code(
                        "\n".join(
                            failures[:100]
                        )
                    )

        else:

            if mode == "ATH":

                st.subheader(
                    "🔥 Top 10 Near / Above ATH"
                )

            else:

                st.subheader(
                    "🩸 Top 10 Near / Below ATL"
                )

            st.caption(
                "Previous ATH/ATL excludes the latest daily "
                "candle so the live price can actually break "
                "the previous extreme. Values represent the "
                "available CoinDCX Futures history, not "
                "guaranteed lifetime exchange-wide extremes."
            )

            table = []

            for i, z in enumerate(
                results,
                1
            ):

                table.append(
                    {
                        "#": i,
                        "Coin": z["symbol"],
                        "Current":
                            fmt(z["price"]),
                        (
                            "ATH"
                            if mode == "ATH"
                            else "ATL"
                        ):
                            fmt(z["extreme"]),
                        "Distance":
                            f'{z["distance"]:+.2f}%',
                        "24h":
                            f'{z["change"]:+.2f}%',
                        "7d":
                            f'{z["change7"]:+.2f}%',
                        "RSI":
                            (
                                f'{z["rsi"]:.1f}'
                                if pd.notna(z["rsi"])
                                else "—"
                            ),
                        "ADX":
                            (
                                f'{z["adx"]:.1f}'
                                if pd.notna(z["adx"])
                                else "—"
                            ),
                        "Vol/20D":
                            (
                                f'{z["volume_ratio"]:.1f}x'
                                if pd.notna(
                                    z["volume_ratio"]
                                )
                                else "—"
                            ),
                        "Status":
                            z["status"]
                    }
                )

            st.dataframe(
                pd.DataFrame(table),
                use_container_width=True,
                hide_index=True
            )

            for i, z in enumerate(
                results,
                1
            ):

                direction = (
                    "above"
                    if z["distance"] > 0
                    else
                    "below"
                )

                st.write(
                    f"**{i}. {z['symbol']} — "
                    f"{z['status']}** | "
                    f"Price {fmt(z['price'])} | "
                    f"Previous {mode} "
                    f"{fmt(z['extreme'])} | "
                    f"{abs(z['distance']):.2f}% "
                    f"{direction} | "
                    f"24h {z['change']:+.2f}% | "
                    f"7d {z['change7']:+.2f}% | "
                    f"RSI {z['rsi']:.1f} | "
                    f"ADX {z['adx']:.1f}"
                )

    except Exception as e:

        st.error(
            f"{mode} scanner failed: "
            f"{type(e).__name__}: {e}"
        )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "Analysis only. No API keys, orders, balances or "
    "withdrawals are used. For a spot holding, SHORT BIAS "
    "is a downside-risk warning, not automatically a sell "
    "instruction."
)
