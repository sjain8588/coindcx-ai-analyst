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
st.caption(
    "Analysis only • Top 5 futures candidates • "
    "Built for volatile/meme-coin momentum • No trading"
)


# ============================================================
# COINDCX DATA FUNCTIONS
# ============================================================

@st.cache_data(ttl=30)
def get_active_instruments(margin="USDT"):
    url = f"{API}/exchange/v1/derivatives/futures/data/active_instruments"

    r = requests.get(
        url,
        params=[("margin_currency_short_name[]", margin)],
        timeout=20
    )

    r.raise_for_status()

    data = r.json()

    if not isinstance(data, list):
        raise RuntimeError(
            f"Unexpected active-instruments response: {data}"
        )

    return data


@st.cache_data(ttl=5)
def get_futures_prices():
    r = requests.get(
        f"{PUBLIC}/market_data/v3/current_prices/futures/rt",
        timeout=20
    )

    r.raise_for_status()

    data = r.json()

    return data.get("prices", {}) if isinstance(data, dict) else {}


@st.cache_data(ttl=30)
def get_futures_candles(pair, resolution, from_ts, to_ts):

    params = {
        "pair": pair,
        "from": int(from_ts),
        "to": int(to_ts),
        "resolution": resolution,
        "pcode": "f",
    }

    r = requests.get(
        f"{PUBLIC}/market_data/candlesticks",
        params=params,
        timeout=30
    )

    r.raise_for_status()

    data = r.json()

    rows = data.get("data", []) if isinstance(data, dict) else data

    if not isinstance(rows, list):
        raise RuntimeError(
            f"Unexpected candle response: {data}"
        )

    d = pd.DataFrame(rows)

    if d.empty:
        return d

    for c in ["open", "high", "low", "close", "volume"]:
        if c not in d.columns:
            raise RuntimeError(
                f"Candle response missing '{c}' column for {pair}: "
                f"{list(d.columns)}"
            )

        d[c] = pd.to_numeric(
            d[c],
            errors="coerce"
        )

    if "time" not in d.columns:
        raise RuntimeError(
            f"Candle response missing 'time' column for {pair}"
        )

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
def get_futures_orderbook(pair):

    r = requests.get(
        f"{PUBLIC}/market_data/v3/orderbook/{pair}-futures/50",
        timeout=15
    )

    r.raise_for_status()

    return r.json()


@st.cache_data(ttl=5)
def get_futures_trades(pair):

    r = requests.get(
        f"{API}/exchange/v1/derivatives/futures/data/trades",
        params={"pair": pair},
        timeout=15
    )

    r.raise_for_status()

    x = r.json()

    return x if isinstance(x, list) else []


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def indicators(d):

    x = d.copy()

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    for n in [20, 50, 100, 200]:
        x[f"ema{n}"] = x.close.ewm(
            span=n,
            adjust=False
        ).mean()

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = x.close.diff()

    gain = delta.clip(
        lower=0
    ).ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    loss = (
        -delta.clip(upper=0)
    ).ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    rs = gain / loss.replace(
        0,
        np.nan
    )

    x["rsi"] = 100 - 100 / (1 + rs)

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    e12 = x.close.ewm(
        span=12,
        adjust=False
    ).mean()

    e26 = x.close.ewm(
        span=26,
        adjust=False
    ).mean()

    x["macd"] = e12 - e26

    x["macd_signal"] = x.macd.ewm(
        span=9,
        adjust=False
    ).mean()

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    tr = pd.concat(
        [
            x.high - x.low,
            (x.high - x.close.shift()).abs(),
            (x.low - x.close.shift()).abs()
        ],
        axis=1
    ).max(axis=1)

    x["atr"] = tr.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    # --------------------------------------------------------
    # Volume MA
    # --------------------------------------------------------

    x["volma"] = x.volume.rolling(20).mean()

    # --------------------------------------------------------
    # Bollinger Bands
    # --------------------------------------------------------

    x["bbmid"] = x.close.rolling(20).mean()

    x["bbstd"] = x.close.rolling(20).std()

    x["bbup"] = (
        x.bbmid +
        2 * x.bbstd
    )

    x["bblow"] = (
        x.bbmid -
        2 * x.bbstd
    )

    # --------------------------------------------------------
    # ADX / DI
    # --------------------------------------------------------

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

    atr = x.atr.replace(
        0,
        np.nan
    )

    pdi = (
        100 *
        plus.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean() /
        atr
    )

    mdi = (
        100 *
        minus.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean() /
        atr
    )

    dx = (
        100 *
        (pdi - mdi).abs() /
        (pdi + mdi).replace(
            0,
            np.nan
        )
    )

    x["adx"] = dx.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    x["pdi"] = pdi
    x["mdi"] = mdi

    return x


# ============================================================
# MARKET STRUCTURE
# ============================================================

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


# ============================================================
# TECHNICAL DIRECTION
# ============================================================

def technical_side(d):

    x = d.iloc[-1]

    long_points = 0
    short_points = 0

    ema_pairs = [
        (x.close, x.ema20),
        (x.ema20, x.ema50),
        (x.ema50, x.ema100),
        (x.ema100, x.ema200),
    ]

    for a, b in ema_pairs:

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
        int(x.volume > x.volma)
    )

    short_points += (
        int(30 < x.rsi < 50)
        +
        int(x.macd < x.macd_signal)
        +
        int(x.mdi > x.pdi)
        +
        int(x.volume > x.volma)
    )

    return long_points, short_points


# ============================================================
# FUTURES MICROSTRUCTURE
# ============================================================

def micro(pair):

    ob = None
    flow = None

    # --------------------------------------------------------
    # ORDER BOOK
    # --------------------------------------------------------

    try:

        b = get_futures_orderbook(pair)

        bids = b.get("bids", {})
        asks = b.get("asks", {})

        B = [
            (float(p), float(q))
            for p, q in bids.items()
        ]

        A = [
            (float(p), float(q))
            for p, q in asks.items()
        ]

        if B and A:

            best_bid = max(
                p for p, _ in B
            )

            best_ask = min(
                p for p, _ in A
            )

            mid = (
                best_bid +
                best_ask
            ) / 2

            near_b = sum(
                q
                for p, q in B
                if p >= mid * 0.995
            )

            near_a = sum(
                q
                for p, q in A
                if p <= mid * 1.005
            )

            if near_b + near_a:
                ob = (
                    near_b - near_a
                ) / (
                    near_b + near_a
                )
            else:
                ob = 0

    except Exception:
        pass

    # --------------------------------------------------------
    # TRADE FLOW
    # --------------------------------------------------------

    try:

        trs = get_futures_trades(pair)

        buy = 0.0
        sell = 0.0

        for t in trs:

            q = float(
                t.get("quantity", 0)
                or 0
            )

            # CoinDCX futures docs define is_maker;
            # use it only as a flow proxy.
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
        else:
            flow = 0

    except Exception:
        pass

    return ob, flow


# ============================================================
# SCORE CANDIDATE
# ============================================================

def score_candidate(
    pair,
    price_row,
    d1,
    h1,
    m15
):

    x1 = d1.iloc[-1]
    xh = h1.iloc[-1]
    x15 = m15.iloc[-1]

    change = float(
        price_row.get("pc", 0)
        or 0
    )

    move = abs(change)

    b1, s1 = technical_side(d1)
    bh, sh = technical_side(h1)
    b15, s15 = technical_side(m15)

    ob, flow = micro(pair)

    # --------------------------------------------------------
    # Direction follows the user's strategy:
    # strongest gainers = LONG candidates
    # strongest losers = SHORT candidates
    # --------------------------------------------------------

    side = (
        "LONG"
        if change > 0
        else "SHORT"
    )

    # --------------------------------------------------------
    # LONG SCORE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # SHORT SCORE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CHASE PENALTY
    # --------------------------------------------------------

    ext = (
        abs(
            float(
                x1.close - x1.ema20
            )
        )
        /
        max(
            float(x1.atr),
            1e-12
        )
    )

    chase_penalty = (
        max(
            0,
            (ext - 2.0)
        )
        * 7
    )

    raw -= min(
        25,
        chase_penalty
    )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence = max(
        35,
        min(
            95,
            50 + raw * 0.45
        )
    )

    # --------------------------------------------------------
    # HISTORY QUALITY
    # --------------------------------------------------------

    history_factor = min(
        1.0,
        min(
            len(d1) / 120.0,
            len(h1) / 240.0,
            len(m15) / 240.0
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

    # --------------------------------------------------------
    # MULTI-TIMEFRAME ALIGNMENT
    # --------------------------------------------------------

    aligned = (

        (
            side == "LONG"
            and b1 > s1
            and bh >= sh
            and b15 >= s15
            and x15.macd > x15.macd_signal
        )

        or

        (
            side == "SHORT"
            and s1 > b1
            and sh >= bh
            and s15 >= b15
            and x15.macd < x15.macd_signal
        )
    )

    # --------------------------------------------------------
    # EXTREME RSI
    # --------------------------------------------------------

    extreme = (
        (
            side == "LONG"
            and x1.rsi >= 78
        )
        or
        (
            side == "SHORT"
            and x1.rsi <= 22
        )
    )

    signal = (
        side
        if (
            aligned
            and not extreme
            and confidence >= 68
        )
        else "WAIT"
    )

    # --------------------------------------------------------
    # SUPPORT / RESISTANCE
    # --------------------------------------------------------

    recent = d1.tail(90)

    support1 = float(
        recent.low.quantile(0.15)
    )

    support2 = float(
        recent.low.quantile(0.05)
    )

    resistance1 = float(
        recent.high.quantile(0.85)
    )

    resistance2 = float(
        recent.high.quantile(0.95)
    )

    # --------------------------------------------------------
    # ENTRY / STOP / TARGETS
    # --------------------------------------------------------

    price = float(
        x15.close
    )

    atr = float(
        x15.atr
    )

    if signal == "LONG":

        sl = price - 1.4 * atr

        risk = price - sl

        tp1 = price + 1.5 * risk

        tp2 = price + 2.5 * risk

    elif signal == "SHORT":

        sl = price + 1.4 * atr

        risk = sl - price

        tp1 = price - 1.5 * risk

        tp2 = price - 2.5 * risk

    else:

        sl = np.nan
        tp1 = np.nan
        tp2 = np.nan

    return {

        "pair": pair,

        "change": change,

        "side": side,

        "signal": signal,

        "confidence": confidence,

        "price": price,

        "support1": support1,

        "support2": support2,

        "resistance1": resistance1,

        "resistance2": resistance2,

        "rsi": float(x1.rsi),

        "adx": float(xh.adx),

        "structure": structure(d1),

        "ob": ob,

        "flow": flow,

        "ext": ext,

        "score": raw,

        "entry": price,

        "sl": sl,

        "tp1": tp1,

        "tp2": tp2,

        "d1": d1,

        "h1": h1,

        "m15": m15
    }


# ============================================================
# FORMAT NUMBERS
# ============================================================

def fmt(v):

    if v is None or pd.isna(v):
        return "—"

    return (
        f"{v:,.8f}"
        .rstrip("0")
        .rstrip(".")
    )


# ============================================================
# USER SETTINGS
# ============================================================

margin = st.selectbox(
    "Futures margin market",
    ["USDT", "INR"],
    index=0
)

meme_only = st.checkbox(
    "Meme-focused scan",
    value=True
)


# ============================================================
# MEME FILTER
# ============================================================

MEME_WORDS = {
    "DOGE",
    "SHIB",
    "PEPE",
    "BONK",
    "FLOKI",
    "WIF",
    "BOME",
    "MEME",
    "BRETT",
    "MOG",
    "TURBO",
    "MEW",
    "NEIRO",
    "BABYDOGE",
    "1000SHIB",
    "1000PEPE",
    "1000BONK",
    "1000FLOKI",
    "1000LUNC",
    "PONKE",
    "MYRO",
    "SLERF",
    "LADYS",
    "DEGEN",
    "MOTHER",
    "MAGA",
    "TRUMP"
}


if meme_only:

    st.caption(
        "Meme focus uses a broad symbol/name keyword filter; "
        "turn it off to scan every active futures instrument."
    )


# ============================================================
# MY INVESTED COIN
# ============================================================

st.divider()

st.header(
    "💼 My Invested Coin — Long / Short Check"
)

st.caption(
    "Enter a coin you already hold. The scanner will find "
    "its CoinDCX Futures contract and evaluate whether the "
    "current setup favors LONG, SHORT, or WAIT."
)


def normalize_coin_input(text):

    q = (
        text.strip()
        .upper()
        .replace("/", "")
        .replace("-", "")
        .replace("_", "")
    )

    # Remove common futures quote suffixes
    # so VELVETUSDT becomes VELVET.

    for quote in (
        "USDT",
        "INR",
        "USDC"
    ):

        if (
            q.endswith(quote)
            and len(q) > len(quote)
        ):

            q = q[
                :-len(quote)
            ]

            break

    # Do not blindly strip B from normal tickers.
    # Only strip it when matching against a
    # CoinDCX futures contract.

    if q.startswith("B") and len(q) > 2:
        pass

    return q


coin_input = st.text_input(
    "Coin / Futures pair",
    placeholder="Example: DOGE, SHIB, PEPE, B-DOGE_USDT",
    help=(
        "You can enter DOGE, DOGEUSDT, or the CoinDCX "
        "futures pair such as B-DOGE_USDT."
    )
)


avg_price_input = st.number_input(
    "Optional: your average entry price",
    min_value=0.0,
    value=0.0,
    step=0.00000001,
    format="%.8f",
    help=(
        "Enter your actual average price if you want the "
        "scanner to compare the current price with your cost."
    )
)


# ============================================================
# ANALYZE MY COIN
# ============================================================

if st.button("📊 Analyze My Coin"):

    try:

        if not coin_input.strip():

            st.warning(
                "Enter a coin first, for example DOGE or SHIB."
            )

            st.stop()

        prices = get_futures_prices()

        # Search both common quote markets.

        requested = normalize_coin_input(
            coin_input
        )

        margin_list = (
            [margin]
            +
            [
                q
                for q in ("USDT", "INR")
                if q != margin
            ]
        )

        candidates = []

        seen = set()

        for qmargin in margin_list:

            try:

                active = get_active_instruments(
                    qmargin
                )

            except Exception:

                continue

            for pair in active:

                pair_u = str(
                    pair
                ).upper()

                p = prices.get(pair)

                if not p:
                    continue

                symbol = str(
                    p.get("mkt", "")
                ).upper()

                names = {

                    pair_u
                    .replace("-", "")
                    .replace("_", ""),

                    symbol
                    .replace("-", "")
                    .replace("_", "")
                }

                matched = False

                for name in names:

                    n = name

                    for prefix in (
                        "B",
                        "I"
                    ):

                        if n.startswith(prefix):
                            n2 = n[1:]
                        else:
                            n2 = n

                        if (
                            n2 ==
                            requested + qmargin
                            or
                            n2 ==
                            requested
                        ):
                            matched = True

                        if n2.startswith(
                            requested + qmargin
                        ):
                            matched = True

                if (
                    matched
                    and pair not in seen
                ):

                    candidates.append(
                        (
                            pair,
                            p,
                            symbol,
                            qmargin
                        )
                    )

                    seen.add(pair)

        if not candidates:

            available = []

            for qmargin in margin_list:

                try:

                    active = get_active_instruments(
                        qmargin
                    )

                    for pair in active:

                        s = str(
                            pair
                        ).upper()

                        if (
                            requested
                            in
                            s.replace(
                                "-",
                                ""
                            ).replace(
                                "_",
                                ""
                            )
                        ):

                            available.append(s)

                except Exception:
                    pass

            hint = ", ".join(
                available[:8]
            )

            extra = (
                f" Possible matching contracts: {hint}"
                if hint
                else ""
            )

            st.error(
                f"No active CoinDCX Futures contract was found "
                f"for '{coin_input}'. Tried "
                f"{', '.join(margin_list)} margin markets."
                f"{extra}"
            )

            st.stop()

        # Prefer user's selected margin market,
        # then USDT.

        candidates.sort(
            key=lambda z: (
                0 if z[3] == margin else 1,
                len(z[0])
            )
        )

        pair, p, symbol, found_margin = candidates[0]

        if found_margin != margin:

            st.info(
                f"Found {symbol} in the "
                f"**{found_margin}** Futures market; "
                f"analyzing that contract."
            )

        now = int(
            time.time()
        )

        d1 = get_futures_candles(
            pair,
            "1D",
            now - 400 * 86400,
            now
        )

        h1 = get_futures_candles(
            pair,
            "60",
            now - 120 * 86400,
            now
        )

        m15 = get_futures_candles(
            pair,
            "15",
            now - 30 * 86400,
            now
        )

        # New listings can have limited history.

        if (
            len(d1) < 10
            or
            len(h1) < 30
            or
            len(m15) < 30
        ):

            st.error(
                f"CoinDCX returned too little history for "
                f"{symbol}: "
                f"1D={len(d1)}, "
                f"1H={len(h1)}, "
                f"15m={len(m15)}. "
                f"Need at least 10 / 30 / 30 candles."
            )

            st.stop()

        x = score_candidate(
            pair,
            p,
            indicators(d1),
            indicators(h1),
            indicators(m15)
        )

        st.subheader(
            f"{symbol}  •  {pair}"
        )

        current = float(
            x["price"]
        )

        if avg_price_input > 0:

            pnl_pct = (
                (
                    current -
                    avg_price_input
                )
                /
                avg_price_input
                *
                100
            )

            st.metric(
                "Your position vs current price",
                f"{pnl_pct:+.2f}%"
            )

        # ----------------------------------------------------
        # SIGNAL INTERPRETATION
        # ----------------------------------------------------

        if x["signal"] == "LONG":

            st.success(
                "🟢 LONG BIAS — Current Futures structure "
                "supports the bullish direction. For an existing "
                "spot holding, this means the trend is currently "
                "favorable to the long side."
            )

        elif x["signal"] == "SHORT":

            st.error(
                "🔴 SHORT BIAS — Current Futures structure "
                "supports the bearish direction. For an existing "
                "spot holding, this is a warning that downside "
                "momentum is stronger."
            )

        else:

            st.warning(
                "🟡 WAIT — The coin is not giving a sufficiently "
                "clean long or short setup right now. Avoid making "
                "a directional decision from momentum alone."
            )

        a, b, c, d = st.columns(4)

        a.metric(
            "Current price",
            fmt(current)
        )

        b.metric(
            "24h move",
            f'{x["change"]:+.2f}%'
        )

        c.metric(
            "Signal confidence",
            f'{x["confidence"]:.0f}%'
        )

        d.metric(
            "1D RSI",
            f'{x["rsi"]:.1f}'
        )

        st.caption(
            f"History available: "
            f"1D {len(x['d1'])} candles • "
            f"1H {len(x['h1'])} candles • "
            f"15m {len(x['m15'])} candles"
        )

        if min(
            len(x["d1"]),
            len(x["h1"]),
            len(x["m15"])
        ) < 60:

            st.info(
                "This is a newer/limited-history Futures "
                "contract. The signal is intentionally given "
                "a lower confidence ceiling."
            )

        st.write(
            f"**1D structure:** {x['structure']}  |  "
            f"**1H ADX:** {x['adx']:.1f}  |  "
            f"**Support:** "
            f"{fmt(x['support1'])} / "
            f"{fmt(x['support2'])}  |  "
            f"**Resistance:** "
            f"{fmt(x['resistance1'])} / "
            f"{fmt(x['resistance2'])}"
        )

        if x["signal"] in (
            "LONG",
            "SHORT"
        ):

            a, b, c = st.columns(3)

            a.metric(
                "Reference entry",
                fmt(x["entry"])
            )

            b.metric(
                "Stop loss",
                fmt(x["sl"])
            )

            c.metric(
                "TP1 / TP2",
                f'{fmt(x["tp1"])} / {fmt(x["tp2"])}'
            )

        # ----------------------------------------------------
        # EXPLANATION
        # ----------------------------------------------------

        with st.expander(
            "Why did the scanner choose this direction?"
        ):

            st.write(
                "The decision combines the 1D, 1H and 15m "
                "trend with EMA 20/50/100/200, RSI, MACD, "
                "ADX/DI, ATR, volume, Bollinger Bands, "
                "market structure, support/resistance and "
                "futures microstructure."
            )

            st.write(
                f"Daily RSI: {x['rsi']:.1f} | "
                f"1H ADX: {x['adx']:.1f} | "
                f"Order-book imbalance: "
                f"{'not available' if x['ob'] is None else f'{x['ob'] * 100:.2f}%'} | "
                f"Trade-flow proxy: "
                f"{'not available' if x['flow'] is None else f'{x['flow'] * 100:.2f}%'}"
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
            f"My Coin analysis failed: {e}"
        )

        st.code(
            str(e)
        )


# ============================================================
# SCAN TOP FUTURES
# ============================================================

if st.button(
    "🔍 Scan Top Futures",
    type="primary"
):

    try:

        active = get_active_instruments(
            margin
        )

        prices = get_futures_prices()

        # ----------------------------------------------------
        # KEEP INSTRUMENTS WITH CURRENT PRICE
        # ----------------------------------------------------

        rows = []

        for pair in active:

            p = prices.get(pair)

            if not p:
                continue

            symbol = str(
                p.get(
                    "mkt",
                    pair
                )
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

            pc = float(
                p.get("pc", 0)
                or 0
            )

            if pc == 0:
                continue

            rows.append(
                (
                    pair,
                    p,
                    symbol
                )
            )

        if not rows:

            st.error(
                "No matching futures instruments were "
                "returned. Try turning off Meme-focused scan."
            )

            st.stop()

        # ----------------------------------------------------
        # SORT BY LARGEST 24H MOVEMENT
        # ----------------------------------------------------

        rows.sort(
            key=lambda z: abs(
                float(
                    z[1].get(
                        "pc",
                        0
                    )
                    or 0
                )
            ),
            reverse=True
        )

        # ----------------------------------------------------
        # ANALYZE STRONGEST MOVERS
        # ----------------------------------------------------

        results = []

        now = int(
            time.time()
        )

        scan_errors = []

        for pair, p, symbol in rows[:12]:

            try:

                d1 = get_futures_candles(
                    pair,
                    "1D",
                    now - 400 * 86400,
                    now
                )

                h1 = get_futures_candles(
                    pair,
                    "60",
                    now - 120 * 86400,
                    now
                )

                m15 = get_futures_candles(
                    pair,
                    "15",
                    now - 30 * 86400,
                    now
                )

                # ==================================================
                # IMPORTANT FIX:
                #
                # OLD BUG:
                # len(d1h)
                #
                # CORRECT:
                # len(h1)
                #
                # d1h did not exist and caused every candidate
                # to fail with NameError.
                # ==================================================

                if (
                    len(d1) < 10
                    or
                    len(h1) < 30
                    or
                    len(m15) < 30
                ):

                    scan_errors.append(
                        f"{symbol}: insufficient candles "
                        f"(1D={len(d1)}, "
                        f"1H={len(h1)}, "
                        f"15m={len(m15)})"
                    )

                    continue

                result = score_candidate(
                    pair,
                    p,
                    indicators(d1),
                    indicators(h1),
                    indicators(m15)
                )

                results.append(
                    (
                        symbol,
                        result
                    )
                )

            except Exception as e:

                # Do not silently hide scanner failures.
                scan_errors.append(
                    f"{symbol}: {type(e).__name__}: {e}"
                )

                continue

        # ----------------------------------------------------
        # NO RESULTS
        # ----------------------------------------------------

        if not results:

            st.error(
                "Futures instruments were found, but no "
                "candidate returned enough futures candle "
                "data for analysis."
            )

            if scan_errors:

                with st.expander(
                    "🔎 Scanner diagnostics"
                ):

                    for error in scan_errors:

                        st.write(
                            f"• {error}"
                        )

            st.stop()

        # ----------------------------------------------------
        # SORT RESULTS
        # ----------------------------------------------------

        results.sort(
            key=lambda z: z[1]["score"],
            reverse=True
        )

        results = results[:5]

        # ----------------------------------------------------
        # DISPLAY TOP 5
        # ----------------------------------------------------

        st.header(
            "🎯 Today's Top 5 Futures Candidates"
        )

        st.caption(
            "First filter: largest 24h movers. Second filter: "
            "trend, EMA 20/50/100/200, RSI, MACD, ADX/DI, ATR, "
            "volume, Bollinger Bands, daily support/resistance "
            "and futures liquidity."
        )

        for i, (symbol, x) in enumerate(
            results,
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

            with st.container(
                border=True
            ):

                st.subheader(
                    f"{i}. {symbol}   "
                    f"{icon} {x['signal']}"
                )

                a, b, c, d = st.columns(4)

                a.metric(
                    "24h move",
                    f'{x["change"]:.2f}%'
                )

                b.metric(
                    "Confidence",
                    f'{x["confidence"]:.0f}%'
                )

                c.metric(
                    "Price",
                    fmt(x["price"])
                )

                d.metric(
                    "Daily RSI",
                    f'{x["rsi"]:.1f}'
                )

                st.write(
                    f"**Support:** "
                    f"{fmt(x['support2'])} / "
                    f"{fmt(x['support1'])}   |   "
                    f"**Resistance:** "
                    f"{fmt(x['resistance1'])} / "
                    f"{fmt(x['resistance2'])}"
                )

                st.caption(
                    f"History available: "
                    f"1D {len(x['d1'])} candles • "
                    f"1H {len(x['h1'])} candles • "
                    f"15m {len(x['m15'])} candles"
                )

                if x["signal"] in (
                    "LONG",
                    "SHORT"
                ):

                    a, b, c, d = st.columns(4)

                    a.metric(
                        "Entry reference",
                        fmt(x["entry"])
                    )

                    b.metric(
                        "Stop Loss",
                        fmt(x["sl"])
                    )

                    c.metric(
                        "TP1",
                        fmt(x["tp1"])
                    )

                    d.metric(
                        "TP2",
                        fmt(x["tp2"])
                    )

                else:

                    direction = (
                        "long"
                        if x["side"] == "LONG"
                        else "short"
                    )

                    st.warning(
                        f"🟡 WAIT — {symbol} is a strong "
                        f"{direction} mover, but the "
                        f"confirmation/entry quality is not "
                        f"strong enough. Do not chase."
                    )

                # ------------------------------------------------
                # ADVANCED ANALYSIS
                # ------------------------------------------------

                with st.expander(
                    "Advanced analysis"
                ):

                    st.write(
                        f"1D structure: **{x['structure']}**"
                    )

                    st.write(
                        f"EMA20/50/100/200: "
                        f"{fmt(x['d1'].iloc[-1].ema20)} / "
                        f"{fmt(x['d1'].iloc[-1].ema50)} / "
                        f"{fmt(x['d1'].iloc[-1].ema100)} / "
                        f"{fmt(x['d1'].iloc[-1].ema200)}"
                    )

                    if pd.isna(
                        x["d1"].iloc[-1].ema200
                    ):

                        st.info(
                            "This futures contract does not yet "
                            "have 200 daily candles. EMA200 is "
                            "therefore not used as a bearish/bullish "
                            "vote for this coin."
                        )

                    st.write(
                        f"1H ADX: {x['adx']:.1f} | "
                        f"1D RSI: {x['rsi']:.1f}"
                    )

                    if x["ob"] is not None:

                        st.write(
                            f"Near-price futures order-book "
                            f"imbalance: "
                            f"{x['ob'] * 100:.2f}%"
                        )

                    if x["flow"] is not None:

                        st.write(
                            f"Futures trade-flow proxy: "
                            f"{x['flow'] * 100:.2f}%"
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

        # ----------------------------------------------------
        # OPTIONAL DIAGNOSTICS
        # ----------------------------------------------------

        if scan_errors:

            with st.expander(
                f"🔎 Scanner diagnostics "
                f"({len(scan_errors)} skipped)"
            ):

                for error in scan_errors:

                    st.write(
                        f"• {error}"
                    )

    except Exception as e:

        st.error(
            f"Scanner failed: {e}"
        )

        st.code(
            str(e)
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "The My Invested Coin tool is an analysis signal, "
    "not a command to open a leveraged position. For a "
    "spot holding, a SHORT BIAS is best interpreted as a "
    "downside-risk warning unless you intentionally hedge "
    "with futures.\n\n"
    "Analysis only. CoinDCX futures endpoints used: active "
    "instruments, futures current prices, futures candlesticks, "
    "futures trades and futures order book. No API keys, orders, "
    "balances or withdrawals are used."
)
