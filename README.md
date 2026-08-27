# CoinDCX AI Market Analyst

Analysis-only Streamlit app for mobile/desktop use.

## Features
- Public CoinDCX market/ticker data
- 15m, 1H, 4H and 1D analysis
- EMA 9/20/50/100/200
- RSI, MACD, ATR and volume
- Support/resistance
- LONG BIAS / SHORT BIAS / WAIT
- Illustrative SL/TP levels
- No API keys
- No order or withdrawal functionality

## Run
pip install -r requirements.txt
streamlit run app.py

CoinDCX may change public endpoint formats. If candles stop working, update only `fetch_candles()` in `app.py`.
