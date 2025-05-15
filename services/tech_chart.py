import yfinance as yf
import pandas_ta as ta

def get_technical_charts(ticker):
    df = yf.download(ticker, period="3mo")
    ma = df.ta.sma(length=20).tolist()
    rsi = df.ta.rsi().tolist()
    bb = df.ta.bbands().dropna().to_dict(orient="list")
    stoch = df.ta.stoch().dropna().to_dict(orient="list")
    return {"ma": ma, "rsi": rsi, "bb": bb, "stoch": stoch}
