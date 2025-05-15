import pandas as pd
from prophet import Prophet
import yfinance as yf

def forecast_price(ticker):
    df = yf.download(ticker, period="6mo")["Close"].reset_index()
    df = df.rename(columns={"Date": "ds", "Close": "y"})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return df.to_dict(orient="records"), forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_dict(orient="records")
