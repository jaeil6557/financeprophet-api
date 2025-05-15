from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import datetime as dt
import yfinance as yf
import FinanceDataReader as fdr
import pyupbit
import requests
import os
import json

app = FastAPI()

class StockRequest(BaseModel):
    ticker: str

class AIRequest(BaseModel):
    ticker: str
    type: str
    openai_api_key: str

@app.get("/search")
def search_stocks(market: str = "stock"):
    if market == "stock":
        kospi = fdr.StockListing("KRX")
        return kospi[['Code', 'Name']].to_dict(orient="records")
    elif market == "crypto":
        tickers = pyupbit.get_tickers(fiat="KRW")
        return [{"ticker": t} for t in tickers]
    else:
        raise HTTPException(status_code=400, detail="Unsupported market type.")

@app.post("/price_and_forecast")
def price_and_forecast(req: StockRequest):
    try:
        from prophet import Prophet

        today = dt.date.today()
        start = today - dt.timedelta(days=730)

        # Load historical data
        if req.ticker.startswith("KRW-"):
            df_raw = pyupbit.get_ohlcv(req.ticker, count=730)
            df_raw = df_raw.reset_index()
            df_raw.rename(columns={"close": "종가", "index": "날짜"}, inplace=True)
        else:
            df_raw = fdr.DataReader(req.ticker, start)
            df_raw = df_raw.reset_index()
            df_raw.rename(columns={"Close": "종가", "Date": "날짜"}, inplace=True)

        df_raw['날짜'] = pd.to_datetime(df_raw['날짜'])
        data = pd.DataFrame({"ds": df_raw['날짜'], "y": df_raw['종가']})

        # Train model and forecast
        model = Prophet(daily_seasonality=True)
        model.fit(data)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        forecast = forecast[['ds', 'yhat']].tail(30).rename(columns={"ds": "date", "yhat": "predicted_price"})
        forecast['date'] = forecast['date'].astype(str)

        # Extract recent actuals (last 30 days)
        recent = df_raw.tail(30)[['날짜', '종가']].rename(columns={"날짜": "date", "종가": "price"})
        recent['date'] = recent['date'].astype(str)

        return {
            "historical": recent.to_dict(orient="records"),
            "forecast": forecast.to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai_analysis")
def ai_analysis(req: AIRequest):
    try:
        if req.type == "stock":
            df = fdr.DataReader(req.ticker, dt.datetime.now() - dt.timedelta(days=120))
            df = df.drop(columns=['Change'])
        elif req.type == "crypto":
            df = pyupbit.get_ohlcv(req.ticker, count=120)
            df.rename(columns=lambda x: x.capitalize(), inplace=True)
            df = df.drop(columns=['Value'])
        else:
            raise HTTPException(status_code=400, detail="Invalid type. Should be 'stock' or 'crypto'.")

        df = df.tail(60)

        stock_data = pd.DataFrame({
            "date": df.index,
            "close": df['Close'],
            "high": df['High'],
            "low": df['Low'],
            "open": df['Open'],
            "volume": df['Volume']
        })

        openai_url = "https://api.openai.com/v1/chat/completions"
        headers = {
            'Authorization': f'Bearer {req.openai_api_key}',
            'Content-Type': 'application/json'
        }

        instructions = f"""
        As a securities expert, analyze the stock/crypto data below and give a concise trading recommendation.
        Focus only on the most recent data.
        {stock_data.to_json(orient='records', date_format='iso')}
        """

        request_data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": "Analyze and give me a final decision: buy/sell/hold."}
            ],
            "max_tokens": 200,
            "temperature": 0.5
        }

        response = requests.post(openai_url, headers=headers, json=request_data)
        response.raise_for_status()
        result = response.json()

        content = result['choices'][0]['message']['content']

        return {"analysis": content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chart")
def get_ohlcv(req: StockRequest):
    try:
        if req.ticker.startswith("KRW-"):
            df = pyupbit.get_ohlcv(req.ticker, count=365)
            df = df.reset_index()
        else:
            df = fdr.DataReader(req.ticker, dt.datetime.now() - dt.timedelta(days=365))
            df = df.reset_index()

        return df.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

