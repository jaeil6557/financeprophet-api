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

@app.post("/current_price")
def get_current_price(req: StockRequest):
    try:
        if req.ticker.startswith("KRW-"):
            df = pyupbit.get_ohlcv(req.ticker, count=30)
            df = df.reset_index()
            df['날짜'] = df['index'].astype(str)
            df = df.rename(columns={"close": "현재가"})
            price_data = df[['날짜', '현재가']].rename(columns={"현재가": "price"})
        else:
            df = fdr.DataReader(req.ticker, dt.datetime.now() - dt.timedelta(days=30))
            df = df.reset_index()
            df['날짜'] = df['Date'].astype(str)
            df = df.rename(columns={"Close": "현재가"})
            price_data = df[['날짜', '현재가']].rename(columns={"현재가": "price"})

        return price_data.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_price")
def predict_price(req: StockRequest):
    try:
        from prophet import Prophet

        today = dt.date.today()
        start = today - dt.timedelta(days=730)

        if req.ticker.startswith("KRW-"):
            df = pyupbit.get_ohlcv(req.ticker, count=730)
            df = df.reset_index()
            df.rename(columns={"close": "종가", "index": "날짜"}, inplace=True)
        else:
            df = fdr.DataReader(req.ticker, start)
            df = df.reset_index()
            df.rename(columns={"Close": "종가", "Date": "날짜"}, inplace=True)

        data = pd.DataFrame({"ds": df['날짜'], "y": df['종가']})

        model = Prophet(daily_seasonality=True)
        model.fit(data)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        first_day_price = forecast.iloc[-30]['yhat']
        last_day_price = forecast.iloc[-1]['yhat']

        return {"ticker": req.ticker, "first_day_price": round(first_day_price, 2), "last_day_price": round(last_day_price, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_chart")
def predict_chart(req: StockRequest):
    try:
        from prophet import Prophet

        today = dt.date.today()
        start = today - dt.timedelta(days=730)

        if req.ticker.startswith("KRW-"):
            df = pyupbit.get_ohlcv(req.ticker, count=730)
            df = df.reset_index()
            df.rename(columns={"close": "종가", "index": "날짜"}, inplace=True)
        else:
            df = fdr.DataReader(req.ticker, start)
            df = df.reset_index()
            df.rename(columns={"Close": "종가", "Date": "날짜"}, inplace=True)

        data = pd.DataFrame({"ds": df['날짜'], "y": df['종가']})

        model = Prophet(daily_seasonality=True)
        model.fit(data)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        result = forecast[['ds', 'yhat']].tail(30)
        result = result.rename(columns={"ds": "date", "yhat": "predicted_price"})
        result['date'] = result['date'].astype(str)

        return result.to_dict(orient="records")

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
