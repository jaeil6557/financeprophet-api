from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware

from services.stock_info import get_stock_info
from services.ai_analysis import analyze_stock
from services.forecast import forecast_price
from services.tech_chart import get_technical_charts

app = FastAPI(title="FinanceProphet API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TickerRequest(BaseModel):
    ticker: str

class AIAnalysisRequest(BaseModel):
    ticker: str
    indicators: Dict[str, float]

@app.post("/stock/info")
def stock_info(req: TickerRequest):
    data = get_stock_info(req.ticker)
    if not data:
        raise HTTPException(status_code=404, detail="Stock info not found")
    return data

@app.post("/stock/ai-analysis")
def ai_analysis(req: AIAnalysisRequest):
    result = analyze_stock(req.ticker, req.indicators)
    return result

@app.post("/stock/forecast")
def stock_forecast(req: TickerRequest):
    chart, forecast = forecast_price(req.ticker)
    return {"ticker": req.ticker, "chart": chart, "forecast": forecast}

@app.post("/stock/tech-chart")
def tech_chart(req: TickerRequest):
    result = get_technical_charts(req.ticker)
    return result
