import pandas as pd
import pandas_ta as ta

def analyze_stock(ticker, indicators):
    rsi = indicators.get("RSI", 50)
    decision = "관망"
    reason = "RSI 중립권"
    if rsi < 30:
        decision = "매수"
        reason = "RSI 과매도 구간"
    elif rsi > 70:
        decision = "매도"
        reason = "RSI 과매수 구간"
    return {"decision": decision, "reason": reason}
