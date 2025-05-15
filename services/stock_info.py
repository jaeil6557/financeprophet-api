import FinanceDataReader as fdr

def get_stock_info(ticker):
    try:
        df = fdr.DataReader(ticker)
        latest = df.iloc[-1]
        return {
            "current_price": latest["Close"],
            "change": latest["Change"],
            "change_rate": round((latest["Change"] / latest["Close"] * 100), 2),
            "volume": latest["Volume"],
            "fundamentals": {"PER": 10.5, "EPS": 3500, "ROE": 8.7}
        }
    except:
        return None
