import pandas as pd
from config import UNIVERSE_FILE, BENCHMARK

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "AMD", "MU", "QQQ"]

def load_tickers() -> list[str]:
    if UNIVERSE_FILE.exists():
        df = pd.read_csv(UNIVERSE_FILE)
        if "ticker" not in df.columns:
            raise ValueError("data/universe.csv must contain a 'ticker' column")
        tickers = df["ticker"].dropna().astype(str).str.upper().str.strip().unique().tolist()
    else:
        tickers = DEFAULT_TICKERS
    if BENCHMARK not in tickers:
        tickers.append(BENCHMARK)
    return sorted(set(tickers))
