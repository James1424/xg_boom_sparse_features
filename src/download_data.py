import time
from pathlib import Path
import pandas as pd
import yfinance as yf

from config import PRICE_FILE, START_DATE, END_DATE
from universe import load_tickers


def _normalize_download(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["date", "ticker", "adj_close", "close", "volume"])

    frames = []
    if isinstance(raw.columns, pd.MultiIndex):
        # yfinance group_by='ticker' returns first level=ticker.
        first_level = raw.columns.get_level_values(0)
        if any(t in first_level for t in tickers):
            for t in tickers:
                if t not in first_level:
                    continue
                sub = raw[t].copy()
                adj_col = "Adj Close" if "Adj Close" in sub.columns else "Close"
                if adj_col not in sub.columns:
                    continue
                tmp = pd.DataFrame({
                    "date": sub.index,
                    "ticker": t,
                    "adj_close": sub[adj_col].values,
                    "close": sub["Close"].values if "Close" in sub.columns else sub[adj_col].values,
                    "volume": sub["Volume"].values if "Volume" in sub.columns else pd.NA,
                })
                frames.append(tmp)
        else:
            # first level may be OHLC field, second ticker.
            second_level = raw.columns.get_level_values(1)
            for t in tickers:
                if t not in second_level:
                    continue
                sub = raw.xs(t, axis=1, level=1).copy()
                adj_col = "Adj Close" if "Adj Close" in sub.columns else "Close"
                if adj_col not in sub.columns:
                    continue
                tmp = pd.DataFrame({
                    "date": sub.index,
                    "ticker": t,
                    "adj_close": sub[adj_col].values,
                    "close": sub["Close"].values if "Close" in sub.columns else sub[adj_col].values,
                    "volume": sub["Volume"].values if "Volume" in sub.columns else pd.NA,
                })
                frames.append(tmp)
    else:
        # Single ticker fallback.
        t = tickers[0]
        adj_col = "Adj Close" if "Adj Close" in raw.columns else "Close"
        frames.append(pd.DataFrame({
            "date": raw.index,
            "ticker": t,
            "adj_close": raw[adj_col].values,
            "close": raw["Close"].values if "Close" in raw.columns else raw[adj_col].values,
            "volume": raw["Volume"].values if "Volume" in raw.columns else pd.NA,
        }))

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        return pd.DataFrame(columns=["date", "ticker", "adj_close", "close", "volume"])
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out = out.dropna(subset=["adj_close"]).sort_values(["ticker", "date"])
    return out


def download_prices(batch_size: int = 40) -> pd.DataFrame:
    tickers = load_tickers()
    all_frames = []
    print(f"Downloading {len(tickers)} tickers from yfinance...")
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"Batch {i//batch_size + 1}: {batch[0]} ... {batch[-1]}")
        raw = yf.download(
            tickers=batch,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=False,
            group_by="ticker",
            progress=False,
            threads=True,
        )
        all_frames.append(_normalize_download(raw, batch))
        time.sleep(0.5)

    prices = pd.concat(all_frames, ignore_index=True)
    prices = prices.drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"])
    PRICE_FILE.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(PRICE_FILE, index=False)
    print(f"Saved {PRICE_FILE} with {len(prices):,} rows and {prices['ticker'].nunique()} tickers")
    return prices


if __name__ == "__main__":
    download_prices()
