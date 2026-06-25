import json
import numpy as np
import pandas as pd

from config import (
    PRICE_FILE,
    PANEL_FILE,
    EXCEL_PANEL_FILE,
    FEATURE_MANIFEST_FILE,
    FIRST_SAMPLE_MONTH,
    BENCHMARK,
    FEATURE_COLUMNS,
)


def _month_end_prices(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices["month"] = prices["date"].dt.to_period("M").dt.to_timestamp("M")
    idx = prices.groupby(["ticker", "month"])["date"].idxmax()
    return prices.loc[idx].copy().sort_values(["ticker", "month"])


def _add_daily_ma_features(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy().sort_values(["ticker", "date"])
    prices["date"] = pd.to_datetime(prices["date"])
    g = prices.groupby("ticker", group_keys=False)
    for w in [5, 10, 20, 30]:
        prices[f"ma{w}"] = g["adj_close"].transform(lambda s: s.rolling(w, min_periods=w).mean())
        prices[f"ma{w}_prev21"] = g[f"ma{w}"].shift(21)
        prices[f"ma{w}_slope_1m"] = prices[f"ma{w}"] / prices[f"ma{w}_prev21"] - 1
    prices["month"] = prices["date"].dt.to_period("M").dt.to_timestamp("M")
    idx = prices.groupby(["ticker", "month"])["date"].idxmax()
    cols = ["ticker", "month", "adj_close"] + [f"ma{w}" for w in [5, 10, 20, 30]] + [f"ma{w}_slope_1m" for w in [5, 10, 20, 30]]
    return prices.loc[idx, cols].copy()


def _rolling_drawdown_and_vol(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy().sort_values(["ticker", "date"])
    prices["date"] = pd.to_datetime(prices["date"])
    prices["daily_ret"] = prices.groupby("ticker")["adj_close"].pct_change()
    g = prices.groupby("ticker", group_keys=False)
    for days, name in [(63, "3m"), (126, "6m"), (252, "12m")]:
        prices[f"high_{name}"] = g["adj_close"].transform(lambda s: s.rolling(days, min_periods=max(20, days // 2)).max())
        prices[f"drawdown_{name}"] = prices["adj_close"] / prices[f"high_{name}"] - 1
        if name in ["3m", "6m"]:
            prices[f"volatility_{name}"] = g["daily_ret"].transform(lambda s: s.rolling(days, min_periods=max(20, days // 2)).std() * np.sqrt(252))
    prices["month"] = prices["date"].dt.to_period("M").dt.to_timestamp("M")
    idx = prices.groupby(["ticker", "month"])["date"].idxmax()
    cols = ["ticker", "month", "drawdown_3m", "drawdown_6m", "drawdown_12m", "volatility_3m", "volatility_6m"]
    return prices.loc[idx, cols].copy()


def _drop_noisy_columns(panel: pd.DataFrame) -> pd.DataFrame:
    # Hard rule: this project deliberately removes cross-sectional ranks and categorical tags.
    drop_cols = []
    forbidden_prefixes = ("rank_", "pct_")
    forbidden_names = {"sector_group", "industry_group", "theme", "universe_tag", "market_cap", "size_bucket"}
    for col in panel.columns:
        if col.startswith(forbidden_prefixes) or col in forbidden_names:
            drop_cols.append(col)
    if drop_cols:
        panel = panel.drop(columns=drop_cols, errors="ignore")
    return panel


def _write_manifest(panel: pd.DataFrame, available_features: list[str]) -> None:
    lines = []
    lines.append("# Clean Momentum Panel Feature Manifest")
    lines.append("")
    lines.append("This panel intentionally excludes rank_*, pct_*, sector_group, industry_group, theme, universe_tag, market_cap, and size_bucket.")
    lines.append("The model is forced to learn momentum shape, trend, pullback, volatility, volume, and QQQ regime features.")
    lines.append("")
    lines.append(f"Rows: {len(panel):,}")
    lines.append(f"Columns: {len(panel.columns):,}")
    lines.append(f"Tickers: {panel['ticker'].nunique():,}")
    lines.append(f"Months: {panel['month'].nunique():,}")
    lines.append("")
    lines.append("## Model features")
    for c in available_features:
        lines.append(f"- {c}")
    FEATURE_MANIFEST_FILE.write_text("\n".join(lines), encoding="utf-8")


def build_panel() -> pd.DataFrame:
    if not PRICE_FILE.exists():
        raise FileNotFoundError(f"Missing {PRICE_FILE}. Run: python src/download_data.py")

    prices = pd.read_csv(PRICE_FILE, parse_dates=["date"])
    required_price_cols = {"date", "ticker", "adj_close", "volume"}
    missing = required_price_cols - set(prices.columns)
    if missing:
        raise ValueError(f"prices_daily.csv missing columns: {missing}")

    monthly = _month_end_prices(prices)
    ma = _add_daily_ma_features(prices)
    risk = _rolling_drawdown_and_vol(prices)

    panel = monthly[["ticker", "month", "adj_close", "volume"]].copy()
    panel = panel.merge(ma.drop(columns=["adj_close"]), on=["ticker", "month"], how="left")
    panel = panel.merge(risk, on=["ticker", "month"], how="left")
    panel = panel.sort_values(["ticker", "month"])
    g = panel.groupby("ticker")

    # Momentum levels.
    for k in [1, 2, 3, 4, 5, 6, 7, 9, 12]:
        panel[f"mom_{k}m"] = g["adj_close"].pct_change(k)

    # 4/5/6 month core momentum shape.
    core_cols = ["mom_4m", "mom_5m", "mom_6m"]
    panel["core_mom_456_avg"] = panel[core_cols].mean(axis=1)
    panel["core_mom_456_min"] = panel[core_cols].min(axis=1)
    panel["core_mom_456_max"] = panel[core_cols].max(axis=1)
    panel["core_mom_456_std"] = panel[core_cols].std(axis=1)
    panel["core_mom_456_positive_count"] = (panel[core_cols] > 0).sum(axis=1).astype(float)
    panel["core_mom_456_all_positive"] = (panel["core_mom_456_positive_count"] == 3).astype(float)
    panel["mom_4m_vs_6m"] = panel["mom_4m"] - panel["mom_6m"]
    panel["mom_5m_vs_6m"] = panel["mom_5m"] - panel["mom_6m"]

    # Momentum decomposition.
    panel["mom_6m_first3m"] = g["adj_close"].shift(3) / g["adj_close"].shift(6) - 1
    panel["mom_6m_last3m"] = panel["adj_close"] / g["adj_close"].shift(3) - 1
    panel["mom_6m_acceleration"] = panel["mom_6m_last3m"] - panel["mom_6m_first3m"]
    panel["mom_3m_vs_6m"] = panel["mom_3m"] - panel["mom_6m"]

    # Moving-average trend.
    for w in [5, 10, 20, 30]:
        panel[f"price_ma{w}_ratio"] = panel["adj_close"] / panel[f"ma{w}"] - 1
    panel["ma5_above_ma10"] = (panel["ma5"] > panel["ma10"]).astype(float)
    panel["ma10_above_ma20"] = (panel["ma10"] > panel["ma20"]).astype(float)
    panel["ma20_above_ma30"] = (panel["ma20"] > panel["ma30"]).astype(float)
    panel["ma_trend_score"] = panel[["ma5_above_ma10", "ma10_above_ma20", "ma20_above_ma30"]].sum(axis=1)
    slope_cols = [f"ma{w}_slope_1m" for w in [5, 10, 20, 30]]
    panel["ma_slope_score"] = (panel[slope_cols] > 0).sum(axis=1).astype(float)

    # Drawdown / pullback.
    for name in ["3m", "6m", "12m"]:
        panel[f"drawdown_{name}_abs"] = -panel[f"drawdown_{name}"]
    panel["near_3m_high"] = (panel["drawdown_3m"] > -0.05).astype(float)
    panel["near_6m_high"] = (panel["drawdown_6m"] > -0.05).astype(float)
    panel["deep_3m_drawdown"] = (panel["drawdown_3m"] < -0.15).astype(float)
    panel["deep_6m_drawdown"] = (panel["drawdown_6m"] < -0.20).astype(float)

    # Volume / liquidity: keep only a few non-rank raw features.
    panel["volume_change_1m"] = g["volume"].pct_change(1)
    panel["volume_change_3m"] = g["volume"].pct_change(3)
    panel["volume_ma3"] = g["volume"].transform(lambda s: s.rolling(3, min_periods=2).mean())
    panel["volume_ratio_3m"] = panel["volume"] / panel["volume_ma3"] - 1

    # Risk-adjusted momentum.
    panel["return_vol_ratio_3m"] = panel["mom_3m"] / panel["volatility_3m"].replace(0, np.nan)
    panel["return_vol_ratio_6m"] = panel["mom_6m"] / panel["volatility_6m"].replace(0, np.nan)

    # Labels: future returns.
    for h in [1, 2, 3]:
        panel[f"future_return_{h}m"] = g["adj_close"].shift(-h) / panel["adj_close"] - 1
    panel["future_max_return_1_3m"] = panel[["future_return_1m", "future_return_2m", "future_return_3m"]].max(axis=1)

    # Benchmark regime features from QQQ.
    qqq = panel[panel["ticker"] == BENCHMARK].copy()
    qqq_cols = ["month", "mom_1m", "mom_3m", "mom_6m", "mom_12m", "ma_trend_score"]
    qqq = qqq[qqq_cols]
    qqq = qqq.rename(columns={
        "mom_1m": "qqq_mom_1m",
        "mom_3m": "qqq_mom_3m",
        "mom_6m": "qqq_mom_6m",
        "mom_12m": "qqq_mom_12m",
        "ma_trend_score": "qqq_trend_score",
    })
    panel = panel.merge(qqq, on="month", how="left")
    panel = panel[panel["ticker"] != BENCHMARK]
    panel = panel[panel["month"] >= pd.Timestamp(FIRST_SAMPLE_MONTH)]

    # Clean invalid numerics.
    numeric_cols = panel.select_dtypes(include=[np.number]).columns
    panel[numeric_cols] = panel[numeric_cols].replace([np.inf, -np.inf], np.nan)
    panel[numeric_cols] = panel[numeric_cols].mask(panel[numeric_cols].abs() > 1e12)

    # Drop categories/ranks if somehow present.
    panel = _drop_noisy_columns(panel)

    # Keep metadata + selected features + labels only.
    available_features = [c for c in FEATURE_COLUMNS if c in panel.columns]
    label_cols = ["future_return_1m", "future_return_2m", "future_return_3m", "future_max_return_1_3m"]
    keep_cols = ["month", "ticker", "adj_close"] + available_features + label_cols
    keep_cols = [c for c in keep_cols if c in panel.columns]
    panel = panel[keep_cols].sort_values(["month", "ticker"]).reset_index(drop=True)

    PANEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PANEL_FILE, index=False)
    try:
        panel.to_excel(EXCEL_PANEL_FILE, index=False)
    except Exception as exc:
        print(f"Excel export skipped: {exc}")
    _write_manifest(panel, available_features)

    print(f"Saved {PANEL_FILE} with {len(panel):,} rows, {len(panel.columns):,} columns, {panel['ticker'].nunique():,} tickers")
    print(f"Saved {FEATURE_MANIFEST_FILE}")
    return panel


if __name__ == "__main__":
    build_panel()
