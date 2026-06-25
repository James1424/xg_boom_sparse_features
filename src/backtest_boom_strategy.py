import numpy as np
import pandas as pd

from config import OUTPUT_DIR, TOP_N

PRED_FILE = OUTPUT_DIR / "boom_predictions_test.csv"
BACKTEST_FILE = OUTPUT_DIR / "boom_top3_backtest.csv"
SUMMARY_FILE = OUTPUT_DIR / "boom_strategy_summary.csv"
BASELINE_FILE = OUTPUT_DIR / "boom_baseline_comparison.csv"


def equity_curve(returns: pd.Series) -> pd.Series:
    return (1 + returns.fillna(0)).cumprod()


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min())


def annualized_return(monthly_returns: pd.Series) -> float:
    if len(monthly_returns) == 0:
        return np.nan
    total = float((1 + monthly_returns).prod())
    years = len(monthly_returns) / 12
    return total ** (1 / years) - 1 if years > 0 and total > 0 else np.nan


def strategy_from_score(df: pd.DataFrame, score_col: str, name: str, n: int = TOP_N) -> pd.DataFrame:
    rows = []
    for month, g in df.groupby("month"):
        top = g.sort_values(score_col, ascending=False).head(n)
        rows.append({
            "month": month,
            "strategy": name,
            "selected_tickers": ", ".join(top["ticker"].astype(str)),
            "avg_score": top[score_col].mean(),
            "return_1m": top["future_return_1m"].mean(),
            "return_2m": top["future_return_2m"].mean(),
            "return_3m": top["future_return_3m"].mean(),
            "future_max_return_1_3m": top["future_max_return_1_3m"].mean(),
            "boom_hit_rate": top["boom_label"].mean(),
        })
    out = pd.DataFrame(rows).sort_values("month")
    out["equity_1m_rebalanced"] = equity_curve(out["return_1m"])
    return out


def summarize(bt: pd.DataFrame) -> dict:
    r = bt["return_1m"]
    eq = bt["equity_1m_rebalanced"]
    return {
        "strategy": bt["strategy"].iloc[0],
        "months": len(bt),
        "total_return_1m_rebalanced": float(eq.iloc[-1] - 1) if len(eq) else np.nan,
        "annualized_return_1m_rebalanced": annualized_return(r),
        "avg_monthly_return_1m": float(r.mean()),
        "monthly_win_rate_1m": float((r > 0).mean()),
        "max_drawdown_1m_rebalanced": max_drawdown(eq) if len(eq) else np.nan,
        "avg_future_max_return_1_3m": float(bt["future_max_return_1_3m"].mean()),
        "avg_boom_hit_rate": float(bt["boom_hit_rate"].mean()),
    }


def main() -> None:
    if not PRED_FILE.exists():
        raise FileNotFoundError(f"Missing {PRED_FILE}. Run python src/train_boom_classifier.py first.")
    df = pd.read_csv(PRED_FILE, parse_dates=["month"])

    strategies = [("boom_probability", "xgb_boom_probability")]
    for col in ["mom_3m", "mom_4m", "mom_5m", "mom_6m", "core_mom_456_avg", "mom_6m_acceleration"]:
        if col in df.columns:
            strategies.append((col, f"baseline_{col}"))

    all_bt = []
    summaries = []
    for score_col, name in strategies:
        bt = strategy_from_score(df, score_col, name, TOP_N)
        all_bt.append(bt)
        summaries.append(summarize(bt))

    all_bt_df = pd.concat(all_bt, ignore_index=True)
    summary_df = pd.DataFrame(summaries).sort_values("total_return_1m_rebalanced", ascending=False)

    xgb_bt = all_bt_df[all_bt_df["strategy"] == "xgb_boom_probability"].copy()
    xgb_bt.to_csv(BACKTEST_FILE, index=False)
    summary_df.to_csv(SUMMARY_FILE, index=False)
    summary_df.to_csv(BASELINE_FILE, index=False)

    print(f"Saved {BACKTEST_FILE}")
    print(f"Saved {SUMMARY_FILE}")
    print(f"Saved {BASELINE_FILE}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
