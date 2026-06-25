import pandas as pd
from config import PROJECT_ROOT, PANEL_FILE, FEATURE_MANIFEST_FILE, OUTPUT_DIR

README_FILE = PROJECT_ROOT / "README.md"
METRICS_FILE = OUTPUT_DIR / "boom_model_metrics.csv"
LIVE_FILE = OUTPUT_DIR / "boom_live_candidates.csv"
BACKTEST_FILE = OUTPUT_DIR / "boom_top3_backtest.csv"
SUMMARY_FILE = OUTPUT_DIR / "boom_strategy_summary.csv"
IMPORTANCE_FILE = OUTPUT_DIR / "boom_feature_importance.csv"
BASELINE_FILE = OUTPUT_DIR / "boom_baseline_comparison.csv"


def fmt_pct(x):
    if pd.isna(x):
        return ""
    return f"{x*100:.2f}%"


def fmt_num(x):
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def table(df):
    if df.empty:
        return "_No data available._"
    return df.to_markdown(index=False)


def main() -> None:
    lines = []
    lines.append("# Boom Momentum XGBoost Classifier")
    lines.append("")
    lines.append("This project builds a clean monthly momentum panel and trains an XGBoost classifier to identify stocks with future 1–3 month boom potential.")
    lines.append("")
    lines.append("The project intentionally removes noisy cross-sectional/categorical features: `rank_*`, `pct_*`, `sector_group`, `industry_group`, `theme`, `universe_tag`, `market_cap`, and `size_bucket`. The model is forced to learn momentum shape, especially 4/5/6-month momentum, plus a small number of trend, pullback, volatility, volume, and QQQ regime features.")
    lines.append("")
    lines.append("## Pipeline")
    lines.append("```bash\npip install -r requirements.txt\npython src/download_data.py\npython src/build_clean_panel.py\npython src/check_panel.py\npython src/train_boom_classifier.py\npython src/backtest_boom_strategy.py\npython src/update_readme.py\n```")
    lines.append("")
    lines.append("## Target")
    lines.append("```text\nfuture_max_return_1_3m = max(future_return_1m, future_return_2m, future_return_3m)\nboom_label = 1 if future_max_return_1_3m is in the monthly top 10%, otherwise 0\n```")
    lines.append("")

    if PANEL_FILE.exists():
        panel = pd.read_csv(PANEL_FILE, parse_dates=["month"])
        lines.append("## Clean panel summary")
        summary = pd.DataFrame({
            "metric": ["rows", "columns", "tickers", "months", "first_month", "last_month"],
            "value": [len(panel), len(panel.columns), panel["ticker"].nunique(), panel["month"].nunique(), panel["month"].min().date(), panel["month"].max().date()],
        })
        lines.append(table(summary))
        lines.append("")

    if METRICS_FILE.exists():
        metrics = pd.read_csv(METRICS_FILE)
        lines.append("## Train / validation / test metrics")
        show = metrics.copy()
        for c in show.columns:
            if c not in ["dataset", "rows", "months"]:
                show[c] = show[c].map(fmt_num)
        lines.append(table(show))
        lines.append("")

    if LIVE_FILE.exists():
        live = pd.read_csv(LIVE_FILE, parse_dates=["month"]).head(15)
        lines.append("## Latest live boom candidates")
        show_cols = ["month", "ticker", "boom_probability", "mom_4m", "mom_5m", "mom_6m", "core_mom_456_avg", "ma_trend_score", "drawdown_3m"]
        show = live[[c for c in show_cols if c in live.columns]].copy()
        if "month" in show.columns:
            show["month"] = show["month"].dt.strftime("%Y-%m-%d")
        for c in ["boom_probability", "mom_4m", "mom_5m", "mom_6m", "core_mom_456_avg", "drawdown_3m"]:
            if c in show.columns:
                show[c] = show[c].map(fmt_pct)
        lines.append(table(show))
        lines.append("")

    if SUMMARY_FILE.exists():
        summ = pd.read_csv(SUMMARY_FILE)
        lines.append("## Strategy and baseline comparison")
        show = summ.copy()
        for c in show.columns:
            if c != "strategy":
                if "return" in c or "rate" in c or "drawdown" in c:
                    show[c] = show[c].map(fmt_pct)
                elif c not in ["months"]:
                    show[c] = show[c].map(fmt_num)
        lines.append(table(show))
        lines.append("")

    if BACKTEST_FILE.exists():
        bt = pd.read_csv(BACKTEST_FILE, parse_dates=["month"]).tail(12)
        lines.append("## Recent XGB Top-3 backtest months")
        show = bt.copy()
        show["month"] = show["month"].dt.strftime("%Y-%m-%d")
        for c in ["avg_score", "return_1m", "return_2m", "return_3m", "future_max_return_1_3m", "boom_hit_rate"]:
            if c in show.columns:
                show[c] = show[c].map(fmt_pct)
        lines.append(table(show[["month", "selected_tickers", "avg_score", "return_1m", "future_max_return_1_3m", "boom_hit_rate"]]))
        lines.append("")

    if IMPORTANCE_FILE.exists():
        imp = pd.read_csv(IMPORTANCE_FILE).head(25)
        lines.append("## Top feature importance")
        lines.append(table(imp))
        lines.append("")

    if FEATURE_MANIFEST_FILE.exists():
        lines.append("## Feature design note")
        lines.append("See `outputs/feature_manifest.txt` for the exact clean feature list.")
        lines.append("")

    lines.append("## Correctness notes")
    lines.append("- The model does not try to predict exact return size; it predicts probability of joining the future monthly top-10% boom group.")
    lines.append("- The split is time-based: train <= 2022, validation 2023–2024, test >= 2025.")
    lines.append("- The latest live candidates table does not use future returns; it only uses the newest available panel features.")
    lines.append("- This is a research backtest, not investment advice.")

    README_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Updated {README_FILE}")


if __name__ == "__main__":
    main()
