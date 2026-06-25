import pandas as pd
from config import PANEL_FILE, FEATURE_COLUMNS


def main() -> None:
    if not PANEL_FILE.exists():
        raise FileNotFoundError(f"Missing {PANEL_FILE}. Run python src/build_clean_panel.py first.")
    df = pd.read_csv(PANEL_FILE, parse_dates=["month"])
    forbidden = [c for c in df.columns if c.startswith("rank_") or c.startswith("pct_") or c in ["sector_group", "industry_group", "theme", "universe_tag", "market_cap", "size_bucket"]]
    if forbidden:
        raise ValueError(f"Forbidden noisy columns found: {forbidden}")
    features = [c for c in FEATURE_COLUMNS if c in df.columns]
    print("Panel check passed.")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"Tickers: {df['ticker'].nunique():,}")
    print(f"Months: {df['month'].nunique():,}")
    print(f"First month: {df['month'].min().date()}")
    print(f"Last month: {df['month'].max().date()}")
    print(f"Model features available: {len(features)}")
    missing_rates = df[features].isna().mean().sort_values(ascending=False).head(12)
    print("\nTop feature missing rates:")
    print(missing_rates.to_string())


if __name__ == "__main__":
    main()
