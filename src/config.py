from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

UNIVERSE_FILE = DATA_DIR / "universe.csv"
PRICE_FILE = DATA_DIR / "prices_daily.csv"
PANEL_FILE = OUTPUT_DIR / "ml_panel_clean_momentum.csv"
EXCEL_PANEL_FILE = OUTPUT_DIR / "ml_panel_clean_momentum.xlsx"
FEATURE_MANIFEST_FILE = OUTPUT_DIR / "feature_manifest.txt"

BENCHMARK = "QQQ"
START_DATE = "2015-01-01"
END_DATE = None
FIRST_SAMPLE_MONTH = "2016-01-31"

TRAIN_END = "2022-12-31"
VALID_END = "2024-12-31"
TEST_START = "2025-01-01"

TOP_N = 3
BOOM_TOP_PCT = 0.10

# Strictly no rank_*, pct_*, sector_group, industry_group, theme, universe_tag.
# The model is forced to learn price-action / momentum-shape features only.
CORE_MOMENTUM_FEATURES = [
    "mom_4m", "mom_5m", "mom_6m",
    "core_mom_456_avg", "core_mom_456_min", "core_mom_456_max", "core_mom_456_std",
    "core_mom_456_positive_count", "core_mom_456_all_positive",
    "mom_4m_vs_6m", "mom_5m_vs_6m", "mom_6m_acceleration",
]

FEATURE_COLUMNS = [
    # Momentum levels. 4/5/6 months are intentionally kept central.
    "mom_1m", "mom_2m", "mom_3m", "mom_4m", "mom_5m", "mom_6m", "mom_7m", "mom_9m", "mom_12m",

    # Core 4/5/6 month momentum shape.
    "core_mom_456_avg", "core_mom_456_min", "core_mom_456_max", "core_mom_456_std",
    "core_mom_456_positive_count", "core_mom_456_all_positive",
    "mom_4m_vs_6m", "mom_5m_vs_6m",

    # Momentum acceleration / persistence.
    "mom_6m_first3m", "mom_6m_last3m", "mom_6m_acceleration", "mom_3m_vs_6m",

    # Trend shape.
    "price_ma5_ratio", "price_ma10_ratio", "price_ma20_ratio", "price_ma30_ratio",
    "ma5_above_ma10", "ma10_above_ma20", "ma20_above_ma30", "ma_trend_score",
    "ma5_slope_1m", "ma10_slope_1m", "ma20_slope_1m", "ma30_slope_1m", "ma_slope_score",

    # Risk / pullback context.
    "drawdown_3m", "drawdown_6m", "drawdown_12m",
    "drawdown_3m_abs", "drawdown_6m_abs", "drawdown_12m_abs",
    "near_3m_high", "near_6m_high", "deep_3m_drawdown", "deep_6m_drawdown",
    "volatility_3m", "volatility_6m",
    "return_vol_ratio_3m", "return_vol_ratio_6m",

    # A small number of liquidity/volume features only.
    "volume_change_1m", "volume_change_3m", "volume_ratio_3m",

    # Market regime.
    "qqq_mom_1m", "qqq_mom_3m", "qqq_mom_6m", "qqq_mom_12m", "qqq_trend_score",
]

MODEL_PARAMS = {
    "n_estimators": 700,
    "max_depth": 3,
    "learning_rate": 0.025,
    "subsample": 0.80,
    "colsample_bytree": 0.75,
    "min_child_weight": 8,
    "reg_alpha": 0.20,
    "reg_lambda": 6.00,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "random_state": 42,
    "n_jobs": -1,
}
