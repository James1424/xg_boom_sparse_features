import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from config import (
    PANEL_FILE,
    OUTPUT_DIR,
    MODELS_DIR,
    FEATURE_COLUMNS,
    CORE_MOMENTUM_FEATURES,
    MODEL_PARAMS,
    TRAIN_END,
    VALID_END,
    BOOM_TOP_PCT,
)

METRICS_FILE = OUTPUT_DIR / "boom_model_metrics.csv"
PRED_ALL_FILE = OUTPUT_DIR / "boom_predictions_all.csv"
PRED_TEST_FILE = OUTPUT_DIR / "boom_predictions_test.csv"
LIVE_FILE = OUTPUT_DIR / "boom_live_candidates.csv"
IMPORTANCE_FILE = OUTPUT_DIR / "boom_feature_importance.csv"
MODEL_FILE = MODELS_DIR / "xgb_boom_classifier.json"
FEATURES_FILE = MODELS_DIR / "boom_feature_columns.json"


def add_boom_label(df: pd.DataFrame, top_pct: float = BOOM_TOP_PCT) -> pd.DataFrame:
    out = df.copy()
    if "future_max_return_1_3m" not in out.columns:
        out["future_max_return_1_3m"] = out[["future_return_1m", "future_return_2m", "future_return_3m"]].max(axis=1)
    # Monthly percentile: 1 means biggest future max return in that month.
    pct = out.groupby("month")["future_max_return_1_3m"].rank(pct=True, ascending=True)
    out["future_max_return_pct"] = pct
    out["boom_label"] = (out["future_max_return_pct"] >= 1 - top_pct).astype(int)
    return out


def time_split(df: pd.DataFrame):
    train = df[df["month"] <= pd.Timestamp(TRAIN_END)].copy()
    valid = df[(df["month"] > pd.Timestamp(TRAIN_END)) & (df["month"] <= pd.Timestamp(VALID_END))].copy()
    test = df[df["month"] > pd.Timestamp(VALID_END)].copy()
    return train, valid, test


def clean_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = df[features].replace([np.inf, -np.inf], np.nan).copy()
    # Median imputation is fitted on the supplied frame. For stricter production use, save train medians.
    return X.fillna(X.median(numeric_only=True)).astype(float)


def monthly_top_precision(data: pd.DataFrame, k: int) -> float:
    vals = []
    for _, g in data.groupby("month"):
        top = g.sort_values("boom_probability", ascending=False).head(k)
        if len(top):
            vals.append(top["boom_label"].mean())
    return float(np.mean(vals)) if vals else np.nan


def monthly_top_return(data: pd.DataFrame, k: int, ret_col: str) -> float:
    vals = []
    for _, g in data.groupby("month"):
        top = g.sort_values("boom_probability", ascending=False).head(k)
        if len(top) and ret_col in top.columns:
            vals.append(top[ret_col].mean())
    return float(np.nanmean(vals)) if vals else np.nan


def decile_spread(data: pd.DataFrame) -> tuple[float, float, float]:
    top_vals, bottom_vals = [], []
    for _, g in data.groupby("month"):
        if len(g) < 10:
            continue
        n = max(1, int(len(g) * 0.10))
        s = g.sort_values("boom_probability", ascending=False)
        top_vals.append(s.head(n)["future_max_return_1_3m"].mean())
        bottom_vals.append(s.tail(n)["future_max_return_1_3m"].mean())
    top = float(np.nanmean(top_vals)) if top_vals else np.nan
    bottom = float(np.nanmean(bottom_vals)) if bottom_vals else np.nan
    return top, bottom, top - bottom if pd.notna(top) and pd.notna(bottom) else np.nan


def evaluate(name: str, data: pd.DataFrame) -> dict:
    y = data["boom_label"].astype(int)
    p = data["boom_probability"].astype(float)
    pred = (p >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y, p)
    except Exception:
        auc = np.nan
    try:
        pr_auc = average_precision_score(y, p)
    except Exception:
        pr_auc = np.nan
    top_dec, bot_dec, spread = decile_spread(data)
    return {
        "dataset": name,
        "rows": len(data),
        "months": data["month"].nunique(),
        "positive_rate": y.mean(),
        "auc": auc,
        "pr_auc": pr_auc,
        "accuracy_at_0_5": accuracy_score(y, pred),
        "precision_at_0_5": precision_score(y, pred, zero_division=0),
        "recall_at_0_5": recall_score(y, pred, zero_division=0),
        "precision_at_top3_monthly": monthly_top_precision(data, 3),
        "precision_at_top5_monthly": monthly_top_precision(data, 5),
        "precision_at_top10_monthly": monthly_top_precision(data, 10),
        "top3_future_return_1m": monthly_top_return(data, 3, "future_return_1m"),
        "top3_future_return_2m": monthly_top_return(data, 3, "future_return_2m"),
        "top3_future_return_3m": monthly_top_return(data, 3, "future_return_3m"),
        "top3_future_max_return_1_3m": monthly_top_return(data, 3, "future_max_return_1_3m"),
        "top_decile_future_max_return": top_dec,
        "bottom_decile_future_max_return": bot_dec,
        "top_minus_bottom_spread": spread,
    }


def main() -> None:
    if not PANEL_FILE.exists():
        raise FileNotFoundError(f"Missing {PANEL_FILE}. Run python src/download_data.py and python src/build_clean_panel.py first.")

    df = pd.read_csv(PANEL_FILE, parse_dates=["month"])
    features = [c for c in FEATURE_COLUMNS if c in df.columns]
    forbidden = [c for c in features if c.startswith("rank_") or c.startswith("pct_") or c in ["sector_group", "industry_group", "theme", "universe_tag"]]
    if forbidden:
        raise ValueError(f"Forbidden features found: {forbidden}")

    # Rows without future labels are kept for live inference, but removed from supervised training.
    supervised = df.dropna(subset=["future_return_1m", "future_return_2m", "future_return_3m", "future_max_return_1_3m"]).copy()
    supervised = add_boom_label(supervised)
    supervised = supervised.dropna(subset=features).copy()

    train, valid, test = time_split(supervised)
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError(f"Not enough rows after split: train={len(train)}, valid={len(valid)}, test={len(test)}")

    X_train = clean_features(train, features)
    y_train = train["boom_label"].astype(int)
    X_valid = clean_features(valid, features)
    y_valid = valid["boom_label"].astype(int)

    # Class imbalance: top 10% positives.
    pos = y_train.sum()
    neg = len(y_train) - pos
    params = MODEL_PARAMS.copy()
    params["scale_pos_weight"] = float(neg / pos) if pos > 0 else 1.0

    model = XGBClassifier(**params)

    fit_kwargs = {
        "X": X_train,
        "y": y_train,
        "eval_set": [(X_valid, y_valid)],
        "verbose": 50,
    }

    # XGBoost sklearn supports feature_weights in many recent versions. Use it when available.
    feature_weights = np.ones(len(features), dtype=float)
    for i, c in enumerate(features):
        if c in CORE_MOMENTUM_FEATURES or c in ["mom_4m", "mom_5m", "mom_6m"]:
            feature_weights[i] = 3.0
        elif c.startswith("mom_") or c.startswith("core_mom"):
            feature_weights[i] = 2.0
    try:
        model.fit(**fit_kwargs, feature_weights=feature_weights)
    except TypeError:
        model.fit(**fit_kwargs)

    def predict_frame(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
        out = frame.copy()
        out["dataset"] = dataset
        out["boom_probability"] = model.predict_proba(clean_features(out, features))[:, 1]
        return out

    train_p = predict_frame(train, "train")
    valid_p = predict_frame(valid, "valid")
    test_p = predict_frame(test, "test")
    all_pred = pd.concat([train_p, valid_p, test_p], ignore_index=True)

    metrics = pd.DataFrame([
        evaluate("train", train_p),
        evaluate("valid", valid_p),
        evaluate("test", test_p),
    ])
    metrics.to_csv(METRICS_FILE, index=False)

    keep_cols = [
        "dataset", "month", "ticker", "boom_probability", "boom_label", "future_max_return_pct",
        "future_return_1m", "future_return_2m", "future_return_3m", "future_max_return_1_3m",
    ] + features
    all_pred[[c for c in keep_cols if c in all_pred.columns]].to_csv(PRED_ALL_FILE, index=False)
    test_p[[c for c in keep_cols if c in test_p.columns]].to_csv(PRED_TEST_FILE, index=False)

    # Live inference on the latest month, even if future labels are not available.
    latest_month = df["month"].max()
    live = df[df["month"] == latest_month].copy()
    live = live.dropna(subset=features)
    live["boom_probability"] = model.predict_proba(clean_features(live, features))[:, 1]
    live = live.sort_values("boom_probability", ascending=False)
    live_cols = ["month", "ticker", "boom_probability"] + features
    live[[c for c in live_cols if c in live.columns]].to_csv(LIVE_FILE, index=False)

    imp = pd.DataFrame({"feature": features, "importance": model.feature_importances_})
    imp["is_core_momentum"] = imp["feature"].isin(CORE_MOMENTUM_FEATURES)
    imp = imp.sort_values("importance", ascending=False)
    imp.to_csv(IMPORTANCE_FILE, index=False)

    model.save_model(MODEL_FILE)
    FEATURES_FILE.write_text(json.dumps(features, indent=2), encoding="utf-8")

    print("Saved:")
    for p in [METRICS_FILE, PRED_ALL_FILE, PRED_TEST_FILE, LIVE_FILE, IMPORTANCE_FILE, MODEL_FILE, FEATURES_FILE]:
        print(f"  {p}")
    print("\nMetrics:")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
