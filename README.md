# Boom Momentum XGBoost Classifier

This repository builds a **clean monthly momentum panel** and trains a **boom-stock classifier**. The goal is not to predict exact return size. The goal is to identify stocks whose current momentum shape resembles past stocks that later entered the future winner group.

## Core idea

```text
future_max_return_1_3m = max(future_return_1m, future_return_2m, future_return_3m)
boom_label = 1 if future_max_return_1_3m is in the monthly top 10%, otherwise 0
```

The model outputs:

```text
boom_probability
```

Each month, the strategy selects the Top 3 stocks by `boom_probability`.

## What this version deliberately removes

This project removes features that may distract the model from momentum shape:

```text
rank_*
pct_*
sector_group
industry_group
theme
universe_tag
market_cap
size_bucket
```

The model focuses on:

```text
4/5/6-month momentum
core 4/5/6 momentum shape
momentum acceleration
moving-average trend
pullback/drawdown context
volatility context
small number of volume features
QQQ market regime
```

## Pipeline

```bash
pip install -r requirements.txt
python src/download_data.py
python src/build_clean_panel.py
python src/check_panel.py
python src/train_boom_classifier.py
python src/backtest_boom_strategy.py
python src/update_readme.py
```

## Outputs

```text
outputs/ml_panel_clean_momentum.csv
outputs/boom_model_metrics.csv
outputs/boom_predictions_all.csv
outputs/boom_predictions_test.csv
outputs/boom_live_candidates.csv
outputs/boom_feature_importance.csv
outputs/boom_top3_backtest.csv
outputs/boom_strategy_summary.csv
outputs/boom_baseline_comparison.csv
models/xgb_boom_classifier.json
models/boom_feature_columns.json
```

## Model evaluation

The model reports train / validation / test metrics, including:

```text
AUC
PR-AUC
Precision@Top3 monthly
Precision@Top5 monthly
Precision@Top10 monthly
Top3 future 1M / 2M / 3M return
Top3 future max return 1–3M
Top decile vs bottom decile spread
```

It also compares the XGBoost boom strategy against simple baselines:

```text
mom_3m Top 3
mom_4m Top 3
mom_5m Top 3
mom_6m Top 3
core_mom_456_avg Top 3
mom_6m_acceleration Top 3
```

## GitHub Actions

The workflow `.github/workflows/build_train.yml` can be run manually or weekly. It downloads data, builds the clean panel, trains the model, backtests it, updates README, and commits outputs.

