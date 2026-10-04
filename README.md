# Boom Momentum XGBoost Classifier

This project builds a clean monthly momentum panel and trains an XGBoost classifier to identify stocks with future 1–3 month boom potential.

The project intentionally removes noisy cross-sectional/categorical features: `rank_*`, `pct_*`, `sector_group`, `industry_group`, `theme`, `universe_tag`, `market_cap`, and `size_bucket`. The model is forced to learn momentum shape, especially 4/5/6-month momentum, plus a small number of trend, pullback, volatility, volume, and QQQ regime features.

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

## Target
```text
future_max_return_1_3m = max(future_return_1m, future_return_2m, future_return_3m)
boom_label = 1 if future_max_return_1_3m is in the monthly top 10%, otherwise 0
```

## Clean panel summary
| metric      | value      |
|:------------|:-----------|
| rows        | 5809       |
| columns     | 63         |
| tickers     | 53         |
| months      | 130        |
| first_month | 2016-01-31 |
| last_month  | 2026-10-31 |

## Train / validation / test metrics
| dataset   |   rows |   months |   positive_rate |    auc |   pr_auc |   accuracy_at_0_5 |   precision_at_0_5 |   recall_at_0_5 |   precision_at_top3_monthly |   precision_at_top5_monthly |   precision_at_top10_monthly |   top3_future_return_1m |   top3_future_return_2m |   top3_future_return_3m |   top3_future_max_return_1_3m |   top_decile_future_max_return |   bottom_decile_future_max_return |   top_minus_bottom_spread |
|:----------|-------:|---------:|----------------:|-------:|---------:|------------------:|-------------------:|----------------:|----------------------------:|----------------------------:|-----------------------------:|------------------------:|------------------------:|------------------------:|------------------------------:|-------------------------------:|----------------------------------:|--------------------------:|
| train     |   3169 |       84 |          0.1048 | 0.9858 |   0.8921 |            0.9205 |             0.5707 |          0.9729 |                      0.8849 |                      0.7    |                       0.3929 |                  0.1319 |                  0.249  |                  0.3519 |                        0.4068 |                         0.3971 |                            0.0741 |                    0.323  |
| valid     |   1223 |       24 |          0.1071 | 0.6834 |   0.2296 |            0.7016 |             0.1821 |          0.5115 |                      0.2917 |                      0.275  |                       0.2333 |                  0.0763 |                  0.1351 |                  0.186  |                        0.3459 |                         0.3014 |                            0.0932 |                    0.2082 |
| test      |   1005 |       19 |          0.1124 | 0.6246 |   0.1633 |            0.6119 |             0.1529 |          0.5398 |                      0.1579 |                      0.1895 |                       0.2    |                  0.037  |                  0.1015 |                  0.1013 |                        0.2743 |                         0.2993 |                            0.0897 |                    0.2096 |

## Latest live boom candidates
| month      | ticker   | boom_probability   | mom_4m   | mom_5m   | mom_6m   | core_mom_456_avg   |   ma_trend_score | drawdown_3m   |
|:-----------|:---------|:-------------------|:---------|:---------|:---------|:-------------------|-----------------:|:--------------|
| 2026-10-31 | MU       | 89.09%             | -6.86%   | 10.72%   | 107.88%  | 37.24%             |                2 | -2.05%        |
| 2026-10-31 | DELL     | 88.11%             | 30.59%   | 33.86%   | 169.66%  | 78.04%             |                2 | -4.40%        |
| 2026-10-31 | AMD      | 87.72%             | 9.12%    | 22.83%   | 78.82%   | 36.92%             |                2 | 0.00%         |
| 2026-10-31 | NOW      | 87.54%             | 35.35%   | 8.05%    | 52.17%   | 31.86%             |                0 | -9.20%        |
| 2026-10-31 | MRVL     | 87.20%             | -8.57%   | 32.86%   | 64.92%   | 29.73%             |                3 | 0.00%         |
| 2026-10-31 | INTC     | 83.21%             | -14.54%  | 4.05%    | 26.30%   | 5.27%              |                2 | -6.33%        |
| 2026-10-31 | FLNC     | 82.14%             | -61.72%  | -59.69%  | -37.52%  | -52.98%            |                1 | -54.54%       |
| 2026-10-31 | CRWD     | 80.05%             | 41.54%   | 47.76%   | 142.32%  | 77.21%             |                3 | 0.00%         |
| 2026-10-31 | LRCX     | 79.86%             | -19.72%  | 9.40%    | 35.00%   | 8.23%              |                3 | -1.50%        |
| 2026-10-31 | AMAT     | 79.37%             | -25.23%  | 20.12%   | 37.21%   | 10.70%             |                3 | -10.27%       |
| 2026-10-31 | PANW     | 78.61%             | 18.25%   | 43.15%   | 124.87%  | 62.09%             |                3 | 0.00%         |
| 2026-10-31 | SNOW     | 78.37%             | 34.00%   | 33.45%   | 149.90%  | 72.45%             |                3 | -4.33%        |
| 2026-10-31 | DDOG     | 77.92%             | 6.48%    | 12.08%   | 109.71%  | 42.75%             |                3 | -3.79%        |
| 2026-10-31 | HPE      | 75.48%             | 54.08%   | 61.96%   | 142.30%  | 86.11%             |                3 | 0.00%         |
| 2026-10-31 | KLAC     | 73.50%             | -31.35%  | 7.78%    | 18.48%   | -1.70%             |                3 | -10.54%       |

## Strategy and baseline comparison
| strategy                     |   months | total_return_1m_rebalanced   | annualized_return_1m_rebalanced   | avg_monthly_return_1m   | monthly_win_rate_1m   | max_drawdown_1m_rebalanced   | avg_future_max_return_1_3m   | avg_boom_hit_rate   |
|:-----------------------------|---------:|:-----------------------------|:----------------------------------|:------------------------|:----------------------|:-----------------------------|:-----------------------------|:--------------------|
| baseline_core_mom_456_avg    |       19 | 207.06%                      | 103.10%                           | 8.11%                   | 63.16%                | -26.15%                      | 39.80%                       | 31.58%              |
| baseline_mom_4m              |       19 | 190.66%                      | 96.19%                            | 7.65%                   | 63.16%                | -26.10%                      | 34.61%                       | 29.82%              |
| baseline_mom_6m_acceleration |       19 | 173.39%                      | 88.74%                            | 6.55%                   | 73.68%                | -27.37%                      | 34.28%                       | 26.32%              |
| baseline_mom_5m              |       19 | 170.78%                      | 87.60%                            | 7.47%                   | 63.16%                | -25.15%                      | 34.44%                       | 29.82%              |
| baseline_mom_6m              |       19 | 140.81%                      | 74.20%                            | 6.80%                   | 68.42%                | -33.70%                      | 36.77%                       | 29.82%              |
| baseline_mom_3m              |       19 | 60.44%                       | 34.80%                            | 4.29%                   | 68.42%                | -33.70%                      | 35.20%                       | 28.07%              |
| xgb_boom_probability         |       19 | 34.68%                       | 20.69%                            | 3.70%                   | 63.16%                | -54.59%                      | 27.43%                       | 15.79%              |

## Recent XGB Top-3 backtest months
| month      | selected_tickers   | avg_score   | return_1m   | future_max_return_1_3m   | boom_hit_rate   |
|:-----------|:-------------------|:------------|:------------|:-------------------------|:----------------|
| 2025-08-31 | SMR, OKLO, PLTR    | 88.85%      | 23.96%      | 45.91%                   | 33.33%          |
| 2025-09-30 | SOUN, OKLO, SMR    | 79.80%      | 17.72%      | 17.72%                   | 0.00%           |
| 2025-10-31 | ARM, ORCL, SMR     | 84.29%      | -32.90%     | -32.90%                  | 0.00%           |
| 2025-11-30 | SMR, FLNC, OKLO    | 81.34%      | -16.64%     | 10.40%                   | 33.33%          |
| 2025-12-31 | SMR, OKLO, MDB     | 82.95%      | 7.60%       | 7.60%                    | 0.00%           |
| 2026-01-31 | FLNC, MDB, SNOW    | 79.87%      | -24.55%     | -24.55%                  | 0.00%           |
| 2026-02-28 | FSLY, FLNC, INTC   | 85.64%      | 12.43%      | 74.97%                   | 33.33%          |
| 2026-03-31 | FLNC, MU, FSLY     | 77.00%      | 9.50%       | 91.02%                   | 33.33%          |
| 2026-04-30 | FSLY, NOW, ARM     | 69.63%      | 26.38%      | 33.05%                   | 0.00%           |
| 2026-05-31 | FSLY, GEV, TSM     | 76.20%      | 13.05%      | 21.36%                   | 0.00%           |
| 2026-06-30 | FSLY, MU, VRT      | 85.29%      | -11.04%     | 3.12%                    | 0.00%           |
| 2026-07-31 | ARM, MRVL, INTC    | 93.49%      | 4.34%       | 35.58%                   | 33.33%          |

## Top feature importance
| feature             |   importance | is_core_momentum   |
|:--------------------|-------------:|:-------------------|
| volatility_6m       |    0.0385999 | False              |
| ma20_above_ma30     |    0.0295524 | False              |
| mom_6m_last3m       |    0.0281394 | False              |
| volatility_3m       |    0.0262804 | False              |
| mom_7m              |    0.0244255 | False              |
| mom_12m             |    0.0242416 | False              |
| qqq_mom_12m         |    0.0239958 | False              |
| core_mom_456_avg    |    0.0237638 | True               |
| drawdown_3m_abs     |    0.0234279 | False              |
| drawdown_3m         |    0.0231888 | False              |
| core_mom_456_max    |    0.022775  | True               |
| core_mom_456_min    |    0.0223367 | True               |
| drawdown_12m        |    0.0222148 | False              |
| drawdown_12m_abs    |    0.0219719 | False              |
| mom_3m              |    0.0216814 | False              |
| qqq_mom_3m          |    0.0211373 | False              |
| mom_4m              |    0.0207927 | True               |
| ma30_slope_1m       |    0.0205602 | False              |
| mom_2m              |    0.0204987 | False              |
| core_mom_456_std    |    0.0201297 | True               |
| price_ma20_ratio    |    0.0200297 | False              |
| mom_9m              |    0.0200044 | False              |
| mom_6m_acceleration |    0.0199683 | True               |
| qqq_mom_1m          |    0.0195941 | False              |
| mom_6m_first3m      |    0.01942   | False              |

## Feature design note
See `outputs/feature_manifest.txt` for the exact clean feature list.

## Correctness notes
- The model does not try to predict exact return size; it predicts probability of joining the future monthly top-10% boom group.
- The split is time-based: train <= 2022, validation 2023–2024, test >= 2025.
- The latest live candidates table does not use future returns; it only uses the newest available panel features.
- This is a research backtest, not investment advice.