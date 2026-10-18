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
| rows        | 5939       |
| columns     | 63         |
| tickers     | 54         |
| months      | 130        |
| first_month | 2016-01-31 |
| last_month  | 2026-10-31 |

## Train / validation / test metrics
| dataset   |   rows |   months |   positive_rate |    auc |   pr_auc |   accuracy_at_0_5 |   precision_at_0_5 |   recall_at_0_5 |   precision_at_top3_monthly |   precision_at_top5_monthly |   precision_at_top10_monthly |   top3_future_return_1m |   top3_future_return_2m |   top3_future_return_3m |   top3_future_max_return_1_3m |   top_decile_future_max_return |   bottom_decile_future_max_return |   top_minus_bottom_spread |
|:----------|-------:|---------:|----------------:|-------:|---------:|------------------:|-------------------:|----------------:|----------------------------:|----------------------------:|-----------------------------:|------------------------:|------------------------:|------------------------:|------------------------------:|-------------------------------:|----------------------------------:|--------------------------:|
| train     |   3253 |       84 |          0.1039 | 0.9841 |   0.8837 |            0.9158 |             0.555  |          0.9556 |                      0.881  |                      0.7024 |                       0.3952 |                  0.1311 |                  0.251  |                  0.3547 |                        0.411  |                         0.3934 |                            0.074  |                    0.3194 |
| valid     |   1247 |       24 |          0.1051 | 0.6888 |   0.2198 |            0.6961 |             0.1788 |          0.5267 |                      0.2778 |                      0.2667 |                       0.2167 |                  0.0608 |                  0.1151 |                  0.1858 |                        0.3407 |                         0.3063 |                            0.1004 |                    0.2059 |
| test      |   1024 |       19 |          0.1104 | 0.6542 |   0.177  |            0.6104 |             0.1659 |          0.6283 |                      0.2281 |                      0.2    |                       0.2105 |                  0.0498 |                  0.1304 |                  0.152  |                        0.3341 |                         0.3098 |                            0.0919 |                    0.2179 |

## Latest live boom candidates
| month      | ticker   | boom_probability   | mom_4m   | mom_5m   | mom_6m   | core_mom_456_avg   |   ma_trend_score | drawdown_3m   |
|:-----------|:---------|:-------------------|:---------|:---------|:---------|:-------------------|-----------------:|:--------------|
| 2026-10-31 | HPE      | 93.72%             | 22.52%   | 28.79%   | 92.67%   | 47.99%             |                2 | -24.95%       |
| 2026-10-31 | DELL     | 92.18%             | 4.87%    | 7.49%    | 116.54%  | 42.97%             |                1 | -23.23%       |
| 2026-10-31 | CRWD     | 86.03%             | 19.19%   | 24.43%   | 104.05%  | 49.22%             |                2 | -18.46%       |
| 2026-10-31 | NET      | 77.73%             | 20.80%   | 22.53%   | 44.56%   | 29.30%             |                1 | -17.91%       |
| 2026-10-31 | NOW      | 76.97%             | 43.00%   | 14.15%   | 60.76%   | 39.30%             |                3 | -4.44%        |
| 2026-10-31 | AMD      | 75.80%             | -8.98%   | 2.45%    | 49.16%   | 14.21%             |                1 | -18.58%       |
| 2026-10-31 | CRM      | 75.78%             | 53.92%   | 26.50%   | 36.94%   | 39.12%             |                1 | -8.81%        |
| 2026-10-31 | PANW     | 74.57%             | 0.29%    | 21.41%   | 90.72%   | 37.47%             |                2 | -18.55%       |
| 2026-10-31 | MU       | 74.10%             | -23.84%  | -9.46%   | 69.99%   | 12.23%             |                1 | -19.90%       |
| 2026-10-31 | MRVL     | 73.50%             | -23.48%  | 11.19%   | 38.02%   | 8.57%              |                2 | -20.60%       |
| 2026-10-31 | FLNC     | 71.71%             | -70.82%  | -69.28%  | -52.38%  | -64.16%            |                0 | -62.99%       |
| 2026-10-31 | DDOG     | 71.43%             | -5.09%   | -0.10%   | 86.94%   | 27.25%             |                2 | -15.74%       |
| 2026-10-31 | FSLY     | 70.82%             | 22.39%   | 26.48%   | -11.03%  | 12.61%             |                1 | -25.15%       |
| 2026-10-31 | AI       | 70.77%             | 24.53%   | 5.11%    | 28.20%   | 19.28%             |                3 | -6.91%        |
| 2026-10-31 | ZS       | 70.42%             | 46.92%   | 48.41%   | 58.69%   | 51.34%             |                3 | -11.27%       |

## Strategy and baseline comparison
| strategy                     |   months | total_return_1m_rebalanced   | annualized_return_1m_rebalanced   | avg_monthly_return_1m   | monthly_win_rate_1m   | max_drawdown_1m_rebalanced   | avg_future_max_return_1_3m   | avg_boom_hit_rate   |
|:-----------------------------|---------:|:-----------------------------|:----------------------------------|:------------------------|:----------------------|:-----------------------------|:-----------------------------|:--------------------|
| baseline_core_mom_456_avg    |       19 | 207.06%                      | 103.10%                           | 8.11%                   | 63.16%                | -26.15%                      | 39.59%                       | 31.58%              |
| baseline_mom_4m              |       19 | 190.66%                      | 96.19%                            | 7.65%                   | 63.16%                | -26.10%                      | 34.40%                       | 29.82%              |
| baseline_mom_5m              |       19 | 170.78%                      | 87.60%                            | 7.47%                   | 63.16%                | -25.15%                      | 34.23%                       | 29.82%              |
| baseline_mom_6m_acceleration |       19 | 153.19%                      | 79.81%                            | 6.01%                   | 73.68%                | -27.37%                      | 33.69%                       | 26.32%              |
| baseline_mom_6m              |       19 | 140.81%                      | 74.20%                            | 6.80%                   | 68.42%                | -33.70%                      | 36.58%                       | 29.82%              |
| xgb_boom_probability         |       19 | 80.19%                       | 45.05%                            | 4.98%                   | 63.16%                | -36.56%                      | 33.41%                       | 22.81%              |
| baseline_mom_3m              |       19 | 60.44%                       | 34.80%                            | 4.29%                   | 68.42%                | -33.70%                      | 35.06%                       | 28.07%              |

## Recent XGB Top-3 backtest months
| month      | selected_tickers   | avg_score   | return_1m   | future_max_return_1_3m   | boom_hit_rate   |
|:-----------|:-------------------|:------------|:------------|:-------------------------|:----------------|
| 2025-08-31 | SMR, PLTR, SOUN    | 86.68%      | 14.60%      | 30.92%                   | 0.00%           |
| 2025-09-30 | SOUN, OKLO, FLNC   | 82.83%      | 40.99%      | 40.99%                   | 33.33%          |
| 2025-10-31 | ORCL, LRCX, SMR    | 83.44%      | -26.48%     | -10.01%                  | 33.33%          |
| 2025-11-30 | FSLY, SMR, FLNC    | 81.18%      | -13.71%     | 36.02%                   | 66.67%          |
| 2025-12-31 | FLNC, OKLO, SMR    | 82.86%      | 29.96%      | 29.96%                   | 33.33%          |
| 2026-01-31 | FLNC, SNOW, MU     | 77.16%      | -20.90%     | -12.47%                  | 0.00%           |
| 2026-02-28 | FSLY, FLNC, INTC   | 88.16%      | 12.43%      | 74.97%                   | 33.33%          |
| 2026-03-31 | FLNC, MU, FSLY     | 82.39%      | 9.50%       | 91.02%                   | 33.33%          |
| 2026-04-30 | FSLY, ARM, VECO    | 72.14%      | 17.98%      | 36.79%                   | 0.00%           |
| 2026-05-31 | FSLY, GEV, VRT     | 71.53%      | 10.27%      | 18.58%                   | 0.00%           |
| 2026-06-30 | FSLY, FLNC, SMCI   | 83.57%      | -3.23%      | 16.64%                   | 0.00%           |
| 2026-07-31 | ARM, MRVL, MU      | 94.65%      | 10.09%      | 30.38%                   | 33.33%          |

## Top feature importance
| feature             |   importance | is_core_momentum   |
|:--------------------|-------------:|:-------------------|
| volatility_6m       |    0.0411327 | False              |
| volatility_3m       |    0.0279446 | False              |
| mom_7m              |    0.0264821 | False              |
| core_mom_456_max    |    0.0260283 | True               |
| mom_12m             |    0.0253854 | False              |
| drawdown_12m        |    0.0252781 | False              |
| drawdown_3m         |    0.0251239 | False              |
| qqq_mom_12m         |    0.0246011 | False              |
| mom_6m_last3m       |    0.0232108 | False              |
| drawdown_6m         |    0.0230751 | False              |
| drawdown_12m_abs    |    0.0228884 | False              |
| core_mom_456_avg    |    0.0226487 | True               |
| mom_2m              |    0.0224125 | False              |
| ma30_slope_1m       |    0.0223068 | False              |
| mom_3m              |    0.0220872 | False              |
| qqq_mom_3m          |    0.0219267 | False              |
| core_mom_456_min    |    0.0212964 | True               |
| core_mom_456_std    |    0.0212662 | True               |
| mom_6m              |    0.0211787 | True               |
| drawdown_3m_abs     |    0.0206376 | False              |
| price_ma30_ratio    |    0.0205163 | False              |
| ma5_slope_1m        |    0.0204514 | False              |
| price_ma20_ratio    |    0.0204389 | False              |
| mom_6m_first3m      |    0.0201493 | False              |
| mom_6m_acceleration |    0.0201052 | True               |

## Feature design note
See `outputs/feature_manifest.txt` for the exact clean feature list.

## Correctness notes
- The model does not try to predict exact return size; it predicts probability of joining the future monthly top-10% boom group.
- The split is time-based: train <= 2022, validation 2023–2024, test >= 2025.
- The latest live candidates table does not use future returns; it only uses the newest available panel features.
- This is a research backtest, not investment advice.