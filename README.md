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
| rows        | 5831       |
| columns     | 63         |
| tickers     | 54         |
| months      | 128        |
| first_month | 2016-01-31 |
| last_month  | 2026-08-31 |

## Train / validation / test metrics
| dataset   |   rows |   months |   positive_rate |    auc |   pr_auc |   accuracy_at_0_5 |   precision_at_0_5 |   recall_at_0_5 |   precision_at_top3_monthly |   precision_at_top5_monthly |   precision_at_top10_monthly |   top3_future_return_1m |   top3_future_return_2m |   top3_future_return_3m |   top3_future_max_return_1_3m |   top_decile_future_max_return |   bottom_decile_future_max_return |   top_minus_bottom_spread |
|:----------|-------:|---------:|----------------:|-------:|---------:|------------------:|-------------------:|----------------:|----------------------------:|----------------------------:|-----------------------------:|------------------------:|------------------------:|------------------------:|------------------------------:|-------------------------------:|----------------------------------:|--------------------------:|
| train     |   3253 |       84 |          0.1039 | 0.9841 |   0.8837 |            0.9158 |             0.555  |          0.9556 |                      0.881  |                      0.7024 |                       0.3952 |                  0.1311 |                  0.251  |                  0.3547 |                        0.411  |                         0.3934 |                            0.074  |                    0.3194 |
| valid     |   1247 |       24 |          0.1051 | 0.6888 |   0.2198 |            0.6961 |             0.1788 |          0.5267 |                      0.2778 |                      0.2667 |                       0.2167 |                  0.0608 |                  0.1151 |                  0.1858 |                        0.3407 |                         0.3063 |                            0.1004 |                    0.2059 |
| test      |    916 |       17 |          0.1103 | 0.6759 |   0.2043 |            0.6321 |             0.1722 |          0.6139 |                      0.2549 |                      0.2235 |                       0.2353 |                  0.0516 |                  0.1267 |                  0.1751 |                        0.3537 |                         0.3328 |                            0.0972 |                    0.2356 |

## Latest live boom candidates
| month      | ticker   | boom_probability   | mom_4m   | mom_5m   | mom_6m   | core_mom_456_avg   |   ma_trend_score | drawdown_3m   |
|:-----------|:---------|:-------------------|:---------|:---------|:---------|:-------------------|-----------------:|:--------------|
| 2026-08-31 | DELL     | 93.27%             | 135.28%  | 200.46%  | 233.02%  | 189.59%            |                3 | -0.75%        |
| 2026-08-31 | CRWD     | 92.97%             | 94.68%   | 122.28%  | 133.29%  | 116.75%            |                3 | -3.80%        |
| 2026-08-31 | MRVL     | 91.55%             | 34.47%   | 124.32%  | 171.99%  | 110.26%            |                2 | -29.82%       |
| 2026-08-31 | MU       | 88.16%             | 87.91%   | 187.65%  | 135.76%  | 137.11%            |                2 | -19.92%       |
| 2026-08-31 | PANW     | 87.72%             | 114.29%  | 139.69%  | 158.04%  | 137.34%            |                3 | -2.96%        |
| 2026-08-31 | NET      | 86.89%             | 54.06%   | 53.04%   | 83.39%   | 63.50%             |                3 | -4.55%        |
| 2026-08-31 | ARM      | 86.11%             | 32.86%   | 84.72%   | 119.25%  | 78.95%             |                2 | -36.41%       |
| 2026-08-31 | HPE      | 85.73%             | 104.66%  | 147.30%  | 176.02%  | 142.66%            |                3 | -1.86%        |
| 2026-08-31 | SNOW     | 81.57%             | 141.02%  | 118.09%  | 95.31%   | 118.14%            |                3 | -2.51%        |
| 2026-08-31 | INTC     | 81.47%             | 8.49%    | 132.27%  | 124.73%  | 88.50%             |                2 | -27.27%       |
| 2026-08-31 | FSLY     | 80.15%             | 18.51%   | 2.99%    | 56.54%   | 26.01%             |                3 | -0.30%        |
| 2026-08-31 | PATH     | 80.01%             | 55.44%   | 44.23%   | 49.21%   | 49.63%             |                3 | -4.02%        |
| 2026-08-31 | ANET     | 78.39%             | 15.12%   | 61.93%   | 48.93%   | 41.99%             |                3 | -5.55%        |
| 2026-08-31 | AOSL     | 76.54%             | -29.15%  | 38.85%   | 46.45%   | 18.72%             |                1 | -42.29%       |
| 2026-08-31 | CEVA     | 76.08%             | 8.97%    | 78.16%   | 59.54%   | 48.89%             |                0 | -33.52%       |

## Strategy and baseline comparison
| strategy                     |   months | total_return_1m_rebalanced   | annualized_return_1m_rebalanced   | avg_monthly_return_1m   | monthly_win_rate_1m   | max_drawdown_1m_rebalanced   | avg_future_max_return_1_3m   | avg_boom_hit_rate   |
|:-----------------------------|---------:|:-----------------------------|:----------------------------------|:------------------------|:----------------------|:-----------------------------|:-----------------------------|:--------------------|
| baseline_core_mom_456_avg    |       17 | 315.79%                      | 173.43%                           | 10.60%                  | 70.59%                | -20.26%                      | 43.17%                       | 35.29%              |
| baseline_mom_5m              |       17 | 261.77%                      | 147.85%                           | 9.83%                   | 70.59%                | -20.26%                      | 37.27%                       | 33.33%              |
| baseline_mom_4m              |       17 | 259.94%                      | 146.97%                           | 9.54%                   | 64.71%                | -16.64%                      | 36.84%                       | 33.33%              |
| baseline_mom_6m              |       17 | 234.35%                      | 134.44%                           | 9.07%                   | 70.59%                | -20.53%                      | 39.99%                       | 31.37%              |
| baseline_mom_6m_acceleration |       17 | 230.32%                      | 132.44%                           | 8.01%                   | 76.47%                | -14.57%                      | 37.39%                       | 29.41%              |
| baseline_mom_3m              |       17 | 131.25%                      | 80.71%                            | 6.51%                   | 70.59%                | -29.65%                      | 39.00%                       | 31.37%              |
| xgb_boom_probability         |       17 | 69.14%                       | 44.92%                            | 5.16%                   | 64.71%                | -36.56%                      | 35.37%                       | 25.49%              |

## Recent XGB Top-3 backtest months
| month      | selected_tickers   | avg_score   | return_1m   | future_max_return_1_3m   | boom_hit_rate   |
|:-----------|:-------------------|:------------|:------------|:-------------------------|:----------------|
| 2025-06-30 | OKLO, SMCI, SMR    | 83.38%      | 28.01%      | 48.87%                   | 33.33%          |
| 2025-07-31 | SMCI, OKLO, GEV    | 77.01%      | -13.53%     | 18.20%                   | 33.33%          |
| 2025-08-31 | SMR, PLTR, SOUN    | 86.68%      | 14.60%      | 30.92%                   | 0.00%           |
| 2025-09-30 | SOUN, OKLO, FLNC   | 82.83%      | 40.99%      | 40.99%                   | 33.33%          |
| 2025-10-31 | ORCL, LRCX, SMR    | 83.44%      | -26.48%     | -10.01%                  | 33.33%          |
| 2025-11-30 | FSLY, SMR, FLNC    | 81.18%      | -13.71%     | 36.02%                   | 66.67%          |
| 2025-12-31 | FLNC, OKLO, SMR    | 82.86%      | 29.96%      | 29.96%                   | 33.33%          |
| 2026-01-31 | FLNC, SNOW, MU     | 77.16%      | -20.90%     | -12.47%                  | 0.00%           |
| 2026-02-28 | FSLY, FLNC, INTC   | 88.16%      | 12.43%      | 74.97%                   | 33.33%          |
| 2026-03-31 | FLNC, MU, FSLY     | 82.39%      | 9.50%       | 91.02%                   | 33.33%          |
| 2026-04-30 | FSLY, ARM, VECO    | 72.14%      | 17.98%      | 36.79%                   | 0.00%           |
| 2026-05-31 | FSLY, GEV, VRT     | 71.53%      | 10.27%      | 31.98%                   | 33.33%          |

## Top feature importance
| feature             |   importance | is_core_momentum   |
|:--------------------|-------------:|:-------------------|
| volatility_6m       |    0.0411326 | False              |
| volatility_3m       |    0.0279447 | False              |
| mom_7m              |    0.0264821 | False              |
| core_mom_456_max    |    0.0260282 | True               |
| mom_12m             |    0.0253853 | False              |
| drawdown_12m        |    0.025278  | False              |
| drawdown_3m         |    0.025124  | False              |
| qqq_mom_12m         |    0.024601  | False              |
| mom_6m_last3m       |    0.0232107 | False              |
| drawdown_6m         |    0.0230754 | False              |
| drawdown_12m_abs    |    0.0228884 | False              |
| core_mom_456_avg    |    0.0226486 | True               |
| mom_2m              |    0.0224124 | False              |
| ma30_slope_1m       |    0.0223066 | False              |
| mom_3m              |    0.0220872 | False              |
| qqq_mom_3m          |    0.0219267 | False              |
| core_mom_456_min    |    0.0212964 | True               |
| core_mom_456_std    |    0.0212661 | True               |
| mom_6m              |    0.0211787 | True               |
| drawdown_3m_abs     |    0.0206375 | False              |
| price_ma30_ratio    |    0.0205162 | False              |
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