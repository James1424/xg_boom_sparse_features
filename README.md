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
| train     |   3169 |       84 |          0.1051 | 0.9835 |   0.8779 |            0.9157 |             0.5575 |          0.961  |                      0.8571 |                      0.681  |                       0.3917 |                  0.1287 |                  0.2486 |                  0.3486 |                        0.4019 |                         0.3878 |                            0.0764 |                    0.3114 |
| valid     |   1223 |       24 |          0.1071 | 0.6608 |   0.1959 |            0.7089 |             0.172  |          0.4504 |                      0.2361 |                      0.2167 |                       0.2083 |                  0.0579 |                  0.0626 |                  0.1172 |                        0.272  |                         0.276  |                            0.1014 |                    0.1746 |
| test      |   1005 |       19 |          0.1124 | 0.6496 |   0.176  |            0.6308 |             0.1623 |          0.5487 |                      0.1579 |                      0.2105 |                       0.2211 |                  0.0587 |                  0.0714 |                  0.1143 |                        0.2818 |                         0.3178 |                            0.107  |                    0.2108 |

## Latest live boom candidates
| month      | ticker   | boom_probability   | mom_4m   | mom_5m   | mom_6m   | core_mom_456_avg   |   ma_trend_score | drawdown_3m   |
|:-----------|:---------|:-------------------|:---------|:---------|:---------|:-------------------|-----------------:|:--------------|
| 2026-10-31 | MU       | 90.79%             | -10.84%  | 5.99%    | 99.00%   | 31.38%             |                2 | -6.23%        |
| 2026-10-31 | NOW      | 89.39%             | 41.88%   | 13.26%   | 59.51%   | 38.22%             |                1 | -4.82%        |
| 2026-10-31 | MRVL     | 86.17%             | -7.55%   | 34.35%   | 66.76%   | 31.19%             |                3 | -4.07%        |
| 2026-10-31 | SMCI     | 85.28%             | 42.72%   | -9.18%   | 52.77%   | 28.77%             |                3 | -6.85%        |
| 2026-10-31 | DELL     | 85.11%             | 36.06%   | 39.47%   | 180.94%  | 85.49%             |                3 | -0.40%        |
| 2026-10-31 | AMD      | 81.16%             | 4.68%    | 17.83%   | 71.54%   | 31.35%             |                3 | -6.36%        |
| 2026-10-31 | CEG      | 79.90%             | 20.19%   | 3.75%    | -4.48%   | 6.49%              |                2 | -0.78%        |
| 2026-10-31 | CRWD     | 73.28%             | 44.16%   | 50.50%   | 146.81%  | 80.49%             |                3 | -1.37%        |
| 2026-10-31 | DDOG     | 72.68%             | 12.64%   | 18.56%   | 121.85%  | 51.01%             |                3 | 0.00%         |
| 2026-10-31 | INTC     | 72.13%             | -25.02%  | -8.70%   | 10.82%   | -7.63%             |                2 | -17.81%       |
| 2026-10-31 | BILL     | 69.55%             | 23.98%   | 21.10%   | 17.97%   | 21.02%             |                1 | -12.63%       |
| 2026-10-31 | FSLY     | 69.12%             | 59.59%   | 64.93%   | 16.02%   | 46.84%             |                3 | -2.40%        |
| 2026-10-31 | VRT      | 67.47%             | -27.47%  | -23.07%  | -26.06%  | -25.53%            |                2 | -20.27%       |
| 2026-10-31 | SNOW     | 64.90%             | 44.95%   | 44.35%   | 170.31%  | 86.54%             |                3 | 0.00%         |
| 2026-10-31 | MSFT     | 64.24%             | 43.71%   | 19.06%   | 31.75%   | 31.51%             |                3 | 0.00%         |

## Strategy and baseline comparison
| strategy                     |   months | total_return_1m_rebalanced   | annualized_return_1m_rebalanced   | avg_monthly_return_1m   | monthly_win_rate_1m   | max_drawdown_1m_rebalanced   | avg_future_max_return_1_3m   | avg_boom_hit_rate   |
|:-----------------------------|---------:|:-----------------------------|:----------------------------------|:------------------------|:----------------------|:-----------------------------|:-----------------------------|:--------------------|
| baseline_core_mom_456_avg    |       19 | 207.06%                      | 103.10%                           | 8.11%                   | 63.16%                | -26.15%                      | 39.93%                       | 31.58%              |
| baseline_mom_4m              |       19 | 190.66%                      | 96.19%                            | 7.65%                   | 63.16%                | -26.10%                      | 34.61%                       | 31.58%              |
| baseline_mom_5m              |       19 | 170.78%                      | 87.60%                            | 7.47%                   | 63.16%                | -25.15%                      | 34.56%                       | 29.82%              |
| baseline_mom_6m_acceleration |       19 | 153.19%                      | 79.81%                            | 6.01%                   | 73.68%                | -27.37%                      | 34.10%                       | 26.32%              |
| baseline_mom_6m              |       19 | 140.81%                      | 74.20%                            | 6.80%                   | 68.42%                | -33.70%                      | 37.17%                       | 29.82%              |
| xgb_boom_probability         |       19 | 108.62%                      | 59.11%                            | 5.87%                   | 63.16%                | -33.73%                      | 28.18%                       | 15.79%              |
| baseline_mom_3m              |       19 | 60.44%                       | 34.80%                            | 4.29%                   | 68.42%                | -33.70%                      | 35.58%                       | 28.07%              |

## Recent XGB Top-3 backtest months
| month      | selected_tickers   | avg_score   | return_1m   | future_max_return_1_3m   | boom_hit_rate   |
|:-----------|:-------------------|:------------|:------------|:-------------------------|:----------------|
| 2025-08-31 | SMR, SOUN, PLTR    | 88.55%      | 14.60%      | 30.92%                   | 0.00%           |
| 2025-09-30 | OKLO, SOUN, SMR    | 82.41%      | 17.72%      | 17.72%                   | 0.00%           |
| 2025-10-31 | ARM, SMR, LRCX     | 84.03%      | -25.51%     | -9.03%                   | 33.33%          |
| 2025-11-30 | FSLY, SMR, BILL    | 77.47%      | -11.03%     | 20.04%                   | 33.33%          |
| 2025-12-31 | OKLO, FLNC, SMR    | 80.63%      | 29.96%      | 29.96%                   | 33.33%          |
| 2026-01-31 | NOW, FLNC, ASML    | 79.56%      | -18.37%     | -18.37%                  | 0.00%           |
| 2026-02-28 | FSLY, FLNC, ZS     | 86.66%      | 11.99%      | 22.97%                   | 0.00%           |
| 2026-03-31 | FLNC, MU, FSLY     | 78.52%      | 9.50%       | 91.02%                   | 33.33%          |
| 2026-04-30 | FSLY, NOW, FLNC    | 71.27%      | 22.06%      | 31.26%                   | 0.00%           |
| 2026-05-31 | FSLY, GEV, KLAC    | 72.71%      | 27.25%      | 35.56%                   | 33.33%          |
| 2026-06-30 | FSLY, MU, SMCI     | 85.08%      | -2.81%      | 24.04%                   | 0.00%           |
| 2026-07-31 | MRVL, ARM, INTC    | 94.40%      | 4.34%       | 33.65%                   | 33.33%          |

## Top feature importance
| feature                     |   importance | is_core_momentum   |
|:----------------------------|-------------:|:-------------------|
| volatility_6m               |    0.0363021 | False              |
| mom_12m                     |    0.0266928 | False              |
| volatility_3m               |    0.0258385 | False              |
| drawdown_12m                |    0.0252865 | False              |
| qqq_mom_12m                 |    0.0245545 | False              |
| core_mom_456_max            |    0.0235718 | True               |
| drawdown_3m_abs             |    0.0232497 | False              |
| drawdown_3m                 |    0.0232229 | False              |
| drawdown_6m_abs             |    0.0226152 | False              |
| core_mom_456_avg            |    0.0218261 | True               |
| ma30_slope_1m               |    0.0218083 | False              |
| drawdown_12m_abs            |    0.0216784 | False              |
| core_mom_456_std            |    0.0213392 | True               |
| mom_6m_last3m               |    0.0211233 | False              |
| mom_7m                      |    0.0209366 | False              |
| core_mom_456_positive_count |    0.0208303 | True               |
| qqq_mom_3m                  |    0.0207877 | False              |
| mom_4m                      |    0.0205702 | True               |
| price_ma20_ratio            |    0.0201439 | False              |
| mom_6m_acceleration         |    0.0197003 | True               |
| ma20_above_ma30             |    0.0196308 | False              |
| mom_2m                      |    0.0194471 | False              |
| mom_3m                      |    0.0193374 | False              |
| mom_9m                      |    0.0192884 | False              |
| mom_6m_first3m              |    0.018941  | False              |

## Feature design note
See `outputs/feature_manifest.txt` for the exact clean feature list.

## Correctness notes
- The model does not try to predict exact return size; it predicts probability of joining the future monthly top-10% boom group.
- The split is time-based: train <= 2022, validation 2023–2024, test >= 2025.
- The latest live candidates table does not use future returns; it only uses the newest available panel features.
- This is a research backtest, not investment advice.