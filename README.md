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
| rows        | 5885       |
| columns     | 63         |
| tickers     | 54         |
| months      | 129        |
| first_month | 2016-01-31 |
| last_month  | 2026-09-30 |

## Train / validation / test metrics
| dataset   |   rows |   months |   positive_rate |    auc |   pr_auc |   accuracy_at_0_5 |   precision_at_0_5 |   recall_at_0_5 |   precision_at_top3_monthly |   precision_at_top5_monthly |   precision_at_top10_monthly |   top3_future_return_1m |   top3_future_return_2m |   top3_future_return_3m |   top3_future_max_return_1_3m |   top_decile_future_max_return |   bottom_decile_future_max_return |   top_minus_bottom_spread |
|:----------|-------:|---------:|----------------:|-------:|---------:|------------------:|-------------------:|----------------:|----------------------------:|----------------------------:|-----------------------------:|------------------------:|------------------------:|------------------------:|------------------------------:|-------------------------------:|----------------------------------:|--------------------------:|
| train     |   3253 |       84 |          0.1039 | 0.9844 |   0.8855 |            0.9139 |             0.5492 |          0.9586 |                      0.8889 |                      0.7071 |                       0.3952 |                  0.1337 |                  0.2526 |                  0.3571 |                        0.4127 |                         0.3962 |                            0.0756 |                    0.3206 |
| valid     |   1247 |       24 |          0.1051 | 0.6883 |   0.2205 |            0.6961 |             0.1771 |          0.5191 |                      0.2778 |                      0.2583 |                       0.2167 |                  0.0506 |                  0.0955 |                  0.1735 |                        0.3267 |                         0.3052 |                            0.1017 |                    0.2035 |
| test      |    970 |       18 |          0.1103 | 0.6591 |   0.182  |            0.6206 |             0.1645 |          0.5981 |                      0.2222 |                      0.2    |                       0.2056 |                  0.0514 |                  0.1188 |                  0.1518 |                        0.3305 |                         0.3134 |                            0.0751 |                    0.2384 |

## Latest live boom candidates
| month      | ticker   | boom_probability   | mom_4m   | mom_5m   | mom_6m   | core_mom_456_avg   |   ma_trend_score | drawdown_3m   |
|:-----------|:---------|:-------------------|:---------|:---------|:---------|:-------------------|-----------------:|:--------------|
| 2026-09-30 | INTC     | 85.50%             | -16.46%  | 1.40%    | 117.09%  | 34.01%             |                2 | -32.03%       |
| 2026-09-30 | MU       | 83.66%             | 4.71%    | 96.60%   | 200.95%  | 100.76%            |                3 | -16.22%       |
| 2026-09-30 | MRVL     | 81.27%             | 9.08%    | 35.40%   | 125.86%  | 56.78%             |                1 | -28.00%       |
| 2026-09-30 | AMD      | 75.65%             | -7.47%   | 34.72%   | 134.76%  | 54.00%             |                0 | -17.79%       |
| 2026-09-30 | ARM      | 74.95%             | -28.65%  | 19.86%   | 66.64%   | 19.28%             |                0 | -42.64%       |
| 2026-09-30 | LRCX     | 72.58%             | -3.24%   | 19.39%   | 44.09%   | 20.08%             |                1 | -29.00%       |
| 2026-09-30 | SMCI     | 72.06%             | -14.10%  | 44.49%   | 73.87%   | 34.75%             |                3 | -10.00%       |
| 2026-09-30 | DELL     | 71.31%             | 24.73%   | 151.26%  | 220.86%  | 132.28%            |                3 | 0.00%         |
| 2026-09-30 | HPE      | 71.26%             | 21.17%   | 81.27%   | 119.03%  | 73.83%             |                1 | -13.07%       |
| 2026-09-30 | BILL     | 66.64%             | 32.79%   | 29.37%   | 28.36%   | 30.17%             |                2 | -4.19%        |
| 2026-09-30 | OKLO     | 66.50%             | -38.29%  | -43.08%  | -16.78%  | -32.72%            |                1 | -32.53%       |
| 2026-09-30 | ADBE     | 65.79%             | 2.82%    | 8.29%    | 9.64%    | 6.92%              |                3 | -8.98%        |
| 2026-09-30 | PATH     | 65.58%             | 29.61%   | 47.48%   | 36.85%   | 37.98%             |                3 | -18.64%       |
| 2026-09-30 | MDB      | 62.26%             | 9.89%    | 47.01%   | 50.65%   | 35.85%             |                1 | -21.93%       |
| 2026-09-30 | CEVA     | 60.80%             | -32.59%  | -11.76%  | 44.27%   | -0.02%             |                0 | -45.80%       |

## Strategy and baseline comparison
| strategy                     |   months | total_return_1m_rebalanced   | annualized_return_1m_rebalanced   | avg_monthly_return_1m   | monthly_win_rate_1m   | max_drawdown_1m_rebalanced   | avg_future_max_return_1_3m   | avg_boom_hit_rate   |
|:-----------------------------|---------:|:-----------------------------|:----------------------------------|:------------------------|:----------------------|:-----------------------------|:-----------------------------|:--------------------|
| baseline_core_mom_456_avg    |       18 | 207.25%                      | 111.35%                           | 8.56%                   | 66.67%                | -26.10%                      | 39.98%                       | 33.33%              |
| baseline_mom_5m              |       18 | 170.95%                      | 94.35%                            | 7.89%                   | 66.67%                | -25.10%                      | 34.45%                       | 31.48%              |
| baseline_mom_4m              |       18 | 165.98%                      | 91.97%                            | 7.56%                   | 61.11%                | -26.10%                      | 34.00%                       | 31.48%              |
| baseline_mom_6m_acceleration |       18 | 139.90%                      | 79.21%                            | 6.04%                   | 72.22%                | -27.37%                      | 34.03%                       | 27.78%              |
| baseline_mom_6m              |       18 | 121.66%                      | 70.00%                            | 6.69%                   | 66.67%                | -33.70%                      | 36.35%                       | 29.63%              |
| xgb_boom_probability         |       18 | 83.83%                       | 50.07%                            | 5.14%                   | 61.11%                | -26.42%                      | 33.05%                       | 22.22%              |
| baseline_mom_3m              |       18 | 53.31%                       | 32.96%                            | 4.27%                   | 66.67%                | -33.70%                      | 35.42%                       | 29.63%              |

## Recent XGB Top-3 backtest months
| month      | selected_tickers   | avg_score   | return_1m   | future_max_return_1_3m   | boom_hit_rate   |
|:-----------|:-------------------|:------------|:------------|:-------------------------|:----------------|
| 2025-07-31 | SMCI, OKLO, GEV    | 77.93%      | -13.53%     | 18.20%                   | 33.33%          |
| 2025-08-31 | SMR, PLTR, SOUN    | 87.25%      | 14.60%      | 30.92%                   | 0.00%           |
| 2025-09-30 | SOUN, OKLO, FLNC   | 81.59%      | 40.99%      | 40.99%                   | 33.33%          |
| 2025-10-31 | LRCX, ORCL, ARM    | 83.56%      | -14.73%     | 1.75%                    | 33.33%          |
| 2025-11-30 | FSLY, SMR, FLNC    | 80.45%      | -13.71%     | 36.02%                   | 66.67%          |
| 2025-12-31 | FLNC, OKLO, SMR    | 82.39%      | 29.96%      | 29.96%                   | 33.33%          |
| 2026-01-31 | FLNC, MU, SNOW     | 77.67%      | -20.90%     | -12.47%                  | 0.00%           |
| 2026-02-28 | FSLY, FLNC, INTC   | 88.66%      | 12.43%      | 74.97%                   | 33.33%          |
| 2026-03-31 | FLNC, MU, FSLY     | 82.04%      | 9.50%       | 91.02%                   | 33.33%          |
| 2026-04-30 | FSLY, ARM, AOSL    | 73.44%      | 14.25%      | 22.43%                   | 0.00%           |
| 2026-05-31 | FSLY, GEV, VRT     | 71.02%      | 10.27%      | 18.58%                   | 0.00%           |
| 2026-06-30 | FSLY, FLNC, SMCI   | 82.99%      | -3.23%      | 9.73%                    | 0.00%           |

## Top feature importance
| feature             |   importance | is_core_momentum   |
|:--------------------|-------------:|:-------------------|
| volatility_6m       |    0.0409636 | False              |
| volatility_3m       |    0.0276567 | False              |
| core_mom_456_max    |    0.0262328 | True               |
| mom_7m              |    0.0261293 | False              |
| mom_12m             |    0.0254207 | False              |
| drawdown_12m        |    0.0251475 | False              |
| drawdown_3m         |    0.0249839 | False              |
| qqq_mom_12m         |    0.0245578 | False              |
| mom_6m_last3m       |    0.0233444 | False              |
| drawdown_12m_abs    |    0.0232047 | False              |
| core_mom_456_avg    |    0.0229752 | True               |
| drawdown_6m         |    0.0228392 | False              |
| mom_3m              |    0.0223768 | False              |
| ma30_slope_1m       |    0.0221851 | False              |
| mom_2m              |    0.0221622 | False              |
| qqq_mom_3m          |    0.0218756 | False              |
| core_mom_456_min    |    0.0213712 | True               |
| mom_6m              |    0.0211243 | True               |
| core_mom_456_std    |    0.0211034 | True               |
| drawdown_3m_abs     |    0.0207423 | False              |
| price_ma30_ratio    |    0.0205449 | False              |
| price_ma20_ratio    |    0.0204216 | False              |
| mom_6m_acceleration |    0.0203838 | True               |
| mom_9m              |    0.0203669 | False              |
| mom_4m              |    0.0201742 | True               |

## Feature design note
See `outputs/feature_manifest.txt` for the exact clean feature list.

## Correctness notes
- The model does not try to predict exact return size; it predicts probability of joining the future monthly top-10% boom group.
- The split is time-based: train <= 2022, validation 2023–2024, test >= 2025.
- The latest live candidates table does not use future returns; it only uses the newest available panel features.
- This is a research backtest, not investment advice.