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
| rows        | 5703       |
| columns     | 63         |
| tickers     | 53         |
| months      | 128        |
| first_month | 2016-01-31 |
| last_month  | 2026-08-31 |

## Train / validation / test metrics
| dataset   |   rows |   months |   positive_rate |    auc |   pr_auc |   accuracy_at_0_5 |   precision_at_0_5 |   recall_at_0_5 |   precision_at_top3_monthly |   precision_at_top5_monthly |   precision_at_top10_monthly |   top3_future_return_1m |   top3_future_return_2m |   top3_future_return_3m |   top3_future_max_return_1_3m |   top_decile_future_max_return |   bottom_decile_future_max_return |   top_minus_bottom_spread |
|:----------|-------:|---------:|----------------:|-------:|---------:|------------------:|-------------------:|----------------:|----------------------------:|----------------------------:|-----------------------------:|------------------------:|------------------------:|------------------------:|------------------------------:|-------------------------------:|----------------------------------:|--------------------------:|
| train     |   3169 |       84 |          0.1051 | 0.985  |   0.8871 |            0.9208 |             0.5735 |          0.961  |                      0.877  |                      0.7    |                       0.3929 |                  0.1302 |                  0.2473 |                  0.3476 |                        0.4037 |                         0.3905 |                            0.0752 |                    0.3152 |
| valid     |   1223 |       24 |          0.1071 | 0.6818 |   0.2    |            0.6893 |             0.1715 |          0.4962 |                      0.2361 |                      0.25   |                       0.2208 |                  0.0838 |                  0.0897 |                  0.1453 |                        0.2952 |                         0.2807 |                            0.0899 |                    0.1908 |
| test      |    899 |       17 |          0.1123 | 0.6743 |   0.197  |            0.6485 |             0.1772 |          0.5842 |                      0.1961 |                      0.2235 |                       0.2176 |                  0.0605 |                  0.1185 |                  0.1588 |                        0.3214 |                         0.3487 |                            0.1047 |                    0.244  |

## Latest live boom candidates
| month      | ticker   | boom_probability   | mom_4m   | mom_5m   | mom_6m   | core_mom_456_avg   |   ma_trend_score | drawdown_3m   |
|:-----------|:---------|:-------------------|:---------|:---------|:---------|:-------------------|-----------------:|:--------------|
| 2026-08-31 | DELL     | 95.25%             | 118.71%  | 179.30%  | 209.57%  | 169.19%            |                2 | -7.74%        |
| 2026-08-31 | HPE      | 86.86%             | 82.35%   | 120.34%  | 145.93%  | 116.21%            |                1 | -12.55%       |
| 2026-08-31 | INTC     | 84.20%             | -5.30%   | 102.74%  | 96.16%   | 64.53%             |                1 | -36.52%       |
| 2026-08-31 | MRVL     | 84.17%             | 31.20%   | 118.86%  | 165.37%  | 105.14%            |                2 | -31.53%       |
| 2026-08-31 | ARM      | 82.01%             | 13.66%   | 58.02%   | 87.56%   | 53.08%             |                1 | -45.60%       |
| 2026-08-31 | PANW     | 81.92%             | 107.22%  | 131.78%  | 149.52%  | 129.51%            |                1 | -6.16%        |
| 2026-08-31 | SNOW     | 80.44%             | 140.35%  | 117.48%  | 94.76%   | 117.53%            |                1 | -2.78%        |
| 2026-08-31 | CRWD     | 79.54%             | 95.98%   | 123.76%  | 134.85%  | 118.20%            |                2 | -4.19%        |
| 2026-08-31 | AOSL     | 78.66%             | -43.29%  | 11.15%   | 17.23%   | -4.97%             |                0 | -53.81%       |
| 2026-08-31 | MU       | 77.58%             | 80.41%   | 176.17%  | 126.35%  | 127.64%            |                2 | -23.12%       |
| 2026-08-31 | QCOM     | 76.90%             | -8.23%   | 27.97%   | 16.50%   | 12.08%             |                1 | -34.08%       |
| 2026-08-31 | ANET     | 75.15%             | 13.13%   | 59.13%   | 46.35%   | 39.54%             |                2 | -7.18%        |
| 2026-08-31 | MDB      | 74.23%             | 78.06%   | 82.47%   | 35.97%   | 65.50%             |                2 | -5.44%        |
| 2026-08-31 | SMCI     | 73.93%             | 35.33%   | 62.85%   | 14.48%   | 37.55%             |                3 | -26.09%       |
| 2026-08-31 | PATH     | 73.37%             | 76.21%   | 63.51%   | 69.15%   | 69.63%             |                3 | -0.98%        |

## Strategy and baseline comparison
| strategy                     |   months | total_return_1m_rebalanced   | annualized_return_1m_rebalanced   | avg_monthly_return_1m   | monthly_win_rate_1m   | max_drawdown_1m_rebalanced   | avg_future_max_return_1_3m   | avg_boom_hit_rate   |
|:-----------------------------|---------:|:-----------------------------|:----------------------------------|:------------------------|:----------------------|:-----------------------------|:-----------------------------|:--------------------|
| baseline_core_mom_456_avg    |       17 | 325.52%                      | 177.94%                           | 10.77%                  | 70.59%                | -20.26%                      | 41.80%                       | 37.25%              |
| baseline_mom_4m              |       17 | 268.98%                      | 151.32%                           | 9.70%                   | 64.71%                | -16.64%                      | 36.74%                       | 35.29%              |
| baseline_mom_5m              |       17 | 261.38%                      | 147.66%                           | 9.83%                   | 70.59%                | -20.26%                      | 37.40%                       | 33.33%              |
| baseline_mom_6m              |       17 | 249.09%                      | 141.69%                           | 9.38%                   | 70.59%                | -20.53%                      | 38.29%                       | 31.37%              |
| baseline_mom_6m_acceleration |       17 | 201.71%                      | 118.04%                           | 7.44%                   | 76.47%                | -14.57%                      | 37.45%                       | 31.37%              |
| baseline_mom_3m              |       17 | 131.25%                      | 80.71%                            | 6.51%                   | 70.59%                | -29.65%                      | 38.84%                       | 33.33%              |
| xgb_boom_probability         |       17 | 90.26%                       | 57.47%                            | 6.05%                   | 70.59%                | -31.20%                      | 32.14%                       | 19.61%              |

## Recent XGB Top-3 backtest months
| month      | selected_tickers   | avg_score   | return_1m   | future_max_return_1_3m   | boom_hit_rate   |
|:-----------|:-------------------|:------------|:------------|:-------------------------|:----------------|
| 2025-06-30 | OKLO, SMCI, NET    | 79.00%      | 21.06%      | 43.09%                   | 33.33%          |
| 2025-07-31 | SMCI, GEV, CRWD    | 74.62%      | -14.50%     | 0.23%                    | 0.00%           |
| 2025-08-31 | SMR, OKLO, PLTR    | 87.70%      | 23.96%      | 45.91%                   | 33.33%          |
| 2025-09-30 | SOUN, OKLO, INTC   | 79.93%      | 15.90%      | 16.47%                   | 0.00%           |
| 2025-10-31 | SMR, ORCL, AMD     | 81.96%      | -31.20%     | -28.70%                  | 0.00%           |
| 2025-11-30 | FLNC, FSLY, MU     | 81.46%      | 2.92%       | 65.39%                   | 100.00%         |
| 2025-12-31 | SMR, OKLO, FLNC    | 83.93%      | 29.96%      | 29.96%                   | 33.33%          |
| 2026-01-31 | FLNC, SNOW, MDB    | 81.76%      | -24.55%     | -24.55%                  | 0.00%           |
| 2026-02-28 | FSLY, FLNC, INTC   | 89.04%      | 12.43%      | 74.97%                   | 33.33%          |
| 2026-03-31 | FLNC, MU, FSLY     | 79.49%      | 9.50%       | 91.02%                   | 33.33%          |
| 2026-04-30 | FSLY, NOW, FLNC    | 74.03%      | 22.06%      | 31.26%                   | 0.00%           |
| 2026-05-31 | FSLY, MU, DELL     | 74.98%      | 8.24%       | 19.05%                   | 0.00%           |

## Top feature importance
| feature                     |   importance | is_core_momentum   |
|:----------------------------|-------------:|:-------------------|
| volatility_6m               |    0.0387694 | False              |
| volatility_3m               |    0.0288471 | False              |
| mom_12m                     |    0.0283497 | False              |
| qqq_mom_12m                 |    0.0267172 | False              |
| drawdown_3m                 |    0.024239  | False              |
| drawdown_3m_abs             |    0.0234216 | False              |
| core_mom_456_avg            |    0.0227648 | True               |
| drawdown_12m_abs            |    0.0224401 | False              |
| qqq_mom_3m                  |    0.0220642 | False              |
| core_mom_456_max            |    0.0216163 | True               |
| ma_trend_score              |    0.0215957 | False              |
| mom_3m                      |    0.0213413 | False              |
| drawdown_6m                 |    0.0210586 | False              |
| drawdown_12m                |    0.0209689 | False              |
| mom_6m_first3m              |    0.0206939 | False              |
| drawdown_6m_abs             |    0.020658  | False              |
| mom_4m                      |    0.0206237 | True               |
| core_mom_456_std            |    0.0202009 | True               |
| mom_7m                      |    0.0200688 | False              |
| core_mom_456_min            |    0.019777  | True               |
| price_ma30_ratio            |    0.019765  | False              |
| ma30_slope_1m               |    0.0196865 | False              |
| core_mom_456_positive_count |    0.0196351 | True               |
| mom_6m                      |    0.0195897 | True               |
| qqq_mom_6m                  |    0.0195431 | False              |

## Feature design note
See `outputs/feature_manifest.txt` for the exact clean feature list.

## Correctness notes
- The model does not try to predict exact return size; it predicts probability of joining the future monthly top-10% boom group.
- The split is time-based: train <= 2022, validation 2023–2024, test >= 2025.
- The latest live candidates table does not use future returns; it only uses the newest available panel features.
- This is a research backtest, not investment advice.