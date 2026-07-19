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
| rows        | 5777       |
| columns     | 63         |
| tickers     | 54         |
| months      | 127        |
| first_month | 2016-01-31 |
| last_month  | 2026-07-31 |

## Train / validation / test metrics
| dataset   |   rows |   months |   positive_rate |    auc |   pr_auc |   accuracy_at_0_5 |   precision_at_0_5 |   recall_at_0_5 |   precision_at_top3_monthly |   precision_at_top5_monthly |   precision_at_top10_monthly |   top3_future_return_1m |   top3_future_return_2m |   top3_future_return_3m |   top3_future_max_return_1_3m |   top_decile_future_max_return |   bottom_decile_future_max_return |   top_minus_bottom_spread |
|:----------|-------:|---------:|----------------:|-------:|---------:|------------------:|-------------------:|----------------:|----------------------------:|----------------------------:|-----------------------------:|------------------------:|------------------------:|------------------------:|------------------------------:|-------------------------------:|----------------------------------:|--------------------------:|
| train     |   3253 |       84 |          0.1039 | 0.9841 |   0.8836 |            0.9158 |             0.555  |          0.9556 |                      0.881  |                      0.7024 |                       0.3952 |                  0.1311 |                  0.251  |                  0.3547 |                        0.411  |                         0.3934 |                            0.074  |                    0.3194 |
| valid     |   1247 |       24 |          0.1051 | 0.6888 |   0.2198 |            0.6961 |             0.1788 |          0.5267 |                      0.2778 |                      0.2667 |                       0.2167 |                  0.0608 |                  0.1151 |                  0.1858 |                        0.3407 |                         0.3063 |                            0.1004 |                    0.2059 |
| test      |    862 |       16 |          0.1102 | 0.6797 |   0.2003 |            0.6311 |             0.1768 |          0.6421 |                      0.25   |                      0.225  |                       0.2437 |                  0.0484 |                  0.1333 |                  0.1732 |                        0.3542 |                         0.3416 |                            0.1017 |                    0.24   |

## Latest live boom candidates
| month      | ticker   | boom_probability   | mom_4m   | mom_5m   | mom_6m   | core_mom_456_avg   |   ma_trend_score | drawdown_3m   |
|:-----------|:---------|:-------------------|:---------|:---------|:---------|:-------------------|-----------------:|:--------------|
| 2026-07-31 | MRVL     | 95.84%             | 90.63%   | 131.14%  | 139.26%  | 120.34%            |                0 | -40.36%       |
| 2026-07-31 | ARM      | 94.68%             | 76.62%   | 109.64%  | 153.60%  | 113.29%            |                0 | -39.20%       |
| 2026-07-31 | INTC     | 91.63%             | 115.36%  | 108.38%  | 104.52%  | 109.42%            |                1 | -32.57%       |
| 2026-07-31 | MU       | 91.23%             | 151.33%  | 105.99%  | 104.74%  | 120.69%            |                1 | -30.03%       |
| 2026-07-31 | FSLY     | 90.13%             | -28.70%  | 8.37%    | 124.00%  | 34.56%             |                3 | -35.97%       |
| 2026-07-31 | AOSL     | 89.09%             | 41.11%   | 48.83%   | 41.49%   | 43.81%             |                0 | -41.35%       |
| 2026-07-31 | VECO     | 88.77%             | 54.37%   | 71.04%   | 67.37%   | 64.26%             |                1 | -37.05%       |
| 2026-07-31 | DELL     | 86.64%             | 142.23%  | 168.48%  | 247.40%  | 186.04%            |                2 | -14.94%       |
| 2026-07-31 | ORCL     | 85.88%             | -13.47%  | -12.45%  | -22.66%  | -16.19%            |                0 | -48.88%       |
| 2026-07-31 | AMAT     | 84.56%             | 55.16%   | 42.44%   | 64.74%   | 54.11%             |                1 | -26.74%       |
| 2026-07-31 | KLAC     | 82.93%             | 44.68%   | 39.73%   | 49.37%   | 44.59%             |                1 | -29.49%       |
| 2026-07-31 | VRT      | 81.20%             | 15.58%   | 13.65%   | 55.60%   | 28.28%             |                1 | -23.02%       |
| 2026-07-31 | AMD      | 78.72%             | 143.70%  | 147.62%  | 109.42%  | 133.58%            |                1 | -14.66%       |
| 2026-07-31 | LRCX     | 78.19%             | 46.74%   | 34.21%   | 34.45%   | 38.47%             |                1 | -27.70%       |
| 2026-07-31 | ASML     | 76.46%             | 32.60%   | 20.74%   | 23.24%   | 25.53%             |                1 | -12.16%       |

## Strategy and baseline comparison
| strategy                     |   months | total_return_1m_rebalanced   | annualized_return_1m_rebalanced   | avg_monthly_return_1m   | monthly_win_rate_1m   | max_drawdown_1m_rebalanced   | avg_future_max_return_1_3m   | avg_boom_hit_rate   |
|:-----------------------------|---------:|:-----------------------------|:----------------------------------|:------------------------|:----------------------|:-----------------------------|:-----------------------------|:--------------------|
| baseline_core_mom_456_avg    |       16 | 287.69%                      | 176.29%                           | 10.81%                  | 68.75%                | -20.26%                      | 44.96%                       | 37.50%              |
| baseline_mom_5m              |       16 | 237.32%                      | 148.90%                           | 10.00%                  | 68.75%                | -20.26%                      | 38.68%                       | 35.42%              |
| baseline_mom_6m_acceleration |       16 | 214.17%                      | 135.98%                           | 8.19%                   | 75.00%                | -14.57%                      | 38.95%                       | 31.25%              |
| baseline_mom_4m              |       16 | 210.13%                      | 133.70%                           | 9.13%                   | 62.50%                | -16.64%                      | 37.68%                       | 33.33%              |
| baseline_mom_6m              |       16 | 192.31%                      | 123.56%                           | 8.74%                   | 68.75%                | -20.53%                      | 41.13%                       | 33.33%              |
| baseline_mom_3m              |       16 | 119.94%                      | 80.60%                            | 6.59%                   | 68.75%                | -29.65%                      | 40.66%                       | 33.33%              |
| xgb_boom_probability         |       16 | 53.38%                       | 37.83%                            | 4.84%                   | 62.50%                | -36.56%                      | 35.42%                       | 25.00%              |

## Recent XGB Top-3 backtest months
| month      | selected_tickers   | avg_score   | return_1m   | future_max_return_1_3m   | boom_hit_rate   |
|:-----------|:-------------------|:------------|:------------|:-------------------------|:----------------|
| 2025-05-31 | OKLO, NET, PLTR    | 80.91%      | 9.23%       | 30.42%                   | 0.00%           |
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
| 2026-04-30 | FSLY, ARM, VECO    | 72.14%      | 17.98%      | 34.23%                   | 0.00%           |

## Top feature importance
| feature             |   importance | is_core_momentum   |
|:--------------------|-------------:|:-------------------|
| volatility_6m       |    0.0411258 | False              |
| volatility_3m       |    0.0279443 | False              |
| mom_7m              |    0.0264818 | False              |
| core_mom_456_max    |    0.026028  | True               |
| mom_12m             |    0.025385  | False              |
| drawdown_12m        |    0.0252778 | False              |
| drawdown_3m         |    0.0251301 | False              |
| qqq_mom_12m         |    0.0246007 | False              |
| mom_6m_last3m       |    0.0232105 | False              |
| drawdown_6m         |    0.0230748 | False              |
| drawdown_12m_abs    |    0.0228883 | False              |
| core_mom_456_avg    |    0.0226484 | True               |
| mom_2m              |    0.022412  | False              |
| ma30_slope_1m       |    0.0223065 | False              |
| mom_3m              |    0.0220869 | False              |
| qqq_mom_3m          |    0.0219264 | False              |
| core_mom_456_min    |    0.0212961 | True               |
| core_mom_456_std    |    0.0212657 | True               |
| mom_6m              |    0.0211786 | True               |
| drawdown_3m_abs     |    0.0206533 | False              |
| price_ma30_ratio    |    0.020516  | False              |
| ma5_slope_1m        |    0.0204511 | False              |
| price_ma20_ratio    |    0.0204386 | False              |
| mom_6m_first3m      |    0.020149  | False              |
| mom_6m_acceleration |    0.020105  | True               |

## Feature design note
See `outputs/feature_manifest.txt` for the exact clean feature list.

## Correctness notes
- The model does not try to predict exact return size; it predicts probability of joining the future monthly top-10% boom group.
- The split is time-based: train <= 2022, validation 2023–2024, test >= 2025.
- The latest live candidates table does not use future returns; it only uses the newest available panel features.
- This is a research backtest, not investment advice.