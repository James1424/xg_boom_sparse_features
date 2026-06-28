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
| rows        | 5723       |
| columns     | 63         |
| tickers     | 54         |
| months      | 126        |
| first_month | 2016-01-31 |
| last_month  | 2026-06-30 |

## Train / validation / test metrics
| dataset   |   rows |   months |   positive_rate |    auc |   pr_auc |   accuracy_at_0_5 |   precision_at_0_5 |   recall_at_0_5 |   precision_at_top3_monthly |   precision_at_top5_monthly |   precision_at_top10_monthly |   top3_future_return_1m |   top3_future_return_2m |   top3_future_return_3m |   top3_future_max_return_1_3m |   top_decile_future_max_return |   bottom_decile_future_max_return |   top_minus_bottom_spread |
|:----------|-------:|---------:|----------------:|-------:|---------:|------------------:|-------------------:|----------------:|----------------------------:|----------------------------:|-----------------------------:|------------------------:|------------------------:|------------------------:|------------------------------:|-------------------------------:|----------------------------------:|--------------------------:|
| train     |   3253 |       84 |          0.1039 | 0.9844 |   0.8821 |            0.9176 |             0.5608 |          0.9556 |                      0.881  |                      0.7    |                       0.394  |                  0.1307 |                  0.2483 |                  0.3512 |                        0.4078 |                         0.3912 |                            0.0706 |                    0.3206 |
| valid     |   1247 |       24 |          0.1051 | 0.6913 |   0.2156 |            0.6961 |             0.1737 |          0.5038 |                      0.2778 |                      0.25   |                       0.2208 |                  0.0737 |                  0.105  |                  0.1707 |                        0.333  |                         0.2894 |                            0.0972 |                    0.1922 |
| test      |    808 |       15 |          0.1101 | 0.6962 |   0.2086 |            0.6448 |             0.1867 |          0.6629 |                      0.2667 |                      0.2667 |                       0.2533 |                  0.0475 |                  0.1288 |                  0.1834 |                        0.3604 |                         0.3629 |                            0.0966 |                    0.2663 |

## Latest live boom candidates
| month      | ticker   | boom_probability   | mom_4m   | mom_5m   | mom_6m   | core_mom_456_avg   |   ma_trend_score | drawdown_3m   |
|:-----------|:---------|:-------------------|:---------|:---------|:---------|:-------------------|-----------------:|:--------------|
| 2026-06-30 | FSLY     | 84.43%             | -10.51%  | 84.97%   | 68.07%   | 47.51%             |                1 | -48.93%       |
| 2026-06-30 | MU       | 80.88%             | 174.71%  | 173.04%  | 296.90%  | 214.88%            |                3 | -6.69%        |
| 2026-06-30 | AVGO     | 78.07%             | 14.65%   | 10.58%   | 5.86%    | 10.36%             |                0 | -24.08%       |
| 2026-06-30 | VRT      | 77.78%             | 19.30%   | 63.33%   | 87.70%   | 56.78%             |                2 | -19.19%       |
| 2026-06-30 | FLNC     | 76.78%             | 24.00%   | -37.37%  | -2.58%   | -5.32%             |                1 | -30.96%       |
| 2026-06-30 | INTC     | 75.73%             | 181.34%  | 176.14%  | 247.75%  | 201.74%            |                3 | -8.95%        |
| 2026-06-30 | ARM      | 73.87%             | 162.28%  | 217.26%  | 205.80%  | 195.11%            |                2 | -23.94%       |
| 2026-06-30 | AOSL     | 73.01%             | 109.38%  | 99.05%   | 122.06%  | 110.16%            |                2 | -17.50%       |
| 2026-06-30 | AMAT     | 70.95%             | 68.58%   | 94.96%   | 144.52%  | 102.69%            |                3 | -6.16%        |
| 2026-06-30 | QCOM     | 70.36%             | 34.38%   | 26.20%   | 11.84%   | 24.14%             |                1 | -24.27%       |
| 2026-06-30 | GEV      | 68.68%             | 19.77%   | 44.05%   | 60.21%   | 41.35%             |                2 | -9.03%        |
| 2026-06-30 | KLAC     | 68.41%             | 63.30%   | 74.57%   | 105.16%  | 81.01%             |                3 | -7.62%        |
| 2026-06-30 | MRVL     | 68.24%             | 226.73%  | 238.20%  | 214.30%  | 226.41%            |                2 | -15.69%       |
| 2026-06-30 | SMCI     | 67.02%             | -5.43%   | 5.22%    | 4.65%    | 1.48%              |                2 | -38.95%       |
| 2026-06-30 | AMD      | 64.92%             | 160.52%  | 120.33%  | 143.55%  | 141.46%            |                3 | -5.45%        |

## Strategy and baseline comparison
| strategy                     |   months | total_return_1m_rebalanced   | annualized_return_1m_rebalanced   | avg_monthly_return_1m   | monthly_win_rate_1m   | max_drawdown_1m_rebalanced   | avg_future_max_return_1_3m   | avg_boom_hit_rate   |
|:-----------------------------|---------:|:-----------------------------|:----------------------------------|:------------------------|:----------------------|:-----------------------------|:-----------------------------|:--------------------|
| baseline_mom_4m              |       15 | 214.17%                      | 149.88%                           | 9.82%                   | 66.67%                | -16.64%                      | 38.95%                       | 35.56%              |
| baseline_core_mom_456_avg    |       15 | 206.49%                      | 144.98%                           | 9.77%                   | 66.67%                | -20.26%                      | 43.79%                       | 37.78%              |
| baseline_mom_5m              |       15 | 166.67%                      | 119.17%                           | 8.90%                   | 66.67%                | -20.26%                      | 37.48%                       | 35.56%              |
| baseline_mom_6m_acceleration |       15 | 160.04%                      | 114.80%                           | 7.34%                   | 73.33%                | -14.57%                      | 38.56%                       | 35.56%              |
| baseline_mom_6m              |       15 | 131.09%                      | 95.45%                            | 7.55%                   | 66.67%                | -20.53%                      | 39.70%                       | 33.33%              |
| baseline_mom_3m              |       15 | 108.90%                      | 80.28%                            | 6.68%                   | 66.67%                | -29.65%                      | 40.85%                       | 37.78%              |
| xgb_boom_probability         |       15 | 50.79%                       | 38.90%                            | 4.75%                   | 60.00%                | -26.42%                      | 36.04%                       | 26.67%              |

## Recent XGB Top-3 backtest months
| month      | selected_tickers   | avg_score   | return_1m   | future_max_return_1_3m   | boom_hit_rate   |
|:-----------|:-------------------|:------------|:------------|:-------------------------|:----------------|
| 2025-04-30 | SOUN, PLTR, CRWD   | 85.88%      | 10.00%      | 22.65%                   | 0.00%           |
| 2025-05-31 | NET, OKLO, PLTR    | 82.03%      | 9.23%       | 30.42%                   | 0.00%           |
| 2025-06-30 | OKLO, SMCI, SMR    | 83.77%      | 28.01%      | 48.87%                   | 33.33%          |
| 2025-07-31 | SMCI, OKLO, GEV    | 77.11%      | -13.53%     | 18.20%                   | 33.33%          |
| 2025-08-31 | SMR, PLTR, SOUN    | 87.18%      | 14.60%      | 30.92%                   | 0.00%           |
| 2025-09-30 | SOUN, OKLO, FLNC   | 82.83%      | 40.99%      | 40.99%                   | 33.33%          |
| 2025-10-31 | LRCX, ARM, ORCL    | 83.72%      | -14.73%     | 1.75%                    | 33.33%          |
| 2025-11-30 | FSLY, SMR, FLNC    | 80.51%      | -13.71%     | 36.02%                   | 66.67%          |
| 2025-12-31 | FLNC, OKLO, SMR    | 82.18%      | 29.96%      | 29.96%                   | 33.33%          |
| 2026-01-31 | FLNC, MU, SNOW     | 78.23%      | -20.90%     | -12.47%                  | 0.00%           |
| 2026-02-28 | FSLY, FLNC, INTC   | 88.09%      | 12.43%      | 74.97%                   | 33.33%          |
| 2026-03-31 | FLNC, MU, FSLY     | 81.87%      | 9.50%       | 87.37%                   | 33.33%          |

## Top feature importance
| feature             |   importance | is_core_momentum   |
|:--------------------|-------------:|:-------------------|
| volatility_6m       |    0.0410881 | False              |
| volatility_3m       |    0.0274162 | False              |
| core_mom_456_max    |    0.0263243 | True               |
| mom_6m_last3m       |    0.0257889 | False              |
| mom_7m              |    0.0254936 | False              |
| mom_12m             |    0.025153  | False              |
| drawdown_12m        |    0.0251122 | False              |
| drawdown_3m         |    0.0248688 | False              |
| qqq_mom_12m         |    0.0241699 | False              |
| drawdown_6m         |    0.0237088 | False              |
| core_mom_456_avg    |    0.0230209 | True               |
| ma30_slope_1m       |    0.0224457 | False              |
| mom_2m              |    0.0224261 | False              |
| drawdown_12m_abs    |    0.0222413 | False              |
| mom_3m              |    0.0217028 | False              |
| qqq_mom_3m          |    0.02148   | False              |
| core_mom_456_std    |    0.0210686 | True               |
| mom_6m              |    0.0209686 | True               |
| core_mom_456_min    |    0.0206956 | True               |
| price_ma20_ratio    |    0.0205719 | False              |
| mom_6m_acceleration |    0.0205187 | True               |
| drawdown_3m_abs     |    0.0204565 | False              |
| price_ma30_ratio    |    0.0204482 | False              |
| mom_4m              |    0.0200797 | True               |
| mom_9m              |    0.0200117 | False              |

## Feature design note
See `outputs/feature_manifest.txt` for the exact clean feature list.

## Correctness notes
- The model does not try to predict exact return size; it predicts probability of joining the future monthly top-10% boom group.
- The split is time-based: train <= 2022, validation 2023–2024, test >= 2025.
- The latest live candidates table does not use future returns; it only uses the newest available panel features.
- This is a research backtest, not investment advice.