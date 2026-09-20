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
| train     |   3253 |       84 |          0.1039 | 0.9842 |   0.8851 |            0.9148 |             0.5521 |          0.9556 |                      0.8849 |                      0.7    |                       0.3964 |                  0.1314 |                  0.2516 |                  0.3556 |                        0.4117 |                         0.3954 |                            0.0723 |                    0.323  |
| valid     |   1247 |       24 |          0.1051 | 0.6884 |   0.2184 |            0.6929 |             0.1736 |          0.5115 |                      0.2639 |                      0.2667 |                       0.2167 |                  0.0561 |                  0.1024 |                  0.1753 |                        0.3289 |                         0.3113 |                            0.1026 |                    0.2087 |
| test      |    970 |       18 |          0.1103 | 0.6631 |   0.1834 |            0.6237 |             0.1709 |          0.6262 |                      0.2222 |                      0.2    |                       0.2167 |                  0.0411 |                  0.114  |                  0.147  |                        0.3207 |                         0.3095 |                            0.0873 |                    0.2222 |

## Latest live boom candidates
| month      | ticker   | boom_probability   | mom_4m   | mom_5m   | mom_6m   | core_mom_456_avg   |   ma_trend_score | drawdown_3m   |
|:-----------|:---------|:-------------------|:---------|:---------|:---------|:-------------------|-----------------:|:--------------|
| 2026-09-30 | ARM      | 90.65%             | -21.99%  | 31.04%   | 82.19%   | 30.41%             |                1 | -32.40%       |
| 2026-09-30 | MRVL     | 85.05%             | 19.18%   | 47.93%   | 146.78%  | 71.30%             |                3 | -20.64%       |
| 2026-09-30 | CEVA     | 83.08%             | -24.29%  | -0.88%   | 62.04%   | 12.29%             |                2 | -37.47%       |
| 2026-09-30 | FLNC     | 81.67%             | -61.23%  | -39.90%  | -46.80%  | -49.31%            |                0 | -70.94%       |
| 2026-09-30 | INTC     | 79.93%             | -5.30%   | 14.94%   | 146.09%  | 51.91%             |                2 | -22.95%       |
| 2026-09-30 | MU       | 79.52%             | 4.63%    | 96.45%   | 200.72%  | 100.60%            |                2 | -16.28%       |
| 2026-09-30 | CRWD     | 78.54%             | 30.04%   | 113.26%  | 143.49%  | 95.60%             |                3 | -3.28%        |
| 2026-09-30 | HPE      | 75.32%             | 41.94%   | 112.35%  | 156.58%  | 103.62%            |                2 | -1.89%        |
| 2026-09-30 | PATH     | 73.51%             | 14.25%   | 30.00%   | 20.63%   | 21.63%             |                1 | -28.28%       |
| 2026-09-30 | NOW      | 72.32%             | 8.92%    | 53.40%   | 29.57%   | 30.63%             |                3 | -8.46%        |
| 2026-09-30 | DELL     | 71.29%             | 35.18%   | 172.31%  | 247.75%  | 151.75%            |                3 | -3.46%        |
| 2026-09-30 | SNOW     | 71.09%             | 30.08%   | 143.59%  | 120.42%  | 98.03%             |                1 | -6.74%        |
| 2026-09-30 | NET      | 70.51%             | 33.82%   | 57.88%   | 56.83%   | 49.51%             |                2 | -3.10%        |
| 2026-09-30 | LRCX     | 69.40%             | -9.39%   | 11.81%   | 34.94%   | 12.45%             |                0 | -33.51%       |
| 2026-09-30 | AMD      | 69.36%             | 8.47%    | 57.92%   | 175.19%  | 80.53%             |                3 | -3.63%        |

## Strategy and baseline comparison
| strategy                     |   months | total_return_1m_rebalanced   | annualized_return_1m_rebalanced   | avg_monthly_return_1m   | monthly_win_rate_1m   | max_drawdown_1m_rebalanced   | avg_future_max_return_1_3m   | avg_boom_hit_rate   |
|:-----------------------------|---------:|:-----------------------------|:----------------------------------|:------------------------|:----------------------|:-----------------------------|:-----------------------------|:--------------------|
| baseline_core_mom_456_avg    |       18 | 207.25%                      | 111.35%                           | 8.56%                   | 66.67%                | -26.10%                      | 40.47%                       | 33.33%              |
| baseline_mom_5m              |       18 | 170.95%                      | 94.35%                            | 7.89%                   | 66.67%                | -25.10%                      | 34.89%                       | 31.48%              |
| baseline_mom_4m              |       18 | 165.98%                      | 91.97%                            | 7.56%                   | 61.11%                | -26.10%                      | 34.48%                       | 31.48%              |
| baseline_mom_6m_acceleration |       18 | 139.90%                      | 79.21%                            | 6.04%                   | 72.22%                | -27.37%                      | 34.46%                       | 27.78%              |
| baseline_mom_6m              |       18 | 121.66%                      | 70.00%                            | 6.69%                   | 66.67%                | -33.70%                      | 36.64%                       | 29.63%              |
| baseline_mom_3m              |       18 | 53.31%                       | 32.96%                            | 4.27%                   | 66.67%                | -33.70%                      | 35.72%                       | 29.63%              |
| xgb_boom_probability         |       18 | 49.13%                       | 30.53%                            | 4.11%                   | 61.11%                | -36.56%                      | 32.07%                       | 22.22%              |

## Recent XGB Top-3 backtest months
| month      | selected_tickers   | avg_score   | return_1m   | future_max_return_1_3m   | boom_hit_rate   |
|:-----------|:-------------------|:------------|:------------|:-------------------------|:----------------|
| 2025-07-31 | SMCI, OKLO, GEV    | 76.76%      | -13.53%     | 18.20%                   | 33.33%          |
| 2025-08-31 | SMR, PLTR, SMCI    | 86.55%      | 11.90%      | 27.50%                   | 0.00%           |
| 2025-09-30 | SOUN, OKLO, FLNC   | 82.48%      | 40.99%      | 40.99%                   | 33.33%          |
| 2025-10-31 | LRCX, ORCL, SMR    | 83.48%      | -26.48%     | -10.01%                  | 33.33%          |
| 2025-11-30 | FSLY, SMR, FLNC    | 81.02%      | -13.71%     | 36.02%                   | 66.67%          |
| 2025-12-31 | FLNC, OKLO, SMR    | 82.40%      | 29.96%      | 29.96%                   | 33.33%          |
| 2026-01-31 | FLNC, SNOW, MU     | 77.42%      | -20.90%     | -12.47%                  | 0.00%           |
| 2026-02-28 | FSLY, FLNC, INTC   | 88.21%      | 12.43%      | 74.97%                   | 33.33%          |
| 2026-03-31 | FLNC, MU, FSLY     | 82.26%      | 9.50%       | 91.02%                   | 33.33%          |
| 2026-04-30 | FSLY, ARM, AOSL    | 72.75%      | 14.25%      | 22.43%                   | 0.00%           |
| 2026-05-31 | FSLY, GEV, GOOGL   | 71.44%      | 6.25%       | 14.56%                   | 0.00%           |
| 2026-06-30 | FSLY, FLNC, SMCI   | 83.82%      | -3.23%      | 11.10%                   | 0.00%           |

## Top feature importance
| feature             |   importance | is_core_momentum   |
|:--------------------|-------------:|:-------------------|
| volatility_6m       |    0.0407515 | False              |
| volatility_3m       |    0.0281964 | False              |
| mom_7m              |    0.0264377 | False              |
| core_mom_456_max    |    0.0257721 | True               |
| mom_12m             |    0.0254725 | False              |
| drawdown_3m         |    0.0250916 | False              |
| drawdown_12m        |    0.0247254 | False              |
| qqq_mom_12m         |    0.0244842 | False              |
| mom_6m_last3m       |    0.0240533 | False              |
| drawdown_12m_abs    |    0.0230571 | False              |
| drawdown_6m         |    0.0227387 | False              |
| core_mom_456_avg    |    0.0226947 | True               |
| mom_2m              |    0.0223767 | False              |
| ma30_slope_1m       |    0.0223166 | False              |
| mom_3m              |    0.0221995 | False              |
| qqq_mom_3m          |    0.0217907 | False              |
| core_mom_456_min    |    0.0212558 | True               |
| core_mom_456_std    |    0.0211361 | True               |
| mom_6m              |    0.0209    | True               |
| drawdown_3m_abs     |    0.0206628 | False              |
| price_ma20_ratio    |    0.0204111 | False              |
| price_ma30_ratio    |    0.0203919 | False              |
| ma5_slope_1m        |    0.0202219 | False              |
| mom_6m_acceleration |    0.0202011 | True               |
| mom_4m              |    0.0201876 | True               |

## Feature design note
See `outputs/feature_manifest.txt` for the exact clean feature list.

## Correctness notes
- The model does not try to predict exact return size; it predicts probability of joining the future monthly top-10% boom group.
- The split is time-based: train <= 2022, validation 2023–2024, test >= 2025.
- The latest live candidates table does not use future returns; it only uses the newest available panel features.
- This is a research backtest, not investment advice.