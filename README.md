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
| train     |   3253 |       84 |          0.1039 | 0.9842 |   0.8827 |            0.9164 |             0.5569 |          0.9556 |                      0.881  |                      0.7    |                       0.3952 |                  0.1318 |                  0.2511 |                  0.3554 |                        0.4108 |                         0.393  |                            0.0734 |                    0.3196 |
| valid     |   1247 |       24 |          0.1051 | 0.6881 |   0.2197 |            0.6937 |             0.1723 |          0.5038 |                      0.25   |                      0.2417 |                       0.2208 |                  0.0494 |                  0.096  |                  0.1591 |                        0.3108 |                         0.2973 |                            0.0943 |                    0.203  |
| test      |   1024 |       19 |          0.1104 | 0.6568 |   0.1758 |            0.6133 |             0.1623 |          0.6018 |                      0.2281 |                      0.2    |                       0.2    |                  0.0526 |                  0.1268 |                  0.1481 |                        0.331  |                         0.311  |                            0.095  |                    0.216  |

## Latest live boom candidates
| month      | ticker   | boom_probability   | mom_4m   | mom_5m   | mom_6m   | core_mom_456_avg   |   ma_trend_score | drawdown_3m   |
|:-----------|:---------|:-------------------|:---------|:---------|:---------|:-------------------|-----------------:|:--------------|
| 2026-10-31 | DELL     | 93.80%             | 6.11%    | 8.77%    | 119.11%  | 44.66%             |                0 | -22.32%       |
| 2026-10-31 | FSLY     | 84.54%             | 32.95%   | 37.41%   | -3.35%   | 22.34%             |                0 | -18.69%       |
| 2026-10-31 | HPE      | 84.33%             | 25.99%   | 32.43%   | 98.12%   | 52.18%             |                1 | -22.83%       |
| 2026-10-31 | SMCI     | 83.41%             | 12.31%   | -28.53%  | 20.22%   | 1.33%              |                0 | -26.70%       |
| 2026-10-31 | NET      | 81.70%             | 32.80%   | 34.70%   | 58.92%   | 42.14%             |                0 | -9.75%        |
| 2026-10-31 | MU       | 81.08%             | -23.95%  | -9.60%   | 69.74%   | 12.06%             |                0 | -20.02%       |
| 2026-10-31 | NOW      | 79.47%             | 60.11%   | 27.81%   | 80.00%   | 55.98%             |                3 | 0.00%         |
| 2026-10-31 | PANW     | 77.04%             | 11.59%   | 35.10%   | 112.22%  | 52.97%             |                1 | -9.37%        |
| 2026-10-31 | DDOG     | 75.24%             | 3.57%    | 9.02%    | 103.99%  | 38.86%             |                1 | -8.05%        |
| 2026-10-31 | AMD      | 75.22%             | -11.04%  | 0.13%    | 45.78%   | 11.63%             |                1 | -20.42%       |
| 2026-10-31 | MRVL     | 71.71%             | -23.25%  | 11.52%   | 38.43%   | 8.90%              |                1 | -20.37%       |
| 2026-10-31 | AI       | 71.41%             | 37.07%   | 15.69%   | 41.11%   | 31.29%             |                3 | 0.00%         |
| 2026-10-31 | FLNC     | 70.45%             | -71.88%  | -70.39%  | -54.11%  | -65.46%            |                0 | -64.33%       |
| 2026-10-31 | CRWD     | 69.87%             | 35.01%   | 40.94%   | 131.13%  | 69.03%             |                1 | -7.63%        |
| 2026-10-31 | BILL     | 69.70%             | 40.21%   | 36.95%   | 33.42%   | 36.86%             |                2 | -1.19%        |

## Strategy and baseline comparison
| strategy                     |   months | total_return_1m_rebalanced   | annualized_return_1m_rebalanced   | avg_monthly_return_1m   | monthly_win_rate_1m   | max_drawdown_1m_rebalanced   | avg_future_max_return_1_3m   | avg_boom_hit_rate   |
|:-----------------------------|---------:|:-----------------------------|:----------------------------------|:------------------------|:----------------------|:-----------------------------|:-----------------------------|:--------------------|
| baseline_core_mom_456_avg    |       19 | 207.06%                      | 103.10%                           | 8.11%                   | 63.16%                | -26.15%                      | 39.59%                       | 31.58%              |
| baseline_mom_4m              |       19 | 190.66%                      | 96.19%                            | 7.65%                   | 63.16%                | -26.10%                      | 34.40%                       | 29.82%              |
| baseline_mom_5m              |       19 | 170.78%                      | 87.60%                            | 7.47%                   | 63.16%                | -25.15%                      | 34.23%                       | 29.82%              |
| baseline_mom_6m_acceleration |       19 | 153.19%                      | 79.81%                            | 6.01%                   | 73.68%                | -27.37%                      | 33.69%                       | 26.32%              |
| baseline_mom_6m              |       19 | 140.81%                      | 74.20%                            | 6.80%                   | 68.42%                | -33.70%                      | 36.58%                       | 29.82%              |
| xgb_boom_probability         |       19 | 97.61%                       | 53.76%                            | 5.26%                   | 63.16%                | -26.42%                      | 33.10%                       | 22.81%              |
| baseline_mom_3m              |       19 | 60.44%                       | 34.80%                            | 4.29%                   | 68.42%                | -33.70%                      | 35.06%                       | 28.07%              |

## Recent XGB Top-3 backtest months
| month      | selected_tickers   | avg_score   | return_1m   | future_max_return_1_3m   | boom_hit_rate   |
|:-----------|:-------------------|:------------|:------------|:-------------------------|:----------------|
| 2025-08-31 | SMR, PLTR, SMCI    | 86.66%      | 11.90%      | 27.50%                   | 0.00%           |
| 2025-09-30 | SOUN, OKLO, FLNC   | 83.05%      | 40.99%      | 40.99%                   | 33.33%          |
| 2025-10-31 | LRCX, ORCL, ARM    | 83.01%      | -14.73%     | 1.75%                    | 33.33%          |
| 2025-11-30 | FSLY, SMR, FLNC    | 80.20%      | -13.71%     | 36.02%                   | 66.67%          |
| 2025-12-31 | FLNC, OKLO, SMR    | 82.30%      | 29.96%      | 29.96%                   | 33.33%          |
| 2026-01-31 | FLNC, SNOW, MU     | 77.35%      | -20.90%     | -12.47%                  | 0.00%           |
| 2026-02-28 | FSLY, FLNC, INTC   | 87.99%      | 12.43%      | 74.97%                   | 33.33%          |
| 2026-03-31 | FLNC, MU, FSLY     | 81.41%      | 9.50%       | 91.02%                   | 33.33%          |
| 2026-04-30 | FSLY, ARM, AOSL    | 73.10%      | 14.25%      | 22.43%                   | 0.00%           |
| 2026-05-31 | FSLY, GEV, VRT     | 71.51%      | 10.27%      | 18.58%                   | 0.00%           |
| 2026-06-30 | FSLY, FLNC, SMCI   | 84.22%      | -3.23%      | 16.64%                   | 0.00%           |
| 2026-07-31 | ARM, MRVL, MU      | 94.39%      | 10.09%      | 30.38%                   | 33.33%          |

## Top feature importance
| feature             |   importance | is_core_momentum   |
|:--------------------|-------------:|:-------------------|
| volatility_6m       |    0.0410857 | False              |
| volatility_3m       |    0.0280414 | False              |
| mom_7m              |    0.0263273 | False              |
| core_mom_456_max    |    0.0256386 | True               |
| mom_6m_last3m       |    0.025522  | False              |
| mom_12m             |    0.0253647 | False              |
| drawdown_12m        |    0.0252065 | False              |
| qqq_mom_12m         |    0.02456   | False              |
| drawdown_3m         |    0.0243192 | False              |
| core_mom_456_avg    |    0.0230647 | True               |
| drawdown_6m         |    0.0227905 | False              |
| drawdown_12m_abs    |    0.0222597 | False              |
| mom_3m              |    0.0221605 | False              |
| ma30_slope_1m       |    0.0219951 | False              |
| mom_2m              |    0.0218238 | False              |
| qqq_mom_3m          |    0.0216058 | False              |
| drawdown_3m_abs     |    0.0212352 | False              |
| core_mom_456_std    |    0.0209975 | True               |
| core_mom_456_min    |    0.0209059 | True               |
| price_ma20_ratio    |    0.0204796 | False              |
| price_ma30_ratio    |    0.0203607 | False              |
| mom_6m              |    0.0203539 | True               |
| mom_6m_acceleration |    0.0202492 | True               |
| mom_6m_first3m      |    0.0202448 | False              |
| mom_1m              |    0.020237  | False              |

## Feature design note
See `outputs/feature_manifest.txt` for the exact clean feature list.

## Correctness notes
- The model does not try to predict exact return size; it predicts probability of joining the future monthly top-10% boom group.
- The split is time-based: train <= 2022, validation 2023–2024, test >= 2025.
- The latest live candidates table does not use future returns; it only uses the newest available panel features.
- This is a research backtest, not investment advice.