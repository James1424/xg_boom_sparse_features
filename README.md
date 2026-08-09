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
| train     |   3253 |       84 |          0.1039 | 0.9842 |   0.8827 |            0.9164 |             0.5569 |          0.9556 |                      0.881  |                      0.7    |                       0.3952 |                  0.1318 |                  0.2511 |                  0.3554 |                        0.4108 |                         0.393  |                            0.0734 |                    0.3196 |
| valid     |   1247 |       24 |          0.1051 | 0.6881 |   0.2197 |            0.6937 |             0.1723 |          0.5038 |                      0.25   |                      0.2417 |                       0.2208 |                  0.0494 |                  0.096  |                  0.1591 |                        0.3108 |                         0.2973 |                            0.0943 |                    0.203  |
| test      |    916 |       17 |          0.1103 | 0.6825 |   0.194  |            0.6419 |             0.1766 |          0.6139 |                      0.2353 |                      0.2118 |                       0.2176 |                  0.0547 |                  0.1227 |                  0.1616 |                        0.3425 |                         0.3274 |                            0.0985 |                    0.2289 |

## Latest live boom candidates
| month      | ticker   | boom_probability   | mom_4m   | mom_5m   | mom_6m   | core_mom_456_avg   |   ma_trend_score | drawdown_3m   |
|:-----------|:---------|:-------------------|:---------|:---------|:---------|:-------------------|-----------------:|:--------------|
| 2026-08-31 | DELL     | 92.56%             | 117.53%  | 177.78%  | 207.89%  | 167.73%            |                3 | -2.89%        |
| 2026-08-31 | CRWD     | 89.78%             | 92.41%   | 119.69%  | 130.57%  | 114.22%            |                3 | 0.00%         |
| 2026-08-31 | MRVL     | 89.47%             | 32.47%   | 120.98%  | 167.94%  | 107.13%            |                1 | -30.86%       |
| 2026-08-31 | MU       | 88.31%             | 69.72%   | 159.80%  | 112.93%  | 114.15%            |                1 | -27.68%       |
| 2026-08-31 | HPE      | 88.09%             | 85.53%   | 124.17%  | 150.21%  | 119.97%            |                3 | -4.94%        |
| 2026-08-31 | SMCI     | 84.81%             | 13.61%   | 36.71%   | -3.89%   | 15.48%             |                3 | -37.95%       |
| 2026-08-31 | INTC     | 83.86%             | 7.59%    | 130.34%  | 122.87%  | 86.93%             |                1 | -27.88%       |
| 2026-08-31 | VECO     | 81.47%             | 5.90%    | 55.91%   | 72.74%   | 44.85%             |                1 | -36.43%       |
| 2026-08-31 | ARM      | 80.10%             | 34.35%   | 86.79%   | 121.71%  | 80.95%             |                1 | -35.70%       |
| 2026-08-31 | FSLY     | 78.41%             | -9.09%   | -20.99%  | 20.08%   | -3.33%             |                3 | -11.79%       |
| 2026-08-31 | AMD      | 77.30%             | 36.35%   | 137.61%  | 141.43%  | 105.13%            |                1 | -16.79%       |
| 2026-08-31 | SNOW     | 75.99%             | 142.17%  | 119.13%  | 96.24%   | 119.18%            |                3 | 0.00%         |
| 2026-08-31 | AOSL     | 75.97%             | -16.92%  | 62.82%   | 71.73%   | 39.21%             |                1 | -32.33%       |
| 2026-08-31 | SMR      | 75.27%             | -21.19%  | -9.41%   | -23.58%  | -18.06%            |                2 | -29.61%       |
| 2026-08-31 | PANW     | 74.56%             | 102.91%  | 126.96%  | 144.33%  | 124.73%            |                2 | -0.68%        |

## Strategy and baseline comparison
| strategy                     |   months | total_return_1m_rebalanced   | annualized_return_1m_rebalanced   | avg_monthly_return_1m   | monthly_win_rate_1m   | max_drawdown_1m_rebalanced   | avg_future_max_return_1_3m   | avg_boom_hit_rate   |
|:-----------------------------|---------:|:-----------------------------|:----------------------------------|:------------------------|:----------------------|:-----------------------------|:-----------------------------|:--------------------|
| baseline_core_mom_456_avg    |       17 | 315.79%                      | 173.43%                           | 10.60%                  | 70.59%                | -20.26%                      | 43.00%                       | 35.29%              |
| baseline_mom_5m              |       17 | 261.77%                      | 147.85%                           | 9.83%                   | 70.59%                | -20.26%                      | 37.09%                       | 33.33%              |
| baseline_mom_4m              |       17 | 259.94%                      | 146.97%                           | 9.54%                   | 64.71%                | -16.64%                      | 36.66%                       | 33.33%              |
| baseline_mom_6m              |       17 | 234.35%                      | 134.44%                           | 9.07%                   | 70.59%                | -20.53%                      | 39.81%                       | 31.37%              |
| baseline_mom_6m_acceleration |       17 | 230.32%                      | 132.44%                           | 8.01%                   | 76.47%                | -14.57%                      | 37.22%                       | 29.41%              |
| baseline_mom_3m              |       17 | 131.25%                      | 80.71%                            | 6.51%                   | 70.59%                | -29.65%                      | 38.83%                       | 31.37%              |
| xgb_boom_probability         |       17 | 85.49%                       | 54.67%                            | 5.47%                   | 64.71%                | -26.42%                      | 34.25%                       | 23.53%              |

## Recent XGB Top-3 backtest months
| month      | selected_tickers   | avg_score   | return_1m   | future_max_return_1_3m   | boom_hit_rate   |
|:-----------|:-------------------|:------------|:------------|:-------------------------|:----------------|
| 2025-06-30 | OKLO, SMCI, SMR    | 82.94%      | 28.01%      | 48.87%                   | 33.33%          |
| 2025-07-31 | SMCI, OKLO, GEV    | 78.03%      | -13.53%     | 18.20%                   | 33.33%          |
| 2025-08-31 | SMR, PLTR, SMCI    | 86.66%      | 11.90%      | 27.50%                   | 0.00%           |
| 2025-09-30 | SOUN, OKLO, FLNC   | 83.05%      | 40.99%      | 40.99%                   | 33.33%          |
| 2025-10-31 | LRCX, ORCL, ARM    | 83.01%      | -14.73%     | 1.75%                    | 33.33%          |
| 2025-11-30 | FSLY, SMR, FLNC    | 80.20%      | -13.71%     | 36.02%                   | 66.67%          |
| 2025-12-31 | FLNC, OKLO, SMR    | 82.30%      | 29.96%      | 29.96%                   | 33.33%          |
| 2026-01-31 | FLNC, SNOW, MU     | 77.35%      | -20.90%     | -12.47%                  | 0.00%           |
| 2026-02-28 | FSLY, FLNC, INTC   | 87.99%      | 12.43%      | 74.97%                   | 33.33%          |
| 2026-03-31 | FLNC, MU, FSLY     | 81.41%      | 9.50%       | 91.02%                   | 33.33%          |
| 2026-04-30 | FSLY, ARM, AOSL    | 73.10%      | 14.25%      | 22.43%                   | 0.00%           |
| 2026-05-31 | FSLY, GEV, VRT     | 71.51%      | 10.27%      | 18.90%                   | 0.00%           |

## Top feature importance
| feature             |   importance | is_core_momentum   |
|:--------------------|-------------:|:-------------------|
| volatility_6m       |    0.0410853 | False              |
| volatility_3m       |    0.0280412 | False              |
| mom_7m              |    0.0263271 | False              |
| core_mom_456_max    |    0.0256385 | True               |
| mom_6m_last3m       |    0.0255219 | False              |
| mom_12m             |    0.0253645 | False              |
| drawdown_12m        |    0.0252063 | False              |
| qqq_mom_12m         |    0.0245597 | False              |
| drawdown_3m         |    0.024319  | False              |
| core_mom_456_avg    |    0.0230645 | True               |
| drawdown_6m         |    0.0227903 | False              |
| drawdown_12m_abs    |    0.0222596 | False              |
| mom_3m              |    0.0221603 | False              |
| ma30_slope_1m       |    0.021995  | False              |
| mom_2m              |    0.0218184 | False              |
| qqq_mom_3m          |    0.0216057 | False              |
| drawdown_3m_abs     |    0.0212351 | False              |
| core_mom_456_std    |    0.0209973 | True               |
| core_mom_456_min    |    0.0209058 | True               |
| price_ma20_ratio    |    0.0204794 | False              |
| price_ma30_ratio    |    0.0203606 | False              |
| mom_6m              |    0.0203538 | True               |
| mom_6m_acceleration |    0.0202491 | True               |
| mom_6m_first3m      |    0.0202447 | False              |
| mom_1m              |    0.0202368 | False              |

## Feature design note
See `outputs/feature_manifest.txt` for the exact clean feature list.

## Correctness notes
- The model does not try to predict exact return size; it predicts probability of joining the future monthly top-10% boom group.
- The split is time-based: train <= 2022, validation 2023–2024, test >= 2025.
- The latest live candidates table does not use future returns; it only uses the newest available panel features.
- This is a research backtest, not investment advice.