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
| train     |   3253 |       84 |          0.1039 | 0.9842 |   0.8827 |            0.9164 |             0.5569 |          0.9556 |                       0.881 |                      0.7    |                       0.3952 |                  0.1318 |                  0.2511 |                  0.3554 |                        0.4108 |                         0.393  |                            0.0734 |                    0.3196 |
| valid     |   1247 |       24 |          0.1051 | 0.6881 |   0.2197 |            0.6937 |             0.1723 |          0.5038 |                       0.25  |                      0.2417 |                       0.2208 |                  0.0494 |                  0.096  |                  0.1591 |                        0.3108 |                         0.2973 |                            0.0943 |                    0.203  |
| test      |    862 |       16 |          0.1102 | 0.6826 |   0.1975 |            0.6381 |             0.1761 |          0.6211 |                       0.25  |                      0.225  |                       0.2313 |                  0.0517 |                  0.129  |                  0.168  |                        0.352  |                         0.3423 |                            0.1008 |                    0.2416 |

## Latest live boom candidates
| month      | ticker   | boom_probability   | mom_4m   | mom_5m   | mom_6m   | core_mom_456_avg   |   ma_trend_score | drawdown_3m   |
|:-----------|:---------|:-------------------|:---------|:---------|:---------|:-------------------|-----------------:|:--------------|
| 2026-07-31 | ARM      | 96.15%             | 58.44%   | 88.07%   | 127.50%  | 91.33%             |                0 | -45.46%       |
| 2026-07-31 | MRVL     | 95.29%             | 89.50%   | 129.77%  | 137.84%  | 119.04%            |                0 | -40.71%       |
| 2026-07-31 | MU       | 91.78%             | 143.65%  | 99.70%   | 98.49%   | 113.95%            |                0 | -32.17%       |
| 2026-07-31 | VECO     | 89.59%             | 47.67%   | 63.61%   | 60.10%   | 57.13%             |                0 | -39.79%       |
| 2026-07-31 | INTC     | 89.35%             | 104.40%  | 97.76%   | 94.10%   | 98.75%             |                0 | -36.00%       |
| 2026-07-31 | FSLY     | 88.92%             | -22.02%  | 18.51%   | 144.97%  | 47.15%             |                3 | -29.98%       |
| 2026-07-31 | DELL     | 88.13%             | 148.15%  | 175.05%  | 255.90%  | 193.04%            |                1 | -12.86%       |
| 2026-07-31 | AOSL     | 84.65%             | 43.23%   | 51.07%   | 43.62%   | 45.97%             |                0 | -40.47%       |
| 2026-07-31 | PWR      | 83.06%             | 21.60%   | 18.56%   | 40.66%   | 26.94%             |                0 | -15.00%       |
| 2026-07-31 | VRT      | 82.06%             | -3.58%   | -5.18%   | 29.81%   | 7.02%              |                0 | -35.78%       |
| 2026-07-31 | AMD      | 80.90%             | 134.06%  | 137.83%  | 101.14%  | 124.34%            |                0 | -18.03%       |
| 2026-07-31 | ASML     | 75.13%             | 23.76%   | 12.69%   | 15.03%   | 17.16%             |                0 | -18.01%       |
| 2026-07-31 | LRCX     | 73.25%             | 37.24%   | 25.52%   | 25.75%   | 29.50%             |                0 | -32.38%       |
| 2026-07-31 | AMAT     | 73.21%             | 48.72%   | 36.53%   | 57.90%   | 47.71%             |                0 | -29.78%       |
| 2026-07-31 | KLAC     | 73.19%             | 24.32%   | 20.07%   | 28.36%   | 24.25%             |                0 | -39.41%       |

## Strategy and baseline comparison
| strategy                     |   months | total_return_1m_rebalanced   | annualized_return_1m_rebalanced   | avg_monthly_return_1m   | monthly_win_rate_1m   | max_drawdown_1m_rebalanced   | avg_future_max_return_1_3m   | avg_boom_hit_rate   |
|:-----------------------------|---------:|:-----------------------------|:----------------------------------|:------------------------|:----------------------|:-----------------------------|:-----------------------------|:--------------------|
| baseline_core_mom_456_avg    |       16 | 287.69%                      | 176.29%                           | 10.81%                  | 68.75%                | -20.26%                      | 45.12%                       | 37.50%              |
| baseline_mom_5m              |       16 | 237.32%                      | 148.90%                           | 10.00%                  | 68.75%                | -20.26%                      | 38.84%                       | 35.42%              |
| baseline_mom_6m_acceleration |       16 | 214.17%                      | 135.98%                           | 8.19%                   | 75.00%                | -14.57%                      | 39.11%                       | 31.25%              |
| baseline_mom_4m              |       16 | 210.13%                      | 133.70%                           | 9.13%                   | 62.50%                | -16.64%                      | 37.84%                       | 33.33%              |
| baseline_mom_6m              |       16 | 192.31%                      | 123.56%                           | 8.74%                   | 68.75%                | -20.53%                      | 41.29%                       | 33.33%              |
| baseline_mom_3m              |       16 | 119.94%                      | 80.60%                            | 6.59%                   | 68.75%                | -29.65%                      | 40.82%                       | 33.33%              |
| xgb_boom_probability         |       16 | 68.21%                       | 47.70%                            | 5.17%                   | 62.50%                | -26.42%                      | 35.20%                       | 25.00%              |

## Recent XGB Top-3 backtest months
| month      | selected_tickers   | avg_score   | return_1m   | future_max_return_1_3m   | boom_hit_rate   |
|:-----------|:-------------------|:------------|:------------|:-------------------------|:----------------|
| 2025-05-31 | NET, OKLO, PLTR    | 81.91%      | 9.23%       | 30.42%                   | 0.00%           |
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