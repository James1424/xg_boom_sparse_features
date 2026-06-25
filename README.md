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
| train     |   3253 |       84 |          0.1039 | 0.9842 |   0.8827 |            0.9164 |             0.5569 |          0.9556 |                      0.881  |                      0.7    |                       0.3952 |                  0.1318 |                  0.2511 |                  0.3554 |                        0.4108 |                         0.393  |                            0.0734 |                    0.3196 |
| valid     |   1247 |       24 |          0.1051 | 0.6881 |   0.2197 |            0.6937 |             0.1723 |          0.5038 |                      0.25   |                      0.2417 |                       0.2208 |                  0.0494 |                  0.096  |                  0.1591 |                        0.3108 |                         0.2973 |                            0.0943 |                    0.203  |
| test      |    808 |       15 |          0.1101 | 0.6966 |   0.2055 |            0.646  |             0.1853 |          0.6517 |                      0.2667 |                      0.24   |                       0.24   |                  0.0457 |                  0.1265 |                  0.1757 |                        0.3534 |                         0.34   |                            0.1079 |                    0.2321 |

## Latest live boom candidates
| month      | ticker   | boom_probability   | mom_4m   | mom_5m   | mom_6m   | core_mom_456_avg   |   ma_trend_score | drawdown_3m   |
|:-----------|:---------|:-------------------|:---------|:---------|:---------|:-------------------|-----------------:|:--------------|
| 2026-06-30 | VRT      | 89.44%             | 24.20%   | 70.04%   | 95.40%   | 63.21%             |                1 | -15.88%       |
| 2026-06-30 | AOSL     | 82.75%             | 99.81%   | 89.95%   | 111.91%  | 100.56%            |                2 | -21.27%       |
| 2026-06-30 | MU       | 81.00%             | 154.37%  | 152.83%  | 267.52%  | 191.58%            |                3 | -13.44%       |
| 2026-06-30 | FSLY     | 80.38%             | -14.33%  | 77.08%   | 60.90%   | 41.22%             |                1 | -51.10%       |
| 2026-06-30 | INTC     | 76.73%             | 188.64%  | 183.30%  | 256.78%  | 209.57%            |                3 | -6.59%        |
| 2026-06-30 | AMAT     | 76.19%             | 58.39%   | 83.18%   | 129.75%  | 90.44%             |                3 | -8.00%        |
| 2026-06-30 | FLNC     | 75.86%             | 27.09%   | -35.81%  | -0.15%   | -2.96%             |                1 | -29.24%       |
| 2026-06-30 | LRCX     | 75.52%             | 60.55%   | 60.85%   | 119.37%  | 80.26%             |                3 | -8.48%        |
| 2026-06-30 | KLAC     | 75.39%             | 57.94%   | 68.84%   | 98.42%   | 75.07%             |                3 | -10.66%       |
| 2026-06-30 | TSM      | 74.05%             | 18.29%   | 34.05%   | 45.81%   | 32.72%             |                3 | -5.74%        |
| 2026-06-30 | MRVL     | 71.57%             | 238.89%  | 250.78%  | 226.00%  | 238.56%            |                3 | -12.56%       |
| 2026-06-30 | ARM      | 71.20%             | 181.74%  | 240.81%  | 228.50%  | 217.02%            |                3 | -18.29%       |
| 2026-06-30 | AVGO     | 70.70%             | 20.01%   | 15.75%   | 10.80%   | 15.52%             |                1 | -20.54%       |
| 2026-06-30 | VECO     | 69.25%             | 132.85%  | 127.86%  | 148.99%  | 136.57%            |                3 | -14.31%       |
| 2026-06-30 | DELL     | 69.21%             | 194.03%  | 280.46%  | 247.40%  | 240.63%            |                3 | -6.85%        |

## Strategy and baseline comparison
| strategy                     |   months | total_return_1m_rebalanced   | annualized_return_1m_rebalanced   | avg_monthly_return_1m   | monthly_win_rate_1m   | max_drawdown_1m_rebalanced   | avg_future_max_return_1_3m   | avg_boom_hit_rate   |
|:-----------------------------|---------:|:-----------------------------|:----------------------------------|:------------------------|:----------------------|:-----------------------------|:-----------------------------|:--------------------|
| baseline_mom_4m              |       15 | 214.17%                      | 149.88%                           | 9.82%                   | 66.67%                | -16.64%                      | 38.40%                       | 35.56%              |
| baseline_core_mom_456_avg    |       15 | 206.49%                      | 144.98%                           | 9.77%                   | 66.67%                | -20.26%                      | 42.99%                       | 37.78%              |
| baseline_mom_5m              |       15 | 166.67%                      | 119.17%                           | 8.90%                   | 66.67%                | -20.26%                      | 36.93%                       | 35.56%              |
| baseline_mom_6m_acceleration |       15 | 160.04%                      | 114.80%                           | 7.34%                   | 73.33%                | -14.57%                      | 38.65%                       | 33.33%              |
| baseline_mom_6m              |       15 | 131.09%                      | 95.45%                            | 7.55%                   | 66.67%                | -20.53%                      | 38.90%                       | 33.33%              |
| baseline_mom_3m              |       15 | 108.90%                      | 80.28%                            | 6.68%                   | 66.67%                | -29.65%                      | 40.93%                       | 35.56%              |
| xgb_boom_probability         |       15 | 47.24%                       | 36.27%                            | 4.57%                   | 60.00%                | -26.42%                      | 35.34%                       | 26.67%              |

## Recent XGB Top-3 backtest months
| month      | selected_tickers   | avg_score   | return_1m   | future_max_return_1_3m   | boom_hit_rate   |
|:-----------|:-------------------|:------------|:------------|:-------------------------|:----------------|
| 2025-04-30 | SOUN, PLTR, CRWD   | 85.25%      | 10.00%      | 22.65%                   | 0.00%           |
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
| 2026-03-31 | FLNC, MU, FSLY     | 81.41%      | 9.50%       | 80.27%                   | 33.33%          |

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