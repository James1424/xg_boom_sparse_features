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
| train     |   3253 |       84 |          0.1039 | 0.9845 |   0.8835 |            0.9179 |             0.5611 |          0.9645 |                      0.873  |                      0.6976 |                       0.3952 |                  0.1319 |                  0.2494 |                  0.3518 |                        0.408  |                         0.3941 |                            0.0754 |                    0.3188 |
| valid     |   1247 |       24 |          0.1051 | 0.6902 |   0.2158 |            0.7009 |             0.1799 |          0.5191 |                      0.3333 |                      0.275  |                       0.2292 |                  0.0726 |                  0.1515 |                  0.2211 |                        0.371  |                         0.3036 |                            0.1038 |                    0.1997 |
| test      |    970 |       18 |          0.1103 | 0.6591 |   0.1828 |            0.6165 |             0.1576 |          0.5701 |                      0.2037 |                      0.2    |                       0.2111 |                  0.0328 |                  0.0994 |                  0.1207 |                        0.2898 |                         0.3033 |                            0.0985 |                    0.2048 |

## Latest live boom candidates
| month      | ticker   | boom_probability   | mom_4m   | mom_5m   | mom_6m   | core_mom_456_avg   |   ma_trend_score | drawdown_3m   |
|:-----------|:---------|:-------------------|:---------|:---------|:---------|:-------------------|-----------------:|:--------------|
| 2026-09-30 | INTC     | 89.07%             | -10.24%  | 8.95%    | 133.27%  | 43.99%             |                2 | -26.96%       |
| 2026-09-30 | MRVL     | 82.69%             | 15.20%   | 43.00%   | 138.54%  | 65.58%             |                2 | -23.96%       |
| 2026-09-30 | ZS       | 76.60%             | 17.76%   | 25.91%   | 17.29%   | 20.32%             |                1 | -12.66%       |
| 2026-09-30 | SNOW     | 76.25%             | 28.74%   | 141.07%  | 118.13%  | 95.98%             |                3 | -7.71%        |
| 2026-09-30 | DELL     | 73.57%             | 35.00%   | 171.94%  | 247.28%  | 151.41%            |                3 | 0.00%         |
| 2026-09-30 | OKLO     | 72.17%             | -45.84%  | -50.04%  | -26.96%  | -40.95%            |                1 | -40.79%       |
| 2026-09-30 | MU       | 70.29%             | 0.45%    | 88.61%   | 188.72%  | 92.59%             |                3 | -19.62%       |
| 2026-09-30 | PATH     | 69.28%             | 17.32%   | 33.50%   | 23.87%   | 24.90%             |                1 | -26.35%       |
| 2026-09-30 | CEVA     | 68.84%             | -27.44%  | -5.01%   | 55.30%   | 7.62%              |                1 | -41.65%       |
| 2026-09-30 | ARM      | 68.02%             | -25.05%  | 25.90%   | 75.03%   | 25.29%             |                1 | -39.75%       |
| 2026-09-30 | AMD      | 65.64%             | 0.01%    | 45.60%   | 153.71%  | 66.44%             |                2 | -11.15%       |
| 2026-09-30 | FSLY     | 65.12%             | 30.37%   | -8.30%   | -20.30%  | 0.59%              |                1 | -22.85%       |
| 2026-09-30 | AOSL     | 64.67%             | -43.15%  | -40.64%  | 16.34%   | -22.49%            |                1 | -47.77%       |
| 2026-09-30 | SMCI     | 63.44%             | -13.00%  | 46.35%   | 76.11%   | 36.49%             |                3 | -0.40%        |
| 2026-09-30 | HPE      | 63.08%             | 44.68%   | 116.45%  | 161.54%  | 107.56%            |                2 | 0.00%         |

## Strategy and baseline comparison
| strategy                     |   months | total_return_1m_rebalanced   | annualized_return_1m_rebalanced   | avg_monthly_return_1m   | monthly_win_rate_1m   | max_drawdown_1m_rebalanced   | avg_future_max_return_1_3m   | avg_boom_hit_rate   |
|:-----------------------------|---------:|:-----------------------------|:----------------------------------|:------------------------|:----------------------|:-----------------------------|:-----------------------------|:--------------------|
| baseline_core_mom_456_avg    |       18 | 207.25%                      | 111.35%                           | 8.56%                   | 66.67%                | -26.10%                      | 40.34%                       | 33.33%              |
| baseline_mom_5m              |       18 | 170.95%                      | 94.35%                            | 7.89%                   | 66.67%                | -25.10%                      | 34.78%                       | 31.48%              |
| baseline_mom_4m              |       18 | 165.98%                      | 91.97%                            | 7.56%                   | 61.11%                | -26.10%                      | 34.35%                       | 31.48%              |
| baseline_mom_6m_acceleration |       18 | 139.90%                      | 79.21%                            | 6.04%                   | 72.22%                | -27.37%                      | 34.18%                       | 27.78%              |
| baseline_mom_6m              |       18 | 121.66%                      | 70.00%                            | 6.69%                   | 66.67%                | -33.70%                      | 36.45%                       | 29.63%              |
| baseline_mom_3m              |       18 | 53.31%                       | 32.96%                            | 4.27%                   | 66.67%                | -33.70%                      | 35.53%                       | 29.63%              |
| xgb_boom_probability         |       18 | 32.62%                       | 20.71%                            | 3.28%                   | 55.56%                | -26.42%                      | 28.98%                       | 20.37%              |

## Recent XGB Top-3 backtest months
| month      | selected_tickers   | avg_score   | return_1m   | future_max_return_1_3m   | boom_hit_rate   |
|:-----------|:-------------------|:------------|:------------|:-------------------------|:----------------|
| 2025-07-31 | SMCI, OKLO, GEV    | 80.07%      | -13.53%     | 18.20%                   | 33.33%          |
| 2025-08-31 | SMR, PLTR, SOUN    | 86.76%      | 14.60%      | 30.92%                   | 0.00%           |
| 2025-09-30 | SOUN, OKLO, FLNC   | 81.42%      | 40.99%      | 40.99%                   | 33.33%          |
| 2025-10-31 | ORCL, LRCX, ARM    | 84.33%      | -14.73%     | 1.75%                    | 33.33%          |
| 2025-11-30 | SMR, FSLY, FLNC    | 80.73%      | -13.71%     | 36.02%                   | 66.67%          |
| 2025-12-31 | FLNC, OKLO, SMR    | 81.52%      | 29.96%      | 29.96%                   | 33.33%          |
| 2026-01-31 | FLNC, MDB, NOW     | 76.92%      | -22.91%     | -22.91%                  | 0.00%           |
| 2026-02-28 | FSLY, FLNC, INTC   | 87.27%      | 12.43%      | 74.97%                   | 33.33%          |
| 2026-03-31 | FLNC, MU, FSLY     | 83.20%      | 9.50%       | 91.02%                   | 33.33%          |
| 2026-04-30 | FSLY, AVGO, AOSL   | 70.75%      | -6.07%      | 1.91%                    | 0.00%           |
| 2026-05-31 | FSLY, GEV, VRT     | 71.89%      | 10.27%      | 18.58%                   | 0.00%           |
| 2026-06-30 | FSLY, FLNC, MU     | 83.68%      | -11.73%     | -6.43%                   | 0.00%           |

## Top feature importance
| feature          |   importance | is_core_momentum   |
|:-----------------|-------------:|:-------------------|
| volatility_6m    |    0.0404997 | False              |
| volatility_3m    |    0.0273909 | False              |
| core_mom_456_max |    0.0257857 | True               |
| mom_12m          |    0.0255443 | False              |
| drawdown_12m     |    0.0253674 | False              |
| qqq_mom_12m      |    0.0250976 | False              |
| mom_6m_last3m    |    0.025068  | False              |
| mom_7m           |    0.0245786 | False              |
| drawdown_12m_abs |    0.0239146 | False              |
| drawdown_3m      |    0.0233391 | False              |
| core_mom_456_avg |    0.0229149 | True               |
| ma30_slope_1m    |    0.022846  | False              |
| drawdown_6m      |    0.0220848 | False              |
| mom_2m           |    0.0215911 | False              |
| mom_3m           |    0.0215417 | False              |
| mom_9m           |    0.0213985 | False              |
| mom_6m           |    0.0213273 | True               |
| price_ma30_ratio |    0.0212387 | False              |
| qqq_mom_3m       |    0.0207825 | False              |
| core_mom_456_std |    0.0207684 | True               |
| mom_6m_first3m   |    0.0206333 | False              |
| core_mom_456_min |    0.0205471 | True               |
| mom_1m           |    0.0202452 | False              |
| price_ma20_ratio |    0.0200893 | False              |
| mom_4m           |    0.0200535 | True               |

## Feature design note
See `outputs/feature_manifest.txt` for the exact clean feature list.

## Correctness notes
- The model does not try to predict exact return size; it predicts probability of joining the future monthly top-10% boom group.
- The split is time-based: train <= 2022, validation 2023–2024, test >= 2025.
- The latest live candidates table does not use future returns; it only uses the newest available panel features.
- This is a research backtest, not investment advice.