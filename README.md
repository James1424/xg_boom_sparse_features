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
| rows        | 5993       |
| columns     | 63         |
| tickers     | 54         |
| months      | 131        |
| first_month | 2016-01-31 |
| last_month  | 2026-11-30 |

## Train / validation / test metrics
| dataset   |   rows |   months |   positive_rate |    auc |   pr_auc |   accuracy_at_0_5 |   precision_at_0_5 |   recall_at_0_5 |   precision_at_top3_monthly |   precision_at_top5_monthly |   precision_at_top10_monthly |   top3_future_return_1m |   top3_future_return_2m |   top3_future_return_3m |   top3_future_max_return_1_3m |   top_decile_future_max_return |   bottom_decile_future_max_return |   top_minus_bottom_spread |
|:----------|-------:|---------:|----------------:|-------:|---------:|------------------:|-------------------:|----------------:|----------------------------:|----------------------------:|-----------------------------:|------------------------:|------------------------:|------------------------:|------------------------------:|-------------------------------:|----------------------------------:|--------------------------:|
| train     |   3253 |       84 |          0.1039 | 0.9842 |   0.8827 |            0.9164 |             0.5569 |          0.9556 |                      0.881  |                      0.7    |                       0.3952 |                  0.1318 |                  0.2511 |                  0.3554 |                        0.4108 |                         0.393  |                            0.0734 |                    0.3196 |
| valid     |   1247 |       24 |          0.1051 | 0.6881 |   0.2197 |            0.6937 |             0.1723 |          0.5038 |                      0.25   |                      0.2417 |                       0.2208 |                  0.0494 |                  0.096  |                  0.1591 |                        0.3108 |                         0.2973 |                            0.0943 |                    0.203  |
| test      |   1078 |       20 |          0.1104 | 0.6543 |   0.1755 |            0.6067 |             0.1604 |          0.605  |                      0.2333 |                      0.2    |                       0.2    |                  0.0588 |                  0.1162 |                  0.1237 |                        0.3233 |                         0.303  |                            0.0916 |                    0.2115 |

## Latest live boom candidates
| month      | ticker   | boom_probability   | mom_4m   | mom_5m   | mom_6m   | core_mom_456_avg   |   ma_trend_score | drawdown_3m   |
|:-----------|:---------|:-------------------|:---------|:---------|:---------|:-------------------|-----------------:|:--------------|
| 2026-11-30 | ZS       | 84.79%             | 30.73%   | 40.04%   | 41.46%   | 37.41%             |                2 | -24.11%       |
| 2026-11-30 | DELL     | 73.11%             | -18.84%  | -23.62%  | -21.71%  | -21.39%            |                0 | -44.09%       |
| 2026-11-30 | FLNC     | 71.01%             | -62.17%  | -73.49%  | -72.09%  | -69.25%            |                0 | -60.14%       |
| 2026-11-30 | FSLY     | 70.94%             | -30.05%  | -13.67%  | -10.78%  | -18.17%            |                0 | -47.20%       |
| 2026-11-30 | MU       | 67.37%             | -16.15%  | -40.20%  | -28.92%  | -28.42%            |                0 | -37.11%       |
| 2026-11-30 | MRVL     | 66.98%             | -24.70%  | -52.58%  | -31.09%  | -36.12%            |                0 | -50.79%       |
| 2026-11-30 | SOUN     | 65.92%             | -32.95%  | -36.48%  | -54.33%  | -41.25%            |                0 | -45.05%       |
| 2026-11-30 | SMR      | 64.95%             | -39.67%  | -49.35%  | -59.91%  | -49.64%            |                0 | -54.56%       |
| 2026-11-30 | PLTR     | 64.48%             | 30.33%   | 37.46%   | 2.45%    | 23.41%             |                0 | -23.28%       |
| 2026-11-30 | SNOW     | 58.81%             | -5.93%   | 8.40%    | 7.96%    | 3.47%              |                0 | -25.21%       |
| 2026-11-30 | OKLO     | 56.76%             | -39.35%  | -55.00%  | -64.79%  | -53.05%            |                0 | -49.90%       |
| 2026-11-30 | ORCL     | 54.13%             | -10.27%  | -20.21%  | -48.21%  | -26.23%            |                0 | -28.30%       |
| 2026-11-30 | BILL     | 53.94%             | 16.13%   | 44.94%   | 41.57%   | 34.21%             |                3 | 0.00%         |
| 2026-11-30 | GOOGL    | 50.63%             | -19.98%  | -20.26%  | -25.03%  | -21.76%            |                0 | -19.77%       |
| 2026-11-30 | PATH     | 50.11%             | 1.88%    | 19.60%   | 10.92%   | 10.80%             |                2 | -30.37%       |

## Strategy and baseline comparison
| strategy                     |   months | total_return_1m_rebalanced   | annualized_return_1m_rebalanced   | avg_monthly_return_1m   | monthly_win_rate_1m   | max_drawdown_1m_rebalanced   | avg_future_max_return_1_3m   | avg_boom_hit_rate   |
|:-----------------------------|---------:|:-----------------------------|:----------------------------------|:------------------------|:----------------------|:-----------------------------|:-----------------------------|:--------------------|
| baseline_core_mom_456_avg    |       20 | 240.87%                      | 108.71%                           | 8.25%                   | 65.00%                | -26.15%                      | 38.16%                       | 30.00%              |
| baseline_mom_4m              |       20 | 214.30%                      | 98.80%                            | 7.67%                   | 65.00%                | -26.10%                      | 33.09%                       | 28.33%              |
| baseline_mom_5m              |       20 | 200.60%                      | 93.55%                            | 7.65%                   | 65.00%                | -25.15%                      | 33.07%                       | 28.33%              |
| baseline_mom_6m              |       20 | 178.35%                      | 84.82%                            | 7.24%                   | 70.00%                | -33.70%                      | 35.53%                       | 30.00%              |
| xgb_boom_probability         |       20 | 132.50%                      | 65.90%                            | 5.88%                   | 65.00%                | -26.42%                      | 32.33%                       | 23.33%              |
| baseline_mom_6m_acceleration |       20 | 120.31%                      | 60.63%                            | 5.06%                   | 70.00%                | -33.30%                      | 32.11%                       | 26.67%              |
| baseline_mom_3m              |       20 | 40.14%                       | 22.44%                            | 3.44%                   | 65.00%                | -39.40%                      | 32.76%                       | 26.67%              |

## Recent XGB Top-3 backtest months
| month      | selected_tickers   | avg_score   | return_1m   | future_max_return_1_3m   | boom_hit_rate   |
|:-----------|:-------------------|:------------|:------------|:-------------------------|:----------------|
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
| 2026-08-31 | DELL, MRVL, SMCI   | 88.77%      | 17.65%      | 17.65%                   | 33.33%          |

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