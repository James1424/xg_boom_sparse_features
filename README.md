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
| train     |   3253 |       84 |          0.1039 | 0.9844 |   0.8855 |            0.9139 |             0.5492 |          0.9586 |                      0.8889 |                      0.7071 |                       0.3952 |                  0.1337 |                  0.2526 |                  0.3571 |                        0.4127 |                         0.3962 |                            0.0756 |                    0.3206 |
| valid     |   1247 |       24 |          0.1051 | 0.6883 |   0.2205 |            0.6961 |             0.1771 |          0.5191 |                      0.2778 |                      0.2583 |                       0.2167 |                  0.0506 |                  0.0955 |                  0.1735 |                        0.3267 |                         0.3052 |                            0.1017 |                    0.2035 |
| test      |    862 |       16 |          0.1102 | 0.6801 |   0.1984 |            0.6357 |             0.177  |          0.6316 |                      0.25   |                      0.225  |                       0.2313 |                  0.0534 |                  0.1312 |                  0.1788 |                        0.3516 |                         0.3442 |                            0.0874 |                    0.2568 |

## Latest live boom candidates
| month      | ticker   | boom_probability   | mom_4m   | mom_5m   | mom_6m   | core_mom_456_avg   |   ma_trend_score | drawdown_3m   |
|:-----------|:---------|:-------------------|:---------|:---------|:---------|:-------------------|-----------------:|:--------------|
| 2026-07-31 | ARM      | 87.06%             | 113.77%  | 153.74%  | 206.94%  | 158.15%            |                0 | -26.41%       |
| 2026-07-31 | MU       | 86.08%             | 189.92%  | 137.62%  | 136.18%  | 154.57%            |                1 | -19.29%       |
| 2026-07-31 | INTC     | 86.02%             | 148.90%  | 140.82%  | 136.37%  | 142.03%            |                1 | -22.07%       |
| 2026-07-31 | MRVL     | 85.65%             | 138.25%  | 188.88%  | 199.02%  | 175.38%            |                1 | -25.46%       |
| 2026-07-31 | FSLY     | 84.96%             | -32.59%  | 2.46%    | 111.78%  | 27.22%             |                2 | -39.46%       |
| 2026-07-31 | VECO     | 84.95%             | 69.52%   | 87.83%   | 83.80%   | 80.38%             |                1 | -30.88%       |
| 2026-07-31 | DELL     | 83.67%             | 165.84%  | 194.65%  | 281.26%  | 213.92%            |                3 | -6.65%        |
| 2026-07-31 | FLNC     | 81.33%             | 17.44%   | 3.99%    | -47.48%  | -8.68%             |                0 | -42.10%       |
| 2026-07-31 | AOSL     | 80.96%             | 69.49%   | 78.77%   | 69.95%   | 72.74%             |                0 | -29.56%       |
| 2026-07-31 | VRT      | 78.66%             | 27.28%   | 25.15%   | 71.34%   | 41.26%             |                2 | -15.23%       |
| 2026-07-31 | AMAT     | 76.44%             | 76.50%   | 62.03%   | 87.39%   | 75.31%             |                2 | -16.67%       |
| 2026-07-31 | NET      | 71.83%             | 30.08%   | 55.87%   | 51.34%   | 45.76%             |                2 | -2.68%        |
| 2026-07-31 | GEV      | 69.87%             | 25.11%   | 25.09%   | 50.45%   | 33.55%             |                2 | -7.09%        |
| 2026-07-31 | PLTR     | 68.41%             | -13.32%  | -7.58%   | -13.51%  | -11.47%            |                2 | -21.08%       |
| 2026-07-31 | CRWD     | 68.38%             | 91.78%   | 101.28%  | 69.62%   | 87.56%             |                3 | -6.12%        |

## Strategy and baseline comparison
| strategy                     |   months | total_return_1m_rebalanced   | annualized_return_1m_rebalanced   | avg_monthly_return_1m   | monthly_win_rate_1m   | max_drawdown_1m_rebalanced   | avg_future_max_return_1_3m   | avg_boom_hit_rate   |
|:-----------------------------|---------:|:-----------------------------|:----------------------------------|:------------------------|:----------------------|:-----------------------------|:-----------------------------|:--------------------|
| baseline_core_mom_456_avg    |       16 | 287.69%                      | 176.29%                           | 10.81%                  | 68.75%                | -20.26%                      | 44.87%                       | 37.50%              |
| baseline_mom_5m              |       16 | 237.32%                      | 148.90%                           | 10.00%                  | 68.75%                | -20.26%                      | 38.59%                       | 35.42%              |
| baseline_mom_6m_acceleration |       16 | 214.17%                      | 135.98%                           | 8.19%                   | 75.00%                | -14.57%                      | 38.86%                       | 31.25%              |
| baseline_mom_4m              |       16 | 210.13%                      | 133.70%                           | 9.13%                   | 62.50%                | -16.64%                      | 37.59%                       | 33.33%              |
| baseline_mom_6m              |       16 | 192.31%                      | 123.56%                           | 8.74%                   | 68.75%                | -20.53%                      | 41.03%                       | 33.33%              |
| baseline_mom_3m              |       16 | 119.94%                      | 80.60%                            | 6.59%                   | 68.75%                | -29.65%                      | 40.57%                       | 33.33%              |
| xgb_boom_probability         |       16 | 72.27%                       | 50.37%                            | 5.34%                   | 62.50%                | -26.42%                      | 35.16%                       | 25.00%              |

## Recent XGB Top-3 backtest months
| month      | selected_tickers   | avg_score   | return_1m   | future_max_return_1_3m   | boom_hit_rate   |
|:-----------|:-------------------|:------------|:------------|:-------------------------|:----------------|
| 2025-05-31 | NET, OKLO, PLTR    | 81.70%      | 9.23%       | 30.42%                   | 0.00%           |
| 2025-06-30 | OKLO, SMCI, SMR    | 82.73%      | 28.01%      | 48.87%                   | 33.33%          |
| 2025-07-31 | SMCI, OKLO, GEV    | 77.93%      | -13.53%     | 18.20%                   | 33.33%          |
| 2025-08-31 | SMR, PLTR, SOUN    | 87.25%      | 14.60%      | 30.92%                   | 0.00%           |
| 2025-09-30 | SOUN, OKLO, FLNC   | 81.59%      | 40.99%      | 40.99%                   | 33.33%          |
| 2025-10-31 | LRCX, ORCL, ARM    | 83.56%      | -14.73%     | 1.75%                    | 33.33%          |
| 2025-11-30 | FSLY, SMR, FLNC    | 80.45%      | -13.71%     | 36.02%                   | 66.67%          |
| 2025-12-31 | FLNC, OKLO, SMR    | 82.39%      | 29.96%      | 29.96%                   | 33.33%          |
| 2026-01-31 | FLNC, MU, SNOW     | 77.67%      | -20.90%     | -12.47%                  | 0.00%           |
| 2026-02-28 | FSLY, FLNC, INTC   | 88.66%      | 12.43%      | 74.97%                   | 33.33%          |
| 2026-03-31 | FLNC, MU, FSLY     | 82.04%      | 9.50%       | 91.02%                   | 33.33%          |
| 2026-04-30 | FSLY, ARM, AOSL    | 73.44%      | 14.25%      | 18.38%                   | 0.00%           |

## Top feature importance
| feature             |   importance | is_core_momentum   |
|:--------------------|-------------:|:-------------------|
| volatility_6m       |    0.0409636 | False              |
| volatility_3m       |    0.0276567 | False              |
| core_mom_456_max    |    0.0262328 | True               |
| mom_7m              |    0.0261293 | False              |
| mom_12m             |    0.0254207 | False              |
| drawdown_12m        |    0.0251475 | False              |
| drawdown_3m         |    0.0249839 | False              |
| qqq_mom_12m         |    0.0245578 | False              |
| mom_6m_last3m       |    0.0233444 | False              |
| drawdown_12m_abs    |    0.0232047 | False              |
| core_mom_456_avg    |    0.0229752 | True               |
| drawdown_6m         |    0.0228392 | False              |
| mom_3m              |    0.0223768 | False              |
| ma30_slope_1m       |    0.0221851 | False              |
| mom_2m              |    0.0221622 | False              |
| qqq_mom_3m          |    0.0218756 | False              |
| core_mom_456_min    |    0.0213712 | True               |
| mom_6m              |    0.0211243 | True               |
| core_mom_456_std    |    0.0211034 | True               |
| drawdown_3m_abs     |    0.0207423 | False              |
| price_ma30_ratio    |    0.0205449 | False              |
| price_ma20_ratio    |    0.0204216 | False              |
| mom_6m_acceleration |    0.0203838 | True               |
| mom_9m              |    0.0203669 | False              |
| mom_4m              |    0.0201742 | True               |

## Feature design note
See `outputs/feature_manifest.txt` for the exact clean feature list.

## Correctness notes
- The model does not try to predict exact return size; it predicts probability of joining the future monthly top-10% boom group.
- The split is time-based: train <= 2022, validation 2023–2024, test >= 2025.
- The latest live candidates table does not use future returns; it only uses the newest available panel features.
- This is a research backtest, not investment advice.