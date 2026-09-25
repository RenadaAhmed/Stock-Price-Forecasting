# Stock Price Prediction with a Hybrid CNN–LSTM Architecture

**MSc Dissertation Project — Multi-Source Deep Learning for U.S. Equity Price Prediction**

> Code is kept private for this repository. This README documents the problem, methodology, and results in full. Feel free to reach out if you'd like to discuss the implementation in more detail.

## Overview

This project investigates whether combining multiple financial data sources — technical indicators, macroeconomic signals, fundamental ratios, and news sentiment — improves stock price forecasting accuracy, and whether a hybrid CNN–LSTM architecture outperforms simpler deep learning baselines. The study spans 11 U.S. equities (including AAPL, TSLA, NVDA, NFLX, MSFT, and GS) over the period March 2018 to June 2025, covering diverse market conditions including the 2023 U.S. banking crisis, the 2024 AI-driven tech rally, and early 2025 geopolitical volatility and systematically tests five temporal window lengths (5, 10, 20, 30, and 60 trading days) to understand how sequence length interacts with each asset's price behavior.

**Three research questions were tested:**
1. Does integrating multiple financial data sources improve predictive accuracy over price data alone?
2. Does a hybrid CNN–LSTM offer an advantage over standalone CNN or LSTM models?
3. How does temporal window length affect forecasting performance, and does it vary by asset?

All three hypotheses were supported by the results.

---

## Data & Coverage

Data was collected from **January 2017 to June 2025** (7.5 years) via **EODHD** (historical prices, financials, news) and **FRED** (macroeconomic indicators), with news sentiment derived using **FinBERT**. The earliest ~14 months (Jan 2017–Mar 2018) served as historical lookback context for the longest sliding windows; core experiments and evaluation span **March 2018 – June 2025**. This range deliberately covers multiple distinct market regimes — U.S.–China trade tensions, the COVID-19 crash and recovery, the 2022–2023 inflation surge and rate hikes, the 2023 U.S. regional banking crisis, and 2024–2025 AI-driven volatility — to test the model's robustness beyond a single, stable period. The held-out test set (the final 15% of each series) covers 29 March 2023 – 30 June 2025 specifically.

**11 U.S. companies across 6 sectors:**

| Sector | Tickers |
|---|---|
| Technology | AAPL, MSFT, NVDA, AMD |
| Healthcare | PFE, ABBV |
| Finance | GS |
| Aerospace | BA |
| Entertainment | NFLX, GOOG |
| Automotive | TSLA |


## Data & Features

Datasets were engineered per ticker, combining:
- **Technical indicators**: EMA, Bollinger Bands, MACD family, RSI, Momentum, Williams %R
- **Macroeconomic signals**: CPI, interest rates, unemployment
- **Fundamental ratios**: EPS, ROE, Debt-to-Equity
- **NLP-derived sentiment**: news sentiment scores, consumer sentiment

Feature selection was performed using **Ridge RFE** and validated with **SHAP** interpretability analysis to identify which signals the model actually relied on.

---

## Model Architecture

Three architectures were trained and compared under identical conditions. All three follow a train (70%) / validation (15%) / test (15%) chronological split with Min-Max normalization fitted on the training set only, to prevent data leakage.

**Standalone CNN** — a deep convolutional stack:
`Conv1D(128, kernel=3) → MaxPooling1D → Dropout(0.3) → Conv1D(64) → MaxPooling1D → Dense(64) → Dense(1)`.
51,521 parameters. Designed to test how much predictive value comes from local, position-invariant pattern extraction alone.

**Standalone LSTM** — 
`LSTM(128) → Dropout(0.3) → Dense(64) → Dense(1)`.
82,561 parameters. A standard sequence-modeling baseline for long-range temporal dependencies.

**Proposed Hybrid CNN–LSTM:**

| Layer | Output Shape | Params |
|---|---|---|
| Conv1D (128 filters, kernel=3, ReLU, L2 reg.) | (T, 128) | 1,280 |
| MaxPooling1D (pool=2) | (T/2, 128) | 0 |
| LSTM (256 units, L2 reg.) | (256) | 394,240 |
| Dropout (0.2) | (256) | 0 |
| Dense (output) | (1) | 257 |
| **Total** | | **395,777** |

The convolutional layer extracts short-term local patterns (spikes, micro-trends, abrupt shifts in technical indicators); the LSTM layer captures longer-range dependencies (macro/sentiment-driven movements unfolding over weeks or months). The model was compiled with the **Adam optimizer** and **Huber loss**—chosen for robustness to outliers, which are common in financial time series.

**Hyperparameter tuning** (grid search + iterative testing on the validation set): CNN filters/kernel size (64 vs. 128 filters, kernel 2 vs. 3 → 128 filters, kernel 3 won), LSTM units (128–512 tested → 256 was the best complexity/performance trade-off), dropout rate (0.1–0.2 → 0.2), learning rate (0.001 vs. 0.0001 → 0.0001 gave more stable convergence), batch size (16 vs. 32 → 16). Early stopping was applied with a patience of 5 epochs.

---


## Results

### Headline result: Overall model comparison

Averaged across all companies and windows:

| Model | RMSE | MAE | MAPE | R² |
|---|---|---|---|---|
| LSTM | 41.39 | 37.70 | 12.52% | -1.365 |
| CNN | 41.39 | 37.70 | 12.52% | -1.365 |
| **Proposed CNN–LSTM** | **12.39** | **9.47** | **3.48%** | **0.794** |

The standalone models produced **negative R² scores overall** — meaning they performed worse than simply predicting the historical mean. The proposed architecture eliminated this failure mode almost entirely, cutting RMSE by roughly two-thirds and MAPE from ~12% down to ~3.5%, while achieving consistently positive R² across nearly every ticker and window combination.

### Training stability

**Training vs validation loss curves across all four model variants**
<img width="985" height="470" alt="Screenshot 2026-09-25 040711" src="https://github.com/user-attachments/assets/8bc6f661-4d82-4599-bd18-4fd5372fb342" />
<img width="480" height="477" alt="Screenshot 2026-09-25 040726" src="https://github.com/user-attachments/assets/4a4a9482-3ad4-41e9-a88c-6fa25d5cc060" />


Loss curves across models tell a consistent story: the proposed CNN–LSTM converges smoothly, with training and validation loss tracking closely and flattening near zero — a strong sign of good generalization without overfitting. The standalone LSTM and CNN configurations show noisier, less stable validation curves by comparison, including a visible validation dip-and-recover pattern in the standalone LSTM — consistent with their weaker held-out performance above.

### Performance across temporal windows (all companies, hybrid model)

| Window (days) | RMSE | MAE | MAPE | R² |
|---|---|---|---|---|
| 5 | 11.31 | 8.39 | 3.07% | **0.855** |
| 10 | 11.15 | 8.24 | 3.15% | 0.841 |
| 20 | 11.80 | 8.94 | 3.28% | 0.840 |
| 30 | 12.24 | 9.45 | 3.43% | 0.819 |
| 60 | 15.46 | 12.33 | 4.47% | 0.616 |

Shorter windows generally performed best — consistent with the model's reliance on fast-adapting, trend-following features (see Feature Importance below).

### Example: NFLX prediction (window = 10)

RMSE: 29.93 · MAPE: 2.40% · MAE: 21.43 · **R² = 0.976** 

<img width="630" height="489" alt="image" src="https://github.com/user-attachments/assets/8ea2ba4a-9be6-4b2d-afc4-c94797658767" />


The model tracked NFLX's sharp 2023–2025 uptrend closely on the held-out test set, including its recovery from the 2022 drawdown.

### Feature importance (SHAP)

The model relied most heavily on **fast-adapting technical signals**, not slow-moving fundamentals:

1. **EMA_20** and **BBM_20 (2.0)** — top-ranked by a clear margin
2. Raw price levels (open, high, low)
3. Momentum indicators — **MOM_10**, **WILLR_14**
4. MACD family, **RSI_14**
5. Sentiment score, EPS, consumer sentiment, ROE — present but consistently lower-ranked

This suggests the model learns primarily from **trend and volatility signals that adapt quickly to changing conditions**, rather than slower macro/fundamental context — which also helps explain why shorter windows tended to outperform.

> **Note:** an earlier SHAP pass on the *baseline* (pre-feature-selection) CNN–LSTM ranked **P/E ratio** as the single most influential feature by a wide margin, followed by momentum/trend indicators (WILLR_14, MACD components, MOM_10). After Ridge RFE feature selection narrowed the feature set for the *proposed* model, EMA_20 and BBM_20 became dominant instead — suggesting the fundamental signal was informative but largely redundant with the technical indicators once a leaner feature set was enforced.

**Feature selection stability:** Ridge RFE was run independently per ticker and window size. For several companies — including AAPL, TSLA, and MSFT — the *exact same* set of 25 features was selected regardless of whether the window was 5 or 60 days, suggesting a stable "core" feature set drives prediction for those tickers independent of temporal framing.

### Asset-specific temporal behavior

Not every stock had the same "best" window — a key finding of the study:

| Ticker | Best window | R² | Interpretation |
|---|---|---|---|
| **NFLX** | 10 days | 0.976 | Stable, trend-following — needs enough context to capture sustained momentum, but not so much it dilutes recent signal |
| **NVDA** | 5 days | 0.757 | High volatility — older data quickly becomes stale or misleading |
| **BA** | 60 days | 0.880 | Anomaly — slower-moving, regulation/industry-driven price action benefits from longer historical context |
| **AMD** | Stable (~0.92 across all windows) | 0.918–0.926 | Robust to window choice, likely due to consistent cyclical pricing behavior |

This finding extends prior literature (Duman & Tağluk, 2024) arguing that **no single optimal temporal window exists across all assets** — window length should be tuned per asset's volatility profile and sector-specific drivers.

---

## Key Takeaways

- **Multi-source data integration measurably improves forecasting accuracy**, even though individual macro/sentiment features had modest standalone SHAP importance — their value is cumulative, helping the model generalize across changing market regimes rather than dominating any single prediction.
- **The hybrid CNN–LSTM architecture consistently and substantially outperformed both standalone CNN and standalone LSTM models**, turning frequently negative R² scores into consistently strong (0.6–0.98) results.
- **Temporal window length is asset-dependent**, not universal — a practical implication for anyone building forecasting pipelines: window size should be treated as a tunable, per-asset hyperparameter rather than a fixed global setting.

---

## Limitations & Future Work

- Performance still varies meaningfully by asset (e.g. NVDA remained the hardest ticker to forecast at longer windows), suggesting further work on adaptive/dynamic window selection per asset.
- Sentiment and macro features could be explored with finer-grained temporal alignment (e.g. event-driven rather than daily sentiment aggregation).

---

## Tech Stack

`Python` · `TensorFlow/Keras` · `SHAP` · `scikit-learn` (Ridge RFE) · `Pandas` / `NumPy` · technical analysis libraries for indicator engineering

---


