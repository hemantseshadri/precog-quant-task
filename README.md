# Quant Research Task: End-to-End Algorithmic Trading Pipeline

This repository contains the code and data processing pipelines for a quantitative research task involving the development of a trading strategy for a universe of 100 anonymized stocks.

## Project Structure

### Part 1: Feature Engineering & Data Cleaning (`part1_feature_engineering.py`)
This script processes the raw OHLCV market data to create a robust set of features for predictive modeling.

**What we accomplished:**
- **Data Cleaning:** Handled missing values using a combination of forward-filling and backward-filling to ensure no data gaps while strictly preventing look-ahead bias.
- **Engineered Features (14 total per asset):**
  - **Momentum**: 1-day returns (`Ret_1d`), 5-day returns (`Ret_5d`), Relative Strength Index (`RSI_14`), MACD, MACD Signal, MACD Histogram.
  - **Trend**: Simple Moving Averages (`SMA_10`, `SMA_50`), Distance from 50-day SMA (`Dist_SMA_50`).
  - **Volatility**: Bollinger Bands (`BB_Upper`, `BB_Lower`), 20-day Annualized Historical Volatility (`Vol_20d`).
  - **Volume**: 20-day Volume SMA (`Vol_SMA_20`), Volume Rate of Change (`Vol_ROC`).

*Output:* Cleaned datasets are saved to the `processed_data/` directory.

---

### Part 2: Model Training & Strategy Formulation (XGBoost)
We trained an XGBoost Regressor to predict each asset's 5-day forward return, then translated those predictions into trading signals: **Long (+1)** for the top 20% predicted returns each day, **Short (-1)** for the bottom 20%, and **Neutral (0)** for the rest.

We iterated through **3 versions**, each improving on the last:

#### V1 — Baseline (`Part2_V1_Baseline.ipynb`)
- **Features:** 14 original technical indicators (RSI, MACD, Bollinger Bands, Volatility, etc.)
- **Model:** XGBoost with basic hyperparameters (`max_depth=4`, `learning_rate=0.01`, `n_estimators=500`)
- **Result:** Gross Return **+3.0%**, Net Return **-18.9%**, Sharpe **-1.42**
- **Problem:** The model's predictions were barely better than random. With 48.7% daily turnover, transaction costs (10 bps) destroyed all profits.

#### V3 — Cross-Sectional Features + Tight Regularization (`Part2_V3_TightReg.ipynb`)
- **New Features (22 total):** Added 6 cross-sectional rank features (ranking each asset vs. its 99 peers each day), plus a mean reversion z-score and momentum acceleration signal.
- **Model:** Aggressive regularization (`max_depth=3`, `reg_lambda=5.0`, `gamma=1.0`, `min_child_weight=100`). Only 4 features survived the regularization filter.
- **Result:** Gross Return **~150%**, Net Return **~130%**, Sharpe **~1.4**
- **Why it improved:** Cross-sectional ranking transformed absolute values into relative comparisons. The model could now answer "Is this asset's volatility high *compared to its peers today*?" instead of just "Is this asset's volatility high?" The tight regularization forced the model to only use features it was extremely confident about, eliminating noise.

#### V2 — Expanded Features + Loosened Regularization (`Part2_Model_Training.ipynb`)
- **New Features (27 total):** Added 5 more engineered features: Relative Volume (volume spike detector), Bollinger %B (position within bands), RSI Deviation (distance from neutral), Volatility-Adjusted Returns, and SMA Crossover Signal.
- **Model:** Slightly loosened regularization (`max_depth=4`, `reg_lambda=3.0`, `gamma=0.5`, `min_child_weight=50`) to allow more features to contribute to the model's decisions.
- **Result:** Broader feature usage across the importance chart, with competitive returns.
- **Why it improved:** The additional features gave the model richer context about volume dynamics and trend structure, while the loosened regularization allowed the model to build more complex decision boundaries.

*Output:* Each version saves signals to Google Drive (`signals_v1.csv`, `signals_v2.csv`, `signals_v3.csv`).

---

### Part 3: Backtesting & Performance Analysis (`Part3_Backtesting.ipynb`)
We simulate the trading strategy over the 2-year out-of-sample test period.

**Backtest Mechanics:**
- **Initial Capital:** $1,000,000
- **Allocation:** Equal-weight across all active Long and Short signals each day.
- **Transaction Costs:** 10 basis points (0.10%) per unit of portfolio turnover.
- **Benchmark:** Equal-Weight Buy & Hold across all 100 assets.

**Key Metrics Calculated:**
- Total Return (Gross & Net of transaction costs)
- Annualized Sharpe Ratio
- Maximum Drawdown
- Average Daily Turnover

*Output:* Performance metrics table and cumulative equity curve chart comparing Strategy (Gross), Strategy (Net), and the Benchmark.

