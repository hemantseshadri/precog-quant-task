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
*(Note: Part 2, Part 3, and Part 4 will be updated here as the project progresses).*
