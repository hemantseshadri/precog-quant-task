import nbformat as nbf

nb = nbf.v4.new_notebook()

md_header = """# Part 2 (V4): Same Model as V3, Reduced Trading Frequency
This version uses the **exact same XGBoost model and features as V3**, but changes the signal generation:
- **10% Long / 10% Short** (highly concentrated positions)
- **Rebalance only 3 days per week** (Mon, Wed, Fri) to drastically reduce turnover and transaction costs.

The model predictions are identical to V3 — only the portfolio construction rules have changed.
"""

code_setup = """import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

data_dir = '/content/drive/MyDrive/precog_data/processed_data'
"""

md_loading = """## 1. Data Loading & Cross-Sectional Features (Same as V3)"""

code_loading = """all_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
df_list = []

for file in all_files:
    df = pd.read_csv(file)
    df['Asset'] = os.path.basename(file).replace('.csv', '')
    df['Target_5d'] = df['Ret_5d'].shift(-5)
    df_list.append(df)

full_df = pd.concat(df_list, ignore_index=True)
full_df['Date'] = pd.to_datetime(full_df['Date'])
full_df.sort_values(by=['Date', 'Asset'], inplace=True)

# Cross-Sectional Features (same as V3)
cross_sectional_cols = ['Ret_1d', 'Ret_5d', 'RSI_14', 'MACD_Hist', 'Dist_SMA_50', 'Vol_20d']
for col in cross_sectional_cols:
    full_df[f'{col}_rank'] = full_df.groupby('Date')[col].rank(pct=True)

full_df['Ret_1d_zscore'] = full_df.groupby('Asset')['Ret_1d'].transform(
    lambda x: (x - x.rolling(20).mean()) / x.rolling(20).std()
)
full_df['Momentum_accel'] = full_df.groupby('Asset')['Ret_5d'].transform(
    lambda x: x - x.shift(5)
)

full_df.dropna(inplace=True)
print(f"Total dataset size: {full_df.shape}")
"""

md_split = """## 2. Train/Test Split (Same as V3)"""

code_split = """dates = full_df['Date'].unique()
split_idx = int(len(dates) * 0.8)
train_dates = dates[:split_idx]
test_dates = dates[split_idx:]

train_df = full_df[full_df['Date'].isin(train_dates)].copy()
test_df = full_df[full_df['Date'].isin(test_dates)].copy()

features = [
    'Ret_1d', 'Ret_5d', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'SMA_10', 'SMA_50', 'Dist_SMA_50', 'BB_Upper', 'BB_Lower',
    'Vol_20d', 'Vol_SMA_20', 'Vol_ROC',
    'Ret_1d_rank', 'Ret_5d_rank', 'RSI_14_rank',
    'MACD_Hist_rank', 'Dist_SMA_50_rank', 'Vol_20d_rank',
    'Ret_1d_zscore', 'Momentum_accel'
]
target = 'Target_5d'

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

print(f"Training samples: {len(X_train):,}")
print(f"Testing samples:  {len(X_test):,}")
"""

md_train = """## 3. XGBoost Model (Same as V3 — Tight Regularization)"""

code_train = """model = xgb.XGBRegressor(
    n_estimators=2000,
    learning_rate=0.005,
    max_depth=3,
    min_child_weight=100,
    subsample=0.7,
    colsample_bytree=0.6,
    gamma=1.0,
    reg_alpha=1.0,
    reg_lambda=5.0,
    random_state=42,
    tree_method='hist',
    n_jobs=-1
)

print("Training XGBoost (same as V3)...")
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=100
)

preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"\\nTest MSE:  {mse:.6f}")
print(f"Test R^2:  {r2:.6f}")
"""

md_strategy = """## 4. V4 Signal Logic: 10% Long/Short + 3 Days Per Week
**What's different from V3:**
- Thresholds changed from Top/Bottom 20% to **Top/Bottom 10%** (more concentrated, higher conviction positions)
- Signals are only generated on **Mon, Wed, Fri** (rebalance 3x per week instead of daily)
- On non-trading days, the signal is set to 0 (hold cash / flat)"""

code_strategy = """test_df['Predicted_Ret_5d'] = preds

# Only generate signals on Monday (0), Wednesday (2), and Friday (4)
trading_days = [0, 2, 4]
test_df['DayOfWeek'] = test_df['Date'].dt.dayofweek

def generate_signals_v4(group):
    day_of_week = group['DayOfWeek'].iloc[0]
    
    # Only trade on Mon, Wed, Fri
    if day_of_week not in trading_days:
        group['Signal'] = 0
        return group
    
    # 10% Long / 10% Short (highly concentrated)
    upper_thresh = group['Predicted_Ret_5d'].quantile(0.90)
    lower_thresh = group['Predicted_Ret_5d'].quantile(0.10)
    
    conditions = [
        (group['Predicted_Ret_5d'] > upper_thresh),
        (group['Predicted_Ret_5d'] < lower_thresh)
    ]
    choices = [1, -1]
    
    group['Signal'] = np.select(conditions, choices, default=0)
    return group

print("Generating V4 signals (10% L/S, 3 days/week)...")
test_df = test_df.groupby('Date').apply(generate_signals_v4)

print("\\nSignal distribution:")
print(test_df['Signal'].value_counts(normalize=True))

# Count actual trading days vs total days
total_days = test_df['Date'].nunique()
active_days = test_df[test_df['Signal'] != 0]['Date'].nunique()
print(f"\\nTrading days: {active_days} out of {total_days} ({active_days/total_days:.1%})")

# Save to Google Drive
file_path = '/content/drive/MyDrive/signals_v4.csv'
test_df[['Date', 'Asset', 'Close', 'Target_5d', 'Predicted_Ret_5d', 'Signal']].to_csv(file_path, index=False)
print(f"\\nSignals saved to: {file_path}")
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(md_header),
    nbf.v4.new_code_cell(code_setup),
    nbf.v4.new_markdown_cell(md_loading),
    nbf.v4.new_code_cell(code_loading),
    nbf.v4.new_markdown_cell(md_split),
    nbf.v4.new_code_cell(code_split),
    nbf.v4.new_markdown_cell(md_train),
    nbf.v4.new_code_cell(code_train),
    nbf.v4.new_markdown_cell(md_strategy),
    nbf.v4.new_code_cell(code_strategy)
]

with open('Part2_V4_ReducedFreq.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Part2_V4_ReducedFreq.ipynb generated successfully!")
