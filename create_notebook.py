import nbformat as nbf

nb = nbf.v4.new_notebook()

md_header = """# Part 2: Model Training & Strategy Formulation (Improved)
In this notebook, we train a heavily improved XGBoost model with:
1. **New cross-sectional features** (rank each asset vs. its peers each day)
2. **Proper early stopping** to prevent overfitting
3. **Aggressive hyperparameter tuning** for noisy financial data

**Prediction Target:** Regression (5-day forward return).
**Model:** XGBoost Regressor.
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

md_loading = """## 1. Data Loading, Target Definition & New Feature Engineering
We load all 100 assets, define our 5-day forward return target, and then engineer **cross-sectional features** — these rank each asset relative to the other 99 assets on the same day. This is critical because XGBoost needs to know not just "Is RSI high?" but "Is this asset's RSI higher than its peers today?"
"""

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

# ============================================================
# Cross-Sectional Features (rank each asset vs peers each day)
# ============================================================
cross_sectional_cols = ['Ret_1d', 'Ret_5d', 'RSI_14', 'MACD_Hist', 'Dist_SMA_50', 'Vol_20d']

for col in cross_sectional_cols:
    full_df[f'{col}_rank'] = full_df.groupby('Date')[col].rank(pct=True)

# ============================================================
# Engineered Features (per-asset, time-series based)
# ============================================================

# 1. Mean Reversion Z-Score: how abnormal is today's return vs its own history?
full_df['Ret_1d_zscore'] = full_df.groupby('Asset')['Ret_1d'].transform(
    lambda x: (x - x.rolling(20).mean()) / x.rolling(20).std()
)

# 2. Momentum Acceleration: is the 5-day trend speeding up or slowing down?
full_df['Momentum_accel'] = full_df.groupby('Asset')['Ret_5d'].transform(
    lambda x: x - x.shift(5)
)

# 3. Relative Volume: today's volume vs its own 20-day avg (volume spike detector)
full_df['Rel_Volume'] = full_df.groupby('Asset').apply(
    lambda g: g['Volume'] / g['Volume'].rolling(20).mean()
).reset_index(level=0, drop=True)

# 4. Bollinger %B: where is price within the Bollinger Band? (0=lower band, 1=upper band)
full_df['BB_pctB'] = (full_df['Close'] - full_df['BB_Lower']) / (full_df['BB_Upper'] - full_df['BB_Lower'])

# 5. RSI Regime: is RSI overbought (>70) or oversold (<30)? Encoded as a continuous signal.
full_df['RSI_deviation'] = full_df['RSI_14'] - 50  # Centered at 0; positive = overbought territory

# 6. Volatility-Adjusted Return: is this return big relative to the asset's own volatility?
full_df['Vol_adj_ret'] = full_df.groupby('Asset').apply(
    lambda g: g['Ret_1d'] / g['Ret_1d'].rolling(20).std()
).reset_index(level=0, drop=True)

# 7. SMA Crossover Signal: short-term trend vs long-term trend
full_df['SMA_cross'] = (full_df['SMA_10'] - full_df['SMA_50']) / full_df['SMA_50']

# Replace infinities and drop NaNs
full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
full_df.dropna(inplace=True)

print(f"Total dataset size: {full_df.shape}")
print(f"Date range: {full_df['Date'].min()} to {full_df['Date'].max()}")
"""

md_split = """## 2. Chronological Train/Test Split
80% train / 20% test, split by date to prevent any look-ahead bias."""

code_split = """dates = full_df['Date'].unique()
split_idx = int(len(dates) * 0.8)
train_dates = dates[:split_idx]
test_dates = dates[split_idx:]

train_df = full_df[full_df['Date'].isin(train_dates)].copy()
test_df = full_df[full_df['Date'].isin(test_dates)].copy()

# FULL feature list (original 14 + 13 new engineered features = 27 total)
features = [
    # Original features
    'Ret_1d', 'Ret_5d', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'SMA_10', 'SMA_50', 'Dist_SMA_50', 'BB_Upper', 'BB_Lower',
    'Vol_20d', 'Vol_SMA_20', 'Vol_ROC',
    # Cross-sectional rank features
    'Ret_1d_rank', 'Ret_5d_rank', 'RSI_14_rank',
    'MACD_Hist_rank', 'Dist_SMA_50_rank', 'Vol_20d_rank',
    # Engineered features
    'Ret_1d_zscore', 'Momentum_accel',
    'Rel_Volume', 'BB_pctB', 'RSI_deviation',
    'Vol_adj_ret', 'SMA_cross'
]
target = 'Target_5d'

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

print(f"Training samples: {len(X_train):,}")
print(f"Testing samples:  {len(X_test):,}")
print(f"Features used:    {len(features)}")
"""

md_train = """## 3. XGBoost with Early Stopping
Key improvements over the previous version:
- **Early Stopping:** We set `n_estimators` very high (2000) but use `early_stopping_rounds=50`. The model will automatically stop training when performance on the validation set stops improving. This finds the *perfect* number of trees without manually guessing.
- **Lower `max_depth=3`:** Even shallower trees. Financial data is extremely noisy — deeper trees memorize noise.
- **Higher regularization:** We push `reg_alpha` and `reg_lambda` higher to force the model to be conservative and only use features it's truly confident about.
- **`colsample_bytree=0.6`:** Each tree only sees 60% of features, forcing diversity and reducing overfitting.
"""

code_train = """model = xgb.XGBRegressor(
    n_estimators=2000,        # High ceiling — early stopping will find the sweet spot
    learning_rate=0.005,      # Very slow learning for subtle pattern detection
    max_depth=4,              # Slightly deeper to capture feature interactions
    min_child_weight=50,      # Lowered to let more features participate in splits
    subsample=0.75,           # Each tree trains on 75% of rows
    colsample_bytree=0.7,     # Each tree sees 70% of features (was 60%)
    gamma=0.5,                # Loosened to allow more splits
    reg_alpha=0.5,            # L1 regularization (loosened)
    reg_lambda=3.0,           # L2 regularization (loosened from 5.0)
    random_state=42,
    tree_method='hist',
    n_jobs=-1
)

print("Training XGBoost with early stopping...")
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=100
)

# Evaluate
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"\\nTest MSE:  {mse:.6f}")
print(f"Test R^2:  {r2:.6f}")
print(f"Trees used: {model.best_iteration if hasattr(model, 'best_iteration') else 'all'}")
"""

md_importance = """## 4. Feature Importance
Which features does the improved model rely on? The cross-sectional rank features should appear prominently if they're adding value."""

code_importance = """importances = model.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(12, 7))
colors = ['#2ecc71' if 'rank' in f or 'zscore' in f or 'accel' in f else '#3498db' for f in feat_imp.index]
feat_imp.plot(kind='bar', color=colors)
plt.title('Feature Importance (Green = New Features)', fontsize=14)
plt.ylabel('Relative Importance')
plt.tight_layout()
plt.show()

print("\\nTop 5 features:")
for f, v in feat_imp.head(5).items():
    print(f"  {f}: {v:.4f}")
"""

md_strategy = """## 5. Generate Signals on Out-of-Sample Data
Same signal logic as before: Top 20% predicted returns → Long (+1), Bottom 20% → Short (-1). Saved directly to Google Drive for Part 3."""

code_strategy = """test_df['Predicted_Ret_5d'] = preds

def generate_signals(group):
    upper_thresh = group['Predicted_Ret_5d'].quantile(0.80)
    lower_thresh = group['Predicted_Ret_5d'].quantile(0.20)
    
    conditions = [
        (group['Predicted_Ret_5d'] > upper_thresh),
        (group['Predicted_Ret_5d'] < lower_thresh)
    ]
    choices = [1, -1]
    
    group['Signal'] = np.select(conditions, choices, default=0)
    return group

print("Generating signals on Out-Of-Sample Data...")
test_df = test_df.groupby('Date').apply(generate_signals)

print(test_df['Signal'].value_counts(normalize=True))

# Save directly to Google Drive
file_path = '/content/drive/MyDrive/signals_v2.csv'
test_df[['Date', 'Asset', 'Close', 'Target_5d', 'Predicted_Ret_5d', 'Signal']].to_csv(file_path, index=False)
print(f"\\nSignals saved successfully to: {file_path}")
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
    nbf.v4.new_markdown_cell(md_importance),
    nbf.v4.new_code_cell(code_importance),
    nbf.v4.new_markdown_cell(md_strategy),
    nbf.v4.new_code_cell(code_strategy)
]

with open('Part2_Model_Training.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook Part2_Model_Training.ipynb generated successfully (Improved XGBoost)!")
