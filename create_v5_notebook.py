import nbformat as nbf

nb = nbf.v4.new_notebook()

md_header = """# Part 2 (V5): V3 Model, 30% Long/Short
This version uses the **exact same XGBoost model and features as V3** (tight regularization, 22 features).
However, it changes the signal generation to trade **30% Long / 30% Short** every day.

**Prediction Target:** Regression (5-day forward return).
**Model:** XGBoost Regressor (Heavy Regularization).
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

md_loading = """## 1. Data Loading, Target Definition & Cross-Sectional Features
We rank each asset relative to its peers on the same day and add mean reversion + momentum signals."""

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

# Mean Reversion Z-Score
full_df['Ret_1d_zscore'] = full_df.groupby('Asset')['Ret_1d'].transform(
    lambda x: (x - x.rolling(20).mean()) / x.rolling(20).std()
)

# Momentum Acceleration
full_df['Momentum_accel'] = full_df.groupby('Asset')['Ret_5d'].transform(
    lambda x: x - x.shift(5)
)

full_df.dropna(inplace=True)

print(f"Total dataset size: {full_df.shape}")
print(f"Date range: {full_df['Date'].min()} to {full_df['Date'].max()}")
"""

md_split = """## 2. Chronological Train/Test Split
80% train / 20% test, split by date."""

code_split = """dates = full_df['Date'].unique()
split_idx = int(len(dates) * 0.8)
train_dates = dates[:split_idx]
test_dates = dates[split_idx:]

train_df = full_df[full_df['Date'].isin(train_dates)].copy()
test_df = full_df[full_df['Date'].isin(test_dates)].copy()

# Feature list (original 14 + 8 cross-sectional/engineered = 22 total)
features = [
    # Original features
    'Ret_1d', 'Ret_5d', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'SMA_10', 'SMA_50', 'Dist_SMA_50', 'BB_Upper', 'BB_Lower',
    'Vol_20d', 'Vol_SMA_20', 'Vol_ROC',
    # Cross-sectional rank features
    'Ret_1d_rank', 'Ret_5d_rank', 'RSI_14_rank',
    'MACD_Hist_rank', 'Dist_SMA_50_rank', 'Vol_20d_rank',
    # Engineered features
    'Ret_1d_zscore', 'Momentum_accel'
]
target = 'Target_5d'

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

print(f"Training samples: {len(X_train):,}")
print(f"Testing samples:  {len(X_test):,}")
print(f"Features used:    {len(features)}")
"""

md_train = """## 3. XGBoost with Aggressive Regularization
Very tight regularization forces the model to only use features it is extremely confident about."""

code_train = """model = xgb.XGBRegressor(
    n_estimators=2000,
    learning_rate=0.005,
    max_depth=3,              # Very shallow trees
    min_child_weight=100,     # Each leaf needs 100+ samples
    subsample=0.7,
    colsample_bytree=0.6,    # Each tree sees only 60% of features
    gamma=1.0,                # Aggressive pruning of weak splits
    reg_alpha=1.0,            # Strong L1 regularization
    reg_lambda=5.0,           # Very strong L2 regularization
    random_state=42,
    tree_method='hist',
    n_jobs=-1
)

print("Training XGBoost (tight regularization)...")
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

md_importance = """## 4. Feature Importance"""

code_importance = """importances = model.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(12, 7))
colors = ['#2ecc71' if 'rank' in f or 'zscore' in f or 'accel' in f else '#3498db' for f in feat_imp.index]
feat_imp.plot(kind='bar', color=colors)
plt.title('Feature Importance - V3 Tight Regularization (Green = New Features)', fontsize=14)
plt.ylabel('Relative Importance')
plt.tight_layout()
plt.show()

print("\\nTop 5 features:")
for f, v in feat_imp.head(5).items():
    print(f"  {f}: {v:.4f}")
"""

md_strategy = """## 5. Generate Signals on Out-of-Sample Data
Top 30% Long (+1), Bottom 30% Short (-1). Saved to Google Drive as signals_v5.csv."""

code_strategy = """test_df['Predicted_Ret_5d'] = preds

def generate_signals(group):
    # CHANGED TO 30% LONG / 30% SHORT
    upper_thresh = group['Predicted_Ret_5d'].quantile(0.70)
    lower_thresh = group['Predicted_Ret_5d'].quantile(0.30)
    
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

# Save to Google Drive
file_path = '/content/drive/MyDrive/signals_v5.csv'
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

with open('Part2_V5_TightReg_30pct.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Part2_V5_TightReg_30pct.ipynb generated successfully!")
