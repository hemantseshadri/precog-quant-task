import nbformat as nbf

nb = nbf.v4.new_notebook()

md_header = """# Part 2 (V1 - Baseline): Model Training with XGBoost
This is our **first approach** using a basic XGBoost Regressor with the original 14 features.

**Prediction Target:** Regression (5-day forward return).
**Model:** XGBoost Regressor (Default Hyperparameters).
**Purpose:** Baseline model to compare against our improved V2 approach.
"""

code_setup = """import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import glob

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

data_dir = '/content/drive/MyDrive/precog_data/processed_data'
"""

md_loading = """## 1. Data Loading and Target Definition
We load the processed CSVs and define our target variable (5-day forward return)."""

code_loading = """all_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
df_list = []

for file in all_files:
    df = pd.read_csv(file)
    df['Asset'] = os.path.basename(file).replace('.csv', '')
    df['Target_5d'] = df['Ret_5d'].shift(-5)
    df_list.append(df)

full_df = pd.concat(df_list, ignore_index=True)
full_df.dropna(inplace=True)

full_df['Date'] = pd.to_datetime(full_df['Date'])
full_df.sort_values(by=['Date', 'Asset'], inplace=True)
print(f"Total dataset size after dropping NaNs: {full_df.shape}")
"""

md_split = """## 2. Chronological Train/Test Split
80% train / 20% test, split by date to prevent look-ahead bias."""

code_split = """dates = full_df['Date'].unique()
split_idx = int(len(dates) * 0.8)
train_dates = dates[:split_idx]
test_dates = dates[split_idx:]

train_df = full_df[full_df['Date'].isin(train_dates)].copy()
test_df = full_df[full_df['Date'].isin(test_dates)].copy()

features = [
    'Ret_1d', 'Ret_5d', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'SMA_10', 'SMA_50', 'Dist_SMA_50', 'BB_Upper', 'BB_Lower',
    'Vol_20d', 'Vol_SMA_20', 'Vol_ROC'
]
target = 'Target_5d'

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Features used: {len(features)}")
"""

md_train = """## 3. Baseline XGBoost Model
Using standard hyperparameters with minimal tuning as our baseline."""

code_train = """model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.5,
    random_state=42,
    tree_method='hist'
)

print("Training baseline XGBoost...")
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"\\nTest MSE: {mse:.6f}")
print(f"Test R^2: {r2:.6f}")
"""

md_importance = """## 4. Feature Importance"""

code_importance = """importances = model.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feat_imp.plot(kind='bar', color='#3498db')
plt.title('Baseline XGBoost Feature Importance')
plt.ylabel('Relative Importance')
plt.tight_layout()
plt.show()
"""

md_strategy = """## 5. Strategy Logic & Signal Export
Top 20% predicted returns -> Long (+1), Bottom 20% -> Short (-1). Saved to Google Drive."""

code_strategy = """test_df['Predicted_Ret_5d'] = model.predict(test_df[features])

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

# Save to Google Drive
file_path = '/content/drive/MyDrive/signals.csv'
test_df[['Date', 'Asset', 'Close', 'Target_5d', 'Predicted_Ret_5d', 'Signal']].to_csv(file_path, index=False)
print(f"Signals saved successfully to: {file_path}")
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

with open('Part2_V1_Baseline.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Part2_V1_Baseline.ipynb generated successfully!")
