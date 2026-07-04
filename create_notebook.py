import nbformat as nbf

nb = nbf.v4.new_notebook()

md_header = """# Part 2: Model Training & Strategy Formulation
In this notebook, we will use the engineered features from Part 1 to train an advanced XGBoost predictive model.

**Prediction Target:** Regression. We will predict the 5-day forward return.
**Model:** XGBoost Regressor (Tuned Hyperparameters).
"""

code_setup = """import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import glob

# Mount Google Drive to save the signals directly to your drive
from google.colab import drive
drive.mount('/content/drive')

# Assuming your processed_data folder was uploaded to this path:
data_dir = '/content/drive/MyDrive/processed_data' 
"""

md_loading = """## 1. Data Loading and Target Definition
We load the processed CSVs, and define our target variable. We shift the `Ret_5d` column backwards by 5 days."""

code_loading = """all_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
df_list = []

for file in all_files:
    df = pd.read_csv(file)
    df['Asset'] = os.path.basename(file).replace('.csv', '')
    
    # Define Target: Shift Ret_5d backwards by 5 days
    df['Target_5d'] = df['Ret_5d'].shift(-5)
    df_list.append(df)

full_df = pd.concat(df_list, ignore_index=True)
full_df.dropna(inplace=True)

# Sort chronologically
full_df['Date'] = pd.to_datetime(full_df['Date'])
full_df.sort_values(by=['Date', 'Asset'], inplace=True)
print(f"Total dataset size after dropping NaNs: {full_df.shape}")
"""

md_split = """## 2. Chronological Train/Test Split
We will train on the first 80% of dates, and test on the remaining 20% to prevent look-ahead bias."""

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
"""

md_train = """## 3. Tuned XGBoost Regression
To beat the market, we are heavily tuning the XGBoost hyperparameters to extract maximum alpha from the features:
- **`learning_rate=0.01` & `n_estimators=500`:** A slower learning rate with more trees allows the model to learn complex, subtle patterns without overfitting.
- **`max_depth=4`:** Shallow trees prevent the model from memorizing financial noise.
- **`reg_alpha` (L1) & `reg_lambda` (L2):** Strong regularization to aggressively punish useless features and prevent the model from becoming overconfident."""

code_train = """model = xgb.XGBRegressor(
    n_estimators=500,        # More trees
    learning_rate=0.01,      # Slower learning
    max_depth=4,             # Shallow trees to prevent overfitting noise
    subsample=0.8,           # Stochastic sampling of rows
    colsample_bytree=0.8,    # Stochastic sampling of columns
    reg_alpha=0.5,           # L1 Regularization (Lasso)
    reg_lambda=1.5,          # L2 Regularization (Ridge)
    random_state=42,
    tree_method='hist'       # Uses GPU if available
)

print("Training highly-tuned XGBoost...")
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

# Evaluate
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"\\nTest MSE: {mse:.6f}")
print(f"Test R^2: {r2:.6f}")
"""

md_importance = """## 4. Feature Importance
Let's see which features the tuned model relies on."""

code_importance = """importances = model.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feat_imp.plot(kind='bar')
plt.title('Tuned XGBoost Feature Importance')
plt.ylabel('Relative Importance')
plt.show()
"""

md_strategy = """## 5. Strategy Logic: Translating Predictions to Signals
We translate the refined predictions into our standard -1, 0, 1 signals (Top 20% Long, Bottom 20% Short) and export to Google Drive."""

code_strategy = """# Generate predictions ONLY for the 20% Out-Of-Sample Test Set
test_df['Predicted_Ret_5d'] = model.predict(test_df[features])

def generate_signals(group):
    # Calculate quantiles for each day
    upper_thresh = group['Predicted_Ret_5d'].quantile(0.80)
    lower_thresh = group['Predicted_Ret_5d'].quantile(0.20)
    
    conditions = [
        (group['Predicted_Ret_5d'] > upper_thresh),
        (group['Predicted_Ret_5d'] < lower_thresh)
    ]
    choices = [1, -1]
    
    group['Signal'] = np.select(conditions, choices, default=0)
    return group

# Apply logic day-by-day
print("Generating signals on Out-Of-Sample Data...")
test_df = test_df.groupby('Date').apply(generate_signals)

# Check signal distribution
print(test_df['Signal'].value_counts(normalize=True))

# Save directly to Google Drive!
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

with open('Part2_Model_Training.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook Part2_Model_Training.ipynb generated successfully for Tuned XGBoost!")
