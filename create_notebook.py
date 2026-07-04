import nbformat as nbf

nb = nbf.v4.new_notebook()

md_header = """# Part 2 (Revised): Model Training with LSTM Neural Network
In this notebook, we use the engineered features from Part 1 to train an **LSTM (Long Short-Term Memory)** Deep Learning model.

Unlike XGBoost, which looks at single days in isolation, an LSTM looks at sequential chunks of time (a 10-day lookback window) to understand the *trajectory* of the price before predicting the 5-day forward return.

**Prediction Target:** Regression (5-day forward return).
**Model:** Keras/TensorFlow LSTM.
"""

code_setup = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Mount Google Drive to save the signals directly to your drive
from google.colab import drive
drive.mount('/content/drive')

# Assuming your processed_data folder was uploaded to this path:
# (Update this if your processed_data folder is located somewhere else)
data_dir = '/content/drive/MyDrive/processed_data' 
"""

md_loading = """## 1. Data Loading and Scaling
Deep learning models are highly sensitive to unscaled data. We will use `StandardScaler` to normalize our 14 features."""

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

features = [
    'Ret_1d', 'Ret_5d', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'SMA_10', 'SMA_50', 'Dist_SMA_50', 'BB_Upper', 'BB_Lower',
    'Vol_20d', 'Vol_SMA_20', 'Vol_ROC'
]

# Scale features (Critical for LSTM)
scaler = StandardScaler()
full_df[features] = scaler.fit_transform(full_df[features])

print(f"Total dataset size: {full_df.shape}")
"""

md_seq = """## 2. Sequence Generation & Train/Test Split
An LSTM requires 3D data: `[Samples, TimeSteps, Features]`. We will create a sliding window of 10 days (TimeSteps) for each asset.
We then split chronologically: first 80% for training, last 20% for testing."""

code_seq = """lookback = 10

def create_sequences(asset_df):
    X, y, dates, assets = [], [], [], []
    feature_data = asset_df[features].values
    target_data = asset_df['Target_5d'].values
    date_data = asset_df['Date'].values
    asset_name = asset_df['Asset'].iloc[0]
    
    for i in range(len(asset_df) - lookback):
        X.append(feature_data[i : i + lookback])
        # The target corresponds to the last day of the sequence
        y.append(target_data[i + lookback - 1])
        dates.append(date_data[i + lookback - 1])
        assets.append(asset_name)
        
    return np.array(X), np.array(y), np.array(dates), np.array(assets)

print("Building 3D sequences... (this may take a minute)")
X_list, y_list, d_list, a_list = [], [], [], []

for asset, group in full_df.groupby('Asset'):
    # ensure sorted by date for each asset
    group = group.sort_values('Date')
    X_a, y_a, d_a, a_a = create_sequences(group)
    X_list.append(X_a)
    y_list.append(y_a)
    d_list.append(d_a)
    a_list.append(a_a)

X_all = np.concatenate(X_list)
y_all = np.concatenate(y_list)
dates_all = np.concatenate(d_list)
assets_all = np.concatenate(a_list)

# Split 80/20 chronologically
unique_dates = np.sort(np.unique(dates_all))
split_idx = int(len(unique_dates) * 0.8)
split_date = unique_dates[split_idx]

train_mask = dates_all < split_date
test_mask = dates_all >= split_date

X_train, y_train = X_all[train_mask], y_all[train_mask]
X_test, y_test = X_all[test_mask], y_all[test_mask]

# Save metadata for test set (we need this to generate signals later)
dates_test = dates_all[test_mask]
assets_test = assets_all[test_mask]
close_test = [] # Optional: We don't strictly need close price to generate the raw signal 1 or -1

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
"""

md_train = """## 3. Build & Train the LSTM Model
We construct a simple but robust LSTM architecture."""

code_train = """model = Sequential([
    LSTM(64, input_shape=(lookback, len(features)), return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# Train the model
print("Training LSTM...")
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=256,
    validation_data=(X_test, y_test),
    verbose=1
)

# Predict on Out-of-Sample Test Set
preds = model.predict(X_test).flatten()
"""

md_strategy = """## 4. Generate Signals and Export
We translate the LSTM's forward return predictions into our standard -1, 0, 1 signals, and save them directly to Google Drive."""

code_strategy = """# Reconstruct a DataFrame for the test set
test_df = pd.DataFrame({
    'Date': dates_test,
    'Asset': assets_test,
    'Target_5d': y_test,
    'Predicted_Ret_5d': preds
})

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
test_df['Close'] = 100 # Dummy close value since backtester uses asset_returns logic in Part 3

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
    nbf.v4.new_markdown_cell(md_seq),
    nbf.v4.new_code_cell(code_seq),
    nbf.v4.new_markdown_cell(md_train),
    nbf.v4.new_code_cell(code_train),
    nbf.v4.new_markdown_cell(md_strategy),
    nbf.v4.new_code_cell(code_strategy)
]

with open('Part2_Model_Training.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook Part2_Model_Training.ipynb generated successfully for LSTM!")
