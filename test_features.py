import pandas as pd

def test_features():
    file_path = 'processed_data/Asset_001.csv'
    print(f"Loading {file_path} for testing...\n")
    df = pd.read_csv(file_path)
    
    print("--- DATASET SHAPE ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")
    
    print("--- MISSING VALUES CHECK ---")
    # It is normal to have missing values equal to the rolling window size at the start
    print(df.isnull().sum().to_string())
    print("\n")
    
    print("--- SAMPLE OF ENGINEERED FEATURES (Last 5 Days) ---")
    # Displaying a subset of columns to ensure they calculated properly
    cols_to_show = ['Date', 'Close', 'SMA_50', 'RSI_14', 'BB_Upper', 'MACD']
    print(df[cols_to_show].tail().to_string(index=False))
    
if __name__ == "__main__":
    test_features()
