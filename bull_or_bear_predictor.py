import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import TimeSeriesSplit
import time
import requests
from io import StringIO

# Configuration
SYMBOL = "SPY"
START_DATE = "2000-01-01"
PLOT_DIR = "plots"
CSV_BACKUP_PATH = f"{SYMBOL}_backup.csv"

def download_stock_data(symbol, start_date, max_retries=3):
    """Download stock data with multiple fallback methods"""
    # Method 1: Try yfinance with period="max"
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt+1}/{max_retries}: Downloading {symbol} data via yfinance...")
            df = yf.download(symbol, period="max")
            if not df.empty:
                print(f"Successfully downloaded {len(df)} trading days via yfinance")
                return df[df.index >= pd.to_datetime(start_date)]
        except Exception as e:
            print(f"yfinance download error: {str(e)}")
        time.sleep(2)
    
    # Method 2: Try direct CSV download from Yahoo
    print("Trying direct CSV download from Yahoo Finance...")
    try:
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}?period1=0&period2={int(time.time())}&interval=1d&events=history"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse CSV
        df = pd.read_csv(StringIO(response.text), parse_dates=['Date'], index_col='Date')
        df = df[df.index >= pd.to_datetime(start_date)]
        print(f"Successfully downloaded {len(df)} trading days via direct CSV")
        return df
    except Exception as e:
        print(f"Direct CSV download failed: {str(e)}")
    
    # Method 3: Load from backup if exists
    if os.path.exists(CSV_BACKUP_PATH):
        print(f"Loading data from backup file: {CSV_BACKUP_PATH}")
        return pd.read_csv(CSV_BACKUP_PATH, parse_dates=['Date'], index_col='Date')
    
    print("All download methods failed")
    return None

def main():
    # Create plot directory if not exists
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Step 1: Data Collection with robust handling
    print(f"\n===== Collecting data for {SYMBOL} from {START_DATE} =====")
    df = download_stock_data(SYMBOL, START_DATE)
    
    if df is None or df.empty:
        print("Fatal error: Could not retrieve data. Exiting.")
        return
    
    # Save backup
    df.to_csv(CSV_BACKUP_PATH)
    print(f"Saved data backup to {CSV_BACKUP_PATH}")
    print(f"Last date: {df.index[-1].strftime('%Y-%m-%d')}")
    
    # Step 2: Feature Engineering
    print("\n===== Creating features =====")
    df['DailyReturn'] = (df['Close'] - df['Open']) / df['Open']
    df['VolumeChange'] = df['Volume'].pct_change()
    df['PrevReturn'] = df['DailyReturn'].shift(1)
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Save original count before cleaning
    original_count = len(df)
    df.dropna(inplace=True)
    print(f"Removed {original_count - len(df)} rows with missing values")
    print(f"Final dataset size: {len(df)} rows")
    
    # Prepare features and target
    features = df[['MA5', 'MA10', 'MA20', 'PrevReturn', 'VolumeChange']]
    target = df['Target']
    
    # Time-series split
    print("\n===== Splitting data =====")
    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(features))
    
    # Use the last fold for final evaluation
    train_index, test_index = splits[-1]
    X_train, X_test = features.iloc[train_index], features.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]
    
    print(f"Training period: {df.index[train_index[0]].date()} to {df.index[train_index[-1]].date()}")
    print(f"Testing period: {df.index[test_index[0]].date()} to {df.index[test_index[-1]].date()}")
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    
    # Step 3: Modeling
    print("\n===== Training model =====")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Step 4: Model Evaluation
    print("\n===== Evaluating model =====")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Bear', 'Bull'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.savefig(f'{PLOT_DIR}/confusion_matrix.png', bbox_inches='tight')
    plt.close()
    print("Saved confusion_matrix.png")
    
    # Step 5: Visualizations
    print("\n===== Generating visualizations =====")
    
    # Price and Moving Averages
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Close'], label='Close Price', alpha=0.7)
    plt.plot(df.index, df['MA20'], label='20-day MA', linestyle='--', linewidth=1.5)
    plt.title(f'{SYMBOL} Price and Moving Averages')
    plt.ylabel('Price (USD)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f'{PLOT_DIR}/moving_averages.png', bbox_inches='tight')
    plt.close()
    print("Saved moving_averages.png")
    
    # Prediction Visualization
    test_dates = df.index[test_index]
    plt.figure(figsize=(14, 7))
    plt.plot(test_dates, df.loc[test_dates, 'Close'], label='Close Price', linewidth=2)
    
    # Plot predicted bull/bear markers
    bull_dates = test_dates[y_pred == 1]
    bear_dates = test_dates[y_pred == 0]
    
    if len(bull_dates) > 0:
        plt.scatter(bull_dates, df.loc[bull_dates, 'Close'], 
                    color='green', marker='^', s=60, alpha=0.8, label='Bull Prediction')
    if len(bear_dates) > 0:
        plt.scatter(bear_dates, df.loc[bear_dates, 'Close'], 
                    color='red', marker='v', s=60, alpha=0.8, label='Bear Prediction')
    
    plt.title(f'{SYMBOL} Price with Predictions (Test Period)')
    plt.ylabel('Price (USD)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f'{PLOT_DIR}/predictions_visual.png', bbox_inches='tight')
    plt.close()
    print("Saved predictions_visual.png")
    
    # Feature Importance
    coeffs = pd.Series(model.coef_[0], index=features.columns)
    plt.figure(figsize=(10, 6))
    coeffs.sort_values().plot(kind='barh', color='skyblue')
    plt.title('Feature Importance (Logistic Regression Coefficients)')
    plt.xlabel('Coefficient Value')
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.savefig(f'{PLOT_DIR}/feature_importance.png', bbox_inches='tight')
    plt.close()
    print("Saved feature_importance.png")
    
    # Daily Returns Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['DailyReturn'], bins=100, kde=True)
    plt.title('Distribution of Daily Returns')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f'{PLOT_DIR}/returns_distribution.png', bbox_inches='tight')
    plt.close()
    print("Saved returns_distribution.png")
    
    print("\n===== Execution completed successfully! =====")
    print(f"Visualizations saved to: {os.path.abspath(PLOT_DIR)}")

# MAIN
if __name__ == "__main__":
    main()
