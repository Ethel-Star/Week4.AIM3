from src.data_loader import load_data, preprocess_data
from src.model import train_random_forest, train_lstm
import pandas as pd

def main():
    # Load and preprocess data
    train_path = r'E:\DS+ML\AIM3\Week4\Data\train.csv'
    test_path = r'E:\DS+ML\AIM3\Week4\Data\test.csv'
    store_path = r'E:\DS+ML\AIM3\Week4\Data\store.csv'
    train_df, test_df, store_df = load_data(train_path, test_path, store_path)
    train_df, test_df, scaler = preprocess_data(train_df, test_df, store_df)

    # Train RandomForestRegressor
    rf_pipeline = train_random_forest(train_df)

    # Train LSTM
    lstm_model, lstm_scaler, window_size = train_lstm(train_df)

if __name__ == "__main__":
    main()