from src.data_loader import load_data, preprocess_data
import pandas as pd

def main():
    # Load raw data
    train_path = r'E:\DS+ML\AIM3\Week4\Data\train.csv'
    test_path = r'E:\DS+ML\AIM3\Week4\Data\test.csv'
    store_path = r'E:\DS+ML\AIM3\Week4\Data\store.csv'
    train_df, test_df, store_df = load_data(train_path, test_path, store_path)

    # Preprocess data
    train_df, test_df, scaler = preprocess_data(train_df, test_df, store_df)

    # Save processed data
    train_df.to_csv('data/processed_train.csv', index=False)
    test_df.to_csv('data/processed_test.csv', index=False)
    print("Data preprocessing complete. Processed data saved to 'data/'.")

if __name__ == "__main__":
    main()