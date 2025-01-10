import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(train_path, test_path, store_path):
    """
    Load datasets from the provided paths.
    
    Args:
        train_path (str): Path to the training data.
        test_path (str): Path to the test data.
        store_path (str): Path to the store data.
    
    Returns:
        tuple: (train_df, test_df, store_df)
    """
    train_df = pd.read_csv(train_path, dtype={'StateHoliday': str})
    test_df = pd.read_csv(test_path)
    store_df = pd.read_csv(store_path)
    return train_df, test_df, store_df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    logging.info("Handling missing values.")
    
    if 'CompetitionDistance' in df.columns:
        df['CompetitionDistance'] = df['CompetitionDistance'].fillna(df['CompetitionDistance'].median())
        logging.info("Filled missing values in CompetitionDistance with median.")
    
    for col in ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
            logging.info(f"Filled missing values in {col} with mode.")
    
    for col in ['Promo2SinceWeek', 'Promo2SinceYear']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
            logging.info(f"Filled missing values in {col} with mode.")
    
    if 'PromoInterval' in df.columns:
        df['PromoInterval'] = df['PromoInterval'].fillna('None')
        logging.info("Filled missing values in PromoInterval with 'None'.")
    
    return df

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles outliers in the dataset using IQR.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with outliers handled.
    """
    logging.info("Handling outliers.")
    
    def cap_outliers(column: str):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    
    outlier_columns = ['CompetitionDistance', 'CompetitionOpenSinceYear']
    for col in outlier_columns:
        if col in df.columns:
            cap_outliers(col)
            logging.info(f"Capped outliers in {col}.")
    
    return df

def preprocess_data(train_df, test_df, store_df):
    """
    Preprocess the datasets.
    
    Args:
        train_df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Test data.
        store_df (pd.DataFrame): Store data.
    
    Returns:
        tuple: (train_df, test_df, scaler)
    """
    # Handle missing values
    train_df = handle_missing_values(train_df)
    test_df = handle_missing_values(test_df)
    store_df = handle_missing_values(store_df)
    
    # Handle outliers
    train_df = handle_outliers(train_df)
    test_df = handle_outliers(test_df)
    store_df = handle_outliers(store_df)
    
    # Feature extraction from 'Date' column
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    test_df['Date'] = pd.to_datetime(test_df['Date'])

    train_df['Weekday'] = train_df['Date'].dt.day_name()
    test_df['Weekday'] = test_df['Date'].dt.day_name()

    train_df['Weekend'] = train_df['Date'].dt.dayofweek.isin([5, 6]).astype(int)
    test_df['Weekend'] = test_df['Date'].dt.dayofweek.isin([5, 6]).astype(int)

    train_df['MonthSegment'] = pd.cut(train_df['Date'].dt.day, bins=[0, 10, 20, 31], labels=['beginning', 'mid', 'end'])
    test_df['MonthSegment'] = pd.cut(test_df['Date'].dt.day, bins=[0, 10, 20, 31], labels=['beginning', 'mid', 'end'])

    # Days to Holidays
    holiday_dates = train_df[(train_df['StateHoliday'] != '0') | (train_df['SchoolHoliday'] == 1)]['Date'].unique()
    holiday_dates = pd.to_datetime(holiday_dates)
    holiday_dates = np.sort(holiday_dates)

    def days_to_next_holiday(date, holiday_dates):
        next_holiday = holiday_dates[holiday_dates >= date]
        return (next_holiday[0] - date).days if len(next_holiday) > 0 else np.nan

    train_df['DaysToHoliday'] = train_df['Date'].apply(lambda x: days_to_next_holiday(x, holiday_dates))
    test_df['DaysToHoliday'] = test_df['Date'].apply(lambda x: days_to_next_holiday(x, holiday_dates))
    train_df['DaysToHoliday'] = train_df['DaysToHoliday'].fillna(365)
    test_df['DaysToHoliday'] = test_df['DaysToHoliday'].fillna(365)

    # Days After Holidays
    def days_since_last_holiday(date, holiday_dates):
        last_holiday = holiday_dates[holiday_dates <= date]
        return (date - last_holiday[-1]).days if len(last_holiday) > 0 else np.nan

    train_df['DaysAfterHoliday'] = train_df['Date'].apply(lambda x: days_since_last_holiday(x, holiday_dates))
    test_df['DaysAfterHoliday'] = test_df['Date'].apply(lambda x: days_since_last_holiday(x, holiday_dates))
    train_df['DaysAfterHoliday'] = train_df['DaysAfterHoliday'].fillna(365)
    test_df['DaysAfterHoliday'] = test_df['DaysAfterHoliday'].fillna(365)

    # Merge datasets
    train_df = train_df.merge(store_df, on='Store', how='left')
    test_df = test_df.merge(store_df, on='Store', how='left')

    # Scale numeric features
    numeric_cols = ['CompetitionDistance', 'Promo2SinceWeek', 'Promo2SinceYear']
    scaler = StandardScaler()
    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

    return train_df, test_df, scaler