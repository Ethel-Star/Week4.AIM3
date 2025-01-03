import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path: str):
    try:
        logging.info(f"Attempting to load data from {file_path}.")
        df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
        logging.info(f"Data loaded successfully from {file_path}.")
        return df
    except FileNotFoundError:
        logging.error(f"Error: File not found at {file_path}")
    except pd.errors.ParserError:
        logging.error(f"Error: Parsing failed for file {file_path}. Check the file format.")
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}")

def convert_date_columns(df):
    logging.info("Converting date columns to datetime.")
    # Loop through all columns and convert to datetime if possible
    for col in df.columns:
        try:
            # Try to convert column to datetime
            df[col] = pd.to_datetime(df[col], errors='coerce')
        except Exception as e:
            # If it cannot be converted, skip it
            logging.warning(f"Column {col} could not be converted to datetime: {e}")
    logging.info("Date columns conversion completed.")
    return df

def detect_missing_values(df: pd.DataFrame, dataset_name="Dataset"):
    logging.info(f"Detecting missing values in {dataset_name}.")
    """
    Detects missing values in the dataset and returns a summary.
    """
    missing_summary = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Values': missing_summary,
        'Percentage Missing': missing_percentage
    })
    if missing_df['Missing Values'].sum() > 0:
        logging.warning(f"{dataset_name} has missing values.")
    else:
        logging.info(f"No missing values in {dataset_name}.")
    return missing_df[missing_df['Missing Values'] > 0]

def outlier_detection(df: pd.DataFrame, dataset_name="Dataset"):
    logging.info(f"Performing outlier detection for {dataset_name}.")
    """
    Detects and visualizes outliers using box plots in a single figure with subplots.
    """
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    num_plots = len(numeric_cols)
    rows = (num_plots // 3) + (num_plots % 3 > 0)  
    cols = 3  
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()  
    
    for i, col in enumerate(numeric_cols):
        cleaned_data = df[col].dropna()
        sns.boxplot(y=cleaned_data, ax=axes[i])  
        axes[i].set_title(f'Outlier Detection for {col}')
    
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f'{dataset_name} - Outlier Detection', fontsize=16)
    plt.show()
    logging.info(f"Outlier detection for {dataset_name} completed.")

def compare_outliers(train_df: pd.DataFrame, store_df: pd.DataFrame):
    logging.info("Comparing outliers between train and store datasets.")
    """
    Compares outliers for both train and store datasets in a single figure with subplots.
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    
    numeric_cols_train = train_df.select_dtypes(include=['float64', 'int64']).columns
    for i, col in enumerate(numeric_cols_train):
        sns.boxplot(y=train_df[col].dropna(), ax=axes[0])
        axes[0].set_title('Train Dataset - Outlier Detection')
    
    numeric_cols_store = store_df.select_dtypes(include=['float64', 'int64']).columns
    for i, col in enumerate(numeric_cols_store):
        sns.boxplot(y=store_df[col].dropna(), ax=axes[1])
        axes[1].set_title('Store Dataset - Outlier Detection')
    
    plt.tight_layout()
    plt.suptitle('Outlier Detection Comparison', fontsize=16)
    plt.show()
    logging.info("Outlier comparison between train and store datasets completed.")

def detect_outliers(df: pd.DataFrame, threshold: float = 1.5, dataset_name="Dataset"):
    logging.info(f"Detecting outliers using IQR method for {dataset_name}.")
    """
    Detects outliers based on the IQR method.
    """
    outlier_info = {}
    
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_info[col] = {
            'Total Outliers': len(outliers),
            'Lower Bound': lower_bound,
            'Upper Bound': upper_bound
        }
    
    logging.info(f"Outlier detection for {dataset_name} completed.")
    return pd.DataFrame.from_dict(outlier_info, orient='index')

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Handling missing values.")
    """
    Handles missing values in the dataset.
    """
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
    logging.info("Handling outliers.")
    """
    Handles outliers in the dataset using IQR.
    """
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


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Preprocessing data by handling missing values and outliers.")
    """
    Preprocess the dataset by handling missing values and outliers.
    """
    df = handle_missing_values(df)
    df = handle_outliers(df)
    logging.info("Preprocessing completed.")
    
    return df

def visualize_missing_values(train, store):
    logging.info("Visualizing missing values for train and store datasets.")
    """
    Visualizes missing values for both the train and store datasets.
    """
    missing_train = detect_missing_values(train)
    missing_store = detect_missing_values(store)
    
    if missing_train.empty and missing_store.empty:
        logging.info("No missing values detected in both datasets.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    if not missing_train.empty:
        missing_train['Percentage Missing'].plot(kind='bar', ax=axes[0], color='skyblue')
        axes[0].set_title('Missing Values - Train Dataset')
        axes[0].set_ylabel('Percentage Missing')
    
    if not missing_store.empty:
        missing_store['Percentage Missing'].plot(kind='bar', ax=axes[1], color='lightgreen')
        axes[1].set_title('Missing Values - Store Dataset')
        axes[1].set_ylabel('Percentage Missing')
    
    plt.tight_layout()
    plt.show()
    logging.info("Missing values visualized.")

def visualize_outliers(train, store):
    logging.info("Visualizing outliers for train and store datasets.")
    """
    Visualizes outliers for both the train and store datasets.
    """
    outliers_train = detect_outliers(train)
    outliers_store = detect_outliers(store)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    outliers_train['Total Outliers'].plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Outliers - Train Dataset')
    axes[0].set_ylabel('Total Outliers')
    
    outliers_store['Total Outliers'].plot(kind='bar', ax=axes[1], color='lightcoral')
    axes[1].set_title('Outliers - Store Dataset')
    axes[1].set_ylabel('Total Outliers')
    
    plt.tight_layout()
    plt.show()
    logging.info("Outliers visualized.")
