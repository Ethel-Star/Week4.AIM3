import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path: str):
    try:
        df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except pd.errors.ParserError:
        print(f"Error: Parsing failed for file {file_path}. Check the file format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def convert_date_columns(df):
    # Loop through all columns and convert to datetime if possible
    for col in df.columns:
        try:
            # Try to convert column to datetime
            df[col] = pd.to_datetime(df[col], errors='coerce')
        except Exception as e:
            # If it cannot be converted, skip it
            pass
    return df

def detect_missing_values(df: pd.DataFrame, dataset_name="Dataset"):
    """
    Detects missing values in the dataset and returns a summary.
    """
    missing_summary = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Values': missing_summary,
        'Percentage Missing': missing_percentage
    })
    print(f"{dataset_name} - Missing Values Summary:")
    print(missing_df[missing_df['Missing Values'] > 0])

    return missing_df[missing_df['Missing Values'] > 0]

def outlier_detection(df: pd.DataFrame, dataset_name="Dataset"):
    """
    Detects and visualizes outliers using box plots in a single figure with subplots.
    
    :param df: DataFrame with numerical columns
    :param dataset_name: The name of the dataset for labeling purposes
    """
    # Get numerical columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Create a grid of subplots (rows, columns)
    num_plots = len(numeric_cols)
    rows = (num_plots // 3) + (num_plots % 3 > 0)  # Automatically determine the number of rows
    cols = 3  # Set the number of columns to 3 for better arrangement
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()  # Flatten the 2D array of axes to make indexing easier
    
    # Loop through numerical columns and plot each box plot in the respective subplot
    for i, col in enumerate(numeric_cols):
        # Drop null values before plotting
        cleaned_data = df[col].dropna()
        sns.boxplot(y=cleaned_data, ax=axes[i])  # Use 'y' for single-column boxplots
        axes[i].set_title(f'Outlier Detection for {col}')
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f'{dataset_name} - Outlier Detection', fontsize=16)
    plt.show()

def compare_outliers(train_df: pd.DataFrame, store_df: pd.DataFrame):
    """
    Compares outliers for both train and store datasets in a single figure with subplots.
    """
    # Create a figure with two subplots (one for train, one for store)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    
    # Plot train dataset outliers
    numeric_cols_train = train_df.select_dtypes(include=['float64', 'int64']).columns
    for i, col in enumerate(numeric_cols_train):
        sns.boxplot(y=train_df[col].dropna(), ax=axes[0])
        axes[0].set_title('Train Dataset - Outlier Detection')
    
    # Plot store dataset outliers
    numeric_cols_store = store_df.select_dtypes(include=['float64', 'int64']).columns
    for i, col in enumerate(numeric_cols_store):
        sns.boxplot(y=store_df[col].dropna(), ax=axes[1])
        axes[1].set_title('Store Dataset - Outlier Detection')
    
    plt.tight_layout()
    plt.suptitle('Outlier Detection Comparison', fontsize=16)
    plt.show()

def detect_outliers(df: pd.DataFrame, threshold: float = 1.5, dataset_name="Dataset"):
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
    
    print(f"{dataset_name} - Outlier Information:")
    return pd.DataFrame.from_dict(outlier_info, orient='index')

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values in the dataset.
    """
    # Handle CompetitionDistance (Fill with Median)
    if 'CompetitionDistance' in df.columns:
        df['CompetitionDistance'] = df['CompetitionDistance'].fillna(df['CompetitionDistance'].median())
    
    # Handle CompetitionOpenSinceMonth & CompetitionOpenSinceYear (Fill with Mode)
    for col in ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Handle Promo2SinceWeek & Promo2SinceYear (Fill with Mode)
    for col in ['Promo2SinceWeek', 'Promo2SinceYear']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Handle PromoInterval (Replace with 'None')
    if 'PromoInterval' in df.columns:
        df['PromoInterval'] = df['PromoInterval'].fillna('None')
    
    return df

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles outliers in the dataset using IQR.
    """
    def cap_outliers(column: str):
        """
        Caps outliers in a given column using the IQR method.
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    
    # Cap outliers for specific columns
    outlier_columns = ['CompetitionDistance', 'CompetitionOpenSinceYear']
    for col in outlier_columns:
        if col in df.columns:
            cap_outliers(col)
    
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset by handling missing values and outliers.
    """
    df = handle_missing_values(df)
    df = handle_outliers(df)
    
    return df

def visualize_missing_values(train, store):
    """
    Visualizes missing values for both the train and store datasets.
    """
    # Detect missing values for train and store datasets
    missing_train = detect_missing_values(train)
    missing_store = detect_missing_values(store)
    
    # Check if any missing values are found, and handle accordingly
    if missing_train.empty and missing_store.empty:
        print("No missing values detected in both datasets.")
        return
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot missing values for train dataset
    if not missing_train.empty:
        missing_train['Percentage Missing'].plot(kind='bar', ax=axes[0], color='skyblue')
        axes[0].set_title('Missing Values - Train Dataset')
        axes[0].set_ylabel('Percentage Missing')
    
    # Plot missing values for store dataset
    if not missing_store.empty:
        missing_store['Percentage Missing'].plot(kind='bar', ax=axes[1], color='lightgreen')
        axes[1].set_title('Missing Values - Store Dataset')
        axes[1].set_ylabel('Percentage Missing')
    
    plt.tight_layout()
    plt.show()


def visualize_outliers(train, store):
    """
    Visualizes outliers for both the train and store datasets.
    """
    # Detect outliers for train and store datasets
    outliers_train = detect_outliers(train)
    outliers_store = detect_outliers(store)
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot outliers for train dataset
    outliers_train['Total Outliers'].plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Outliers - Train Dataset')
    axes[0].set_ylabel('Total Outliers')
    
    # Plot outliers for store dataset
    outliers_store['Total Outliers'].plot(kind='bar', ax=axes[1], color='lightcoral')
    axes[1].set_title('Outliers - Store Dataset')
    axes[1].set_ylabel('Total Outliers')
    
    plt.tight_layout()
    plt.show()