import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
sns.set_palette("Set2")
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def plot_promotion_distribution(train, test):
    """
    Plot the distribution of promotions in the training and test datasets.
    """
    logging.info("Plotting promotion distribution for training and test datasets.")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Distribution of promotions in training set
    sns.countplot(data=train, x='Promo', hue='Promo', palette='Set1', ax=axes[0], legend=False)
    axes[0].set_title('Promotion Distribution in Training Set')

    # Distribution of promotions in test set
    sns.countplot(data=test, x='Promo', hue='Promo', palette='Set2', ax=axes[1], legend=False)
    axes[1].set_title('Promotion Distribution in Test Set')

    plt.tight_layout()
    plt.show()

    logging.info("Promotion distribution plot completed.")


def analyze_sales_holidays(train):
    """
    Analyze and visualize sales behavior during different holiday periods using the 'train' DataFrame.
    """
    logging.info("Analyzing sales during holidays.")

    # Ensure 'StateHoliday' column is consistent
    train['StateHoliday'] = train['StateHoliday'].fillna('0').astype(str)

    # Compute average sales for each holiday state
    sales_holiday = train.groupby(['StateHoliday', 'SchoolHoliday'])['Sales'].mean().reset_index()

    # Create a figure with two subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

    # Bar plot for average sales during holidays
    sns.barplot(data=sales_holiday, x='StateHoliday', y='Sales', hue='SchoolHoliday', palette='muted', ax=axes[0])
    axes[0].set_title('Average Sales During Holidays')
    axes[0].set_xlabel('Holiday State')
    axes[0].set_ylabel('Average Sales')
    axes[0].legend(title='School Holiday')

    # Box plot for sales distribution during holidays
    sns.boxplot(x='StateHoliday', y='Sales', data=train, ax=axes[1])
    axes[1].set_title('Sales Distribution During Holidays')
    axes[1].set_xlabel('Holiday State')
    axes[1].set_ylabel('Sales')

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()

    logging.info("Sales analysis during holidays completed.")


def plot_monthly_sales_trend(train):
    """
    Plot the trend of average sales by month.
    """
    logging.info("Plotting monthly sales trend.")
    # Convert Date column to datetime format
    train['Date'] = pd.to_datetime(train['Date'])

    # Extract Month and Day
    train['Month'] = train['Date'].dt.month
    train['Day'] = train['Date'].dt.day

    # Aggregate sales by Month
    monthly_sales = train.groupby('Month')['Sales'].mean().reset_index()

    # Plot Monthly Sales Trend
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_sales['Month'], monthly_sales['Sales'], marker='o', linestyle='-')
    plt.title('Average Sales by Month (All Years Combined)')
    plt.xlabel('Month')
    plt.ylabel('Average Sales')
    plt.xticks(range(1, 13))
    plt.grid(True)
    plt.show()

    logging.info("Monthly sales trend plotted.")


def analyze_sales_on_key_dates(train):
    """
    Analyze and plot sales on key holiday dates like Christmas, Black Friday, and Easter.
    """
    logging.info("Analyzing sales on key holiday dates.")
    # Ensure 'Date' column is in datetime format
    train['Date'] = pd.to_datetime(train['Date'])

    # Identify key holiday dates (e.g., Christmas, Black Friday, Easter)
    train['DayMonth'] = train['Date'].dt.strftime('%m-%d')
    key_dates = ['12-25', '11-29', '04-20']  # Example: Christmas, Black Friday, Easter

    # Map key dates to holiday names
    holiday_map = {
        '12-25': 'Christmas',
        '11-29': 'Black Friday',
        '04-20': 'Easter'
    }

    # Filter holiday sales and create an explicit copy
    holiday_sales = train[train['DayMonth'].isin(key_dates)].copy()

    # Add holiday name as a new column
    holiday_sales['Holiday'] = holiday_sales['DayMonth'].map(holiday_map)

    # Average sales on key dates
    key_date_sales = holiday_sales.groupby('DayMonth')['Sales'].mean().reset_index()

    # Define a vibrant color palette with gradients
    colors = ['#FF6F61', '#6B5B95', '#88B04B']  # Coral, Purple, Green

    # Create a plot with more flair
    plt.figure(figsize=(12, 6))

    # Create a bar chart with customized color
    bars = plt.bar(key_date_sales['DayMonth'], key_date_sales['Sales'], color=colors)

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 2, f'{height:.2f}', 
                 ha='center', va='bottom', fontsize=12, color='black')

    # Annotate holidays at the bottom of each bar
    for i, bar in enumerate(bars):
        holiday_name = holiday_map[key_date_sales['DayMonth'].iloc[i]]
        plt.text(bar.get_x() + bar.get_width() / 2, -5, holiday_name, 
                 ha='center', va='top', fontsize=12, color='blue', fontweight='bold')

    # Enhance the plot with vibrant styling
    plt.title('Average Sales on Key Holiday Dates', fontsize=18, fontweight='bold', color='#2E4053')
    plt.xlabel('Holiday Dates', fontsize=14, fontweight='bold', color='#34495E')
    plt.ylabel('Average Sales ($)', fontsize=14, fontweight='bold', color='#34495E')

    # Customize tick labels and grid style
    plt.xticks(fontsize=12, rotation=45, color='#34495E')
    plt.yticks(fontsize=12, color='#34495E')
    plt.grid(axis='y', linestyle='--', alpha=0.7, color='#BDC3C7')

    # Add a soft background color to the plot
    plt.gca().set_facecolor('#F7F7F7')

    # Adjust figure layout for better spacing
    plt.tight_layout()

    # Remove top and right spines for a cleaner look
    sns.despine()

    # Show the plot
    plt.show()

    logging.info("Sales analysis on key holiday dates completed.")


def plot_sales_vs_customers(train):
    """
    Plot scatter plot and calculate correlation between Sales and Customers.
    """
    try:
        # Correlation coefficient
        correlation = train['Sales'].corr(train['Customers'])
        logging.info(f"Correlation between Sales and Customers: {correlation}")

        # Scatter plot with enhanced color
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='Customers', y='Sales', data=train, color='purple', alpha=0.7, s=100, edgecolor='black')
        plt.title('Sales vs. Customers', fontsize=16, color='darkblue')
        plt.xlabel('Customers', fontsize=12, color='darkgreen')
        plt.ylabel('Sales', fontsize=12, color='darkgreen')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
    except Exception as e:
        logging.error(f"Error in plot_sales_vs_customers: {e}")

def plot_promo_effectiveness(train):
    """
    Calculate promo sales impact and visualize top-performing stores in promo campaigns.
    """
    try:
        # Calculate promo sales impact
        promo_effectiveness = train.groupby(['Store', 'Promo'])['Sales'].mean().unstack().reset_index()
        promo_effectiveness['Promo_Impact'] = (promo_effectiveness[1] - promo_effectiveness[0]) / promo_effectiveness[0]

        # Identify top and bottom-performing stores in promo campaigns
        top_promo_stores = promo_effectiveness.sort_values(by='Promo_Impact', ascending=False).head(10)
        low_promo_stores = promo_effectiveness.sort_values(by='Promo_Impact').head(10)

        # Convert 'Store' column to categorical type for proper handling by seaborn
        top_promo_stores['Store'] = top_promo_stores['Store'].astype('category')

        # Visualize promo effectiveness with more vibrant color, specify hue to avoid FutureWarning
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Store', y='Promo_Impact', data=top_promo_stores, palette='viridis', edgecolor='black', hue='Store')
        plt.title('Top 10 Stores with Highest Promo Impact', fontsize=16, color='darkblue')
        plt.xlabel('Store', fontsize=12, color='darkgreen')
        plt.ylabel('Promo Impact', fontsize=12, color='darkgreen')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Log success message
        logging.info("Promo effectiveness plot created successfully.")
        
    except Exception as e:
        logging.error(f"Error in plot_promo_effectiveness: {e}")


def plot_customer_trends_by_day(train):
    """
    Analyze and visualize customer trends by day of the week and promo status.
    """
    try:
        # Extract Day of the Week
        train['DayOfWeek'] = train['Date'].dt.dayofweek

        # Group data by DayOfWeek and Promo status
        customer_trend = train.groupby(['DayOfWeek', 'Promo'])['Customers'].mean().reset_index()

        # Visualize customer behavior with a brighter color palette
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=customer_trend, x='DayOfWeek', y='Customers', hue='Promo', palette='Set1', linewidth=2.5)
        plt.title('Customer Trends by Day of Week and Promo Status', fontsize=16, color='darkblue')
        plt.xlabel('Day of the Week (0=Monday, 6=Sunday)', fontsize=12, color='darkgreen')
        plt.ylabel('Average Number of Customers', fontsize=12, color='darkgreen')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()
    except Exception as e:
        logging.error(f"Error in plot_customer_trends_by_day: {e}")

def plot_weekend_vs_weekday_sales(train):
    """
    Compare weekend sales vs weekday sales for stores open all weekdays.
    """
    try:
        # Identify stores open all weekdays
        weekday_open_stores = train.groupby('Store')['DayOfWeek'].nunique().reset_index()
        weekday_open_stores = weekday_open_stores[weekday_open_stores['DayOfWeek'] == 7]

        # Compare weekend sales
        train['IsWeekend'] = train['DayOfWeek'].isin([5, 6])
        sales_by_weekend = train[train['Store'].isin(weekday_open_stores['Store'])].groupby('IsWeekend')['Sales'].mean()

        # Visualize weekend vs weekday sales with a more colorful styling
        sales_by_weekend.plot(kind='bar', color=['orange', 'purple'], figsize=(8, 5))
        plt.title('Sales Comparison on Weekdays vs Weekends for Stores Open All Week', fontsize=16, color='darkblue')
        plt.xlabel('Is Weekend', fontsize=12, color='darkgreen')
        plt.ylabel('Average Sales', fontsize=12, color='darkgreen')
        plt.xticks(rotation=0)
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()
        
        # Log success message
        logging.info("Weekend vs Weekday sales plot created successfully.")
        
    except Exception as e:
        logging.error(f"Error in plot_weekend_vs_weekday_sales: {e}")


def plot_sales_vs_competitor_distance(train_store):
    """
    Check correlation between competitor distance and sales.
    """
    try:
        # Handle missing data for competitor distance
        train_store['CompetitionDistance'] = train_store['CompetitionDistance'].fillna(train_store['CompetitionDistance'].median())
        correlation = train_store[['CompetitionDistance', 'Sales']].corr()
        logging.info(f"Correlation between Competitor Distance and Sales: \n{correlation}")

        # Scatter plot with vibrant color
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=train_store, x='CompetitionDistance', y='Sales', color='orange', alpha=0.7, s=100, edgecolor='black')
        plt.title('Sales vs Competitor Distance', fontsize=16, color='darkblue')
        plt.xlabel('Competitor Distance', fontsize=12, color='darkgreen')
        plt.ylabel('Sales', fontsize=12, color='darkgreen')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()
    except Exception as e:
        logging.error(f"Error in plot_sales_vs_competitor_distance: {e}")

def plot_city_center_sales(train_store):
    """
    Compare sales based on competitor distance for city center vs non-city stores.
    """
    try:
        # Filter city center and non-city stores
        city_center_stores = train_store[train_store['StoreType'] == 'c']
        non_city_stores = train_store[train_store['StoreType'] != 'c']

        # Plot with more colorful styling
        plt.figure(figsize=(12, 6))
        sns.kdeplot(city_center_stores['CompetitionDistance'], label='City Center Stores', fill=True, color='blue', alpha=0.5)
        sns.kdeplot(non_city_stores['CompetitionDistance'], label='Non-City Stores', fill=True, color='red', alpha=0.5)
        plt.title('Distribution of Competitor Distance for City Center vs Non-City Stores', fontsize=16, color='darkblue')
        plt.xlabel('Competitor Distance', fontsize=12, color='darkgreen')
        plt.ylabel('Density', fontsize=12, color='darkgreen')
        plt.legend()
        plt.show()
    except Exception as e:
        logging.error(f"Error in plot_city_center_sales: {e}")
def plot_sales_trend_before_after_competitor_opening(train_store):
    """
    Plot sales trend before and after a competitor opens a new store.
    """
    try:
        logging.info("Starting to process 'plot_sales_trend_before_after_competitor_opening'...")

        # Ensure 'Date' column is in datetime format
        train_store['Date'] = pd.to_datetime(train_store['Date'], errors='coerce')
        logging.info("Date column successfully converted to datetime.")

        # Create 'YearMonth' column for grouping
        train_store['YearMonth'] = train_store['Date'].dt.to_period('M').astype(str)
        logging.info("YearMonth column created for grouping.")

        # Handle competitor open date with a specified format to avoid warnings
        train_store['CompetitorOpenDate'] = pd.to_datetime(
            train_store.apply(
                lambda row: f"{int(row['CompetitionOpenSinceYear'])}-{int(row['CompetitionOpenSinceMonth']):02d}-01"
                if row['CompetitionOpenSinceYear'] > 0 and row['CompetitionOpenSinceMonth'] > 0
                else pd.NaT,
                axis=1
            ),
            errors='coerce'
        )
        logging.info("Competitor opening dates processed.")

        # Filter for stores where the competitor opened during the dataset period
        competitor_change_stores = train_store[train_store['CompetitorOpenDate'].notna()]
        logging.info(f"Filtered stores where competitor opened: {competitor_change_stores.shape[0]} rows.")

        # Calculate average sales before and after the competitor opening
        sales_trend = competitor_change_stores.groupby('YearMonth')['Sales'].mean().reset_index()
        logging.info("Calculated average sales before and after competitor opening.")

        # Convert the earliest competitor opening date to 'YearMonth' format
        earliest_opening_date = competitor_change_stores['CompetitorOpenDate'].min()

        # Check for NaT (Not a Time) and handle it
        if pd.isna(earliest_opening_date):
            logging.warning("No valid competitor opening date found.")
            return  # Exit the function if no valid date exists

        earliest_opening_yearmonth = pd.Period(earliest_opening_date, freq='M').strftime('%Y-%m')
        logging.info(f"Earliest competitor opening date is: {earliest_opening_yearmonth}")

        # Plot sales trends with vibrant color scheme
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=sales_trend, x='YearMonth', y='Sales', color='teal', linewidth=2.5)
        plt.axvline(x=earliest_opening_yearmonth, color='r', linestyle='--', label='Competitor Opening')
        plt.title('Sales Trend Before and After Competitor Reopening', fontsize=16, color='darkblue')
        plt.xlabel('Year-Month', fontsize=12, color='darkgreen')
        plt.ylabel('Average Sales', fontsize=12, color='darkgreen')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()
        logging.info("Sales trend plot generated successfully.")
    
    except Exception as e:
        logging.error(f"Error in plot_sales_trend_before_after_competitor_opening: {e}")


def compare_promo_effect(train):
    """
    Compare sales and customers during promotions vs non-promotions
    and plot the results.
    
    Args:
    train (pd.DataFrame): DataFrame containing 'Sales', 'Customers', and 'Promo' columns.
    
    Returns:
    None
    """
    try:
        promo_effect = train.groupby('Promo').agg({'Sales': 'mean', 'Customers': 'mean'})
        logging.info(f"Promo Effect: \n{promo_effect}")

        # Plot Promo Effect with colorful bar chart
        promo_effect.plot(kind='bar', title='Promo Impact on Sales and Customers', color=['lightgreen', 'lightcoral'])
        plt.title('Promo Impact on Sales and Customers', fontsize=16, color='darkblue')
        plt.xlabel('Promo', fontsize=12, color='darkgreen')
        plt.ylabel('Average Value', fontsize=12, color='darkgreen')
        plt.show()

        # Calculate Sales per Customer for both promo and non-promo
        promo_effect['SalesPerCustomer'] = promo_effect['Sales'] / promo_effect['Customers']
        logging.info(f"Promo Effect with SalesPerCustomer: \n{promo_effect}")
    except Exception as e:
        logging.error(f"Error in compare_promo_effect: {e}")

def compare_assortment_sales(train, store):
    """
    Merge train data with store data to include assortment info, 
    calculate average sales by assortment type, and plot the results.
    
    Args:
    train (pd.DataFrame): DataFrame containing sales data.
    store (pd.DataFrame): DataFrame containing store info with 'Assortment' column.
    
    Returns:
    None
    """
    try:
        # Merge train data with store data to include assortment info
        train_store = pd.merge(train, store, on='Store')

        # Group by Assortment and calculate average sales
        assortment_sales = train_store.groupby('Assortment')['Sales'].mean().reset_index()

        # Visualize assortment type impact on sales with vibrant color
        plt.figure(figsize=(10, 6))
        sns.barplot(data=assortment_sales, x='Assortment', y='Sales', hue='Assortment', palette='viridis', legend=False)
        plt.title('Average Sales by Assortment Type', fontsize=16, color='darkblue')
        plt.xlabel('Assortment Type', fontsize=12, color='darkgreen')
        plt.ylabel('Average Sales', fontsize=12, color='darkgreen')
        plt.show()
    except Exception as e:
        logging.error(f"Error in compare_assortment_sales: {e}")
