import matplotlib.pyplot as plt
import seaborn as sns

def plot_sales_distribution(df):
    sns.histplot(df['Sales'], kde=True)
    plt.title('Sales Distribution')
    plt.show()

def plot_correlation(df):
    # Drop datetime columns
    df = df.drop(columns=['Date'])
    
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

def plot_time_trends(df, x_col, y_col):
    """
    Plot sales trends over time.
    """
    plt.figure(figsize=(12, 6))
    df.groupby(x_col)[y_col].mean().plot()
    plt.title(f"{y_col} Trends Over {x_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()

def plot_store_sales(store_summary):
    """
    Plot top and bottom 10 stores by average sales.
    """
    top_10 = store_summary.nlargest(10, 'Store_avg_sales')
    bottom_10 = store_summary.nsmallest(10, 'Store_avg_sales')
    
    plt.figure(figsize=(14, 6))
    sns.barplot(data=top_10, x='Store', y='Store_avg_sales', palette='viridis')
    plt.title("Top 10 Stores by Average Sales")
    plt.show()
    
    plt.figure(figsize=(14, 6))
    sns.barplot(data=bottom_10, x='Store', y='Store_avg_sales', palette='viridis')
    plt.title("Bottom 10 Stores by Average Sales")
    plt.show()

def plot_weekday_sales(weekday_summary):
    """
    Plot average sales by day of the week.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(data=weekday_summary, x='DayOfWeek', y='Avg_Sales', palette='coolwarm')
    plt.title("Average Sales by Day of the Week")
    plt.show()

def plot_correlation_heatmap(df):
    """
    Plot heatmap of correlations among numerical features.
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])

    # Plot correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()



