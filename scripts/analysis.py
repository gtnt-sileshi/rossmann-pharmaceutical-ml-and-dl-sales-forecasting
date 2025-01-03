import pandas as pd

def sales_correlation(df):
    return df['Sales'].corr(df['Customers'])

def promo_effect(df):
    promo_sales = df[df['Promo'] == 1]['Sales'].mean()
    no_promo_sales = df[df['Promo'] == 0]['Sales'].mean()
    return promo_sales, no_promo_sales
def analyze_store_performance(df):
    """
    Aggregate store-level sales data and return performance summary.
    """
    store_summary = df.groupby('Store')['Sales'].agg(['mean', 'sum']).reset_index()
    store_summary.columns = ['Store', 'Store_avg_sales', 'Store_total_sales']
    return store_summary

def analyze_weekly_trends(df):
    """
    Analyze sales trends by day of the week.
    """
    weekly_summary = df.groupby('DayOfWeek')['Sales'].mean().reset_index()
    weekly_summary.columns = ['DayOfWeek', 'Avg_Sales']
    return weekly_summary

def analyze_holiday_impact(df):
    """
    Compare sales on holidays vs. non-holidays.
    """
    holiday_summary = df.groupby(['StateHoliday', 'SchoolHoliday'])['Sales'].mean().reset_index()
    return holiday_summary
