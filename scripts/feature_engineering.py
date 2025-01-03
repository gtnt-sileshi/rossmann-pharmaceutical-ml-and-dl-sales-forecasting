import pandas as pd

def add_holiday_flag(df, holidays):
    df['HolidayFlag'] = df['Date'].isin(holidays).astype(int)
    return df

def create_time_features(df):
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    df['Day'] = pd.to_datetime(df['Date']).dt.day
    df['Weekday'] = pd.to_datetime(df['Date']).dt.weekday
    return df
def add_weekend_flag(df):
    """
    Add a flag to indicate whether the day is a weekend.
    """
    df['IsWeekend'] = df['DayOfWeek'].isin([6, 7]).astype(int)
    return df

def compute_weekday_averages(df):
    """
    Compute average sales per day of the week and merge with the main dataframe.
    """
    weekday_avg = df.groupby('DayOfWeek')['Sales'].mean().reset_index()
    weekday_avg.columns = ['DayOfWeek', 'DayOfWeek_avg_sales']
    df = df.merge(weekday_avg, on='DayOfWeek', how='left')
    return df
