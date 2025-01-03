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
