import pandas as pd
import numpy as np

def handle_missing_values(df):
    # Example: Fill missing competitor distance with -1
    df['StateHoliday'].fillna(-1, inplace=True)
    return df

def remove_outliers(df, column, method='iqr'):
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        return df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]
    return df
def preprocess_holidays(df):
    """
    Clean and preprocess holiday-related columns.
    Convert StateHoliday and SchoolHoliday to more consistent formats.
    """
    df['StateHoliday'] = df['StateHoliday'].replace(0, '0').astype(str)
    df['SchoolHoliday'] = df['SchoolHoliday'].astype(int)
    return df

def preprocess_date(df):
    """
    Convert Date column to datetime and extract useful time features.
    """
    df['Date'] = pd.to_datetime(df['Date'])
    return df
