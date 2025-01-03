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
