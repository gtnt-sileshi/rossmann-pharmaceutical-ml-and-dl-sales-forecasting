import pandas as pd

def sales_correlation(df):
    return df['Sales'].corr(df['Customers'])

def promo_effect(df):
    promo_sales = df[df['Promo'] == 1]['Sales'].mean()
    no_promo_sales = df[df['Promo'] == 0]['Sales'].mean()
    return promo_sales, no_promo_sales
