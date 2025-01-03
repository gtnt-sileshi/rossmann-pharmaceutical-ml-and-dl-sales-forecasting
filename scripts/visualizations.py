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

