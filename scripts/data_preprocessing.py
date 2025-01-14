import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import numpy as np

def load_data(file_path):
    """Load data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        DataFrame: Loaded data as a pandas DataFrame.
    
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(df):
    """Preprocess the data by scaling numeric features and encoding categorical features.
    
    Args:
        df (DataFrame): The input DataFrame containing sales data.
    
    Returns:
        tuple: Processed features and target variable (X, y).
    """
    # Define available features
    numeric_features = ['Open', 'Customers']
    categorical_features = ['StateHoliday', 'SchoolHoliday']
    
    # Ensure all categorical columns are strings
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Define the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    try:
        # Ensure required columns exist in the DataFrame
        required_columns = numeric_features + categorical_features + ['Sales']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")
        
        # Separate features and target variable
        X = df.drop('Sales', axis=1)
        y = df['Sales']
        
        # Apply transformations
        X_processed = preprocessor.fit_transform(X)
        return X_processed, y
    
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise

def create_lstm_dataset(data, time_steps=1):
    """Create dataset for LSTM by transforming time series data.
    
    Args:
        data (array-like): Input time series data.
        time_steps (int): Number of time steps to consider for LSTM.
    
    Returns:
        tuple: Features and target arrays (X, y).
    """
    X, y = [], []
    for i in range(len(data) - time_steps - 1):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)
def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),  # Define the input shape explicitly
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1)  # Output layer
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model