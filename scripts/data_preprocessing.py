import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import numpy as np

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        DataFrame: Loaded data as a pandas DataFrame.
    
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please check the file path.")
        raise
    except pd.errors.EmptyDataError:
        print(f"Error: The file at {file_path} is empty.")
        raise
    except Exception as e:
        print(f"Unexpected error while loading data: {e}")
        raise

def preprocess_data(df):
    """
    Preprocess the data by scaling numeric features and encoding categorical features.
    
    Args:
        df (DataFrame): The input DataFrame containing sales data.
    
    Returns:
        tuple: Processed features and target variable (X, y).
    
    Raises:
        ValueError: If required columns are missing or data is invalid.
    """
    numeric_features = ['Open', 'Customers']
    categorical_features = ['StateHoliday', 'SchoolHoliday']
    
    # Ensure all categorical columns are strings
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].astype(str)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    try:
        # Check for missing required columns
        required_columns = numeric_features + categorical_features + ['Sales']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in data: {missing_columns}")
        
        # Separate features and target variable
        X = df.drop('Sales', axis=1)
        y = df['Sales']
        
        # Apply preprocessing transformations
        X_processed = preprocessor.fit_transform(X)
        print("Data preprocessing completed successfully.")
        return X_processed, y
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        raise

def create_lstm_dataset(data, time_steps=1):
    """
    Create dataset for LSTM by transforming time series data.
    
    Args:
        data (array-like): Input time series data.
        time_steps (int): Number of time steps to consider for LSTM.
    
    Returns:
        tuple: Features and target arrays (X, y).
    
    Raises:
        ValueError: If input data is not properly formatted.
    """
    try:
        if len(data) < time_steps + 1:
            raise ValueError("Insufficient data to create LSTM dataset.")
        
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:(i + time_steps), 0])
            y.append(data[i + time_steps, 0])
        
        print("LSTM dataset created successfully.")
        return np.array(X), np.array(y)
    except Exception as e:
        print(f"Error creating LSTM dataset: {e}")
        raise

def build_lstm_model(input_shape):
    """
    Build and compile an LSTM model.
    
    Args:
        input_shape (tuple): Shape of the input data (time_steps, features).
    
    Returns:
        Sequential: Compiled LSTM model.
    """
    try:
        model = Sequential([
            Input(shape=input_shape),
            LSTM(50, return_sequences=True),  # First LSTM layer with return_sequences=True
            LSTM(50),  # Second LSTM layer
            Dense(1)  # Output layer for regression
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        print("LSTM model built and compiled successfully.")
        return model
    except Exception as e:
        print(f"Error building LSTM model: {e}")
        raise
