import joblib
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model as load_keras_model
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

def train_rf_model(X, y):
    """
    Train a Random Forest model.
    
    Args:
        X (array-like): Features for training (e.g., a 2D array or DataFrame).
        y (array-like): Target variable (e.g., a 1D array or Series).
    
    Returns:
        RandomForestRegressor: Trained Random Forest model.
    """
    try:
        # Initialize the Random Forest model with default parameters
        model = RandomForestRegressor()
        # Fit the model to the training data
        model.fit(X, y)
    except ValueError as e:
        print(f"ValueError during training Random Forest model: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error during Random Forest model training: {e}")
        raise
    return model

def save_rf_model(model, filename):
    """
    Save the trained Random Forest model to a file.
    
    Args:
        model: Trained Random Forest model.
        filename (str): Path to save the model (e.g., 'rf_model.pkl').
    """
    try:
        joblib.dump(model, filename)
        print(f"Random Forest model saved successfully to {filename}")
    except Exception as e:
        print(f"Error saving Random Forest model: {e}")
        raise

def train_lstm_model(X, y, time_steps=5, epochs=5, batch_size=64):
    """
    Train an LSTM model for time series prediction.
    
    Args:
        X (array-like): Features for training (reshaped into 3D format).
        y (array-like): Target variable.
        time_steps (int): Number of time steps for the LSTM input (default=5).
        epochs (int): Number of training epochs (default=5).
        batch_size (int): Size of training batches (default=64).
    
    Returns:
        Sequential: Trained LSTM model.
    """
    try:
        # Validate input shape compatibility
        if len(X.shape) != 2 or X.shape[0] <= time_steps:
            raise ValueError("X should be a 2D array, and samples must exceed time_steps.")
        
        # Reshape X into (samples, time_steps, features_per_step)
        samples, features = X.shape
        if features % time_steps != 0:
            raise ValueError("Number of features must be divisible by time_steps.")
        X = X.reshape((samples, time_steps, features // time_steps))
        
        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(5, input_shape=(time_steps, features // time_steps)))
        model.add(Dense(1))
        
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Early stopping callback to prevent overfitting
        early_stopping = EarlyStopping(monitor='loss', patience=3, verbose=1)
        
        # Train the model
        history = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stopping])
        print(f"Training completed. Final loss: {history.history['loss'][-1]}")
    except ValueError as e:
        print(f"ValueError during LSTM model training: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error during LSTM model training: {e}")
        raise
    
    return model

def save_lstm_model(model, filename):
    """
    Save the trained LSTM model to a file.
    
    Args:
        model: Trained LSTM model.
        filename (str): Path to save the model (e.g., 'lstm_model.h5').
    """
    try:
        model.save(filename)
        print(f"LSTM model saved successfully to {filename}")
    except Exception as e:
        print(f"Error saving LSTM model: {e}")
        raise

def load_lstm_model(filename):
    """
    Load a trained LSTM model from a file.
    
    Args:
        filename (str): Path to the saved model file (e.g., 'lstm_model.h5').
    
    Returns:
        Sequential: Loaded LSTM model.
    """
    try:
        model = load_keras_model(filename)
        print(f"LSTM model loaded successfully from {filename}")
        return model
    except FileNotFoundError:
        print(f"File not found: {filename}. Please check the file path.")
        raise
    except Exception as e:
        print(f"Unexpected error loading LSTM model: {e}")
        raise
