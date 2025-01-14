import joblib
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model as load_keras_model
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

def train_rf_model(X, y):
    """Train a Random Forest model.
    
    Args:
        X (array-like): Features for training.
        y (array-like): Target variable.
    
    Returns:
        RandomForestRegressor: Trained Random Forest model.
    """
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

def save_rf_model(model, filename):
    """Save the trained Random Forest model to a file.
    
    Args:
        model: Trained model to save.
        filename (str): Path where the model will be saved.
    """
    try:
        joblib.dump(model, filename)
    except Exception as e:
        print(f"Error saving model: {e}")
        raise

def train_lstm_model(X, y, time_steps=5, epochs=5, batch_size=64):
    """Train an optimized LSTM model for time series prediction.
    
    Args:
        X (array-like): Features for training (reshaped as required).
        y (array-like): Target variable.
        time_steps (int): Number of time steps for LSTM input (default=5).
        epochs (int): Number of epochs for training (default=20).
        batch_size (int): Size of training batches (default=64).
    
    Returns:
        Sequential: Trained LSTM model.
    """
    # Ensure X is reshaped to (samples, time_steps, features)
    samples, features = X.shape
    X = X.reshape((samples, time_steps, features // time_steps))
    
    # Define the model architecture
    model = Sequential()
    model.add(LSTM(5, input_shape=(time_steps, features // time_steps)))  # Reduced units, single LSTM layer
    model.add(Dense(1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Early stopping callback
    early_stopping = EarlyStopping(monitor='loss', patience=3, verbose=1)
    
    # Fit the model
    try:
        history = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stopping])
        # Evaluate the loss after training
        loss = history.history['loss'][-1]
        print(f"Optimized LSTM Model Training Loss: {loss}")
    except Exception as e:
        print(f"Error during model training: {e}")
        raise
    
    return model


def save_lstm_model(model, filename):
    """Save the trained LSTM model to a file.
    
    Args:
        model: Trained LSTM model to save.
        filename (str): Path where the model will be saved.
    """
    try:
        model.save(filename)
    except Exception as e:
        print(f"Error saving LSTM model: {e}")
        raise

def load_lstm_model(filename):
    """Load a trained LSTM model from a file.
    
    Args:
        filename (str): Path to the saved model file.
    
    Returns:
        Sequential: Loaded LSTM model.
    """
    try:
        return load_keras_model(filename)
    except Exception as e:
        print(f"Error loading LSTM model: {e}")
        raise