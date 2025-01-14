from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the models
try:
    rf_model = joblib.load('rf_model.pkl')
    lstm_model = load_model('lstm_model.h5')
except Exception as e:
    print(f"Error loading models: {e}")
    raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_rf', methods=['POST'])
def predict_rf():
    try:
        file = request.files['file']
        if not file or not file.filename.endswith('.csv'):
            return render_template('result.html', error="Invalid file type.")

        # Read the CSV file
        data = pd.read_csv(file)

        # Preprocess the data (select relevant columns)
        required_columns = ['Store', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']  # Exclude 'Customers' and 'Sales'
        data = data[required_columns]

        # Convert categorical features to numeric
        data['DayOfWeek'] = data['DayOfWeek'].astype('category').cat.codes
        data['StateHoliday'] = data['StateHoliday'].astype('category').cat.codes
        
        # Handle NaN values before conversion
        data.fillna(0, inplace=True)  # Fill NaN values with 0 or you can choose to drop them

        # Convert 'Open' and 'Promo' to integers
        data['Open'] = data['Open'].astype(int)
        data['Promo'] = data['Promo'].astype(int)

        # Ensure all data is numeric
        data = data.apply(pd.to_numeric, errors='coerce')

        # Drop rows with NaN values after numeric conversion (if any)
        data.dropna(inplace=True)

        # Make predictions
        predictions = rf_model.predict(data)  # Predict for all rows
        
        return render_template('result.html', predictions=predictions.tolist(), model='Random Forest')
    except Exception as e:
        return render_template('result.html', error=f"Prediction error: {str(e)}")

@app.route('/predict_lstm', methods=['POST'])
def predict_lstm():
    try:
        file = request.files['file']
        if not file or not file.filename.endswith('.csv'):
            return render_template('result.html', error="Invalid file type.")

        # Read the CSV file
        data = pd.read_csv(file)

        # Preprocess the data (select relevant columns)
        required_columns = ['Store', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']  # Exclude 'Customers' and 'Sales'
        data = data[required_columns]

        # Convert categorical features to numeric
        data['DayOfWeek'] = data['DayOfWeek'].astype('category').cat.codes
        data['StateHoliday'] = data['StateHoliday'].astype('category').cat.codes
        
        # Handle NaN values before conversion
        data.fillna(0, inplace=True)  # Fill NaN values with 0 or you can choose to drop them

        # Convert 'Open' and 'Promo' to integers
        data['Open'] = data['Open'].astype(int)
        data['Promo'] = data['Promo'].astype(int)

        # Ensure all data is numeric
        data = data.apply(pd.to_numeric, errors='coerce')

        # Drop rows with NaN values after numeric conversion (if any)
        data.dropna(inplace=True)

        # Reshape for LSTM input
        input_array = data.values.reshape((data.shape[0], data.shape[1], 1))  # Adjust for LSTM input shape
        
        # Make predictions
        predictions = lstm_model.predict(input_array)
        
        return render_template('result.html', predictions=predictions.flatten().tolist(), model='LSTM')
    except Exception as e:
        return render_template('result.html', error=f"Prediction error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)