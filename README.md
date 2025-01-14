# Customer Purchasing Behavior - Exploratory Data Analysis and Model Implementation

This project focuses on analyzing customer purchasing behavior across various stores. Using exploratory data analysis (EDA), we aim to uncover patterns and insights that influence sales, customer interactions, and the impact of promotions, holidays, and competitor activity.

## Project Structure

```
project/
│
├── scripts/
│   ├── __init__.py           # Makes this a Python package
│   ├── data_cleaning.py      # Functions for cleaning the data
│   ├── feature_engineering.py # Functions for feature creation
│   ├── analysis.py           # Functions for analysis
│   ├── visualizations.py     # Functions for visualizations
│   ├── data_preprocessing.py # Functions for data processing
│   ├── model_training.py     # Functions for model training
│
├── data/
│   ├── train.csv             # Training data
│   ├── test.csv              # Test data
│
├── logs/
│   ├── eda_process.log       # EDA Logging
│   ├── model.log             # Model Log
│
├── notebooks/
│   ├── eda_task1.ipynb       # Main notebook for EDA
│   ├── main_notebook.ipynb   # Main notebook for model creation and optimization
│
├── templates/
│   ├── index.html            # Index page
│   ├── result.html           # Prediction Result page
│
├── requirements.txt          # Dependencies
├── model_serving.py          # Flask Model serving API
├── lstm_model.h5             # LSTM Model
├── rf_model.pkl              # random forest model
└── README.md                 # Project overview
```

## Goals of the Analysis

### Core Questions:
1. Are promotions distributed similarly between training and test sets?
2. What is the sales behavior before, during, and after holidays?
3. Do seasonal events (e.g., Christmas, Easter) influence sales patterns?
4. How are sales correlated with customer numbers?
5. How effective are promotions in attracting new and existing customers?
6. Can promotions be deployed more strategically?
7. How does customer behavior vary during store opening and closing times?
8. How do weekday vs. weekend operations impact sales?
9. What is the effect of assortment type on sales?
10. How does the distance to competitors affect sales, especially in city centers?
11. How do competitor openings/reopenings impact store performance?

### Deliverables:
- A Jupyter Notebook detailing the findings with annotated plots and summaries.

## Installation

### Prerequisites:
- Python 3.8+
- `pip` (Python package installer)

### Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/gtnt-sileshi/rossmann-pharmaceutical-ml-and-dl-sales-forecasting
   cd project
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start Jupyter Lab:
   ```bash
   jupyter lab
   ```

5. Open `notebooks/eda_task1.ipynb` to begin exploration.

## Methodology

### 1. Data Cleaning
- Handle missing values (e.g., competitor distances).
- Detect and handle outliers to prevent skewed analysis.

### 2. Feature Engineering
- Add holiday flags for specific dates.
- Extract time-based features (e.g., year, month, weekday).

### 3. Analysis
- Analyze correlations, trends, and variability in the data.
- Answer specific business questions using statistical methods.

### 4. Visualization
- Use plots (histograms, scatter plots, heatmaps) to communicate findings.

## Results and Insights
- Findings will be shared in the notebook with accompanying plots and summaries.

## Dependencies
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `matplotlib` & `seaborn`: Data visualization
- `jupyterlab`: Interactive notebook environment

Here's a README file tailored for the models implementation you described earlier:

# Customer Purchasing Behavior - Model Implementation

This project implements machine learning models to predict customer purchasing behavior using the provided datasets. The focus is on developing predictive models that leverage historical sales data, promotions, and customer interactions to forecast future sales.

## Goals of the Model Implementation

The aim of this implementation is to:
1. Train machine learning models to predict sales based on various features.
2. Serve these models through a web application for easy access and predictions.
3. Analyze the impact of promotions, holidays, and customer behavior on sales.

## Implemented Models

### 1. Random Forest Regressor
- A tree-based ensemble model used for regression tasks.
- Trained on historical sales data, considering features such as promotions, holidays, and customer counts.

### 2. Long Short-Term Memory (LSTM) Network
- A recurrent neural network (RNN) architecture suited for time-series forecasting.
- Trained on sequences of sales data to capture temporal dependencies.

## Installation

### Prerequisites:
- Python 3.8+
- `pip` (Python package installer)

### Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/gtnt-sileshi/rossmann-pharmaceutical-ml-and-dl-sales-forecasting
   cd project
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Train the models:
   - For Random Forest:
     ```bash
     python train_rf.py
     ```
   - For LSTM:
     ```bash
     python train_lstm.py
     ```

5. Start the Flask application:
   ```bash
   python model_serving.py
   ```

6. Access the web application at `http://127.0.0.1:5000` to make predictions.

## Model Training

### Random Forest

- The `train_rf.py` script trains a Random Forest model using the training dataset. It preprocesses the data, encodes categorical variables, and handles NaN values before training the model.
- The trained model is saved as `rf_model.pkl`.

### LSTM

- The `train_lstm.py` script trains an LSTM model for time-series forecasting. It reshapes the data for LSTM input and trains the model to capture temporal patterns.
- The trained model is saved as `lstm_model.h5`.

## Model Serving

The `model_serving.py` script sets up a Flask web application to serve the trained models. Users can upload a CSV file containing the relevant features to receive predictions on sales.

### Supported Features
- **Store**: Store identifier
- **DayOfWeek**: Day of the week (1-7)
- **Open**: Store open status (1 = open, 0 = closed)
- **Promo**: Promotion status (1 = active, 0 = not active)
- **StateHoliday**: Indicates if the day is a state holiday
- **SchoolHoliday**: Indicates if the day is a school holiday

## Results and Insights
- Predictions can be made through the web application, and findings will be documented based on the results obtained from the models.

## Dependencies
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `scikit-learn`: Machine learning library for Random Forest
- `tensorflow`: Library for building and training the LSTM model
- `flask`: Web framework for serving the models
- `jupyterlab`: Interactive notebook environment (for exploration)

## Contribution
To contribute:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Create a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.


## Contribution
To contribute:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Create a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

