# Customer Purchasing Behavior - Exploratory Data Analysis

This project focuses on analyzing customer purchasing behavior across various stores. Using exploratory data analysis (EDA), we aim to uncover patterns and insights that influence sales, customer interactions, and the impact of promotions, holidays, and competitor activity.

## Project Structure

```
project/
│
├── eda/
│   ├── __init__.py           # Makes this a Python package
│   ├── data_cleaning.py      # Functions for cleaning the data
│   ├── feature_engineering.py # Functions for feature creation
│   ├── analysis.py           # Functions for analysis
│   ├── visualizations.py     # Functions for visualizations
│
├── data/
│   ├── train.csv             # Training data
│   ├── test.csv              # Test data
│
├── notebooks/
│   ├── eda_task1.ipynb       # Main notebook for EDA
│
├── requirements.txt          # Dependencies
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

