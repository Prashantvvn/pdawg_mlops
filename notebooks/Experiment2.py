import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
import numpy as np
import mlflow
import dagshub
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Initialize DagsHub and MLflow
dagshub.init(repo_owner='Prashantvvn', repo_name='pdawg_mlops', mlflow=True)
mlflow.set_experiment("Experiment2")
mlflow.set_tracking_uri("https://dagshub.com/Prashantvvn/pdawg_mlops.mlflow")

# Load data
file_path = r"C:\Users\bhara\mlops_project\cleaned_multi_stock_data.csv"
data = pd.read_csv(file_path, index_col="Date", parse_dates=True)

# Extract relevant columns
multi_stock_data = data[['AAPL_Close', 'GOOG_Close.1', 'TSLA_Close.2']].dropna()

# Stationarity check and differencing function
def check_stationarity(series, name):
    result = adfuller(series)
    if result[1] > 0.05:
        print(f"{name} is not stationary (p-value = {result[1]}). Differencing needed.")
        return series.diff().dropna()
    else:
        print(f"{name} is stationary (p-value = {result[1]}).")
        return series

# Apply stationarity check and differencing to each column
multi_stock_data_diff = pd.DataFrame()
for col in multi_stock_data.columns:
    multi_stock_data_diff[col] = check_stationarity(multi_stock_data[col], col)

# Train-Test Split
train_size = int(len(multi_stock_data_diff) * 0.8)
train, test = multi_stock_data_diff[:train_size], multi_stock_data_diff[train_size:]

# Fit the VAR Model
with mlflow.start_run():
    model = VAR(train)
    
    # Select optimal lag order based on AIC
    lag_order = model.select_order(maxlags=15).aic
    print(f"Optimal lag order: {lag_order}")
    
    # Fit the VAR model
    var_model = model.fit(lag_order)
    
    # Forecasting
    forecast = var_model.forecast(y=train.values, steps=len(test))
    forecast_df = pd.DataFrame(forecast, index=test.index, columns=train.columns)
    
    # Evaluate performance (RMSE)
    metrics = {}
    for col in test.columns:
        rmse = np.sqrt(mean_squared_error(test[col], forecast_df[col]))
        metrics[f"RMSE_{col}"] = rmse
        print(f"{col} RMSE: {rmse}")
    
    # Log parameters and metrics
    mlflow.log_param("lag_order", lag_order)
    mlflow.log_metrics(metrics)
    
    # Visualization of Actual vs Forecast
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(test.columns):
        plt.subplot(3, 1, i + 1)
        plt.plot(test[col], label='Actual', color='blue')
        plt.plot(forecast_df[col], label='Forecast', color='red')
        plt.title(f"{col} - Actual vs Forecast")
        plt.legend()
    
    plt.tight_layout()
    plt.savefig("VAR_Forecasts.png")
    mlflow.log_artifact("VAR_Forecasts.png")
    
    # Show the plot
    plt.show()

