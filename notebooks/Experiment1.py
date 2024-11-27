import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Switch to 'Agg' backend, which is non-interactive
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
import mlflow
import dagshub
import mlflow.sklearn

# Initialize Dagshub integration
dagshub.init(repo_owner='Prashantvvn', repo_name='pdawg_mlops', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Prashantvvn/pdawg_mlops.mlflow")
mlflow.set_experiment("Experiment1")

# Load the cleaned dataset
file_path = r"C:\Users\bhara\mlops_project\cleaned_multi_stock_data.csv"
data = pd.read_csv(file_path, index_col="Date", parse_dates=True)

# Extract AAPL_Close data for time series analysis
aapl_data = data['AAPL_Close'].dropna()

# Train-Test Split (80% train, 20% test)
train_size = int(len(aapl_data) * 0.8)
train, test = aapl_data[:train_size], aapl_data[train_size:]

# Start MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("p", 5)
    mlflow.log_param("d", 1)
    mlflow.log_param("q", 0)
    
    # Fit an ARIMA model
    model = ARIMA(train, order=(5, 1, 0))
    arima_model = model.fit()
    
    # Forecasting
    forecast = arima_model.forecast(steps=len(test))
    forecast_index = test.index

    # Plot predictions vs actual
    plt.figure(figsize=(10, 5))
    plt.plot(train, label='Train Data')
    plt.plot(test, label='Test Data')
    plt.plot(forecast_index, forecast, label='Forecast', color='red')
    plt.title("ARIMA Model Predictions")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()

    # Save the plot as an artifact
    plot_path = "arima_predictions.png"
    plt.savefig(plot_path)
    plt.close()
    mlflow.log_artifact(plot_path)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test, forecast))
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    
    # Log metrics
    mlflow.log_metric("RMSE", rmse)
    
    # Log the ARIMA model
    mlflow.sklearn.log_model(arima_model, "arima_model")
