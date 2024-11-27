# Core Libraries
import pandas as pd
import numpy as np

# Preprocessing
from sklearn.preprocessing import MinMaxScaler

# Time Series Modeling
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# Deep Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Metrics
from sklearn.metrics import mean_squared_error

# Visualization
import matplotlib.pyplot as plt

# Experiment Tracking
import mlflow
import dagshub

# Initialize MLflow with DagsHub
dagshub.init(repo_owner='Prashantvvn', repo_name='pdawg_mlops', mlflow=True)
mlflow.set_experiment("Experiment4")
mlflow.set_tracking_uri("https://dagshub.com/Prashantvvn/pdawg_mlops.mlflow")

# Load data
file_path = r"C:\Users\bhara\mlops_project\cleaned_multi_stock_data.csv"
data = pd.read_csv(file_path, index_col="Date", parse_dates=True)

# Extract AAPL_Close and drop missing values
aapl_data = data[['AAPL_Close']].dropna()

# Decompose time series into trend, seasonal, and residual
decomposition = seasonal_decompose(aapl_data, model='additive', period=365)
trend = decomposition.trend.dropna()
residual = decomposition.resid.dropna()

# ARIMA Model on Trend
with mlflow.start_run(run_name="ARIMA_Trend_Model"):
    arima_model = ARIMA(trend, order=(5, 1, 0))
    arima_result = arima_model.fit()
    arima_forecast = arima_result.forecast(steps=len(trend))

    # Evaluate ARIMA
    arima_rmse = np.sqrt(mean_squared_error(trend[-len(arima_forecast):], arima_forecast))
    mlflow.log_param("arima_order", (5, 1, 0))
    mlflow.log_metric("arima_rmse", arima_rmse)
    
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(trend[-len(arima_forecast):], label='Actual Trend')
    plt.plot(arima_forecast, label='ARIMA Forecast')
    plt.title("ARIMA Trend Prediction")
    plt.legend()
    plt.savefig("ARIMA_Trend_Predictions.png")
    mlflow.log_artifact("ARIMA_Trend_Predictions.png")
    plt.close()

    # Save ARIMA Model
    arima_result.save("arima_model.pkl")
    mlflow.log_artifact("arima_model.pkl")

# Normalize Residual for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
residual_scaled = scaler.fit_transform(residual.values.reshape(-1, 1))

# Prepare LSTM Input
look_back = 60
X, y = [], []
for i in range(look_back, len(residual_scaled)):
    X.append(residual_scaled[i-look_back:i, 0])
    y.append(residual_scaled[i, 0])
X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Train-Test Split for LSTM
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# LSTM Model on Residual
with mlflow.start_run(run_name="LSTM_Residual_Model"):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
    lstm_model.add(Dense(units=1))
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    # Predictions
    lstm_predictions = lstm_model.predict(X_test)
    lstm_predictions = scaler.inverse_transform(lstm_predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Evaluate LSTM
    lstm_rmse = np.sqrt(mean_squared_error(y_test_actual, lstm_predictions))
    mlflow.log_param("look_back", look_back)
    mlflow.log_param("epochs", 10)
    mlflow.log_param("batch_size", 32)
    mlflow.log_metric("lstm_rmse", lstm_rmse)
    
    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual, label='Actual Residual')
    plt.plot(lstm_predictions, label='LSTM Predicted Residual')
    plt.title("LSTM Residual Prediction")
    plt.legend()
    plt.savefig("LSTM_Residual_Predictions.png")
    mlflow.log_artifact("LSTM_Residual_Predictions.png")
    plt.close()

    # Save LSTM Model
    lstm_model.save("lstm_model.h5")
    mlflow.log_artifact("lstm_model.h5")

# Hybrid Model: Combine ARIMA and LSTM Predictions
with mlflow.start_run(run_name="Hybrid_Model"):
    # Align the length of ARIMA and LSTM predictions
    trend_forecast = arima_forecast[-len(residual):]  # Ensure the length matches the residual series
    lstm_predictions = lstm_predictions.flatten()
    
    # Ensure both arrays have the same length
    min_length = min(len(trend_forecast), len(lstm_predictions))  # Get the minimum length
    trend_forecast = trend_forecast[-min_length:]  # Slice to match length
    lstm_predictions = lstm_predictions[-min_length:]  # Slice to match length
    
    # Generate Hybrid Predictions by adding ARIMA and LSTM
    hybrid_forecast = trend_forecast + lstm_predictions

    # Evaluate Hybrid Model
    hybrid_rmse = np.sqrt(mean_squared_error(aapl_data.iloc[-min_length:].values, hybrid_forecast))
    mlflow.log_metric("hybrid_rmse", hybrid_rmse)

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(aapl_data.iloc[-min_length:].values, label='Actual Stock Price')
    plt.plot(hybrid_forecast, label='Hybrid Forecast')
    plt.title("Hybrid Model (ARIMA + LSTM) Prediction")
    plt.legend()
    plt.savefig("Hybrid_ARIMA_LSTM_Predictions.png")
    mlflow.log_artifact("Hybrid_ARIMA_LSTM_Predictions.png")
    plt.close()

print("Experiment 4 completed and logged to DagsHub!")

