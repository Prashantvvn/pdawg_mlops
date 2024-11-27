import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import mlflow
import dagshub
import matplotlib.pyplot as plt
import numpy as np

# Initialize DagsHub and MLflow
dagshub.init(repo_owner='Prashantvvn', repo_name='pdawg_mlops', mlflow=True)
mlflow.set_experiment("Experiment3")
mlflow.set_tracking_uri("https://dagshub.com/Prashantvvn/pdawg_mlops.mlflow")

# Load data
file_path = r"C:\Users\bhara\mlops_project\cleaned_multi_stock_data.csv"
data = pd.read_csv(file_path, index_col="Date", parse_dates=True)

# Extract AAPL_Close and drop missing values
aapl_data = data[['AAPL_Close']].dropna()

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
aapl_scaled = scaler.fit_transform(aapl_data)

# Create sequences for LSTM input (look back period)
look_back = 60  # 60 previous time steps for prediction
X, y = [], []
for i in range(look_back, len(aapl_scaled)):
    X.append(aapl_scaled[i-look_back:i, 0])  # Use previous 60 time steps as features
    y.append(aapl_scaled[i, 0])  # Use the next time step as the target

X, y = np.array(X), np.array(y)

# Reshape X for LSTM [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# Train-Test Split (80-20)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM Model
with mlflow.start_run():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(units=1))  # Output layer
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Inverse transform predictions to original scale
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Evaluate RMSE
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
    print(f"RMSE: {rmse}")
    
    # Log metrics and parameters
    mlflow.log_param("look_back", look_back)
    mlflow.log_param("epochs", 10)
    mlflow.log_param("batch_size", 32)
    mlflow.log_metric("RMSE", rmse)
    
    # Visualization: Actual vs Forecast
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title("AAPL Stock Price Prediction - Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig("LSTM_AAPL_Predictions.png")
    mlflow.log_artifact("LSTM_AAPL_Predictions.png")
    plt.show()
