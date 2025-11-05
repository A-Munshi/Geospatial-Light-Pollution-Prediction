
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam

# Load the dataset
data_path = "Dubai_Monthly_NTL.csv"  # Adjust the path as per your system
df = pd.read_csv(data_path)

# Ensure the 'Month' column is in datetime format and clean the dataset
df.rename(columns={"Month": "date", "mean_radiance": "Value"}, inplace=True)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna()

# Set the index to the Date column
df.set_index("date", inplace=True)

# Prepare the data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df["Value"].values.reshape(-1, 1))

# Create sequences for LSTM
sequence_length = 12  # Using 12 months (1 year) as input sequence
X, y = [], []
for i in range(sequence_length, len(data_scaled)):
    X.append(data_scaled[i-sequence_length:i, 0])
    y.append(data_scaled[i, 0])
X, y = np.array(X), np.array(y)

# Split data into training and testing sets (80% train, 20% test)
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Reshape input data for LSTM (samples, timesteps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, activation="relu", return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, activation="relu"))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.01), loss="mean_squared_error")

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Predictions and inverse scaling
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Validation Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

# Plotting predictions vs actual values
plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test)), y_test, label="Actual", marker="o")
plt.plot(range(len(y_pred)), y_pred, label="Predicted", linestyle="--")
plt.title("LSTM Model: Actual vs Predicted Mean Radiance")
plt.xlabel("Time Steps (Months)")
plt.ylabel("Mean Radiance")
plt.legend()
plt.grid()
plt.show()

# Future Predictions
last_sequence = data_scaled[-sequence_length:]
last_sequence = last_sequence.reshape(1, sequence_length, 1)
future_predictions = []
for _ in range(120):  # Predicting the next 120 months (10 years)
    pred = model.predict(last_sequence)
    future_predictions.append(pred[0, 0])
    last_sequence = np.append(last_sequence[:, 1:, :], np.reshape(pred, (1, 1, 1)), axis=1)

# Inverse scale the future predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Plot the future forecast
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Value"], label="Historical Data", marker="o")
future_dates = pd.date_range(df.index[-1], periods=121, freq="M")[1:]
plt.plot(future_dates, future_predictions, label="Forecast", linestyle="--")
plt.title("LSTM Model: 10-Year Forecast")
plt.xlabel("Year")
plt.ylabel("Mean Radiance")
plt.legend()
plt.grid()
plt.show()
