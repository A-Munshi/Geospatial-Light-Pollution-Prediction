import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the CSV file
file_path = "Dubai_Monthly_NTL.csv" 
df = pd.read_csv(file_path)

# Ensure the data is sorted by time
df["Date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str) + "-01")
df = df.sort_values("date").set_index("date")

# Target variable
radiance_col = "mean_radiance"  # Replace with actual column name
data = df[radiance_col]

# Train-test split: Use last 2 years (24 months) as the test set
train_data = data[:-24]
test_data = data[-24:]

# Model fitting: SARIMA (adjust seasonal and trend order based on data)
sarima_model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit = sarima_model.fit(disp=False)

# Forecast for test period
forecast_test = model_fit.get_forecast(steps=len(test_data))
forecast_test_mean = forecast_test.predicted_mean
forecast_test_conf = forecast_test.conf_int()

# Forecast for the next 10 years (120 months)
forecast_future = model_fit.get_forecast(steps=120)
forecast_future_mean = forecast_future.predicted_mean
forecast_future_conf = forecast_future.conf_int()

# Validation Metrics
mae = mean_absolute_error(test_data, forecast_test_mean)
rmse = np.sqrt(mean_squared_error(test_data, forecast_test_mean))
mape = np.mean(np.abs((test_data - forecast_test_mean) / test_data)) * 100
print(f"Validation Metrics:\nMAE: {mae}\nRMSE: {rmse}\nMAPE: {mape}%")

# Plot results
plt.figure(figsize=(14, 7))

# Historical data
plt.plot(data, label="Observed Radiance", color="blue")

# Forecasted test data
plt.plot(forecast_test_mean, label="Test Forecast", color="orange")
plt.fill_between(
    forecast_test_mean.index, 
    forecast_test_conf.iloc[:, 0], 
    forecast_test_conf.iloc[:, 1], 
    color="orange", alpha=0.2
)

# Forecasted future data
plt.plot(forecast_future_mean, label="Future Forecast", color="green")
plt.fill_between(
    forecast_future_mean.index, 
    forecast_future_conf.iloc[:, 0], 
    forecast_future_conf.iloc[:, 1], 
    color="green", alpha=0.2
)

plt.title("Monthly Radiance Change in Dubai (Observed and Forecasted)")
plt.xlabel("Year")
plt.ylabel("Radiance (nW/cm²/sr)")
plt.legend()
plt.grid()
plt.show()
# Save predictions to CSV
forecast_future_mean.to_csv("SARIMA_Predictions.csv", header=["predicted_radiance"])



# import matplotlib.pyplot as plt
# import seaborn as sns
# import scipy.stats as stats
# from statsmodels.graphics.tsaplots import plot_acf

# # Calculate residuals (difference between actual and predicted values in the test set)
# residuals = test_data - forecast_test_mean

# # Step 1: Handle Missing or Invalid Values in Residuals
# residuals = residuals.dropna()  # Remove NaN values, if any

# # Step 2: Inspect Residuals for Length and Consistency
# print(residuals.describe())     # Summary statistics
# print(f"Total Residuals: {len(residuals)}")

# # Ensure valid lags for ACF
# max_lags = min(len(residuals) - 1, 24)  # Set maximum lags to the smaller of residual length or 24

# # Step 3: Plot Residuals Over Time
# plt.figure(figsize=(12, 6))
# plt.plot(residuals, label="Residuals", color="purple")
# plt.axhline(0, linestyle="--", color="black", linewidth=1)
# plt.title("Residuals Over Time")
# plt.xlabel("Date")
# plt.ylabel("Residuals (Actual - Predicted Radiance)")
# plt.legend()
# plt.grid()
# plt.show()

# # Step 4: Histogram of Residuals
# plt.figure(figsize=(12, 6))
# sns.histplot(residuals, kde=True, bins=20, color="skyblue")
# plt.title("Histogram of Residuals")
# plt.xlabel("Residuals (nW/cm²/sr)")
# plt.ylabel("Frequency")
# plt.grid()
# plt.show()

# # Step 5: Q-Q Plot (Quantile-Quantile Plot)
# plt.figure(figsize=(12, 6))
# stats.probplot(residuals, dist="norm", plot=plt)
# plt.title("Q-Q Plot of Residuals")
# plt.grid()
# plt.show()

# # Step 6: Residual Autocorrelation Plot (ACF)
# plt.figure(figsize=(12, 6))
# plot_acf(residuals, lags=max_lags, alpha=0.05)  # Use max_lags to ensure no shape mismatch
# plt.title("Autocorrelation of Residuals")
# plt.grid()
# plt.show()
