import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from prophet.diagnostics import cross_validation, performance_metrics
import numpy as np

# Load the dataset
df = pd.read_csv("Dubai_Monthly_NTL.csv")

# Combine year and month into a datetime format for Prophet
df['ds'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str) + '-01')

# Rename the mean_radiance column to 'y' for Prophet
df.rename(columns={'mean_radiance': 'y'}, inplace=True)

# Select only the 'ds' (datetime) and 'y' (mean radiance) columns
df_prophet = df[['ds', 'y']].copy()

# Initialize and fit the Prophet model
model = Prophet()
model.fit(df_prophet)

# Create a future dataframe for the next 10 years (120 months)
future = model.make_future_dataframe(periods=120, freq='MS')  # 'MS' for month start frequency

# Make predictions
forecast = model.predict(future)

# Save forecast to CSV
forecast.to_csv("prophet_forecast.csv", index=False)

# Plot the historical and predicted data
fig1 = model.plot(forecast)
plt.title("Nighttime Light Pollution Prediction for Dubai (2012-2033)")
plt.xlabel("Date")
plt.ylabel("Mean Radiance")
plt.show()

# Extract historical and predicted values for evaluation
historical_data = df_prophet['y']
predicted_historical = forecast['yhat'][:len(historical_data)]
predicted_future = forecast['yhat'][len(historical_data):]
actual_future_dates = forecast['ds'][len(historical_data):]

# Evaluation Metrics for the historical data
mse = mean_squared_error(historical_data, predicted_historical)
mae = mean_absolute_error(historical_data, predicted_historical)
r2 = r2_score(historical_data, predicted_historical)

print("Evaluation Metrics (Historical Data):")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R2): {r2}")

# Calculate MAPE
def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = calculate_mape(historical_data, predicted_historical)
print(f"Mean Absolute Percentage Error (MAPE): {mape}")

# Cross-validation and performance metrics for robustness check
df_cv = cross_validation(model, initial='730 days', period='365 days', horizon='730 days')
# Adjust initial, period, horizon according to your data and needs.
df_p = performance_metrics(df_cv)

print("\nCross-Validation Performance Metrics:")
print(df_p.head())
print(f"Cross Validation Mean Absolute Percentage Error (MAPE): {df_p['mape'].mean()}")
print(f"Cross Validation Root Mean Squared Error (RMSE): {df_p['rmse'].mean()}")

# Plot the forecasted values along with the actual historic values.
plt.figure(figsize=(15, 6))
plt.plot(df_prophet['ds'], df_prophet['y'], label='Historical Data', color='blue')
plt.plot(forecast['ds'], forecast['yhat'], label='Predicted Data', color='red')
plt.plot(actual_future_dates, predicted_future, label='Future Predictions', color='green')
plt.xlabel('Date')
plt.ylabel('Mean Radiance')
plt.title('Historical and Predicted Mean Radiance')
plt.legend()
plt.show()

# Plot the components of the forecast (trend, yearly seasonality, etc.)
fig2 = model.plot_components(forecast)
plt.show()

# Extract only the future predicted data (2024-2033)
future_forecast = forecast[forecast['ds'] >= '2024-01-01']

# Extract predictions for 2033
pred_2033 = future_forecast[future_forecast['ds'].dt.year == 2033]

# Get mean predicted radiance for 2033
mean_2033 = pred_2033['yhat'].mean()
print("Mean Radiance for 2033 (Predicted):", mean_2033)