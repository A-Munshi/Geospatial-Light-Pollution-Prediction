import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the CSV file
data = pd.read_csv("Dubai_Monthly_NTL.csv")  

# Ensure the data is sorted by date
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values(by='date')

# Create features for modeling
data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
data['year_original'] = data['year']  
data['year'] = data['year'] - data['year'].min()

# Define features and target
X = data[['year', 'month_sin', 'month_cos']]
y = data['mean_radiance']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
model = XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)
variance = np.var(y_test)
r_squared = model.score(X_test, y_test)

# Calculate MAPE
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")
print(f"Variance of test data: {variance}")
print(f"R-squared: {r_squared}")

# Prepare future data for prediction
future_years = np.arange(data['year'].max() + 1, data['year'].max() + 11)
future_months = np.tile(np.arange(1, 13), len(future_years))
future_years = np.repeat(future_years, 12)

future_data = pd.DataFrame({
    'year': future_years,
    'month': future_months,
})
future_data['month_sin'] = np.sin(2 * np.pi * future_data['month'] / 12)
future_data['month_cos'] = np.cos(2 * np.pi * future_data['month'] / 12)

# Predict future radiance
future_data['mean_radiance'] = model.predict(future_data[['year', 'month_sin', 'month_cos']])

# Convert year back to original scale
future_data['year_original'] = future_data['year'] + data['year_original'].min()

# Combine past and future data for plotting
combined_data = pd.concat([
    data[['year_original', 'month', 'mean_radiance']],
    future_data[['year_original', 'month', 'mean_radiance']]
])

# Plot the results
plt.figure(figsize=(12, 6))

# Plot past data
past_data = combined_data[combined_data['year_original'] <= data['year_original'].max()]
plt.plot(
    past_data['year_original'] + (past_data['month'] - 1) / 12,
    past_data['mean_radiance'],
    label='Actual Data',
    color='blue'
)

# Plot future predictions
future_data = combined_data[combined_data['year_original'] > data['year_original'].max()]
plt.plot(
    future_data['year_original'] + (future_data['month'] - 1) / 12,
    future_data['mean_radiance'],
    label='Predicted Data',
    color='orange',
    linestyle='--'
)

# Customize the plot
plt.title("Mean Radiance: Past and Predicted (2024-2033)")
plt.xlabel("Year")
plt.ylabel("Mean Radiance")
plt.legend()
plt.grid(True)

# Set x-axis ticks to show years
years = combined_data['year_original'].unique()
plt.xticks(ticks=years, labels=years, rotation=45)
plt.tight_layout()
plt.show()
# Save predictions to CSV
future_data[['year_original', 'month', 'mean_radiance']].to_csv("XGBoost_Predictions.csv", index=False)