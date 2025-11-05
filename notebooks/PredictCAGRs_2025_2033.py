import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Load monthly mean radiance data
df = pd.read_csv('Dubai_Monthly_NTL.csv')  # assume columns: year, month, mean_radiance
annual_radiance = df.groupby('year')['mean_radiance'].mean().reset_index()

# Calculate historical CAGR values (e.g. for 2014 to 2024)
cagr_years = []
cagr_values = []

for i in range(1, len(annual_radiance)):
    vi = annual_radiance.iloc[0]['mean_radiance']
    vf = annual_radiance.iloc[i]['mean_radiance']
    n = annual_radiance.iloc[i]['year'] - annual_radiance.iloc[0]['year']
    cagr = ((vf / vi) ** (1 / n)) - 1
    cagr_years.append(annual_radiance.iloc[i]['year'])
    cagr_values.append(cagr)

# Fit regression model
model = LinearRegression()
X = np.array(cagr_years).reshape(-1, 1)
y = np.array(cagr_values)
model.fit(X, y)

# Predict CAGR for 2025 to 2033
future_years = np.arange(2025, 2034)
predicted_cagrs = model.predict(future_years.reshape(-1, 1))
cagr_changes = np.diff(predicted_cagrs, prepend=cagr_values[-1])

# Save to CSV
cagr_df = pd.DataFrame({
    'year': future_years,
    'predicted_cagr': predicted_cagrs,
    'change_from_last_year': cagr_changes
})
cagr_df.to_csv('Predicted_CAGRs_2025_2033.csv', index=False)
