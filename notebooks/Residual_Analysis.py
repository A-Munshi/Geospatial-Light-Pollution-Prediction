import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Load actual data
actual_data = pd.read_csv("Dubai_Monthly_NTL.csv")
actual_data["date"] = pd.to_datetime(actual_data["year"].astype(str) + "-" + actual_data["month"].astype(str) + "-01")
actual_data = actual_data.sort_values("date")
actual_values = actual_data["mean_radiance"].values[-24:]  # Last 2 years

# Load predicted data
sarima_pred = pd.read_csv("SARIMA_Predictions.csv")["predicted_radiance"].values[:24]
lstm_pred = pd.read_csv("LSTM_Predictions.csv")["predicted_radiance"].values[:24]
xgb_pred = pd.read_csv("XGBoost_Predictions.csv")["mean_radiance"].values[:24]
prop_pred = pd.read_csv("Prophet_Predictions.csv")["predicted_radiance"].values[:24]


# Compute residuals
residuals_sarima = actual_values - sarima_pred
residuals_lstm = actual_values - lstm_pred
residuals_xgb = actual_values - xgb_pred
residuals_prop = actual_values - prop_pred

# Function to plot residual analysis
def residual_analysis(residuals, model_name):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram & KDE Plot
    sns.histplot(residuals, kde=True, bins=15, ax=axes[0, 0])
    axes[0, 0].set_title(f'{model_name} Residuals Distribution')
    
    # Q-Q Plot
    sm.qqplot(residuals, line='s', ax=axes[0, 1])
    axes[0, 1].set_title(f'{model_name} Q-Q Plot')
    
    # Residuals vs Time
    axes[1, 0].plot(actual_data["date"][-24:], residuals, marker='o', linestyle='dashed', color='b')
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_title(f'{model_name} Residuals Over Time')
    
    # Autocorrelation Plot
    sm.graphics.tsa.plot_acf(residuals, ax=axes[1, 1])
    axes[1, 1].set_title(f'{model_name} Residuals Autocorrelation')
    
    plt.tight_layout()
    plt.show()

# Perform residual analysis for each model
residual_analysis(residuals_sarima, "SARIMA")
residual_analysis(residuals_lstm, "LSTM")
residual_analysis(residuals_xgb, "XGBoost")
residual_analysis(residuals_prop, "Prophet")
