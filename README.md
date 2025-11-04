## 🌍 Geospatial Light Pollution Prediction – Dubai (2012–2033)
### Forecasting Nighttime Light Pollution Using Machine Learning and Time-Series Models

### Overview
This project analyzes and forecasts **nighttime light (NTL) pollution** trends in **Dubai, UAE** using satellite-based radiance data from the **Visible Infrared Imaging Radiometer Suite (VIIRS) Day/Night Band (DNB)**.  
By leveraging advanced **time-series forecasting** and **machine learning** models, the project predicts radiance levels up to **2033**, offering insights into **urban expansion**, **artificial illumination growth**, and **environmental sustainability**.

### Objectives
- Analyze historical nighttime light (NTL) patterns from **2012–2024** using VIIRS DNB imagery via **Google Earth Engine (GEE)**.
- Forecast future mean radiance values (2024–2033) using:
  - **SARIMA**
  - **Prophet**
  - **XGBoost**
  - **LSTM**
- Compared model performances using statistical metrics (**MAE, MSE, RMSE, MAPE**).
- Visualized predicted spatial distribution of light pollution to identify **emerging hotspots**.

### Methodology
#### 1. Data Collection
- **Dataset:** VIIRS DNB Monthly Cloud-Free Composites (2012–2024)
- **Platform:** Google Earth Engine (GEE)
- **Processing Steps:**
  - Cloud filtering using QA bitmask.
  - Radiance band extraction and clipping to Dubai’s boundary.
  - Monthly mean computation → generated a univariate time-series dataset.

#### 2. Forecasting Models
| Model | Description | Library Used |
|-------------|-----------------------------------------------------------|-----------------------|
| **SARIMA**  | Captures seasonality and linear temporal dependencies.    |     `statsmodels`     |
| **Prophet** | Handles trend and seasonal decomposition; most accurate.  |       `prophet`       |
| **XGBoost** | Gradient boosting for non-linear dependencies.            |       `xgboost`       |
|  **LSTM**   | Neural network capturing long-term temporal patterns.     | `tensorflow`, `keras` |

#### 3. Evaluation Metrics
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Percentage Error (MAPE)**

#### Results Summary
| Model   |    MAE   |   RMSE   |    MAPE   |        Remarks       |
|---------|----------|----------|-----------|----------------------|
| SARIMA  |   0.84   |   1.07   |   3.48%   |   Stable baseline    |
| Prophet | **0.74** | **0.97** | **3.22%** |     Best overall     |
|  LSTM   |   1.72   |   2.18   |   5.67%   |    Non-linear fit    |
| XGBoost |   1.44   |   1.99   |   4.68%   | Balanced performance |

The **Prophet model** achieved the lowest error values and passed statistical validation through the **Diebold–Mariano test**, confirming its superior forecasting accuracy.

### Spatial Analysis & Visualization
- Predicted spatial radiance (2025–2033) generated using **Compound Annual Growth Rate (CAGR)** projection.
- Each year’s **NTL TIFF** visualizes Dubai’s artificial illumination expansion.
- High illumination zones remain in:
  - *Downtown Dubai*
  - *Business Bay*
  - *Jumeirah*
  - *Dubai Marina*
- Emerging hotspots identified around *Dubai South*, *Al Qudra*, and *Jebel Ali*.

### 📂 Repository Structure

