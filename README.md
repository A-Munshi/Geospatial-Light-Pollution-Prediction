## 🌍 Geospatial Light Pollution Prediction – Dubai (2012–2033)
### Forecasting Nighttime Light Pollution Using Machine Learning and Time-Series Models
<img width="420" height="293" alt="image" src="https://github.com/user-attachments/assets/4bfd7bb2-2948-4480-9cd8-4cb659deb27b" />
Dubai

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
<img width="2415" height="1558" alt="image" src="https://github.com/user-attachments/assets/418c4122-2ddd-41b4-84dd-cb099044345a" />
Graphs showing Predicted vs Historical NTL using the above models

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

<img width="1859" height="1532" alt="Screenshot 2025-04-11 153724" src="https://github.com/user-attachments/assets/2cd7d095-c162-45fc-aefd-676ce367d2f0" />
Monthly VIIRS-derived NTL images of Dubai from January 2024 to December 2024

<img width="2450" height="1549" alt="Screenshot 2025-04-11 165845" src="https://github.com/user-attachments/assets/6cc39ce4-e466-49fe-8847-a5eca0c178bc" />
Predicted NTL images of Dubai from 2025 to 2033

### Repository Structure
```
Geospatial-Light-Pollution-Prediction/
│
├── data/
│ ├── VIIRS_Dubai_2012_2024.csv
│ ├── Dubai_Boundary.shp
│ └── Predicted_NTL_2025_2033.tif
│
├── notebooks/
│ ├── 01_Data_Preprocessing.ipynb
│ ├── 02_SARIMA_Model.ipynb
│ ├── 03_Prophet_Model.ipynb
│ ├── 04_XGBoost_Model.ipynb
│ └── 05_LSTM_Model.ipynb
│
├── results/
│ ├── performance_metrics.csv
│ ├── residual_plots/
│ ├── prediction_graphs/
│ └── ntl_spatial_maps/
│
├── src/
│ ├── preprocessing.py
│ ├── forecast_models.py
│ ├── evaluate.py
│ └── visualize.py
│
├── requirements.txt
└── README.md
```

### Tech Stack
- **Languages:** Python (3.13)
- **Libraries:** pandas, numpy, statsmodels, prophet, xgboost, tensorflow, keras, rasterio, geopandas, matplotlib, seaborn
- **Platform:** Google Earth Engine

### How to Run
1. Clone the repository:
```
git clone https://github.com/A-Munshi/Geospatial-Light-Pollution-Prediction.git
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Run notebooks in order (1 to 5) inside `/notebooks/`.
4. View model outputs and spatial predictions in `/results/`.

### Insights
* NTL levels in Dubai show a **consistent upward trend** (2012–2033).
* **Prophet** model provides the most stable and interpretable forecasts.
* Spatial maps show both **core hotspot persistence** and **peripheral growth**.
* The findings highlight Dubai’s **expanding artificial illumination footprint**, especially beyond 2029.

### Future Work
* Extend analysis to **other major cities** (Singapore, Los Angeles, Kuwait City).
* Combine NTL data with **socioeconomic and land-use indicators**.
* Build a **web-based interactive visualization dashboard**.

## ✍️ Authors
**Anuvab Munshi**, **Saikat Mondal**, **Ayush Shaw**
Department of Computer Applications & Science,
Institute of Engineering & Management, Kolkata
