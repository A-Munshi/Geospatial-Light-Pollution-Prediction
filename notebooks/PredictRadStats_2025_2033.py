import os
import numpy as np
import pandas as pd
import rasterio

# Folder to store predicted images
output_folder = "Pred_NTL_Imgs"
os.makedirs(output_folder, exist_ok=True)

# Function to generate predicted image
def generate_future_image(base_image_path, cagr, years_ahead, output_path):
    with rasterio.open(base_image_path) as src:
        profile = src.profile
        data = src.read(1)
        data = np.where(data == src.nodata, np.nan, data)

        # Apply CAGR
        predicted_data = data * ((1 + cagr) ** years_ahead)
        predicted_data = np.where(np.isnan(data), src.nodata, predicted_data)

        profile.update(dtype='float32')
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(predicted_data.astype('float32'), 1)

# Load predicted CAGRs from the previous step
cagr_df = pd.read_csv('Predicted_CAGRs_2025_2033.csv')

# Generate images and compute statistics
stats = []

for i, row in cagr_df.iterrows():
    year = row['year']
    cagr = row['predicted_cagr']
    years_ahead = year - 2024
    output_path = os.path.join(output_folder, f'predicted_ntl_{year}.tif')
    
    # Generate the image
    generate_future_image('Dubai_NTL_2014.tif', cagr, years_ahead, output_path)

    # Read the image to calculate statistics
    with rasterio.open(output_path) as src:
        data = src.read(1)
        data = np.where(data == src.nodata, np.nan, data)
        stats.append({
            'Year': year,
            'Min Radiance': np.nanmin(data),
            'Max Radiance': np.nanmax(data),
            'Mean Radiance': np.nanmean(data),
            'Median Radiance': np.nanmedian(data)
        })

# Create and display statistics table
stats_df = pd.DataFrame(stats)
print(stats_df)

# Save the table to CSV
stats_df.to_csv("Pred_rad_stats_2025_2033.csv", index=False)
