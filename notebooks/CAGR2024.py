import rasterio
import numpy as np

def calculate_mean_radiance(tiff_path):
    with rasterio.open(tiff_path) as src:
        array = src.read(1)
        array = np.where(array == src.nodata, np.nan, array)
        return np.nanmean(array)

radiance_2014 = calculate_mean_radiance('Dubai_NTL_2014.tif')
radiance_2024 = calculate_mean_radiance('Dubai_NTL_2024.tif')

cagr_2024 = ((radiance_2024 / radiance_2014) ** (1 / 10)) - 1
print("CAGR for 2024:", cagr_2024)
