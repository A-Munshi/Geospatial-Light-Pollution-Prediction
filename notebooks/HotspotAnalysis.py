import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import shape
from rasterio.features import shapes
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

# === Load Predicted NTL Image ===
ntl_path = "Dubai_2033_Predicted.tif"
with rasterio.open(ntl_path) as src:
    ntl_data = src.read(1)
    ntl_transform = src.transform
    ntl_crs = src.crs

# === Set Hotspot Threshold ===
threshold = 30  # Customize as needed
mask = ntl_data > threshold

# === Extract Hotspot Shapes from Raster ===
hotspot_shapes = shapes(ntl_data, mask=mask, transform=ntl_transform)
hotspot_geoms = []
radiances = []

for geom, val in hotspot_shapes:
    if val > threshold:
        hotspot_geoms.append(shape(geom))
        radiances.append(val)

hotspot_gdf = gpd.GeoDataFrame({'radiance': radiances, 'geometry': hotspot_geoms}, crs=ntl_crs)

# === Load Shapefiles ===
location = gpd.read_file("UAE/united_arab_emirates_location.shp").to_crs(ntl_crs)
admin = gpd.read_file("UAE/united_arab_emirates_administrative.shp").to_crs(ntl_crs)

# === Join Hotspots with Locations ===
hotspot_locations = gpd.sjoin(location, hotspot_gdf, how='inner', predicate='intersects')
hotspot_locations = hotspot_locations[hotspot_locations['NAME'].notnull()].drop_duplicates(subset=['NAME'])

# === Join Hotspots with Districts ===
hotspot_admin = gpd.sjoin(hotspot_gdf, admin, how='left', predicate='intersects')
hotspot_by_district = hotspot_admin.groupby('NAME')['radiance'].mean().reset_index().sort_values(by='radiance', ascending=False)
hotspot_by_district.rename(columns={'NAME': 'District', 'radiance': 'Average Radiance'}, inplace=True)

# === Save Ranked CSV ===
hotspot_by_district.to_csv("dubai_hotspots_by_district_2033.csv", index=False)

# === Plotting ===
fig, ax = plt.subplots(figsize=(15, 15))

# Background: Districts
admin.boundary.plot(ax=ax, color='gray', linewidth=0.8)

# Hotspot Layer
hotspot_gdf.plot(
    ax=ax,
    column='radiance',
    cmap=mcolors.ListedColormap(['black', 'blue', 'cyan', 'green', 'yellow', 'red']),
    legend=True,
    vmin=0,
    vmax=50,
    alpha=0.8,
    edgecolor='none'
)

# Plot location names only for identified hotspots
for idx, row in hotspot_locations.iterrows():
    ax.annotate(
        row['NAME'],
        xy=(row.geometry.x, row.geometry.y),
        xytext=(3, 3),
        textcoords="offset points",
        fontsize=9,
        color='black'
    )

# Titles and cleanup
ax.set_title("Dubai 2033 NTL Radiance Hotspots", fontsize=20)
ax.set_xticks([])
ax.set_yticks([])
ax.set_axis_off()

plt.tight_layout()
plt.savefig("dubai_ntl_hotspots_2033.png", dpi=300)
plt.show()
