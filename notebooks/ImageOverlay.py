import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import matplotlib.patches as mpatches
import re
import matplotlib

# Set global font that supports wider character sets
matplotlib.rcParams['font.family'] = 'Arial'  # Or try 'Noto Sans' if installed

# Load the predicted NTL TIFF image
ntl_path = "Dubai_2033_Predicted.tif"
ntl_image = rasterio.open(ntl_path)

# UAE shapefiles for overlay
admin = gpd.read_file("UAE/united_arab_emirates_administrative.shp")
highways = gpd.read_file("UAE/united_arab_emirates_highway.shp")
coastline = gpd.read_file("UAE/united_arab_emirates_coastline.shp")
water = gpd.read_file("UAE/united_arab_emirates_water.shp")
natural = gpd.read_file("UAE/united_arab_emirates_natural.shp")
poi = gpd.read_file("UAE/united_arab_emirates_poi.shp")
location = gpd.read_file("UAE/united_arab_emirates_location.shp")

# Plot setup
fig, ax = plt.subplots(figsize=(15, 15))
ax.set_facecolor('#f3e8dc')  # Skin-like map background

# NTL image preprocessing
ntl_data = ntl_image.read(1)
ntl_bounds = ntl_image.bounds
extent = [ntl_bounds.left, ntl_bounds.right, ntl_bounds.bottom, ntl_bounds.top]
ntl_data = np.clip(ntl_data, 0, 50)

# GEE-like color palette
palette = ['black', 'blue', 'cyan', 'green', 'yellow', 'red']
cmap = ListedColormap(palette)
bounds = [0, 5, 10, 20, 30, 40, 50]
norm = BoundaryNorm(bounds, ncolors=len(palette))

# Plot the NTL image
ax.imshow(ntl_data, cmap=cmap, norm=norm, extent=extent, alpha=0.6)

# Overlay shapefiles
admin.boundary.plot(ax=ax, color='gray', linewidth=1)
highways.plot(ax=ax, color='orange', linewidth=1)
coastline.plot(ax=ax, color='blue', linewidth=1)
water.plot(ax=ax, color='deepskyblue', linewidth=0.7)
natural.plot(ax=ax, color='green', linewidth=0.7)
poi.plot(ax=ax, color='purple', markersize=10)

# Clean and display fewer location names
location = location[location['NAME'].notnull()]
sampled_locations = location.sample(n=min(15, len(location)), random_state=42)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r'[^\x00-\x7F]+', '', text)  # remove non-ASCII chars

for idx, row in sampled_locations.iterrows():
    if row.geometry:
        name = clean_text(row["NAME"])
        ax.annotate(
            text=name,
            xy=(row.geometry.x, row.geometry.y),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8,
            color='black',
            alpha=0.9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.7)
        )

# Custom manual legend (since PatchCollection can’t go in default)
custom_patches = [
    mpatches.Patch(color='gray', label='Admin Boundaries'),
    mpatches.Patch(color='orange', label='Highways'),
    mpatches.Patch(color='blue', label='Coastline'),
    mpatches.Patch(color='deepskyblue', label='Water Bodies'),
    mpatches.Patch(color='green', label='Natural Terrain'),
    mpatches.Patch(color='purple', label='Points of Interest')
]
ax.legend(handles=custom_patches, loc="lower right", fontsize=10)

# Titles and ticks
ax.set_title("Dubai Predicted NTL Image (2033) Over UAE Map", fontsize=18)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
plt.show()
