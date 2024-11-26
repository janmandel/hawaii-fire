from osgeo import gdal, osr
import rasterio
import numpy as np
from ml_sample_generator import load_topography, get_file_paths, load_fire_detection, load_meteorology

# Reproject lon/lat arrays to the raster's CRS using gdal
file_paths = get_file_paths()
topography = load_topography(file_paths)
raster_crs = topography["crs"]

with rasterio.open(file_paths['elevation_path']) as dataset:
    bounds = dataset.bounds
    print(f"Raster bounds: {bounds}")

    # Get the raster bounds
    left, bottom, right, top = bounds

try:
    # Create an OSR SpatialReference object for the raster CRS
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster_crs)
    print("CRS is valid:", raster_srs.ExportToPrettyWkt())
except Exception as e:
    print("Error with raster CRS:", e)

print(f"The Topography CRS is {raster_srs.ExportToPrettyWkt()} ...")

# Create a transformer from WGS84 (EPSG:4326) to the raster CRS
source_srs = osr.SpatialReference()
source_srs.ImportFromEPSG(4326)  # WGS84
target_srs = osr.SpatialReference()
target_srs.ImportFromWkt(raster_crs)  # CRS from the raster file
transform = osr.CoordinateTransformation(source_srs, target_srs)

# Step 1: Load meteorology data
meteorology = load_meteorology(file_paths, start_index=0, end_index=100)

# Using only the upper and lower time bounds
time_lb = meteorology['times'].min()
time_ub = meteorology['times'].max()
print(f"Meteorology time range: {time_lb} to {time_ub}")
confidence_threshold = 70
fire_detection_data = load_fire_detection(file_paths, time_lb, time_ub, confidence_threshold)

# The CRS for the fire detection data is WGS84
lon_array = fire_detection_data['lon']
lat_array = fire_detection_data['lat']
print(f"The types of the lon_array, lat_array: {type(lon_array)}, {type(lat_array)}")
print(f"The min/max of the lon_array, lat_array: {lon_array.min()}/{lon_array.max()}, {lat_array.min()}/{lat_array.max()}")
print(f"The shapes of the lon array, lat array : {lon_array.shape}, {lat_array.shape}")

# Get the raster bounds
left, bottom, right, top = bounds

# Reproject lon/lat to the raster CRS and filter
filtered_lon = []
filtered_lat = []
for lon, lat in zip(lon_array, lat_array):
    try:
        x, y, _ = transform.TransformPoint(lon, lat)
        if left <= x <= right and bottom <= y <= top:
            filtered_lon.append(lon)
            filtered_lat.append(lat)
    except Exception as e:
        print(f"Error transforming point lon={lon}, lat={lat}: {e}")

# Convert back to numpy arrays
filtered_lon = np.array(filtered_lon)
filtered_lat = np.array(filtered_lat)

# Transform coordinates from WGS84 to the CRS of the topography files
print("Calling transformer...")
raster_coords = [transform.TransformPoint(lon, lat) for lon, lat in zip(filtered_lon, filtered_lat)]
raster_lon = [coord[0] for coord in raster_coords]
raster_lat = [coord[1] for coord in raster_coords]
print(f"Transformed raster coordinates: {list(zip(raster_lon, raster_lat))}")

