from osgeo import osr
import numpy as np
import rasterio
from ml_sample_generator import load_topography, get_file_paths, load_fire_detection, load_meteorology

# Load topography
file_paths = get_file_paths()
topography = load_topography(file_paths)
raster_crs = topography["crs"]

# Parse the CRS to extract the standard parallels
raster_srs = osr.SpatialReference()
raster_srs.ImportFromWkt(raster_crs)

# Extract standard parallels from CRS
standard_parallel_1 = raster_srs.GetProjParm("standard_parallel_1")
standard_parallel_2 = raster_srs.GetProjParm("standard_parallel_2")

if standard_parallel_1 is None or standard_parallel_2 is None:
    raise ValueError("Standard parallels are not defined in the CRS.")

# Latitude bounds based on standard parallels
lat_min = min(standard_parallel_1, standard_parallel_2)
lat_max = max(standard_parallel_1, standard_parallel_2)

print(f"Latitude bounds from standard parallels: {lat_min} to {lat_max}")

# Load fire detection data
meteorology = load_meteorology(file_paths, start_index=0, end_index=100)
time_lb = meteorology['times'].min()
time_ub = meteorology['times'].max()
print(f"Meteorology time range: {time_lb} to {time_ub}")
confidence_threshold = 70
fire_detection_data = load_fire_detection(file_paths, time_lb, time_ub, confidence_threshold)
lon_array = fire_detection_data['lon']
lat_array = fire_detection_data['lat']
print("####### PRE FILTERING ########")
print(f"The types of the lon_array, lat_array: {type(lon_array)}, {type(lat_array)}")
print(f"The min/max of the lon_array, lat_array: {lon_array.min()}/{lon_array.max()}, {lat_array.min()}/{lat_array.max()}")
print(f"The shapes of the lon array, lat array : {lon_array.shape}, {lat_array.shape}")
# Filter coordinates within the latitude bounds
valid_indices = (lat_array >= lat_min) & (lat_array <= lat_max)
filtered_lon_array = lon_array[valid_indices]
filtered_lat_array = lat_array[valid_indices]
print("####### POST FILTERING ########")
print(f"Filtered lon_array size: {filtered_lon_array.size}")
print(f"Filtered lat_array size: {filtered_lat_array.size}")
print(f"The min/max of the filtered lon_array, lat_array: {filtered_lon_array.min()}/{filtered_lon_array.max()}, {filtered_lat_array.min()}/{filtered_lat_array.max()}")

# Continue with transforming filtered coordinates to the raster CRS
source_srs = osr.SpatialReference()
source_srs.ImportFromEPSG(4326)  # WGS84
transform = osr.CoordinateTransformation(source_srs, raster_srs)

# Transform the filtered coordinates
transformed_coords = [
    transform.TransformPoint(lon, lat)[:2] for lon, lat in zip(filtered_lon_array, filtered_lat_array)
]
print(f"Transformed coordinates (example): {transformed_coords[:5]}")
