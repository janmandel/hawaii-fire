from ml_sample_generator import load_topography, get_file_paths, load_fire_detection, load_meteorology
from pyproj import CRS, Transformer

# Reproject lon/lat arrays to the raster's CRS using pyproj
file_paths = get_file_paths()
topography = load_topography(file_paths) 
raster_crs = topography["crs"]
print(f"The raster CRS is {raster_crs} ...")
transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
# Step 1: Load meteorology data
meteorology = load_meteorology(file_paths, start_index = 0, end_index = 100)
time_lb = meteorology['times'].min()
time_ub = meteorology['times'].max()
print(f"Meteorology time range: {time_lb} to {time_ub}")
confidence_threshold = 70
fire_detection_data = load_fire_detection(file_paths, time_lb, time_ub, confidence_threshold)
lon_array = fire_detection_data['lon']
lat_array = fire_detection_data['lat']
print(f"The shapes of the lon array, lat array : {lon_array.shape}, {lat_array.shape}")
print("Calling transformer...")
raster_lon, raster_lat = transformer.transform(lon_array, lat_array)
print(f"The tranform for the raster is: {transformer}" )
