from ml_sample_generator import load_topography, get_file_paths, load_fire_detection, load_meteorology
from pyproj import CRS, Transformer

# Reproject lon/lat arrays to the raster's CRS using pyproj
file_paths = get_file_paths()
topography = load_topography(file_paths) 
raster_crs = topography["crs"]
try:
    raster_crs_obj = CRS(raster_crs)  # Replace `raster_crs` with the actual CRS object/string
    print("CRS is valid:", raster_crs_obj)
except Exception as e:
    print("Error with raster CRS:", e)
print(f"The Topography CRS is {raster_crs} ...")
# This is tranforming from the native crs of the topography to WGS84
transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
# Step 1: Load meteorology data
meteorology = load_meteorology(file_paths, start_index = 0, end_index = 100)
# Is using only the upper and lower time bounds
time_lb = meteorology['times'].min()
time_ub = meteorology['times'].max()
print(f"Meteorology time range: {time_lb} to {time_ub}")
confidence_threshold = 70
fire_detection_data = load_fire_detection(file_paths, time_lb, time_ub, confidence_threshold)
# The CRS for the fire detection data is WGS84
# VERIFIED?!
lon_array = fire_detection_data['lon']
lat_array = fire_detection_data['lat']
print(f"The types of the lon_array, lat_array: {type(lon_array)}, {type(lat_array)}")
print(f"The min/max of the lon_array, lat_array: {lon_array.min()}/{lon_array.max()}, {lat_array.min()}/{lat_array.max()}")
print(f"The shapes of the lon array, lat array : {lon_array.shape}, {lat_array.shape}")
# We are tranforming from WGS84 to the CRS of the topography files 
print("Calling transformer...")
raster_lon, raster_lat = transformer.transform(lon_array[0:5], lat_array[0:5])
print(f"The tranform for the raster is: {transformer}" )
