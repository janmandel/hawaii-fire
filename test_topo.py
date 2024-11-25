from ml_sample_generator import load_topography, get_file_paths
from pyproj import CRS, Transformer

# Reproject lon/lat arrays to the raster's CRS using pyproj
file_paths = get_file_paths()
topography = load_topography(file_paths) 
raster_crs = topography["crs"]
print(f"The raster CRS is {raster_crs} ...")
transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
print(f"The tranform for the raster is: {transformer}")
