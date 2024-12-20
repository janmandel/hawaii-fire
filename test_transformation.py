from pyproj import Transformer
import rasterio, pyproj, sys
import numpy as np

# Path to the NFS-mounted raster file
elevation_path = "./LH20_Elev_220.tif"

print('rasterio',rasterio.__version__)
print('pyproj',pyproj.__version__)
print('numpy',np.__version__)
print(f"Python version: {sys.version}")
pyproj.show_versions()


try:
    print("Reading CRS from raster file...")
    with rasterio.open(elevation_path) as dataset:
        raster_crs = dataset.crs.to_string()
    print(f"Raster CRS: {raster_crs}")

    print("Creating transformer and transforming coordinates...")
    transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    lon, lat = -155.3, 19.6  # Example coordinates
    raster_x, raster_y = transformer.transform(lon, lat)
    print(f"Transformation successful: {raster_x}, {raster_y}")
except Exception as e:
    print(f"Error during transformation: {e}")

