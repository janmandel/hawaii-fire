import rasterio
from pyproj import Transformer
import os.path as osp

# Path to the raster file
home = osp.expanduser("~")
base_dir = osp.abspath(osp.join(osp.expanduser("~"), 'p', 'data'))
elevation_path = osp.join(base_dir, 'feat', 'landfire', 'top', 'LF2020_Elev_220_HI', 'LH20_Elev_220.tif')
print('elevation_path =',elevation_path)

try:
    print("Reading CRS from the raster file...")
    with rasterio.open(elevation_path) as dataset:
        raster_crs = dataset.crs.to_string()  # Extract CRS as a string
        print(f"CRS read successfully: {raster_crs}")
except Exception as e:
    print(f"Error reading CRS from raster file: {e}")

