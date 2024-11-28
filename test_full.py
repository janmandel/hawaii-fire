import rasterio
from pyproj import Transformer

# Path to the raster file
elevation_path = "LH20_Elev_220.tif"  # Replace with the actual path

try:
    print("Step 1: Opening raster file...")
    with rasterio.open(elevation_path) as dataset:
        print("Step 2: Raster file opened successfully.")
        print(f"Raster CRS: {dataset.crs}")
        
        print("Step 3: Creating transformer...")
        transformer = Transformer.from_crs("EPSG:4326", dataset.crs.to_string(), always_xy=True)
        print("Step 4: Transformer created successfully.")
        
        print("Step 5: Starting coordinate transformation...")
        raster_x, raster_y = transformer.transform(-155.3, 19.6)
        print(f"Step 6: Transformation complete: {raster_x}, {raster_y}")
except Exception as e:
    print(f"Error: {e}")

