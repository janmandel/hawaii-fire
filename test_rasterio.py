import rasterio

elevation_path = "../data/LH20_Elev_220.tif"  # Replace with your raster file path

try:
    print("Opening raster file...")
    with rasterio.open(elevation_path) as dataset:
        print(f"Raster CRS: {dataset.crs}")
        print(f"Raster Bounds: {dataset.bounds}")
        print(f"Raster Transform: {dataset.transform}")
        print(f"Raster Shape: {dataset.shape}")
    print("Raster file opened successfully!")
except Exception as e:
    print(f"Error opening raster file: {e}")

