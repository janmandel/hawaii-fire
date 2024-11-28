from pyproj import Transformer

# Test CRS transformation from WGS84 to Mercator (EPSG:3857)
try:
    print("Starting a simple CRS transformation test...")
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    lon, lat = -155.3, 19.6  # Example coordinates
    result = transformer.transform(lon, lat)
    print(f"Transformation successful: {result}")
except Exception as e:
    print(f"Error during transformation: {e}")

