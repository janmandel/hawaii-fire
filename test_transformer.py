from pyproj import Transformer

# Use the raster CRS
raster_crs = "PROJCS[\"Albers\",GEOGCS[\"NAD83\",DATUM[\"North_American_Datum_1983\",SPHEROID[\"GRS 1980\",6378137,298.257222101004,AUTHORITY[\"EPSG\",\"7019\"]],AUTHORITY[\"EPSG\",\"6269\"]],PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4269\"]],PROJECTION[\"Albers_Conic_Equal_Area\"],PARAMETER[\"latitude_of_center\",19.5832215821859],PARAMETER[\"longitude_of_center\",-155.432511653083],PARAMETER[\"standard_parallel_1\",18.8709313186648],PARAMETER[\"standard_parallel_2\",20.295511845707],PARAMETER[\"false_easting\",0],PARAMETER[\"false_northing\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]],AXIS[\"Easting\",EAST],AXIS[\"Northing\",NORTH]]"

# Coordinates to transform
lon, lat = -155.3, 19.6

try:
    print("Starting transformation to raster CRS...")
    transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    raster_x, raster_y = transformer.transform(lon, lat)
    print(f"Transformation successful: {raster_x}, {raster_y}")
except Exception as e:
    print(f"Error during transformation: {e}")

