from pyproj import CRS

# get the CRS here

try:
    raster_crs_obj = CRS(raster_crs)  # Replace `raster_crs` with the actual CRS object/string
    print("CRS is valid:", raster_crs_obj)
except Exception as e:
    print("Error with raster CRS:", e)

