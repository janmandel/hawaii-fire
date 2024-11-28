import os
import sys
import hashlib
from os import path as osp 
from pyproj import Transformer
import numpy as np
import random
import rasterio 
import pyproj
from rasterio.transform import rowcol,xy

print('rasterio',rasterio.__version__)
print('pyproj',pyproj.__version__)
print('numpy',np.__version__)
print(f"Python version: {sys.version}")


def get_row_col(lon_array, lat_array, raster_crs, transform, raster_shape, debug):
    """
    Optimized function to compute row and column indices for arrays of longitudes and latitudes.

    Args:
        lon_array (np.ndarray): Array of longitudes in WGS84.
        lat_array (np.ndarray): Array of latitudes in WGS84.
        raster_crs (str): CRS of the raster (e.g., "EPSG:5070").
        transform (Affine): Rasterio affine transform of the raster.

    Returns:
        tuple: Arrays of row and column indices in the raster.
    """
    print('Computing row and column indices for topography and vegetation files...')

    print("Reproject lon/lat arrays to the raster's CRS using pyproj")
    transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    print('done')

    if debug:
        print(f"Debug: The raster CRS is: {raster_crs}")
        print(f"Debug: The raster shape is: {raster_shape}")
        print(f"Debug: The transform for the raster is: {transform}")
        print(f"Debug: lon_array shape: {lon_array.shape}, lat_array shape: {lat_array.shape}")
        print(f"Debug: lon_array: {lon_array}")
        print(f"Debug: lat_array: {lat_array}")
        print(f"Debug: NaNs in lon_array: {np.isnan(lon_array).any()}, NaNs in lat_array: {np.isnan(lat_array).any()}")
        print(f"Debug: lon_array min/max: {lon_array.min()} / {lon_array.max()}")
        print(f"Debug: lat_array min/max: {lat_array.min()} / {lat_array.max()}")
        print("Starting coordinate transformation...")  # Before transformer is built
        try:
            raster_lon, raster_lat = transformer.transform(lon_array[0], lat_array[0])
            print(f"Debug: Single-point transformation successful: {raster_lon}, {raster_lat}")
        except Exception as e:
            print(f"Debug: Error during single-point transformation: {e}")

    print("Transforming coordinates to raster CRS...")  # Before transformation
    raster_lon, raster_lat = transformer.transform(lon_array, lat_array)
    if debug:
        print("Coordinate transformation completed.")  # After transformation
        print("Starting affine transformation for row/col computation...")  # Before affine transformation

    # Calculate row and column indices using vectorized transformation
    inv_transform = ~transform
    cols, rows = inv_transform * (raster_lon, raster_lat)
    print(f"The affine transformation is complete and the row,col values have been extracted...")

    # Round to nearest integer and convert to int
    rows = np.round(rows).astype(int)
    cols = np.round(cols).astype(int)

    if debug:
        # Debugging: Check bounds, reprojected coordinates and other metrics
        print(f"Debug: WGS84 lon min/max (pre-mask): {lon_array.min()} / {lon_array.max()}")
        print(f"Debug: WGS84 lat min/max (pre-mask): {lat_array.min()} / {lat_array.max()}")
        print(f"Debug: Reprojected lon min/max (pre-mask): {raster_lon.min()} / {raster_lon.max()}")
        print(f"Debug: Reprojected lat min/max (pre-mask): {raster_lat.min()} / {raster_lat.max()}")
        print(f"Debug: Rows min/max(pre-mask): {rows.min()}, {rows.max()}")
        print(f"Debug: Cols min/max(pre-mask): {cols.min()}, {cols.max()}")
        print(f"Debug: The shape of rows, cols: {rows.shape, cols.shape}")

    print("Truncating rows, cols, lon_array, and lat_array based on valid raster array inputs...")

    # Create a mask to filter valid row/col indices
    rowcol_mask = (
            (rows >= 0) & (rows < raster_shape[0]) &
            (cols >= 0) & (cols < raster_shape[1])
    )

    # Calculate bounds in raster CRS
    raster_x_min, raster_y_min = transform * (0, raster_shape[0])
    raster_x_max, raster_y_max = transform * (raster_shape[1], 0)

    print("reproject bounds to WGS84")
    transformer = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)

    lon_min, lat_min = transformer.transform(raster_x_min, raster_y_min)
    lon_max, lat_max = transformer.transform(raster_x_max, raster_y_max)

    # Create a valid mask based on coordinate bounds via the raster boundaries
    coord_mask = (
            (lon_array >= lon_min) & (lon_array <= lon_max) &
            (lat_array >= lat_min) & (lat_array <= lat_max)
    )

    # Combine the masks
    valid_mask = rowcol_mask & coord_mask

    # Apply the mask and filter spatial data accordingly
    rows_valid = rows[valid_mask]
    cols_valid = cols[valid_mask]
    lon_array_valid = lon_array[valid_mask]
    lat_array_valid = lat_array[valid_mask]

    if debug:
        print(f"Raster bounds in WGS84: lon_min={lon_min}, lon_max={lon_max}, lat_min={lat_min}, lat_max={lat_max}")
        print(f"Debug: WGS84 lon min/max (post-mask): {lon_array_valid.min()} / {lon_array_valid.max()}")
        print(f"Debug: WGS84 lat min/max (post-mask): {lat_array_valid.min()} / {lat_array_valid.max()}")
        print(f"Debug: Reprojected lon min/max (post-mask): {raster_x_min} / {raster_x_max}")
        print(f"Debug: Reprojected lat min/max (post-mask): {raster_y_min} / {raster_y_max}")
        print(f"Debug: Rows min/max(post-mask): {rows_valid.min()}, {rows_valid.max()}")
        print(f"Debug: Cols min/max(post-mask): {cols_valid.min()}, {cols_valid.max()}")

    return rows_valid, cols_valid, lon_array_valid, lat_array_valid, valid_mask


def calculate_checksum(file_path, algorithm="md5"):
    """
    Calculate the checksum of a file.
    
    :param file_path: Path to the file
    :param algorithm: Hashing algorithm (e.g., 'md5', 'sha256', 'sha1')
    :return: Hexadecimal checksum
    """
    hash_function = getattr(hashlib, algorithm)()  # Get the hash function from hashlib
    with open(file_path, "rb") as file:
        while chunk := file.read(8192):  # Read the file in chunks to handle large files
            hash_function.update(chunk)
    return hash_function.hexdigest()

# Paths to the tif file
home = osp.expanduser("~")
base_dir = osp.abspath(osp.join(osp.expanduser("~"), 'p', 'data'))
elevation_path = osp.join(base_dir, 'feat', 'landfire', 'top', 'LF2020_Elev_220_HI', 'LH20_Elev_220.tif')
print('elevation_path =',elevation_path)
print('file checksum=',calculate_checksum(elevation_path))

# Load the elevation data
with rasterio.open(elevation_path,sharing=False) as elevation_dataset:
    elevation_data = elevation_dataset.read(1)
    elevation_transform = elevation_dataset.transform
    elevation_crs = elevation_dataset.crs
    elevation_nodata = elevation_dataset.nodata
    xmin, ymin, xmax, ymax = elevation_dataset.bounds

# Example lon and lat arrays
lon_array_test = np.array([-155.3])
lat_array_test = np.array([19.6])

# Get rows and cols for all coordinates
rows_valid, cols_valid, lon_array_valid, lat_array_valid, spatial_mask = get_row_col(lon_array_test, lat_array_test, elevation_dataset.crs.to_string(),elevation_transform, elevation_data.shape, True)
