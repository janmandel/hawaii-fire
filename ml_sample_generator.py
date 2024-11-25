# Necessary Imports
## In-house packages
from lonlat_interp import Coord_to_index
from saveload import load
## External packages
import os
from os import path as osp
import netCDF4 as nc
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from pyproj import CRS, Transformer
from dbfread import DBF
from datetime import datetime, timedelta
import psutil
import time
import random

# Functions
def get_file_paths():
    """
    Define and validate file paths.
    Returns a dictionary of paths.
    """
    base_dir = osp.join('/', 'home', 'spearsty', 'p', 'data')
    file_paths = {
        "slope_path": osp.join(base_dir, 'feat', 'landfire', 'top','LF2020_SlpP_220_HI', 'LH20_SlpP_220.tif'),
        "elevation_path": osp.join(base_dir, 'feat', 'landfire', 'top', 'LF2020_Elev_220_HI', 'LH20_Elev_220.tif'),
        "aspect_path": osp.join(base_dir, 'feat', 'landfire', 'top', 'LF2020_Asp_220_HI', 'LH20_Asp_220.tif'),
        "fuelmod_path": osp.join(base_dir, 'feat', 'landfire', 'afbfm', 'LF2022_FBFM13_230_HI', 'LH22_F13_230.tif'),
        "fuelvat_path": osp.join(base_dir, 'feat', 'landfire', 'afbfm', 'LF2022_FBFM13_230_HI', 'LH22_F13_230.tif.vat.dbf'),
        "process_path": osp.join(base_dir, 'feat', 'weather', 'processed_output.nc'),
        "fire_path": osp.join(base_dir, 'targ', 'Hawaii-all_2024-10-29_16:36:26', 'ml_data')
    }
    for name, path in file_paths.items():
        if not osp.exists(path):
            raise FileNotFoundError(f"ERROR: File {name} does not exist at {path}")
    return file_paths

def load_topography(file_paths):
    """
    Load topography data and associated metadata from GeoTIFF files.

    Args:
        file_paths (dict): Dictionary containing paths to elevation, slope, and aspect files.

    Returns:
        dict: Contains topography arrays ('elevation', 'slope', 'aspect'), CRS, and transform.
    """
    print("Loading topography data...")
    with rasterio.open(file_paths['elevation_path']) as elev, \
         rasterio.open(file_paths['slope_path']) as slope, \
         rasterio.open(file_paths['aspect_path']) as aspect:
        return {
            "elevation": elev.read(1),
            "slope": slope.read(1),
            "aspect": aspect.read(1),
            "crs": elev.crs.to_string(),
            "transform": elev.transform,
        }

def load_vegetation(file_paths):
    """
    Load vegetation data and map pixel values to vegetation classes.
    """
    print("Loading vegetation data...")
    fuelmod = rasterio.open(file_paths['fuelmod_path']).read(1)
    vat_df = pd.DataFrame(iter(DBF(file_paths['fuelvat_path']))).sort_values(by='VALUE').reset_index(drop=True)
    value_to_class = dict(zip(vat_df['VALUE'], vat_df['FBFM13']))
    return np.vectorize(value_to_class.get)(fuelmod)

def load_meteorology(file_paths):
    """
    Load meteorology data from a NetCDF file.
    """
    print("Loading meteorology data...")
    data = nc.Dataset(file_paths['process_path'])
    return {
        "rain": data.variables['RAIN'][:, :, :],
        "temp": data.variables['T2'][:, :, :],
        "vapor": data.variables['Q2'][:, :, :],
        "wind_u": data.variables['U10'][:, :, :],
        "wind_v": data.variables['V10'][:, :, :],
        "swdwn": data.variables['SWDOWN'][:, :, :],
        "swup": data.variables['SWUPT'][:, :, :],
        "press": data.variables['PSFC'][:, :, :],
        "lon_grid": data.variables['XLONG'][:, :],
        "lat_grid": data.variables['XLAT'][:, :],
        "times": pd.to_datetime([t.strip() for t in data.variables['times'][:]], format='%Y-%m-%d_%H:%M:%S', errors='coerce')
    }

def load_fire_detection(file_paths, time_lb, time_ub, confidence_threshold):
    """
    Load and process fire detection data.
    Retains all points but filters out those with a label of 1 and confidence < confidence_threshold.
    and time indices that aren't in [time_lb, time_ub]

    """
    print("Loading fire detection data...")
    X, y, c, basetime = load(file_paths['fire_path'])
    time_in_days_raw = X[:, 2]
    dates_fire_actual_raw = basetime + pd.to_timedelta(time_in_days_raw, unit='D')
    dates_fire_raw = dates_fire_actual_raw.floor("h")  # Round to nearest hou

    # Debug: Print initial statistics
    print(f"Total data points: {len(X)}")
    print(f"Number of 'Fire' labels: {np.sum(y == 1)}")
    print(f"Number of 'Fire' labels with confidence < {confidence_threshold}: {np.sum((y == 1) & (c < confidence_threshold))}")

    # Filter out points with label 1, confidence < confidence_threshold,and outside of time bounds
    valid_indices = (
            ~((y == 1) & (c < confidence_threshold)) & (dates_fire_raw >= time_lb) & (dates_fire_raw <= time_ub)
    ) # Keep points not failing this condition
    X_filtered = X[valid_indices]
    y_filtered = y[valid_indices]

    # Debug: Print post-filtering statistics
    print(f"Number of remaining data points: {len(X_filtered)}")
    print(f"Number of remaining 'Fire' labels: {np.sum(y_filtered == 1)}")

    # Extract filtered components
    lon_array = X_filtered[:, 0]
    lat_array = X_filtered[:, 1]
    time_in_days = X_filtered[:, 2]
    dates_fire_actual = basetime + pd.to_timedelta(time_in_days, unit='D')
    dates_fire = dates_fire_actual.floor("h")  # Round to nearest hour

    print(f"Loaded {len(X)} data points, filtered down to {len(X_filtered)} based on confidence and labels.")

    return {
        "lon": lon_array,
        "lat": lat_array,
        "time_days": time_in_days,
        "dates_fire": dates_fire,
        "labels": y_filtered
    }


def compute_time_indices(satellite_times, processed_times, debug):
    """
    Compute the number of hours since the start of processed data for each satellite time.
    Ensure alignment between satellite and processed data timestamps.

    Args:
        satellite_times (pd.Series or pd.DatetimeIndex): Satellite observation times.
        processed_times (pd.Series or pd.DatetimeIndex): Times in the processed meteorology dataset.
        debug (bool): If True, print debug information and validate time indices.

    Returns:
        np.ndarray: Array of computed time indices.
    """
    print("Computing the time indices for the fire detection data...")
    start_time = processed_times[0]
    hours_since_start = (satellite_times - start_time).total_seconds() // 3600
    indices = hours_since_start.astype(int)
    validate = debug
    total_records = len(processed_times)
    progress_interval = max(total_records // 10, 1)  # Log progress every 10%

    if debug:
        print(f"Debug: Time index range: {indices.min()} to {indices.max()}")
        print(f"Debug: First few time indices: {indices[:10]}")
        print(f"Debug: Satellite times min/max: {satellite_times.min()} / {satellite_times.max()}")
        print(f"Debug: Processed times min/max: {processed_times.min()} / {processed_times.max()}")

    if validate:
        print("Validating indices...")
        for i in range(len(indices)):
            idx = indices[i]
            sat_time = satellite_times[i]  # Use direct indexing here

            # Check index validity
            if 0 <= idx < len(processed_times):
                processed_time = processed_times[idx]
                if abs((processed_time - sat_time).total_seconds()) > 3600:
                    if debug:
                        print(f"Debug: Mismatch at index {i}: processed_time={processed_time}, sat_time={sat_time}")
                    raise ValueError(
                        f"Mismatch: Processed time {processed_time} does not match satellite time {sat_time}.")
            else:
                if debug:
                    print(f"Debug: Index {idx} out of bounds at record {i}: sat_time={sat_time}")
                raise IndexError(f"Index {idx} out of bounds for processed data times.")

                # Progress logging
                if (i + 1) % progress_interval == 0 or i + 1 == total_records:
                    print(f"Processed {i + 1} out of {total_records} records...")

    return indices


def calc_rhum(temp_K, mixing_ratio, pressure_pa):
    """
    Calculate relative humidity from temperature and mixing ratio.
    """
    epsilon = 0.622
    es_hpa = 6.112 * np.exp((17.67 * (temp_K - 273.15)) / ((temp_K - 273.15) + 243.5))
    es_pa = es_hpa * 100
    e_pa = (mixing_ratio * pressure_pa) / (epsilon + mixing_ratio)
    return (e_pa / es_pa) * 100


def get_row_col(lon_array, lat_array, raster_crs, transform, debug):
    """
    Get the row and column indices in a raster for arrays of longitudes and latitudes.
    To be used for retrieval of values from GeoTIFFs at longitude and latitude.

    Args:
        lon_array (np.ndarray): Array of longitudes in WGS84.
        lat_array (np.ndarray): Array of latitudes in WGS84.
        raster_crs (str): CRS of the raster (e.g., "EPSG:5070" for Albers Equal Area).
        transform (Affine): Rasterio affine transform of the raster.
        debug (bool): If True, prints debug information.

    Returns:
        tuple: Arrays of row and column indices in the raster.
    """
    print('Computing row and column indices for topography and vegetation files...')

    # Reproject lon, lat arrays to the raster's CRS
    transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    raster_lon, raster_lat = transformer.transform(lon_array, lat_array)

    # Debug: Check reprojected coordinates
    if debug:
        print(f"Debug: Given lon min/max: {lon_array.min()} / {lon_array.max()}")
        print(f"Debug: Given lat min/max: {lat_array.min()} / {lat_array.max()}")
        print(f"Debug: Reprojected lon min/max: {raster_lon.min()} / {raster_lon.max()}")
        print(f"Debug: Reprojected lat min/max: {raster_lat.min()} / {raster_lat.max()}")

    # Calculate row and column indices using vectorized transformation
    inv_transform = ~transform
    cols, rows = inv_transform * (raster_lon, raster_lat)

    # Round to nearest integer and convert to int
    rows = np.round(rows).astype(int)
    cols = np.round(cols).astype(int)

    # Debug: Check row/col indices
    if debug:
        print(f"Debug: Row indices min/max: {rows.min()} / {rows.max()}")
        print(f"Debug: Column indices min/max: {cols.min()} / {cols.max()}")

    return rows, cols

def interpolate_all(satellite_coords, time_indices, interp, meteorology, topography, vegetation, labels, debug):
    """
    Perform batch interpolation for all satellite coordinates and times.
    Only valid points are included in the output.
    """
    print('Entering the interpolation loop...')
    data_interp = []
    total_records = len(satellite_coords)
    progress_interval = max(total_records // 10, 1)  # Log progress every 10%

    # Debug: Validate time indices
    if debug:
        valid_time_range = (0, len(meteorology['times']) - 1)
        invalid_time_indices = np.sum((time_indices < valid_time_range[0]) | (time_indices > valid_time_range[1]))
        print(f"Debug: Total time indices: {len(time_indices)}")
        print(f"Debug: Invalid time indices: {invalid_time_indices}")
        print(f"Debug: First few time indices: {time_indices[:10]}")
        print(f"Debug: The first timestamp is: {time_indices[0]}")
        print(f"Debug: The final timestamp is: {time_indices[-1]}")
        if invalid_time_indices > 0:
            raise ValueError(f"Invalid time indices detected: {invalid_time_indices}")

    # Precompute raster indices
    rows, cols = get_row_col(
        lon_array=satellite_coords[:, 0],
        lat_array=satellite_coords[:, 1],
        raster_crs=topography["crs"],
        transform=topography["transform"],
        debug = debug
    )

    # Debug: Validate raster indices
    if debug:
        out_of_bounds = (
            (rows < 0) | (rows >= topography["elevation"].shape[0]) |
            (cols < 0) | (cols >= topography["elevation"].shape[1])
        )
        print(f"Raster indices out of bounds: {np.sum(out_of_bounds)} / {len(rows)}")

    for idx, ((lon, lat), time_idx, label, row, col) in enumerate(
            zip(satellite_coords, time_indices, labels, rows, cols)):
        try:
            # Interpolate meteorological data
            ia, ja = interp.evaluate(lon, lat)
            i, j = np.round(ia).astype(int), np.round(ja).astype(int)

            # Skip invalid meteorological indices
            if not (0 <= i < meteorology['temp'].shape[1] and 0 <= j < meteorology['temp'].shape[2]):
                if debug:
                    print(f"Skipped invalid meteorological indices: i={i}, j={j}")
                continue

            # Extract meteorological data
            temp_val = meteorology['temp'][time_idx, i, j]
            rain_val = meteorology['rain'][time_idx, i, j]
            rhum_val = calc_rhum(meteorology['temp'][time_idx, i, j], meteorology['vapor'][time_idx, i, j],
                                 meteorology['press'][time_idx, i, j])
            wind_val = np.sqrt(
                meteorology['wind_u'][time_idx, i, j] ** 2 + meteorology['wind_v'][time_idx, i, j] ** 2
            )
            sw_val = meteorology['swdwn'][time_idx, i, j] - meteorology['swup'][time_idx, i, j]

            # Extract raster features
            if 0 <= row < topography["elevation"].shape[0] and 0 <= col < topography["elevation"].shape[1]:
                elevation_val = topography["elevation"][row, col]
                slope_val = topography["slope"][row, col]
                aspect_val = topography["aspect"][row, col]
                fuelmod_val = vegetation[row, col]
            else:
                elevation_val, slope_val, aspect_val, fuelmod_val = np.nan, np.nan, np.nan, "Out of bounds"

            # Debug: Check extracted values
            if debug and idx % progress_interval == 0:
                print(f"Record {idx + 1}: temp={temp_val}, rain={rain_val}, rhum={rhum_val}, "
                      f"elevation={elevation_val}, slope={slope_val}, aspect={aspect_val}")

            # Append results
            data = {
                'date': meteorology['times'][time_idx],
                'lon': lon,
                'lat': lat,
                'temp': temp_val,
                'rain': rain_val,
                'rhum': rhum_val,
                'wind': wind_val,
                'sw': sw_val,
                'elevation': elevation_val,
                'slope': slope_val,
                'aspect': aspect_val,
                'fuelmod': fuelmod_val,
                'label': label,
            }
            data_interp.append(data)

            # Progress logging
            if (idx + 1) % progress_interval == 0 or idx + 1 == total_records:
                print(f"Processed {idx + 1} out of {total_records} records...")

        except Exception as e:
            if debug:
                print(f"Error processing record {idx + 1}: {e}")

    return pd.DataFrame(data_interp)


def test_function(file_paths, subset_start, subset_end, min_fire_detections, confidence_threshold, debug):
    """
    Test the workflow with a subset of the data for debugging or validation.
    Dynamically adjust subset indices to include enough fire detections (label=1).

    Args:
        file_paths (dict): Dictionary containing paths to required data files.
        subset_start (int): Start index for the subset.
        subset_end (int): End index for the subset.
        min_fire_detections (int): Minimum number of fire detections (label=1) to include in the subset.
        confidence_threshold (float): Minimum confidence for filtering fire detections.
        debug (bool): Whether to enable debug logs.

    Returns:
        pd.DataFrame: Interpolated test data.
    """
    print("Running test function...")

    # Step 1: Load meteorology data
    meteorology = load_meteorology(file_paths)
    time_lb = meteorology['times'].min()
    time_ub = meteorology['times'].max()
    print(f"Meteorology time range: {time_lb} to {time_ub}")

    # Step 2: Load topography and vegetation data
    topography = load_topography(file_paths)
    vegetation = load_vegetation(file_paths)

    # Step 3: Load and filter fire detection data
    fire_detection_data = load_fire_detection(file_paths, time_lb, time_ub, confidence_threshold)

    # Extract relevant data
    lon_array = fire_detection_data['lon']
    lat_array = fire_detection_data['lat']
    dates_fire = fire_detection_data['dates_fire']
    labels = fire_detection_data['labels']

    # Step 4: Define subset with sufficient fire and non-fire labels
    if subset_start is None or subset_end is None:
        print("Selecting a subset with sufficient fire detections...")
        fire_indices = np.where(labels == 1)[0]  # Indices of fire labels
        non_fire_indices = np.where(labels == 0)[0]  # Indices of non-fire labels

        if len(fire_indices) < min_fire_detections:
            raise ValueError("Not enough fire detections in the data to satisfy the minimum requirement.")

        # Select fire indices
        selected_fire_indices = np.random.choice(fire_indices, min_fire_detections, replace=False)

        # Limit total subset size (e.g., max 100,000 points)
        max_subset_size = 100000
        remaining_slots = max_subset_size - len(selected_fire_indices)

        # Select non-fire indices to fill the remaining slots
        selected_non_fire_indices = np.random.choice(
            non_fire_indices, min(remaining_slots, len(non_fire_indices)), replace=False
        )

        # Combine and sort indices
        selected_indices = np.sort(np.concatenate([selected_fire_indices, selected_non_fire_indices]))
        subset_start, subset_end = selected_indices[0], selected_indices[-1] + 1  # Adjust range

    # Select subset
    print(f"Selected range: start={subset_start}, end={subset_end}")
    lon_array = lon_array[subset_start:subset_end]
    lat_array = lat_array[subset_start:subset_end]
    dates_fire = dates_fire[subset_start:subset_end]
    labels = labels[subset_start:subset_end]

    # Calculate and log statistics for the subset
    selected_date_range = f"{dates_fire.min()} to {dates_fire.max()}"
    label_counts = pd.Series(labels).value_counts()

    print(f"Selected date range: {selected_date_range}")
    print(f"Subset size: {len(labels)}")
    print(f"Number of 'Fire' labels (1): {label_counts.get(1, 0)}")
    print(f"Number of 'Non-Fire' labels (0): {label_counts.get(0, 0)}")

    # Step 5: Build interpolator
    print("Building interpolator...")
    interp = Coord_to_index(degree=2)
    interp.build(meteorology['lon_grid'], meteorology['lat_grid'])

    # Step 6: Compute time indices
    time_indices = compute_time_indices(dates_fire, meteorology['times'], debug)

    # Step 7: Perform interpolation
    satellite_coords = np.column_stack((lon_array, lat_array))
    interpolated_data = interpolate_all(satellite_coords, time_indices, interp, meteorology, topography, vegetation, labels, debug)

    # Step 8: Save and return results
    print("Saving test results...")
    interpolated_data.to_pickle("test_processed_data.pkl")
    print("Test data saved to 'test_processed_data.pkl'.")
    return interpolated_data

# Main Execution
if __name__ == "__main__":
    # Load and validate paths
    file_paths = get_file_paths()

    # Define parameters
    subset_start = None  # Let the function compute based on fire detections
    subset_end = None
    min_fire_detections = 20
    confidence_threshold = 70

    # Toggle testing mode and debug mode
    test = True  # Set to False to run the full workflow
    debug = True # Set to False when the bugs are gone

    if test:
        test_data = test_function(file_paths, subset_start, subset_end, min_fire_detections, confidence_threshold,
                                  debug)

        if test_data is not None:
            print("Test run completed successfully. Displaying head of the DataFrame:")
            print(test_data.head())  # Fix: Call `.head()` as a method
        else:
            print("Test run failed or returned no data.")
    else:
        print("Skipping testing, running the full workflow...")

        # Load data
        topography = load_topography(file_paths)
        vegetation = load_vegetation(file_paths)
        meteorology = load_meteorology(file_paths)
        time_lb = meteorology['times'].min()
        time_ub = meteorology['times'].max()

        # Load fire detection data
        fire_detection_data = load_fire_detection(file_paths, time_lb, time_ub, confidence_threshold)
        lon_array = fire_detection_data['lon']
        lat_array = fire_detection_data['lat']
        dates_fire = fire_detection_data['dates_fire']
        labels = fire_detection_data['labels']

        # Build interpolator
        print("Building the interpolator...")
        interp = Coord_to_index(degree=2)
        interp.build(meteorology['lon_grid'], meteorology['lat_grid'])

        # Compute time indices
        time_indices = compute_time_indices(dates_fire, meteorology['times'], debug)

        # Perform interpolation
        satellite_coords = np.column_stack((lon_array, lat_array))
        interpolated_data = interpolate_all(satellite_coords, time_indices, interp, meteorology, topography, vegetation, labels, debug)

        # Save interpolated data
        interpolated_data.to_pickle('processed_data.pkl')
        print(f"Interpolated data saved to 'processed_data.pkl'.")
