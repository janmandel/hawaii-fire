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
from rasterio.transform import rowcol, xy
from pyproj import CRS, Transformer
from dbfread import DBF
from datetime import datetime, timedelta
import time
import random
import argparse

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
        # Read data
        elevation_data = elev.read(1)
        slope_data = slope.read(1)
        aspect_data = aspect.read(1)
        crs = elev.crs.to_string()
        transform = elev.transform

        print(f"The crs is: {crs}")
        print(f"The transform is: {transform}")

        # Replace nodata values and values < 0 with NaN and add debug for each variable
        elevation_data = np.where((elevation_data == elev.nodata) | (elevation_data < 0), np.nan, elevation_data)
        slope_data = np.where((slope_data == slope.nodata) | (slope_data < 0), np.nan, slope_data)
        aspect_data = np.where((aspect_data == aspect.nodata) | (aspect_data < 0), np.nan, aspect_data)
        print(f"Topography data: Converted NoData and values < 0 to NaN.")
        print(f"  - NaN values in elevation: {np.isnan(elevation_data).sum()}")
        print(f"  - NaN values in slope: {np.isnan(slope_data).sum()}")
        print(f"  - NaN values in aspect: {np.isnan(aspect_data).sum()}")

        # Return the data
        return {
            "elevation": elevation_data,
            "slope": slope_data,
            "aspect": aspect_data,
            "crs": crs,
            "transform": transform,
        }

def load_vegetation(file_paths):
    """
    Load vegetation data, map pixel values to vegetation classes, and handle specific replacements.
    """
    print("Loading vegetation data...")

    with rasterio.open(file_paths['fuelmod_path']) as fuelmod_dataset:
        # Read data and retrieve nodata value
        fuelmod_data = fuelmod_dataset.read(1)
        fuelmod_nodata = fuelmod_dataset.nodata

        # Replace nodata values with NaN
        if fuelmod_nodata is not None:
            fuelmod_nodata_count = np.sum(fuelmod_data == fuelmod_nodata)
            fuelmod_data = np.where(fuelmod_data == fuelmod_nodata, np.nan, fuelmod_data)
            print(f"Replaced {fuelmod_nodata_count} nodata values in 'fuelmod' with NaN.")
        else:
            print("No nodata value defined for 'fuelmod'.")

    # Load VAT file and map pixel values to vegetation classes
    vat_df = pd.DataFrame(iter(DBF(file_paths['fuelvat_path']))).sort_values(by='VALUE').reset_index(drop=True)
    value_to_class = dict(zip(vat_df['VALUE'], vat_df['FBFM13']))

    # Map fuelmod values to class names
    fuel_classes = np.vectorize(value_to_class.get)(fuelmod_data)

    # Replace 'Barren' and 'Water' with NaN
    replace_classes = ['Barren', 'Water', 'Fill-NoData']
    replace_count = np.isin(fuel_classes, replace_classes).sum()
    fuel_classes = np.where(np.isin(fuel_classes, replace_classes), np.nan, fuel_classes)
    print(f"Replaced {replace_count} values ('Barren', 'Water', 'Fill-NoData') in 'fuelmod' with NaN.")

    return fuel_classes

def load_meteorology(file_paths, start_index = 0, end_index = -1 ):
    """
    Load meteorology data from a NetCDF file.
    """
    print("Loading meteorology data...")
    data = nc.Dataset(file_paths['process_path'])
    return {
        "rain": data.variables['RAIN'][start_index:end_index, :, :],
        "temp": data.variables['T2'][start_index:end_index, :, :],
        "vapor": data.variables['Q2'][start_index:end_index, :, :],
        "wind_u": data.variables['U10'][start_index:end_index, :, :],
        "wind_v": data.variables['V10'][start_index:end_index, :, :],
        "swdwn": data.variables['SWDOWN'][start_index:end_index, :, :],
        "swup": data.variables['SWUPT'][start_index:end_index, :, :],
        "press": data.variables['PSFC'][start_index:end_index, :, :],
        "lon_grid": data.variables['XLONG'][:, :],
        "lat_grid": data.variables['XLAT'][:, :],
        "times": pd.to_datetime([t.strip() for t in data.variables['times'][start_index:end_index]], format='%Y-%m-%d_%H:%M:%S', errors='coerce')
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

    print(f"Loaded {len(X)} data points, filtered down to {len(X_filtered)} based on confidence of labels and time.")

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

def interpolate_all(satellite_coords, time_indices, interp, meteorology, topography, vegetation, labels, debug):
    """
    Perform batch interpolation for all satellite coordinates and times.
    Only valid points are included in the output.
    """

    # Validate and filter time indices separately
    valid_time_mask = (time_indices >= 0) & (time_indices < len(meteorology['times']))
    time_indices = time_indices[valid_time_mask]

    # Debug: Validate time indices
    if debug:
        valid_time_range = (0, len(meteorology['times']) - 1)
        invalid_time_indices = np.sum((time_indices < valid_time_range[0]) | (time_indices > valid_time_range[1]))
        print(f"Debug: Total time indices: {len(time_indices)}")
        print(f"Debug: Invalid time indices: {invalid_time_indices}")
        print(f"Debug: First few time indices: {time_indices[:10]}")
        print(f"Debug: The first timestamp is: {meteorology['times'][time_indices[0]]}")
        print(f"Debug: The final timestamp is: {meteorology['times'][time_indices[-1]]}")
        if invalid_time_indices > 0:
            raise ValueError(f"Invalid time indices detected: {invalid_time_indices}")

    # Ensure all data lengths match
    if len(time_indices) != len(satellite_coords):
        raise ValueError("Mismatch between time_indices and satellite_coords after masking.")

    # Calculate progress intervals
    total_records = len(satellite_coords)
    progress_steps = 20  # Number of updates you want
    progress_intervals = set((np.linspace(0, total_records - 1, progress_steps + 1)).astype(int))
    progress_interval = max(total_records // 10, 1)  # Log progress every 10%

    # Initialize a variable to track the last valid time index
    last_time_idx = None

    # Initialize transformer
    transformer = Transformer.from_crs("EPSG:4326", topography['crs'], always_xy=True)

    # Init List for storing dictionaries of interpolated values
    data_interp = []
    print("Entering the interpolation loop...")

    for idx, ((lon, lat), time_idx, label) in enumerate(
            zip(satellite_coords, time_indices, labels)):
        try:
            # Check if time_idx corresponds to a valid entry in meteorology['times']
            if pd.isna(meteorology['times'][time_idx]) or meteorology['times'][time_idx] == '                   ':
                if debug and time_idx != last_time_idx:
                    print(f"Skipping iteration due to invalid timestamp at index {idx}: time_idx={time_idx}")
                last_time_idx = time_idx  # Update the last checked time index
                continue

            # Reproject lon/lat to raster CRS
            #print("Transforming coordinates to raster CRS...")
            raster_lon, raster_lat = transformer.transform(np.array([lon]), np.array([lat]))

            # Calculate row and column indices
            transform = topography["transform"]
            inv_transform = ~transform
            col, row = inv_transform * (raster_lon, raster_lat)

            # Round indices and convert to integers
            row = int(np.round(row).item())
            col = int(np.round(col).item())

            # Extract raster features
            if 0 <= row < topography["elevation"].shape[0] and 0 <= col < topography["elevation"].shape[1]:
                elevation_val = topography["elevation"][row, col]
                slope_val = topography["slope"][row, col]
                aspect_val = topography["aspect"][row, col]
                fuelmod_val = vegetation[row, col]
            else:
                if debug and idx in progress_intervals:
                    print(f"Skipping iteration due to invalid raster indices for ({lon},{lat}) at row={row}, col={col}")
                continue

            # Check for NaN values in topography or vegetation
            if (
                    np.isnan(elevation_val) or
                    np.isnan(slope_val) or
                    np.isnan(aspect_val) or
                    pd.isna(fuelmod_val)
            ):
                if debug and idx in progress_intervals:
                    print(f"Skipping iteration due to NaN values at row={row}, col={col}")
                continue

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

            # Debug: Check extracted values
            if debug and idx % progress_interval == 0:
                print("=" * 50)  # Separator for clarity
                print(f"Record {idx + 1}")
                print(f"Date: {meteorology['times'][time_idx]}")
                print(f"Corresponding Time index: {time_idx}")
                print(f"Coordinates: Longitude = {lon}, Latitude = {lat}")
                print(f"Interpolated Indices(rounded): i = {i}, j = {j}")
                print(f"Interpolated Raster Indices: Row = {row}, Column = {col}")
                print(f"Label: {label}")
                print("Meteorological Data:")
                print(f"  Temperature: {temp_val}")
                print(f"  Rain: {rain_val}")
                print(f"  Relative Humidity: {rhum_val}")
                print(f"  Wind Speed: {wind_val}")
                print(f"  Net Shortwave Radiation (SWDOWN - SWUP): {sw_val}")
                print("Topographical Data:")
                print(f"  Elevation: {elevation_val}")
                print(f"  Slope: {slope_val}")
                print(f"  Aspect: {aspect_val}")
                print(f"  Fuel Model: {fuelmod_val}")
                print("=" * 50)  # End separator

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
            if idx in progress_intervals:
                print(f"Processed {idx + 1} out of {total_records} records...")

        except Exception as e:
            if debug:
                print(f"Error processing record {idx + 1}: {e}")

    return pd.DataFrame(data_interp)


def test_function(file_paths, subset_start, subset_end, min_fire_detections, max_subset_size, confidence_threshold, debug):
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

    # Step 2: Load topography, vegetation, and row/col mask
    topography = load_topography(file_paths)
    vegetation = load_vegetation(file_paths)

    # Step 3: Load and filter fire detection data
    fire_detection_data = load_fire_detection(file_paths, time_lb, time_ub, confidence_threshold)

    # Extract relevant data
    lon_array = fire_detection_data['lon']
    lat_array = fire_detection_data['lat']
    dates_fire = fire_detection_data['dates_fire']
    labels = fire_detection_data['labels']

    # Step 5: Define subset with sufficient fire detections
    if subset_start is None or subset_end is None:
        print("Selecting a continuous subset with sufficient fire detections...")

        # Sort all relevant arrays by dates_fire
        sorted_indices = np.argsort(dates_fire)
        lon_array = lon_array[sorted_indices]
        lat_array = lat_array[sorted_indices]
        dates_fire = dates_fire[sorted_indices]
        labels = labels[sorted_indices]

        # Find a continuous range with enough fire detections
        total_points = len(dates_fire)
        subset_start = 0
        subset_end = None

        for i in range(total_points - min_fire_detections):
            # Count fire detections in the current range
            fire_count = np.sum(labels[i:i + min_fire_detections] == 1)
            if fire_count >= min_fire_detections:
                subset_start = i
                subset_end = i + min_fire_detections
                break

        if subset_end is None:
            print("Insufficient fire detections. Returning all available data as a fallback.")
            subset_start = 0
            subset_end = total_points

        # Extend range to include non-fire detections up to max_subset_size
        subset_end = min(subset_start + max_subset_size, total_points)

    # Select the continuous subset
    lon_array = lon_array[subset_start:subset_end]
    lat_array = lat_array[subset_start:subset_end]
    dates_fire = dates_fire[subset_start:subset_end]
    labels = labels[subset_start:subset_end]

    # Log subset statistics
    selected_date_range = f"{dates_fire.min()} to {dates_fire.max()}"
    print(f"Selected range: start={subset_start}, end={subset_end}")
    print(f"Selected date range: {selected_date_range}")
    print(f"Subset size: {subset_end - subset_start}")
    print(f"Number of 'Fire' labels (1): {np.sum(labels == 1)}")
    print(f"Number of 'Non-Fire' labels (0): {np.sum(labels == 0)}")

    # Step 6: Build interpolator
    print("Building interpolator...")
    interp = Coord_to_index(degree=2)
    interp.build(meteorology['lon_grid'], meteorology['lat_grid'])

    # Step 7: Compute time indices
    time_indices = compute_time_indices(dates_fire, meteorology['times'], debug)

    # Step 8: Perform interpolation
    satellite_coords = np.column_stack((lon_array, lat_array))
    interpolated_data = interpolate_all(
        satellite_coords,
        time_indices,
        interp,
        meteorology,
        topography,
        vegetation,
        labels,
        debug
    )

    # Step 9: Save and return results
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
    min_fire_detections = 1000
    max_subset_size = 50000000  # Define maximum subset size
    confidence_threshold = 70

    # Toggle testing mode and debug mode
    test = True  # Set to False to run the full workflow
    debug = False # Set to False when the bugs are gone

    if test:
        test_data = test_function(file_paths, subset_start, subset_end, min_fire_detections, max_subset_size, confidence_threshold,
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
