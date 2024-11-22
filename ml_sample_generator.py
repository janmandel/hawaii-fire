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
    Load topography data from GeoTIFF files.
    """
    print("Loading topography data...")
    return {
        "elevation": rasterio.open(file_paths['elevation_path']).read(1),
        "aspect": rasterio.open(file_paths['aspect_path']).read(1),
        "slope": rasterio.open(file_paths['slope_path']).read(1)
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

def compute_time_indices(satellite_times, processed_times, validate=False):
    """
    Compute the number of hours since the start of processed data for each satellite time.
    Ensure alignment between satellite and processed data timestamps.
    """
    print("Computing the time indices for the fire detection data")
    # Calculate hours since the start of processed times
    start_time = processed_times[0]
    hours_since_start = (satellite_times - start_time).total_seconds() // 3600
    indices = hours_since_start.astype(int)

    # Validate indices
    if validate:
        print("Validating indices...")
        for i in range(len(indices)):
            idx = indices[i]
            sat_time = satellite_times[i]

            # Check index validity
            if 0 <= idx < len(processed_times):
                processed_time = processed_times[idx]
                if abs((processed_time - sat_time).total_seconds()) > 3600:
                    raise ValueError(f"Mismatch: Processed time {processed_time} does not match satellite time {sat_time}.")
            else:
                raise IndexError(f"Index {idx} out of bounds for processed data times.")

            # Print progress every 10,000 iterations
            if i % 1000000 == 0:
                print(f"Processed {i} out of {len(indices)} records...")

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

def interpolate_all(satellite_coords, time_indices, interp, variables, labels):
    """
    Perform batch interpolation for all satellite coordinates and times.
    Only valid points are included in the output.
    """
    print('Entering the interpolation loop...')
    data_interp = []
    total_records = len(satellite_coords)
    progress_interval = 1000000

    for idx, ((lon, lat), time_idx, label) in enumerate(zip(satellite_coords, time_indices)):
        ia, ja = interp.evaluate(lon, lat)
        i, j = np.round(ia).astype(int), np.round(ja).astype(int)

        # Skip invalid points
        if not (0 <= i < variables['temp'].shape[1] and 0 <= j < variables['temp'].shape[2]):
            continue

        # Append valid data
        data = {
            'date': variables['times'][time_idx],
            'lon': lon,
            'lat': lat,
            'temp': variables['temp'][time_idx, i, j],
            'rain': variables['rain'][time_idx, i, j],
            'rhum': calc_rhum(variables['temp'][time_idx, i, j], variables['vapor'][time_idx, i, j],variables['press'][time_idx, i, j]),
            'wind': np.sqrt(
                variables['wind_u'][time_idx, i, j]**2 + variables['wind_v'][time_idx, i, j]**2),
            'sw': variables['swdwn'][time_idx, i, j] - variables['swup'][time_idx, i, j],
            'label': label,
        }
        data_interp.append(data)

        # Debug statement for progress
        if (idx + 1) % progress_interval == 0 or idx + 1 == total_records:
            print(f"Processed {idx + 1} out of {total_records} records...")

    return pd.DataFrame(data_interp)

def test_function(file_paths, subset_size, confidence_threshold):
    """
    Test the workflow with a subset of the data for debugging or validation.

    """
    print(f"Running test function with subset size: {subset_size}")

    # Step 1: Load meteorology data
    print("Loading meteorology data...")
    meteorology = load_meteorology(file_paths)
    time_lb = meteorology['times'].min()
    time_ub = meteorology['times'].max()
    print(f"Meteorology time range: {time_lb} to {time_ub}")

    # Step 2: Load and filter fire detection data
    print("Loading and filtering fire detection data...")
    fire_detection_data = load_fire_detection(file_paths, time_lb, time_ub, confidence_threshold)
    lon_array = fire_detection_data['lon'][:subset_size]
    lat_array = fire_detection_data['lat'][:subset_size]
    dates_fire = fire_detection_data['dates_fire'][:subset_size]
    labels = fire_detection_data['labels'][:subset_size]

    # Step 3: Build interpolator
    print("Building interpolator...")
    interp = Coord_to_index(degree=2)
    interp.build(meteorology['lon_grid'], meteorology['lat_grid'])

    # Step 4: Compute time indices
    print("Computing time indices...")
    time_indices = compute_time_indices(dates_fire, meteorology['times'])

    # Step 5: Perform interpolation
    print("Performing interpolation...")
    satellite_coords = np.column_stack((lon_array, lat_array))
    interpolated_data = interpolate_all(satellite_coords, time_indices, interp, meteorology, labels)

    # Step 6: Save and return results
    print("Saving test results...")
    interpolated_data.to_pickle("test_processed_data.pkl")
    print("Test data saved to 'test_processed_data.pkl'.")
    return interpolated_data

# Main Execution
if __name__ == "__main__":
    # Load and validate paths
    file_paths = get_file_paths()

    # Define test parameters
    subset_size = 10000
    confidence_threshold = 70

    # Toggle testing mode
    test = True  # Set to False to run the full workflow

    if test:
        print("Running the test function...")
        test_data = test_function(file_paths, subset_size, confidence_threshold)

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
        time_indices = compute_time_indices(dates_fire, meteorology['times'], validate=False)

        # Perform interpolation
        satellite_coords = np.column_stack((lon_array, lat_array))
        interpolated_data = interpolate_all(satellite_coords, time_indices, interp, meteorology, labels)

        # Save interpolated data
        interpolated_data.to_pickle('processed_data.pkl')
        print(f"Interpolated data saved to 'processed_data.pkl'.")
