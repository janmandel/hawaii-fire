# Necessary Imports
import netCDF4 as nc
import numpy as np
import pandas as pd
from lonlat_interp import Coord_to_index
from ml_sample_generator import load_meteorology
import os
from os import path as osp
from datetime import datetime

def extract_record(record, meteorology):
    """
    Extract meteorology data for a specific record.

    Args:
        record (dict): Information about the record.
        meteorology (dict): Loaded meteorology data.

    Returns:
        dict: Extracted meteorology data for the record.
    """
    lon, lat = record['lon'], record['lat']
    lon_array, lat_array = np.array([lon]), np.array([lat])
    date = pd.Timestamp(record['date'])

    # Debug Record Lon/Lat anda date
    print(f"Record date: {date}")
    print(f"Record Lon: {lon}, Lat: {lat}")
    print(f"lon_grid type: {type(meteorology['lon_grid'])}, shape: {meteorology['lon_grid'].shape}")
    print(f"lat_grid type: {type(meteorology['lat_grid'])}, shape: {meteorology['lat_grid'].shape}")

    # Find the matching time index
    try:
        time_idx = (meteorology['times'] == date).nonzero()[0][0]
    except IndexError:
        raise ValueError(f"Date {date} not found in meteorology times.")

    # Build interpolator
    print("Building interpolator...")
    interp = Coord_to_index(degree=2)
    interp.build(meteorology['lon_grid'], meteorology['lat_grid'])

    # Evaluate interpolator
    print("Evaluating interpolator...")
    ia, ja = interp.evaluate(lon_array, lat_array)
    print(f"Interpolator Output: ia={ia}, ja={ja}")

    # Validate indices
    i, j = np.round(ia).astype(int), np.round(ja).astype(int)
    print(f"Rounded Indices: i={i}, j={j}")
    if not (0 <= i < meteorology['temp'].shape[1] and 0 <= j < meteorology['temp'].shape[2]):
        raise ValueError(f"Invalid indices: Row = {i}, Col = {j}")

    # Extract values
    temp_val = meteorology['temp'][time_idx, i, j]
    rain_val = meteorology['rain'][time_idx, i, j]
    vapor_val = meteorology['vapor'][time_idx, i, j]
    press_val = meteorology['press'][time_idx, i, j]
    wind_u_val = meteorology['wind_u'][time_idx, i, j]
    wind_v_val = meteorology['wind_v'][time_idx, i, j]
    swdown_val = meteorology['swdwn'][time_idx, i, j]
    swup_val = meteorology['swup'][time_idx, i, j]

    # Derived values
    wind_speed = np.sqrt(wind_u_val**2 + wind_v_val**2)
    net_shortwave_radiation = swdown_val - swup_val

    print("=" * 50)
    print(f" True Spatial-Temporal Values; Date: {date}, Longitude: {lon}, Latitude: {lat}") 
    print(f" Extracted Values at Interpolated Indices; Time: {time_idx}, i: {i}, j: {j}")
    print(f"  Temperature: {temp_val}")
    print(f"  Rain: {rain_val}")
    print(f"  Vapor: {vapor_val}")
    print(f"  Pressure: {press_val}")
    print(f"  Wind U: {wind_u_val}")
    print(f"  Wind V: {wind_v_val}")
    print(f"  Wind Speed: {wind_speed}") 
    print(f"  SWDOWN: {swdown_val}, SWUP: {swup_val}")
    print(f"  Net Shortwave Radiation: {net_shortwave_radiation}")
    print("=" * 50)     

    return {
        "Temperature": temp_val,
        "Rain": rain_val,
        "Wind Speed": wind_speed,
        "Net Shortwave Radiation": net_shortwave_radiation
    }


# Main Execution
if __name__ == "__main__":
    # Define file paths
    base_dir = osp.join('/', 'home', 'spearsty', 'p', 'data')
    file_path = {"process_path": osp.join(base_dir, 'feat', 'weather', 'processed_output.nc')}

    # Define the record
    record_1 = {
        "date": datetime.strptime("2011-01-02 11:00:00", "%Y-%m-%d %H:%M:%S"),
        "lon": -154.803955078125,
        "lat": 20.29543304443359
    }

    # Load meteorology data
    meteorology = load_meteorology(file_path, 0, 12)

    # Extract data for the record
    try:
        extracted_data = extract_record(record_1, meteorology)
        #print(f"Extracted Meteorology Data for Record 1: {extracted_data}")
    except Exception as e:
        print(f"Error extracting data for Record 1: {e}")
