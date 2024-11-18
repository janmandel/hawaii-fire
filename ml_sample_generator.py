# The neccessary packages
## In-house packages
import lonlat_interp
from lonlat_interp import test_reproduce_smooth_grid, Coord_to_index, Interpolator
from saveload import save,load
## Pacakages from anywhere else
import os
from os import path as osp
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
from rasterio.plot import show
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from dbfread import DBF
import pygrib
import pickle
from joblib import Parallel, delayed
import time
import psutil
from datetime import datetime, timedelta

# The file paths
## Define the base directory (main)
main = osp.join('/', 'home', 'spearsty', 'p', 'data')
feat = osp.join(main, 'feat')
targ = osp.join(main, 'targ', 'Hawaii-all_2024-10-29_16:36:26')

## topography paths
slope_path = osp.join(feat, 'landfire', 'reprojected', 'slope_reproj.tif')
elevation_path = osp.join(feat, 'landfire', 'reprojected', 'elevation_reproj.tif')
aspect_path = osp.join(feat, 'landfire', 'reprojected', 'aspect_reproj.tif')

## vegetation paths
fuelmod_path = osp.join(feat, 'landfire', 'reprojected', 'fuelmod_reproj.tif')
fuelvat_path = osp.join(feat, 'landfire', 'afbfm', 'LF2022_FBFM13_230_HI', 'LH22_F13_230.tif.vat.dbf')

# meteorology paths (processed wrf outputs)
process_path = osp.join(feat, 'weather', 'processed_output.nc')

## fire detection (red pixel detection....)
fire_path = osp.join(targ, 'ml_data')

# Check if files exist
file_paths = {
    "slope_path": slope_path,
    "elevation_path": elevation_path,
    "aspect_path": aspect_path,
    "fuelmod_path": fuelmod_path,
    "fuelvat_path": fuelvat_path,
    "process_path": process_path,
    "fire_path": fire_path
}

for name, path in file_paths.items():
    if not osp.exists(path):
        print(f"ERROR: File {name} does not exist at {path}")
    else:
        print(f"File {name} loaded successfully: {path}")

# Extraction of the data from the files
procdata = nc.Dataset(process_path)

## spatial
lon_grid = procdata.variables['XLONG'][:, :]
lat_grid = procdata.variables['XLAT'][:, :]

## temporal
date_proc = procdata.variables['times'][:]
date_proc_strings = [t.strip() for t in date_proc]

# Remove empty strings from date_proc_strings
date_proc_strings = [t for t in date_proc_strings if t.strip() != '']
date_proc_times = pd.to_datetime(date_proc_strings, format='%Y-%m-%d_%H:%M:%S', errors ='coerce')

## meteorology
rain = procdata.variables['RAIN'][:, :, :] # 'RAIN', from convective (deep) thunderstorms
temp = procdata.variables['T2'][:, :, :] #'T2', the measured temp 2m above the surface
vapor = procdata.variables['Q2'][:, :, :] # 'Q2', the water-vapor mixing ratio 2m above the surface
wind_u = procdata.variables['U10'][:, :, :]
wind_v = procdata.variables['V10'][:, :, :]

print("Checking meteorology variables...")
required_vars = ['RAIN', 'T2', 'Q2', 'U10', 'V10', 'XLONG', 'XLAT', 'times']

for var in required_vars:
    if var not in procdata.variables:
        print(f"ERROR: Variable '{var}' not found in processed NetCDF file")
    else:
        print(f"Variable '{var}' loaded successfully")


## topography
print("Loading topography data...")
elevation_dataset = rasterio.open(elevation_path)
elevation = elevation_dataset.read(1)
print(f"Elevation data loaded. Shape: {elevation_dataset.read(1).shape}")
aspect_dataset = rasterio.open(aspect_path)
aspect = aspect_dataset.read(1)
print(f"Aspect data loaded. Shape: {aspect_dataset.read(1).shape}")
slope_dataset = rasterio.open(slope_path)
slope = slope_dataset.read(1)
print(f"Aspect data loaded. Shape: {aspect_dataset.read(1).shape}")

## vegetation
print("Loading vegetation data...")
fuelmod_dataset = rasterio.open(fuelmod_path)
fuelmod = fuelmod_dataset.read(1)
print(f"Fuel model data loaded. Shape: {fuelmod_dataset.read(1).shape}")

### Read the VAT file
fuel_vat = DBF(fuelvat_path)
fuel_vat_df = pd.DataFrame(iter(fuel_vat))

### Sort the VAT DataFrame by VALUE
fuel_vat_df_sorted = fuel_vat_df.sort_values(by='VALUE').reset_index(drop=True)
# Create a mapping from pixel values to class names
fuel_value_to_class = dict(zip(fuel_vat_df['VALUE'], fuel_vat_df['FBFM13']))

### Map the Fuel Model data to class names
fuelmod = np.vectorize(fuel_value_to_class.get)(fuelmod)

## fire detection (red pixel detection....)
X, y, c, basetime = load(fire_path) # X is a matrix of lon, lat and time (since base_time), y is fire dectections, c is confidence
lon_array = X[:, 0]
lat_array = X[:, 1]
time_in_days = X[:, 2]
dates_fire_actual = basetime + pd.to_timedelta(time_in_days, unit='D')
dates_fire =  dates_fire_actual.floor("h")

### above is the new setup and below is the old implementation
# Build the interpolator
interp = Coord_to_index(degree = 2)
interp.build(lon_grid, lat_grid)


def calc_rhum(temp_K, mixing_ratio):
    try:
        # Constants
        epsilon = 0.622
        pressure_pa = 1000 * 100  # fixed until we can get the data

        # Saturation vapor pressure
        es_hpa = 6.112 * np.exp((17.67 * (temp_K - 273.15)) / ((temp_K - 273.15) + 243.5))
        es_pa = es_hpa * 100

        # Actual vapor pressure
        e_pa = (mixing_ratio * pressure_pa) / (epsilon + mixing_ratio)

        # Relative humidity
        rh = (e_pa / es_pa) * 100
        return rh
    except Exception as e:
        print(f"Error in calc_rhum: temp_K={temp_K}, mixing_ratio={mixing_ratio}, error={e}")
        return np.nan

# Define the function to interpolate continuous features for each coordinate
def interpolate_data(lon, lat, date_fire):
    try:
        time_index = date_proc_times.get_loc(date_fire)
    except KeyError:
        # date_fire not found in date_proc_times
        return None

    # Interpolate spatially at this time index
    ia, ja = interp.evaluate(lon, lat)
    i, j = np.round(ia).astype(int), np.round(ja).astype(int)

    # Check if indices are within bounds
    if (0 <= i < temp.shape[1]) and (0 <= j < temp.shape[2]):
        data_dict = {
            'date': date_fire,
            'lon': lon,
            'lat': lat,
            'temp': temp[time_index, i, j],
            'rain': rain[time_index, i, j],
            'rhum': calc_rhum(temp[time_index, i, j], vapor[time_index, i, j]),
            'wind': np.sqrt(wind_u[time_index, i, j]**2 + wind_v[time_index, i, j]**2)
        }
        return data_dict
    else:
        # Indices are out of bounds
        return None


# Start timing and resource monitoring
start_time = time.time()
process = psutil.Process(os.getpid())
start_cpu = process.cpu_percent(interval=None)
start_mem = process.memory_info().rss  # in bytes

# Perform interpolation with matched dates
data_interp = []
no_interpolation_indices = []

for idx, (lon, lat, date_fire) in enumerate(zip(lon_array, lat_array, dates_fire)):
    result = interpolate_data(lon, lat, date_fire)
    if result is not None:
        data_interp.append(result)
    else:
        print(f"Interpolation failed for lon={lon}, lat={lat}, date={date_fire}")
        no_interpolation_indices.append(idx)
        data_interp.append({
            'date': date_fire,
            'lon': lon,
            'lat': lat,
            'temp': np.nan,
            'rain': np.nan,
            'rhum': np.nan,
            'wind': np.nan
        })

# Convert the list of dictionaries to a DataFrame for easy handling
df = pd.DataFrame(data_interp)

# Display a summary of the DataFrame and pickle it
print(df.head())
df.to_pickle('processed_data.pkl')
     
# End timing and resource monitoring
end_time = time.time()
end_cpu = process.cpu_percent(interval=None)
end_mem = process.memory_info().rss  # in bytes

# Calculate the differences
total_time = end_time - start_time
cpu_usage = end_cpu - start_cpu
memory_usage = end_mem - start_mem

print(f"Script runtime: {total_time:.2f} seconds")
print(f"CPU usage change: {cpu_usage:.2f}%")
print(f"Memory usage change: {memory_usage / (1024 ** 2):.2f} MB")

# Analyze missing interpolations
num_missing = len(no_interpolation_indices)
total_points = len(lon_array)
print(f"Number of timestamps without valid interpolation: {num_missing} out of {total_points}")
