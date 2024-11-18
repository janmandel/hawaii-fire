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
main = osp.join('home', 'spearsty', 'p', 'data')
feat = osp.join(main, 'feat')
targ = osp.join(main, 'targ', 'Hawaii-all_2024-10-29_16:36:26')

## topography paths
slope_path = osp.join(feat, 'landfire', 'reprojected', 'slope_reproj.tif')
elevation_path = osp.join(feat, 'landfire', 'reprojected', 'elevation_reproj.tif')
aspect_path = osp.join(feat, 'landfire', 'reprojected', 'aspect_reproj.tif')

## vegetation paths
fuelmod_path = osp.join(feat, 'landfire', 'reprojected', 'fuelmod_reproj.tif')
fuelvat_path = osp.join(feat, 'landfire', 'afbfm', ' LF2022_FBFM13_230_HI', 'LH22_F13_230.tif.vat.dbf')

# meteorology paths (processed wrf outputs)
process_path = osp.join(feat, 'weather', 'processed_output.nc')

## fire detection (red pixel detection....)
fire_path = osp.join(targ, 'ml_data')

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

## topography
elevation_dataset = rasterio.open(elevation_path)
elevation = elevation_dataset.read(1)
aspect_dataset = rasterio.open(aspect_path)
aspect = aspect_dataset.read(1)
slope_dataset = rasterio.open(slope_path)
slope = slope_dataset.read(1)

## vegetation
fuelmod_dataset = rasterio.open(fuelmod_path)
fuelmod = fuelmod_dataset.read(1)

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
interp.build(lon_array, lat_array)

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
            'vapor': vapor[time_index, i, j],
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
        no_interpolation_indices.append(idx)
        data_interp.append({
            'date': date_fire,
            'lon': lon,
            'lat': lat,
            'temp': np.nan,
            'rain': np.nan,
            'vapor': np.nan,
            'wind': np.nan
        })

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