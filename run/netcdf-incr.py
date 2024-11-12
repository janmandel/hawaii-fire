from netCDF4 import Dataset
import numpy as np

# Initialize the NetCDF file with an unlimited time dimension
filename = 'incremental_data_2d_latlon.nc'
ncfile = Dataset(filename, 'w', format='NETCDF4')

# Define dimensions
ncfile.createDimension('time', None)  # unlimited time dimension
ncfile.createDimension('y', 20)       # example y-dimension
ncfile.createDimension('x', 10)       # example x-dimension

# Define variables
time = ncfile.createVariable('time', 'f4', ('time',))
lat = ncfile.createVariable('lat', 'f4', ('y', 'x'))
lon = ncfile.createVariable('lon', 'f4', ('y', 'x'))
temp = ncfile.createVariable('temperature', 'f4', ('time', 'y', 'x'), zlib=True, complevel=4, chunksizes=(1, 20, 10))

# Generate 2D latitude and longitude values (e.g., for a curvilinear grid)
# Example: a grid covering an irregular area, with unique lat/lon values for each (y, x) point
lat_vals = np.linspace(-90, 90, 20).reshape(20, 1) + np.zeros((20, 10))  # latitude varying by row
lon_vals = np.linspace(-180, 180, 10).reshape(1, 10) + np.zeros((20, 10))  # longitude varying by column
lat[:, :] = lat_vals
lon[:, :] = lon_vals

# Function to incrementally add data for each time step
def append_data(ncfile, time_index, temperature_data):
    """Append data for a new time step."""
    time[time_index] = time_index  # Assign time value
    temp[time_index, :, :] = temperature_data  # Assign temperature values for this time step
    return time_index + 1

# Initial time index
time_index = 0
batch_size = 5

# Example of incremental data addition with batching
for t in range(0, 25, batch_size):  # simulate 25 time steps in batches
    for _ in range(batch_size):
        temperature_data = np.random.rand(20, 10)  # example data matching the (y, x) shape
        time_index = append_data(ncfile, time_index, temperature_data)
    
    ncfile.sync()  # Batch sync after each batch_size

# Close the file when done
ncfile.close()

