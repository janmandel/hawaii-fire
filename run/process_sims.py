import os
import numpy as np
import logging
from datetime import datetime, timedelta
from netCDF4 import Dataset

# Set up logging
logging.basicConfig(level=logging.INFO)

# Constants and configuration
prefix = "wrfxpy/wksp/wfc-run_hawaii-gfsa-3km-2dom-"
suffix = "-192/wrf/wrfout_d02_"
cycle_duration_hours = 24 * 8  # Total hours each cycle runs
unique_data_duration_hours = 24 * 7  # Hours of unique data in each cycle
spinup_hours = 24  # Hours of spinup data to ignore
output_file = "processed_output.nc"
variable_names = ['TEMP', 'RH', 'WIND']  # Example list of variables to copy

# Create and configure the NetCDF file
ncfile = Dataset(output_file, 'w', format='NETCDF4')
ncfile.createDimension('time', None)  # unlimited time dimension
ncfile.createDimension('string_len', 19)  # length for ISO 8601 format strings

# Define variables: time as character array, and placeholder for XLONG and XLAT
times = ncfile.createVariable('times', 'S1', ('time', 'string_len'))
XLONG_var = None
XLAT_var = None

# Function to format datetime objects to the desired format
def format_time_string(dt):
    return dt.strftime("%Y-%m-%d_%H:%M:%S")

# Function to convert formatted time strings to a char array for NetCDF
def time_to_char_array(time_str):
    return np.array(list(time_str), dtype='S1')

# Initial cycle start time
cycle_start_time = datetime.strptime("2012-04-21_00:00:00", "%Y-%m-%d_%H:%M:%S")
cycle_index = 0
first_file_processed = False

# Loop over each cycle
while True:  # This can run indefinitely; remove break to continue beyond one cycle
    logging.info(f"Starting cycle {cycle_index} starting at {cycle_start_time}")

    # Define the time range for the cycle, excluding spin-up
    start_time = cycle_start_time + timedelta(hours=spinup_hours)
    end_time = start_time + timedelta(hours=unique_data_duration_hours)

    # Loop over each hour in the unique data duration
    time_index = 0
    current_time = start_time
    while current_time < end_time:
        # Generate the file path
        frame_timestr = format_time_string(current_time)
        filepath = f"{prefix}{cycle_start_time.strftime('%Y-%m-%d_%H:%M:%S')}{suffix}{frame_timestr}"

        # Log file processing start
        logging.info(f"Processing file: {filepath}")

        # Check if the file exists
        if os.path.exists(filepath):
            # Open the input file
            with Dataset(filepath, 'r') as src_file:
                # Copy XLONG and XLAT if this is the first file and they haven't been set yet
                if not first_file_processed:
                    XLONG_data = src_file.variables['XLONG'][:]
                    XLAT_data = src_file.variables['XLAT'][:]
                    ncfile.createDimension('y', XLONG_data.shape[0])
                    ncfile.createDimension('x', XLONG_data.shape[1])
                    XLONG_var = ncfile.createVariable('XLONG', 'f4', ('y', 'x'))
                    XLAT_var = ncfile.createVariable('XLAT', 'f4', ('y', 'x'))
                    XLONG_var[:, :] = XLONG_data
                    XLAT_var[:, :] = XLAT_data
                    first_file_processed = True  # Mark that the 2D variables have been copied

                # Copy each variable listed in `variable_names`
                for var_name in variable_names:
                    if var_name not in ncfile.variables:
                        # Create variable in the output file with the same dimensions
                        ncfile.createVariable(var_name, 'f4', ('time', 'y', 'x'), fill_value=np.nan)
                    
                    # Copy data from input file for the current time step
                    ncfile.variables[var_name][time_index, :, :] = src_file.variables[var_name][:]
        else:
            logging.info(f"File missing: {filepath}. Filling with NaN for this time frame.")
            # Fill each variable with NaN for the current time step if the file is missing
            for var_name in variable_names:
                if var_name not in ncfile.variables:
                    # Create variable in the output file with NaNs if it doesn’t exist
                    ncfile.createVariable(var_name, 'f4', ('time', 'y', 'x'), fill_value=np.nan)
                
                ncfile.variables[var_name][time_index, :, :] = np.nan

        # Store the time string in the `times` variable
        times[time_index, :] = time_to_char_array(frame_timestr)

        # Increment time step and time index
        current_time += timedelta(hours=1)
        time_index += 1

    # Sync to ensure data is written for the current cycle
    ncfile.sync()

    # Move to the next cycle start
    cycle_start_time += timedelta(hours=24 * 7)  # Shift to the next cycle
    cycle_index += 1
    break  # Remove this break statement to continue indefinitely or add stop conditions

# Close the NetCDF file
ncfile.close()
