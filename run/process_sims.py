import os
import sys
import logging
from datetime import datetime, timedelta
import numpy as np
import netCDF4 as nc

# Set up logging for tracking progress and issues
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def read_wrf_variable(cycle_start_time, hour_time, variables, prefix, suffix, prev_rain=None):
    """
    Reads specified WRF variables from a file in a given simulation cycle and hour.
    
    Args:
        cycle_start_time (datetime): The start time of the simulation cycle, used for directory naming.
        hour_time (datetime): The exact hour to read within the cycle, used for file naming.
        variables (list of str): List of variable names to read.
        prefix (str): Prefix for the directory name.
        suffix (str): Suffix for the directory name.
        prev_rain (numpy array or None): Accumulated rainfall from the previous hour in the current cycle.
    
    Returns:
        dict: A dictionary where keys are variable names and values are 2D numpy arrays containing data.
              Returns {} if file or directory does not exist.
        numpy array: Updated cumulative rainfall for the current hour, to be used in the next call.
    """
    # Construct directory and file paths
    cycle_dir = f"{prefix}{cycle_start_time.strftime('%Y-%m-%d_%H:%M:%S')}{suffix}"
    file_name = f"wrfout_d02_{hour_time.strftime('%Y-%m-%d_%H:%M:%S')}"
    wrf_file = f"{cycle_dir}/{file_name}"
    
    logging.info(f"Processing file {file_name}")
    
    if not os.path.exists(cycle_dir) or not os.path.exists(wrf_file):
        logging.warning(f"Directory or file {wrf_file} missing, filling with NaNs.")
        return {}, prev_rain

    result = {}
    with nc.Dataset(wrf_file) as ds:
        for var in variables:
            if var == 'RAIN':
                # Calculate rainfall incrementally based on RAINC, RAINSH, and RAINNC
                rain_total = ds.variables['RAINC'][0] + ds.variables['RAINSH'][0] + ds.variables['RAINNC'][0]
                result[var] = rain_total - (prev_rain if prev_rain is not None else rain_total)
                prev_rain = rain_total  # Update prev_rain for the next hour in the current cycle
            else:
                result[var] = ds.variables[var][0]  # Read the variable directly as 2D
    return result, prev_rain


def build_3d_arrays(start_time, end_time, variables, prefix, suffix, omit_first=3):
    """
    Constructs 3D arrays for specified WRF variables across multiple simulation cycles.
    
    Args:
        start_timedate (datetime): Start time for the simulations.
        end_time (datetime): End time for the simulations.
        variables (list of str): List of WRF variable names to read.
        prefix (str): Prefix for the directory name.
        suffix (str): Suffix for the directory name.
        omit_first (int): Number of initial spinup hours to omit in the first cycle.

    Returns:
        dict: A dictionary with 'times' (1D array of datetime objects) and each variable (3D array of data).
    """
    # get dimensions
    hourly_data, prev_rain = read_wrf_variable(start_time, start_time, ['XLONG','XLAT'], prefix, suffix)
    XLONG = hourly_data['XLONG']
    XLAT = hourly_data['XLAT']
    hours = int((end_date - start_date).total_seconds()/3600 + 1)
    print(XLONG.shape)
    nx, ny = XLONG.shape
    #if h != 1 :
    #   logging.error('wrfouts must have one timeframe only')
    #   sys.exit(1)

    logging.info(f'Initializing storage for {hours} hours')
    date = [start_time + timedelta(hours=i) for i in range(hours)]
    data_dict = {'date':date, 'XLONG':XLONG, 'XLAT':XLAT}
    data_dict.update({var: np.full((hours, nx, ny), np.nan, dtype=np.float32) for var in variables})
    
    # Generate the list of cycle start times
    cycle_start_times = []
    current_time = start_date
    while current_time <= end_date:
        cycle_start_times.append(current_time)
        current_time += timedelta(days=7)

    # Iterate over each cycle and process data
    for i, cycle_start_time in enumerate(cycle_start_times):
        logging.info(f"Processing cycle {i} starting at {cycle_start_time}")
        prev_rain = None  # Reset rainfall accumulation for each new cycle
        
        # Loop over each hour in the 8-day cycle
        for hour in range(24 * 8):
            hour_time = cycle_start_time + timedelta(hours=hour)
            
            # Skip initial for spinup
            if (i == 0 and hour < omit_first) or hour < 23:
                continue
            # index in the arrays
            x = (hour_time - start_time).total_seconds()           
            hour = x/3600
            if hour * 3600 != x:
                logging.error(f"Time since start {x} seconds is not full hours")
                sys.exit(1)
            logging.info(f"Reading data for cycle  hour {hour} at {hour_time}")
            # Read data for each variable at the specific hour
            hourly_data, prev_rain = read_wrf_variable(cycle_start_time, hour_time, variables, prefix, suffix, prev_rain)

            for var, data in hourly_data.items():
                data_dict[var][hour,:,:] = hourly_data[var]

    # Return a dictionary with time and variable arrays
    return(data_dict)


def write_to_netcdf(data, output_file):
    """
    Writes the given data dictionary to a NetCDF file.
    
    Args:
        data (dict): Dictionary containing 'Times' (1D datetime array) and 3D arrays for each variable.
        output_file (str): Path to the output NetCDF file.
    """
    with nc.Dataset(output_file, 'w', format='NETCDF4') as ds:
        # Create dimensions
        time_dim = ds.createDimension('time', len(data['Times']))
        lat_dim = ds.createDimension('lat', data[next(iter(data))].shape[1])  # Assuming variable shape consistency
        lon_dim = ds.createDimension('lon', data[next(iter(data))].shape[2])

        # Create time variable
        times = ds.createVariable('Times', 'f8', ('time',))
        times.units = 'hours since 2011-01-01 00:00:00'
        times.calendar = 'gregorian'
        times[:] = nc.date2num(data['Times'], units=times.units, calendar=times.calendar)

        # Create variables for each data array
        for var, values in data.items():
            if var == 'Times':
                continue  # Skip the 'Times' key as it has been handled
            var_data = ds.createVariable(var, 'f4', ('time', 'lat', 'lon'), fill_value=np.nan)
            var_data[:] = values

    logging.info(f"Data successfully written to {output_file}")


if __name__ == "__main__":
    # Set the start and end dates for the simulations
    start_date = datetime(2011, 1, 1)
    end_date = datetime(2024, 11, 2)
    
    # List of WRF variables to read (example list)
    variables = ['RAIN', 'T2', 'Q2', 'U10', 'V10', 'SWDOWN', 'SWUPT']  # Example variables
    
    # Define prefix and suffix for directory names
    prefix = "/data001/projects/jmandel/wrfxpy/wksp/wfc-run_hawaii-gfsa-3km-2dom-"
    suffix = "-192/wrf"
    
    # Run the main function and collect results
    data = build_3d_arrays(start_date, end_date, variables, prefix, suffix)

    file = "data.pkl"

    with open(file,"wb") as f:
        pickle.dump(data,f)

    
    # Specify output NetCDF file path
    # output_file = 'wrf_simulation_output.nc'
    
    # Write results to NetCDF
    # write_to_netcdf(results, output_file)
    
    # Log the completion of the process
    logging.info(f"Data processing and writing to file {file} complete.")

