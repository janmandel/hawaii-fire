import xarray as xr
import numpy as np
import os

# Function to process a single cycle and append it to the output file
def process_cycle(cycle_start_time, cycle_end_time, variable_names, input_dir, output_file):
    time_dim = 'time'
    
    # Initialize a list to store the data for the current cycle
    cycle_data = {var: [] for var in variable_names}
    time_steps = []
    
    # Loop through the 7 days of the simulation, ignoring the first day (spin-up)
    for day in range(1, 8):
        # Construct the filename based on the cycle time (assumes files are named in a specific format)
        start_time = cycle_start_time + pd.Timedelta(days=day)
        end_time = cycle_end_time + pd.Timedelta(days=day)
        filename = construct_filename(start_time, end_time, input_dir)
        
        # Check if the file exists
        if os.path.exists(filename):
            # Load the NetCDF file using xarray
            ds = xr.open_dataset(filename)
            
            # Extract the variables for the current time step
            for var in variable_names:
                cycle_data[var].append(ds[var].values)
            time_steps.append(start_time)
        else:
            # If the file is missing, fill with NaNs
            for var in variable_names:
                cycle_data[var].append(np.nan * np.ones_like(ds[variable_names[0]].shape))
            time_steps.append(start_time)
    
    # Convert cycle data into a DataFrame-like structure (xarray Dataset)
    data_dict = {var: np.array(cycle_data[var]) for var in variable_names}
    data_dict[time_dim] = np.array(time_steps)
    
    # Create an xarray Dataset for this cycle
    cycle_ds = xr.Dataset(data_dict)
    
    # Append to the output file
    if os.path.exists(output_file):
        cycle_ds.to_netcdf(output_file, mode='a', append_dim=time_dim)
    else:
        cycle_ds.to_netcdf(output_file)

# Helper function to construct the file path from start and end times
def construct_filename(start_time, end_time, input_dir):
    # Create a formatted filename with the start and end times
    prefix = start_time.strftime('%Y-%m-%d_%H:%M:%S')
    suffix = end_time.strftime('%Y-%m-%d_%H:%M:%S')
    filename = os.path.join(input_dir, f"prefix{prefix}suffix{suffix}.wrfout")
    return filename

# Example usage:
input_dir = '/path/to/wrfout/files'  # Directory containing the wrfout files
output_file = '/path/to/output.nc'   # Path for the output NetCDF file
cycle_start_time = pd.Timestamp('2017-04-15')
cycle_end_time = pd.Timestamp('2017-04-22')
variable_names = ['T2', 'Q2', 'U10', 'V10']

# Process the cycle and store data in the output file
process_cycle(cycle_start_time, cycle_end_time, variable_names, input_dir, output_file)

