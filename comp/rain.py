import netCDF4 as nc
import numpy as np
import sys

# Define the function to load and process rain variables from wrfout files
def process_rain_variables(file1_path, file2_path):
    # Open the two NetCDF files
    with nc.Dataset(file1_path) as file1, nc.Dataset(file2_path) as file2:
        
        # List of rain-related variables
        rain_vars = ['RAINC', 'RAINSH', 'RAINNC']
        
        # Initialize dictionaries to hold sums and differences
        sums_file1 = {}
        sums_file2 = {}
        differences = {}
        
        # Loop over each rain variable
        for var in rain_vars:
            # Read data for the variable from both files
            data1 = file1.variables[var][:]
            data2 = file2.variables[var][:]
            
            # Sum the data across all timesteps and spatial dimensions
            sum_data1 = np.sum(data1)
            sum_data2 = np.sum(data2)
            
            # Store the sums in the dictionaries
            sums_file1[var] = sum_data1
            sums_file2[var] = sum_data2
            
            # Calculate and store the difference
            differences[var] = sum_data1 - sum_data2
            
            # Print the sums and difference for each variable
            print(f"{var} sum in file 1: {sum_data1}")
            print(f"{var} sum in file 2: {sum_data2}")
            print(f"Difference for {var} (file 1 - file 2): {differences[var]}")
            print("")

# Example usage
# Replace 'wrfout_file1.nc' and 'wrfout_file2.nc' with the actual file paths
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <wrfout_file1> <wrfout_file2>")
    else:
        file1_path = sys.argv[1]
        file2_path = sys.argv[2]
        process_rain_variables(file1_path, file2_path)
