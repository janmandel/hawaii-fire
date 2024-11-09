import sys
from netCDF4 import Dataset
import numpy as np

# Check if correct arguments are provided
if len(sys.argv) < 4:
    print("Usage: python compare_wrf_var.py <variable> <file1> <file2>")
    sys.exit(1)

# Get the variable name and file paths from command-line arguments
file1 = sys.argv[1]
file2 = sys.argv[2]
variable_name = sys.argv[3]

# Open the WRF files
try:
    dataset1 = Dataset(file1, 'r')
    dataset2 = Dataset(file2, 'r')
except IOError as e:
    print(f"Error opening files: {e}")
    sys.exit(1)

# Check if the variable exists in both files
if variable_name not in dataset1.variables:
    print(f"Variable '{variable_name}' not found in {file1}")
    sys.exit(1)
if variable_name not in dataset2.variables:
    print(f"Variable '{variable_name}' not found in {file2}")
    sys.exit(1)

# Read the variable data from both files
var_data1 = dataset1.variables[variable_name][:]
var_data2 = dataset2.variables[variable_name][:]

# Close the datasets
dataset1.close()
dataset2.close()

# Compare the two arrays
if var_data1.shape != var_data2.shape:
    print(f"Shapes do not match: {file1} has {var_data1.shape}, {file2} has {var_data2.shape}")
else:
    difference = np.abs(var_data1 - var_data2)
    #print("Absolute Difference:\n", difference)
    max_diff = np.max(difference)
    print(f"Maximum difference between '{variable_name}' in the two files: {max_diff}")
    rmse = np.sqrt(np.mean((var_data1 - var_data2) ** 2))  # RMSE calculation
    dif_mean = np.mean(var_data1 - var_data2)
    dif_abs_mean = np.mean(np.abs(difference))
    print("Mean Difference:\n", dif_mean)
    print("Mean Absolute Difference:\n", dif_abs_mean)
    print("RMSE:", rmse)
    rms_scale = np.sqrt(np.mean(var_data1 ** 2))  # RMSE calculation
    print("RMS scale L2 norms of var 1:", rms_scale)
    print("relative RMS",rmse/rms_scale)


    # Optionally, check if arrays are exactly equal
    if np.array_equal(var_data1, var_data2):
        print("The variable data is identical in both files.")
    else:
        print("The variable data differs between the two files.")

