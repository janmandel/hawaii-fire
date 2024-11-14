import sys

print('\nTest reader for output_file produced by process_sims.py\n')

if len(sys.argv) < 2:
    print('Usage:\nconda activate wrfx python test_processed_output.py processed_output.nc')
    sys.exit(1)

import os
import numpy as np
from datetime import datetime, timedelta
from netCDF4 import Dataset


def str_to_time(time_str):
    return datetime.strptime(time_str, "%Y-%m-%d_%H:%M:%S")

output_file = "processed_output.nc"
d = Dataset(output_file,'r')

print("File",output_file,":",d)

vars = list(d.variables.keys())

print('variables are:',vars)
 
n = 2
for var in vars:
    print("Variable", var,"shape",d[var].shape)
    if n:
        print('First',n,'rows:\n',d[var][:n])
print('when row data is not available, times should be blank and other variables nan')

ntimes = d['times'].shape[0]
print('Number of rows is',ntimes)
start_time = str_to_time(d['times'][0])
print('Start time is',start_time)

print('testing that row times proceed by hour')
nbl = 0
start = None

def itime_fun(i):
    return start_time + timedelta(hours=i)

for i in range(ntimes):
    time_str = d['times'][i]
    # print('row',i,'time string',time_str)
    itime = itime_fun(i)

    if time_str.isspace():
        if start is None:
            start = i
        # print('row',i,'time',itime,'missing, start',start,itime_fun(start))

    else:
        # string not blank 
        if start is not None:
            print(f"Missing rows from {start} {itime_fun(start)} to {i} {itime_fun(i-1)}")  # Print the range of the stretch
            start = None  # Reset the start for the next stretch

        nbl = nbl + 1 # count non-blank rows

        # test the time string 
        time = str_to_time(time_str)
        if time != itime:
            print('row',i,'time is',time,'should be',itime)
            sys.exit(1)

# Handle the case where the last element ends in a stretch
if start is not None:
    print(f"Missing rows from {start} {itime_fun(start)} to {ntimes} {itime_fun(ntimes-1)}")  # Print the range of the stretch
          
print(nbl,'nonempty rows with data out of',ntimes,'rows')

