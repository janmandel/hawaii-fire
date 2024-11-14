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
for i in range(ntimes):
    time_str = d['times'][i]
    # print('row',i,'time string',time_str)
    if not time_str.isspace():
        # string not blank 
        nbl = nbl + 1 
        time = str_to_time(time_str)
        itime = start_time + timedelta(hours=i)
        if time != itime:
            print('row',i,'time is',time,'should be',itime)
            sys.exit(1)
print(nbl,'nonempty rows with data out of',ntimes,'rows')

