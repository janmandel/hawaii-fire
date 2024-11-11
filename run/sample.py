import netCDF4 as nc
import numpy as np
import pickle
from datetime import datetime

f  = "/data001/projects/jmandel/hawaii-fire/run/wrfxpy/wksp/wfc-run_hawaii-gfsa-3km-2dom-2013-05-04_00:00:00-192/wrf/wrfout_d02_2013-05-04_03:00:00"
f1  = "/data001/projects/jmandel/hawaii-fire/run/wrfxpy/wksp/wfc-run_hawaii-gfsa-3km-2dom-2013-05-04_00:00:00-192/wrf/wrfout_d02_2013-05-04_04:00:00"

d0 = nc.Dataset(f)
d1 = nc.Dataset(f1)

XLONG = d0['XLONG'][:]
XLAT =  d0['XLAT'][:]
T2_0 =    d0['XLAT'][:]
Times_0 = d0['Times'][:]

print(Times_0)

#date0  = datetime.strptime("".join(Times_0.compressed().decode('utf-8')),"%Y-%m-%d %H:%M:%S")
#date1  = datetime.strptime("".join(Times_1.compressed().decode('utf-8')),"%Y-%m-%d %H:%M:%S")

date0 = datetime(2013, 5, 4, 3, 0, 0)
date1 = datetime(2013, 5, 4, 4, 0, 0)
print(date0,date1)

T2_1 =    d1['XLAT'][:]

T2 = np.concatenate((T2_0 ,T2_1),axis=0).reshape(2, 96, 96)

date = [date0, date1]
print('XLONG shape',XLONG.shape)
print(T2_0.shape)
print(T2_1.shape)
print(T2.shape)
print('T2 at time 0=',T2[0,:,:])

data = {"date":date,'XLONG':XLONG[0,:,:], 'XLAT':XLAT[0,:,:], 'T2':T2}

with open("data.pkl","wb") as f:
   pickle.dump(data,f)

