import netCDF4 as nc
import numpy as np

f1 = "/data001/projects/jmandel/hawaii-fire/run/wrfxpy/wksp/wfc-run_hawaii-gfsa-3km-2dom-2013-05-04_00:00:00-192/wrf/wrfout_d02_2013-05-04_03:00:00"
f2 = "/data001/projects/jmandel/hawaii-fire/run/wrfxpy/wksp/wfc-run_hawaii-gfsa-3km-2dom-2013-05-04_00:00:00-192/wrf/wrfout_d02_2013-05-04_05:00:00"
f3 = "/data001/projects/jmandel/hawaii-fire/run/wrfxpy/wksp/wfc-run_hawaii-gfsa-3km-2dom-2013-05-04_00:00:00-192/wrf/wrfout_d02_2013-05-04_06:00:00"

d = nc.Dataset(f1)

XLONG = d['XLONG'][:]
XLAT =  d['XLAT'][:]
T2 =    d['XLAT'][:]

print(XLONG.shape)
print(T2.shape)
