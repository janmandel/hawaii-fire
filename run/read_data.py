import numpy as np
import pickle
from lonlat_interp import Coord_to_index

with open("data.pkl", "rb") as f:
    data = pickle.load(f) 
date = data['date']
T2 = data['T2']
XLONG = data['XLONG']
XLAT = data['XLAT']
print('date is',date)
print('T2 shape is',T2.shape)
print('XLONG shape is',XLONG.shape)
print('XLONG is',XLONG)
print('XLAT is',XLAT)

print('here is hour 0')
print('date 0 is',date[0])
print('T2 at that time is',T2[0,:,:])

interp = Coord_to_index(degree=2)
interp.build(XLONG, XLAT)
lon, lat = np.array([-154.11]),np.array([19.0])

ia,ja = interp.evaluate(lon,lat)

print('point at',lon,lat,'has index coordinates',ia,ja)
i, j = np.round(ia).astype(int), np.round(ja).astype(int)
print('rounded',i,j)
print('Temperature at 2m at that point is',T2[0,i,j])


