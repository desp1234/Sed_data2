import netCDF4 as nc
import numpy as np

stations = ['Rhine_Emmerich', 'Rhine_Illingen', 'Rhine_Maxau']

for station_file in stations:
    f = nc.Dataset(f'done/{station_file}.nc')
    print(f'\n{station_file}:')
    print(f'  Station: {f.station_name}')
    print(f'  Period: {f.data_period}')
    print(f'  Latitude: {f.variables["latitude"][...]}')
    print(f'  Longitude: {f.variables["longitude"][...]}')
    print(f'  Time points: {len(f.variables["time"][:])}')

    discharge = f.variables['discharge'][:]
    ssc = f.variables['ssc'][:]
    sediment_load = f.variables['sediment_load'][:]

    print(f'  Non-missing discharge: {np.sum(discharge != -9999)}')
    print(f'  Non-missing SSC: {np.sum(ssc != -9999)}')
    print(f'  Non-missing sediment_load: {np.sum(sediment_load != -9999)}')

    f.close()
