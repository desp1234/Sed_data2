#!/usr/bin/env python3
"""
从HYDAT NetCDF文件中提取泥沙数据并按站点保存（并行版）
"""

import netCDF4 as nc
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

def read_char_variable(var, index=None):
    """读取字符变量"""
    if index is not None:
        data = var[index]
    else:
        data = var[:]

    if isinstance(data, np.ma.MaskedArray):
        data = data.filled(b'')

    if len(data.shape) == 2:
        result = []
        for row in data:
            try:
                if isinstance(row[0], (bytes, np.bytes_)):
                    s = b''.join(row).decode('utf-8', errors='ignore').strip('\x00').strip()
                else:
                    s = ''.join([chr(ord(c)) if isinstance(c, str) else chr(c) for c in row if c]).strip()
                result.append(s)
            except:
                result.append('')
        return np.array(result)
    else:
        try:
            if isinstance(data[0], (bytes, np.bytes_)):
                return b''.join(data).decode('utf-8', errors='ignore').strip('\x00').strip()
            else:
                return ''.join([chr(ord(c)) if isinstance(c, str) else chr(c) for c in data if c]).strip()
        except:
            return ''


class SedimentDataExtractor:
    def __init__(self, input_nc, output_dir):
        self.input_nc = input_nc
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stations_info = {}

    def load_stations(self, ds):
        print("正在加载站点信息...")

        if 'STATIONS' not in ds.groups:
            raise ValueError("NetCDF文件中没有找到STATIONS表")

        stations = ds.groups['STATIONS']
        station_numbers = read_char_variable(stations.variables['STATION_NUMBER'])

        for i, station_id in enumerate(station_numbers):
            if not station_id:
                continue

            station_meta = {'STATION_NUMBER': station_id}

            for var_name in ['STATION_NAME', 'PROV_TERR_STATE_LOC']:
                if var_name in stations.variables:
                    try:
                        value = read_char_variable(stations.variables[var_name], i)
                        if value:
                            station_meta[var_name] = value
                    except:
                        pass

            for var_name in ['LATITUDE', 'LONGITUDE', 'DRAINAGE_AREA_GROSS', 'DRAINAGE_AREA_EFFECT']:
                if var_name in stations.variables:
                    try:
                        value = stations.variables[var_name][i]
                        if isinstance(value, np.ma.MaskedArray):
                            if not value.mask:
                                value = float(value.data)
                            else:
                                continue
                        if not np.isnan(value) and value != -999:
                            station_meta[var_name] = float(value)
                    except:
                        pass

            self.stations_info[station_id] = station_meta

        print(f"  加载完成: {len(self.stations_info)} 个站点")
        return self.stations_info

    def extract_sed_daily_loads(self, ds, station_id):
        if 'SED_DLY_LOADS' not in ds.groups:
            return None

        sed_loads = ds.groups['SED_DLY_LOADS']
        station_numbers = read_char_variable(sed_loads.variables['STATION_NUMBER'])
        mask = np.array([s == station_id for s in station_numbers])

        if not np.any(mask):
            return None

        years = sed_loads.variables['YEAR'][:][mask]
        months = sed_loads.variables['MONTH'][:][mask]

        if isinstance(years, np.ma.MaskedArray):
            years = years.filled(-999)
        if isinstance(months, np.ma.MaskedArray):
            months = months.filled(-999)

        data_records = []

        for idx, (year, month) in enumerate(zip(years, months)):
            if year == -999 or month == -999:
                continue

            year, month = int(year), int(month)

            try:
                n_days = pd.Timestamp(year=year, month=month, day=1).days_in_month
            except:
                continue

            for day in range(1, n_days + 1):
                load_var = f'LOAD{day}'
                if load_var in sed_loads.variables:
                    load_value = sed_loads.variables[load_var][:][mask][idx]

                    if isinstance(load_value, np.ma.MaskedArray):
                        if load_value.mask:
                            continue
                        load_value = load_value.data

                    if load_value != -999 and not np.isnan(load_value):
                        try:
                            date = pd.Timestamp(year=year, month=month, day=day)
                            data_records.append({'time': date, 'sediment_load': float(load_value)})
                        except:
                            continue

        if not data_records:
            return None

        df = pd.DataFrame(data_records)
        df = df.sort_values('time').reset_index(drop=True)
        return df

    def extract_sed_daily_suscon(self, ds, station_id):
        if 'SED_DLY_SUSCON' not in ds.groups:
            return None

        sed_suscon = ds.groups['SED_DLY_SUSCON']
        station_numbers = read_char_variable(sed_suscon.variables['STATION_NUMBER'])
        mask = np.array([s == station_id for s in station_numbers])

        if not np.any(mask):
            return None

        years = sed_suscon.variables['YEAR'][:][mask]
        months = sed_suscon.variables['MONTH'][:][mask]

        if isinstance(years, np.ma.MaskedArray):
            years = years.filled(-999)
        if isinstance(months, np.ma.MaskedArray):
            months = months.filled(-999)

        data_records = []

        for idx, (year, month) in enumerate(zip(years, months)):
            if year == -999 or month == -999:
                continue

            year, month = int(year), int(month)

            try:
                n_days = pd.Timestamp(year=year, month=month, day=1).days_in_month
            except:
                continue

            for day in range(1, n_days + 1):
                suscon_var = f'SUSCON{day}'
                if suscon_var in sed_suscon.variables:
                    suscon_value = sed_suscon.variables[suscon_var][:][mask][idx]

                    if isinstance(suscon_value, np.ma.MaskedArray):
                        if suscon_value.mask:
                            continue
                        suscon_value = suscon_value.data

                    if suscon_value != -999 and not np.isnan(suscon_value):
                        try:
                            date = pd.Timestamp(year=year, month=month, day=day)
                            data_records.append({'time': date, 'suspended_sediment_concentration': float(suscon_value)})
                        except:
                            continue

        if not data_records:
            return None

        df = pd.DataFrame(data_records)
        df = df.sort_values('time').reset_index(drop=True)
        return df

    def extract_sed_samples(self, ds, station_id):
        if 'SED_SAMPLES' not in ds.groups:
            return None

        sed_samples = ds.groups['SED_SAMPLES']
        station_numbers = read_char_variable(sed_samples.variables['STATION_NUMBER'])
        mask = np.array([s == station_id for s in station_numbers])

        if not np.any(mask):
            return None

        data_records = []

        dates = read_char_variable(sed_samples.variables['DATE'])

        for i, is_match in enumerate(mask):
            if not is_match:
                continue

            try:
                date_str = dates[i]
                if not date_str:
                    continue

                date = pd.to_datetime(date_str, errors='coerce')
                if pd.isna(date):
                    continue

                record = {'time': date}

                for var_name in ['FLOW', 'TEMPERATURE', 'CONCENTRATION', 'DISSOLVED_SOLIDS',
                                 'SAMPLE_DEPTH', 'SV_DEPTH1', 'SV_DEPTH2']:
                    if var_name in sed_samples.variables:
                        try:
                            value = sed_samples.variables[var_name][i]
                            if isinstance(value, np.ma.MaskedArray):
                                if value.mask:
                                    continue
                                value = value.data
                            if value != -999 and not np.isnan(value):
                                record[var_name.lower()] = float(value)
                        except:
                            pass

                for var_name in ['SED_DATA_TYPE', 'SAMPLER_TYPE', 'SAMPLING_VERTICAL_LOCATION']:
                    if var_name in sed_samples.variables:
                        try:
                            value = read_char_variable(sed_samples.variables[var_name], i)
                            if value:
                                record[var_name.lower()] = value
                        except:
                            pass

                if len(record) > 1:
                    data_records.append(record)
            except:
                continue

        if not data_records:
            return None

        df = pd.DataFrame(data_records)
        df = df.sort_values('time').reset_index(drop=True)
        return df

    def create_station_netcdf(self, station_id, sed_loads_df, sed_suscon_df, sed_samples_df):
        station_meta = self.stations_info.get(station_id, {'STATION_NUMBER': station_id})
        output_file = self.output_dir / f"HYDAT_{station_id}_SEDIMENT.nc"

        print(f"  创建文件: {output_file.name}")

        with nc.Dataset(output_file, 'w', format='NETCDF4') as ncfile:
            ncfile.Conventions = "CF-1.8"
            ncfile.title = f"HYDAT Station {station_id} - Sediment Data"
            ncfile.institution = "Water Survey of Canada / Environment and Climate Change Canada"
            ncfile.source = "HYDAT - Canadian Hydrometric Database"
            ncfile.history = f"Created {datetime.now().isoformat()}"
            ncfile.references = "https://www.canada.ca/en/environment-climate-change/services/water-overview/quantity/monitoring/survey/data-products-services/national-archive-hydat.html"

            ncfile.station_id = station_id

            if 'STATION_NAME' in station_meta:
                ncfile.station_name = station_meta['STATION_NAME']

            if 'PROV_TERR_STATE_LOC' in station_meta:
                ncfile.province_territory = station_meta['PROV_TERR_STATE_LOC']

            lat = station_meta.get('LATITUDE', None)
            lon = station_meta.get('LONGITUDE', None)

            if lat is not None and lon is not None:
                ncfile.geospatial_lat_min = lat
                ncfile.geospatial_lat_max = lat
                ncfile.geospatial_lon_min = lon
                ncfile.geospatial_lon_max = lon

                ncfile.createDimension('lat', 1)
                ncfile.createDimension('lon', 1)

                lat_var = ncfile.createVariable('lat', 'f4', ('lat',))
                lat_var.standard_name = "latitude"
                lat_var.long_name = "station latitude"
                lat_var.units = "degrees_north"
                lat_var.axis = "Y"
                lat_var[:] = [lat]

                lon_var = ncfile.createVariable('lon', 'f4', ('lon',))
                lon_var.standard_name = "longitude"
                lon_var.long_name = "station longitude"
                lon_var.units = "degrees_east"
                lon_var.axis = "X"
                lon_var[:] = [lon]

            reference_date = pd.Timestamp('1850-01-01')

            if sed_loads_df is not None and len(sed_loads_df) > 0:
                time_dim = 'time_sed_load'
                ncfile.createDimension(time_dim, len(sed_loads_df))

                time_var = ncfile.createVariable('time_sed_load', 'f8', (time_dim,))
                time_var.standard_name = "time"
                time_var.long_name = "time for sediment load"
                time_var.units = "days since 1850-01-01 00:00:00"
                time_var.calendar = "gregorian"
                time_var.axis = "T"

                time_values = (sed_loads_df['time'] - reference_date).dt.total_seconds() / 86400.0
                time_var[:] = time_values.values

                if lat is not None and lon is not None:
                    sed_load_var = ncfile.createVariable('sediment_load', 'f4',
                                                         (time_dim, 'lat', 'lon'),
                                                         fill_value=-999.0,
                                                         zlib=True, complevel=4)
                    sed_load_var[:, 0, 0] = sed_loads_df['sediment_load'].values
                else:
                    sed_load_var = ncfile.createVariable('sediment_load', 'f4',
                                                         (time_dim,),
                                                         fill_value=-999.0,
                                                         zlib=True, complevel=4)
                    sed_load_var[:] = sed_loads_df['sediment_load'].values

                sed_load_var.long_name = "daily sediment load"
                sed_load_var.units = "tonnes"
                sed_load_var.cell_methods = "time: mean"

            if sed_suscon_df is not None and len(sed_suscon_df) > 0:
                time_dim = 'time_sed_suscon'
                ncfile.createDimension(time_dim, len(sed_suscon_df))

                time_var = ncfile.createVariable('time_sed_suscon', 'f8', (time_dim,))
                time_var.standard_name = "time"
                time_var.long_name = "time for suspended sediment concentration"
                time_var.units = "days since 1850-01-01 00:00:00"
                time_var.calendar = "gregorian"
                time_var.axis = "T"

                time_values = (sed_suscon_df['time'] - reference_date).dt.total_seconds() / 86400.0
                time_var[:] = time_values.values

                if lat is not None and lon is not None:
                    suscon_var = ncfile.createVariable('suspended_sediment_concentration', 'f4',
                                                       (time_dim, 'lat', 'lon'),
                                                       fill_value=-999.0,
                                                       zlib=True, complevel=4)
                    suscon_var[:, 0, 0] = sed_suscon_df['suspended_sediment_concentration'].values
                else:
                    suscon_var = ncfile.createVariable('suspended_sediment_concentration', 'f4',
                                                       (time_dim,),
                                                       fill_value=-999.0,
                                                       zlib=True, complevel=4)
                    suscon_var[:] = sed_suscon_df['suspended_sediment_concentration'].values

                suscon_var.long_name = "daily suspended sediment concentration"
                suscon_var.units = "mg/L"
                suscon_var.cell_methods = "time: mean"

            if sed_samples_df is not None and len(sed_samples_df) > 0:
                time_dim = 'time_sed_sample'
                ncfile.createDimension(time_dim, len(sed_samples_df))

                time_var = ncfile.createVariable('time_sed_sample', 'f8', (time_dim,))
                time_var.standard_name = "time"
                time_var.long_name = "time for sediment samples"
                time_var.units = "days since 1850-01-01 00:00:00"
                time_var.calendar = "gregorian"
                time_var.axis = "T"

                time_values = (sed_samples_df['time'] - reference_date).dt.total_seconds() / 86400.0
                time_var[:] = time_values.values

                for col in sed_samples_df.columns:
                    if col == 'time':
                        continue

                    if sed_samples_df[col].dtype == 'object':
                        max_len = max(len(str(v)) for v in sed_samples_df[col])
                        if max_len > 0:
                            str_values = ', '.join([str(v) for v in sed_samples_df[col][:10]])
                            ncfile.setncattr(f'{col}_sample_values', str_values)
                    else:
                        num_var = ncfile.createVariable(col, 'f4', (time_dim,),
                                                       fill_value=-999.0,
                                                       zlib=True, complevel=4)
                        num_var.long_name = col.replace('_', ' ')
                        num_var[:] = sed_samples_df[col].values

        print(f"    ✓ 完成")


def process_single_station(input_nc, output_dir, station_id, stations_info):
    try:
        with nc.Dataset(input_nc, 'r') as ds:
            extractor = SedimentDataExtractor(input_nc, output_dir)
            extractor.stations_info = stations_info

            sed_loads_df = extractor.extract_sed_daily_loads(ds, station_id)
            sed_suscon_df = extractor.extract_sed_daily_suscon(ds, station_id)
            sed_samples_df = extractor.extract_sed_samples(ds, station_id)

            if (sed_loads_df is None or len(sed_loads_df) == 0) and \
               (sed_suscon_df is None or len(sed_suscon_df) == 0) and \
               (sed_samples_df is None or len(sed_samples_df) == 0):
                return station_id, False, "无泥沙数据"

            extractor.create_station_netcdf(station_id, sed_loads_df, sed_suscon_df, sed_samples_df)

        return station_id, True, "成功"

    except Exception as e:
        return station_id, False, str(e)


# ★★★ 并行版本 process_all_stations ★★★
def process_all_stations_parallel(self):
    print(f"\n{'='*80}")
    print(f"提取泥沙数据（并行模式）")
    print(f"{'='*80}\n")

    with nc.Dataset(self.input_nc, 'r') as ds:
        self.load_stations(ds)

        all_sed_stations = set()
        sed_groups = ['SED_DLY_LOADS', 'SED_DLY_SUSCON', 'SED_SAMPLES']

        for group_name in sed_groups:
            if group_name in ds.groups:
                station_numbers = read_char_variable(ds.groups[group_name].variables['STATION_NUMBER'])
                all_sed_stations.update([sid for sid in station_numbers if sid])

        station_list = sorted(all_sed_stations)

    print(f"找到 {len(station_list)} 个包含泥沙数据的站点\n")
    print(f"并行处理中，CPU 核心数 = {os.cpu_count()}\n")

    success = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {
            executor.submit(process_single_station, self.input_nc, self.output_dir, sid, self.stations_info): sid
            for sid in station_list
        }

        for future in as_completed(futures):
            sid, ok, msg = future.result()
            if ok:
                print(f" ✓ {sid} 处理成功")
                success += 1
            else:
                print(f" ✗ {sid} 失败: {msg}")
                failed += 1

    print(f"\n{'='*80}")
    print(f"并行处理完成!")
    print(f"成功: {success} 个站点")
    print(f"失败/跳过: {failed} 个站点")
    print(f"输出目录: {self.output_dir.absolute()}")
    print(f"{'='*80}\n")


# 替换原方法
SedimentDataExtractor.process_all_stations = process_all_stations_parallel


def main():
    input_file = 'hydat.nc'
    output_dir = 'sediment'

    if not Path(input_file).exists():
        print(f"错误: 文件不存在: {input_file}")
        return

    extractor = SedimentDataExtractor(input_file, output_dir)
    extractor.process_all_stations()


if __name__ == "__main__":
    main()

