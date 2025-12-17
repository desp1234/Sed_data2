#!/usr/bin/env python3
"""
从HYDAT NetCDF文件中提取泥沙数据并按站点保存
"""

import netCDF4 as nc
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

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
        """加载站点基本信息"""
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
        """提取每日泥沙负荷数据"""
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
                            data_records.append({
                                'time': date,
                                'sediment_load': float(load_value)
                            })
                        except:
                            continue

        if not data_records:
            return None

        df = pd.DataFrame(data_records)
        df = df.sort_values('time').reset_index(drop=True)
        return df

    def extract_sed_daily_suscon(self, ds, station_id):
        """提取每日悬浮泥沙浓度数据"""
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
                            data_records.append({
                                'time': date,
                                'suspended_sediment_concentration': float(suscon_value)
                            })
                        except:
                            continue

        if not data_records:
            return None

        df = pd.DataFrame(data_records)
        df = df.sort_values('time').reset_index(drop=True)
        return df

    def extract_sed_samples(self, ds, station_id):
        """提取泥沙样本数据"""
        if 'SED_SAMPLES' not in ds.groups:
            return None

        sed_samples = ds.groups['SED_SAMPLES']
        station_numbers = read_char_variable(sed_samples.variables['STATION_NUMBER'])
        mask = np.array([s == station_id for s in station_numbers])

        if not np.any(mask):
            return None

        data_records = []

        # 读取日期
        dates = read_char_variable(sed_samples.variables['DATE'])

        for i, is_match in enumerate(mask):
            if not is_match:
                continue

            try:
                date_str = dates[i]
                if not date_str:
                    continue

                # 解析日期 (格式可能是 YYYY-MM-DD 等)
                date = pd.to_datetime(date_str, errors='coerce')
                if pd.isna(date):
                    continue

                record = {'time': date}

                # 提取其他字段
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

                # 提取字符字段
                for var_name in ['SED_DATA_TYPE', 'SAMPLER_TYPE', 'SAMPLING_VERTICAL_LOCATION']:
                    if var_name in sed_samples.variables:
                        try:
                            value = read_char_variable(sed_samples.variables[var_name], i)
                            if value:
                                record[var_name.lower()] = value
                        except:
                            pass

                if len(record) > 1:  # 有数据
                    data_records.append(record)
            except:
                continue

        if not data_records:
            return None

        df = pd.DataFrame(data_records)
        df = df.sort_values('time').reset_index(drop=True)
        return df

    def create_station_netcdf(self, station_id, sed_loads_df, sed_suscon_df, sed_samples_df):
        """为单个站点创建包含泥沙数据的NetCDF文件"""

        station_meta = self.stations_info.get(station_id, {'STATION_NUMBER': station_id})
        output_file = self.output_dir / f"HYDAT_{station_id}_SEDIMENT.nc"

        print(f"  创建文件: {output_file.name}")

        with nc.Dataset(output_file, 'w', format='NETCDF4') as ncfile:
            # 全局属性
            ncfile.Conventions = "CF-1.8"
            ncfile.title = f"HYDAT Station {station_id} - Sediment Data"
            ncfile.institution = "Water Survey of Canada / Environment and Climate Change Canada"
            ncfile.source = "HYDAT - Canadian Hydrometric Database"
            ncfile.history = f"Created {datetime.now().isoformat()}"
            ncfile.references = "https://www.canada.ca/en/environment-climate-change/services/water-overview/quantity/monitoring/survey/data-products-services/national-archive-hydat.html"

            # 站点元数据
            ncfile.station_id = station_id

            if 'STATION_NAME' in station_meta:
                ncfile.station_name = station_meta['STATION_NAME']

            if 'PROV_TERR_STATE_LOC' in station_meta:
                ncfile.province_territory = station_meta['PROV_TERR_STATE_LOC']

            # 获取经纬度
            lat = station_meta.get('LATITUDE', None)
            lon = station_meta.get('LONGITUDE', None)

            if lat is not None and lon is not None:
                ncfile.geospatial_lat_min = lat
                ncfile.geospatial_lat_max = lat
                ncfile.geospatial_lon_min = lon
                ncfile.geospatial_lon_max = lon

                # 创建维度
                ncfile.createDimension('lat', 1)
                ncfile.createDimension('lon', 1)

                # 坐标变量
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

            # 每日泥沙负荷数据
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

            # 每日悬浮泥沙浓度数据
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

            # 泥沙样本数据
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

                # 保存样本数据的各个字段
                for col in sed_samples_df.columns:
                    if col == 'time':
                        continue

                    if sed_samples_df[col].dtype == 'object':
                        # 字符串变量 - 简化处理
                        max_len = max(len(str(v)) for v in sed_samples_df[col])
                        if max_len > 0:
                            # 将字符串列保存为属性而不是变量，避免复杂的字符数组处理
                            str_values = ', '.join([str(v) for v in sed_samples_df[col][:10]])  # 只保存前10个作为示例
                            ncfile.setncattr(f'{col}_sample_values', str_values)
                    else:
                        # 数值变量
                        num_var = ncfile.createVariable(col, 'f4', (time_dim,),
                                                       fill_value=-999.0,
                                                       zlib=True, complevel=4)
                        num_var.long_name = col.replace('_', ' ')
                        num_var[:] = sed_samples_df[col].values

        print(f"    ✓ 完成")

    def process_all_stations(self):
        """处理所有站点"""
        print(f"\n{'='*80}")
        print(f"提取泥沙数据")
        print(f"{'='*80}\n")

        with nc.Dataset(self.input_nc, 'r') as ds:
            self.load_stations(ds)

            # 获取所有有泥沙数据的站点
            all_sed_stations = set()

            sed_groups = ['SED_DLY_LOADS', 'SED_DLY_SUSCON', 'SED_SAMPLES']

            for group_name in sed_groups:
                if group_name in ds.groups:
                    station_numbers = read_char_variable(ds.groups[group_name].variables['STATION_NUMBER'])
                    for sid in station_numbers:
                        if sid:
                            all_sed_stations.add(sid)

            station_list = sorted(all_sed_stations)

            print(f"找到 {len(station_list)} 个包含泥沙数据的站点\n")
            print(f"输出目录: {self.output_dir}\n")
            print(f"{'='*80}\n")

            success_count = 0
            failed_count = 0

            for i, station_id in enumerate(station_list, 1):
                try:
                    print(f"[{i}/{len(station_list)}] 处理站点: {station_id}")

                    sed_loads_df = self.extract_sed_daily_loads(ds, station_id)
                    sed_suscon_df = self.extract_sed_daily_suscon(ds, station_id)
                    sed_samples_df = self.extract_sed_samples(ds, station_id)

                    has_data = False
                    if sed_loads_df is not None and len(sed_loads_df) > 0:
                        print(f"    泥沙负荷: {len(sed_loads_df)} 条记录")
                        has_data = True
                    if sed_suscon_df is not None and len(sed_suscon_df) > 0:
                        print(f"    悬浮泥沙浓度: {len(sed_suscon_df)} 条记录")
                        has_data = True
                    if sed_samples_df is not None and len(sed_samples_df) > 0:
                        print(f"    泥沙样本: {len(sed_samples_df)} 条记录")
                        has_data = True

                    if not has_data:
                        print(f"    跳过: 无泥沙数据")
                        failed_count += 1
                        continue

                    self.create_station_netcdf(station_id, sed_loads_df, sed_suscon_df, sed_samples_df)
                    success_count += 1

                except Exception as e:
                    print(f"    ✗ 错误: {e}")
                    import traceback
                    traceback.print_exc()
                    failed_count += 1
                    continue

                print()

        print(f"\n{'='*80}")
        print(f"处理完成!")
        print(f"{'='*80}")
        print(f"成功: {success_count} 个站点")
        print(f"失败/跳过: {failed_count} 个站点")
        print(f"输出目录: {self.output_dir.absolute()}")
        print(f"{'='*80}\n")

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
