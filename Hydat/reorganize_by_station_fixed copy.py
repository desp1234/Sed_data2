#!/usr/bin/env python3
"""
修复版本 - 直接处理MaskedArray和字符数据
"""

import netCDF4 as nc
import numpy as np
import pandas as pd
from datetime import datetime
import os
import sys
from pathlib import Path

def read_char_variable(var, index=None):
    """
    正确读取字符变量，处理MaskedArray
    """
    if index is not None:
        # 读取单个记录
        data = var[index]
    else:
        # 读取所有记录
        data = var[:]
    
    # 处理MaskedArray
    if isinstance(data, np.ma.MaskedArray):
        # 用空字符填充masked值
        data = data.filled(b'')
    
    # 转换为字符串
    if len(data.shape) == 2:
        # 多个字符串 (n, strlen)
        result = []
        for row in data:
            try:
                # 尝试拼接字节
                if isinstance(row[0], (bytes, np.bytes_)):
                    s = b''.join(row).decode('utf-8', errors='ignore').strip('\x00').strip()
                else:
                    s = ''.join([chr(ord(c)) if isinstance(c, str) else chr(c) for c in row if c]).strip()
                result.append(s)
            except:
                result.append('')
        return np.array(result)
    else:
        # 单个字符串 (strlen,)
        try:
            if isinstance(data[0], (bytes, np.bytes_)):
                return b''.join(data).decode('utf-8', errors='ignore').strip('\x00').strip()
            else:
                return ''.join([chr(ord(c)) if isinstance(c, str) else chr(c) for c in data if c]).strip()
        except:
            return ''

class FixedHydatStationReorganizer:
    def __init__(self, input_nc, output_dir):
        self.input_nc = input_nc
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stations_info = {}
        
    def load_stations(self, ds):
        """加载站点信息 - 修复版"""
        print("正在加载站点信息...")
        
        if 'STATIONS' not in ds.groups:
            raise ValueError("NetCDF文件中没有找到STATIONS表")
        
        stations = ds.groups['STATIONS']
        n_stations = len(stations.dimensions['STATIONS_records'])
        
        print(f"  站点总数: {n_stations}")
        
        # 读取所有站点编号
        station_numbers = read_char_variable(stations.variables['STATION_NUMBER'])
        
        # 过滤空站点
        valid_stations = []
        for i, station_id in enumerate(station_numbers):
            if station_id:  # 非空
                valid_stations.append((i, station_id))
        
        print(f"  有效站点: {len(valid_stations)}")
        
        # 收集站点元数据
        for idx, station_id in valid_stations:
            station_meta = {'STATION_NUMBER': station_id}
            
            # 读取所有字段
            for var_name in stations.variables:
                try:
                    var = stations.variables[var_name]
                    
                    if var.dtype == np.dtype('|S1'):
                        # 字符变量
                        value = read_char_variable(var, idx)
                        if value:
                            station_meta[var_name] = value
                    else:
                        # 数值变量
                        value = var[idx]
                        if isinstance(value, np.ma.MaskedArray):
                            if not value.mask:
                                value = value.data
                            else:
                                continue
                        
                        if var.dtype in [np.float32, np.float64]:
                            if not np.isnan(value) and value != -999:
                                station_meta[var_name] = float(value)
                        elif var.dtype in [np.int32, np.int64, np.int16, np.int8]:
                            if value != -999:
                                station_meta[var_name] = int(value)
                        
                except Exception as e:
                    continue
            
            self.stations_info[station_id] = station_meta
        
        print(f"  加载完成: {len(self.stations_info)} 个站点")
        
        return list(self.stations_info.keys())
    
    def extract_daily_flows(self, ds, station_id):
        """提取每日流量数据"""
        if 'DLY_FLOWS' not in ds.groups:
            return None
        
        flows = ds.groups['DLY_FLOWS']
        
        # 读取站点编号
        station_numbers = read_char_variable(flows.variables['STATION_NUMBER'])
        mask = np.array([s == station_id for s in station_numbers])
        
        if not np.any(mask):
            return None
        
        # 提取数据
        years = flows.variables['YEAR'][:][mask]
        months = flows.variables['MONTH'][:][mask]
        
        # 处理MaskedArray
        if isinstance(years, np.ma.MaskedArray):
            years = years.filled(-999)
        if isinstance(months, np.ma.MaskedArray):
            months = months.filled(-999)
        
        # 构建时间序列
        data_records = []
        
        for idx, (year, month) in enumerate(zip(years, months)):
            if year == -999 or month == -999:
                continue
            
            year, month = int(year), int(month)
            
            # 该月天数
            try:
                n_days = pd.Timestamp(year=year, month=month, day=1).days_in_month
            except:
                continue
            
            # 提取每日流量
            for day in range(1, n_days + 1):
                flow_var = f'FLOW{day}'
                if flow_var in flows.variables:
                    flow_value = flows.variables[flow_var][:][mask][idx]
                    
                    # 处理MaskedArray
                    if isinstance(flow_value, np.ma.MaskedArray):
                        if flow_value.mask:
                            continue
                        flow_value = flow_value.data
                    
                    if flow_value != -999 and not np.isnan(flow_value):
                        try:
                            date = pd.Timestamp(year=year, month=month, day=day)
                            data_records.append({
                                'time': date,
                                'discharge': float(flow_value)
                            })
                        except:
                            continue
        
        if not data_records:
            return None
        
        df = pd.DataFrame(data_records)
        df = df.sort_values('time').reset_index(drop=True)
        
        return df
    
    def extract_daily_levels(self, ds, station_id):
        """提取每日水位数据"""
        if 'DLY_LEVELS' not in ds.groups:
            return None
        
        levels = ds.groups['DLY_LEVELS']
        
        station_numbers = read_char_variable(levels.variables['STATION_NUMBER'])
        mask = np.array([s == station_id for s in station_numbers])
        
        if not np.any(mask):
            return None
        
        years = levels.variables['YEAR'][:][mask]
        months = levels.variables['MONTH'][:][mask]
        
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
                level_var = f'LEVEL{day}'
                if level_var in levels.variables:
                    level_value = levels.variables[level_var][:][mask][idx]
                    
                    if isinstance(level_value, np.ma.MaskedArray):
                        if level_value.mask:
                            continue
                        level_value = level_value.data
                    
                    if level_value != -999 and not np.isnan(level_value):
                        try:
                            date = pd.Timestamp(year=year, month=month, day=day)
                            data_records.append({
                                'time': date,
                                'water_level': float(level_value)
                            })
                        except:
                            continue
        
        if not data_records:
            return None
        
        df = pd.DataFrame(data_records)
        df = df.sort_values('time').reset_index(drop=True)
        
        return df
    
    def extract_annual_statistics(self, ds, station_id):
        """提取年度统计数据"""
        if 'ANNUAL_STATISTICS' not in ds.groups:
            return None
        
        ann_stats = ds.groups['ANNUAL_STATISTICS']
        
        station_numbers = read_char_variable(ann_stats.variables['STATION_NUMBER'])
        mask = np.array([s == station_id for s in station_numbers])
        
        if not np.any(mask):
            return None
        
        years = ann_stats.variables['YEAR'][:][mask]
        
        if isinstance(years, np.ma.MaskedArray):
            years = years.filled(-999)
        
        data_records = []
        
        for idx, year in enumerate(years):
            if year == -999:
                continue
            
            record = {'year': int(year)}
            
            for var_name in ['MEAN', 'MIN', 'MAX']:
                if var_name in ann_stats.variables:
                    value = ann_stats.variables[var_name][:][mask][idx]
                    
                    if isinstance(value, np.ma.MaskedArray):
                        if value.mask:
                            continue
                        value = value.data
                    
                    if value != -999 and not np.isnan(value):
                        record[var_name.lower()] = float(value)
            
            if len(record) > 1:  # 有数据
                data_records.append(record)
        
        if not data_records:
            return None
        
        return pd.DataFrame(data_records)
    
    def create_station_netcdf(self, station_id, flow_df, level_df, annual_df):
        """为单个站点创建CF-compliant NetCDF文件"""
        
        station_meta = self.stations_info[station_id]
        output_file = self.output_dir / f"HYDAT_{station_id}.nc"
        
        print(f"  创建文件: {output_file.name}")
        
        with nc.Dataset(output_file, 'w', format='NETCDF4') as ncfile:
            # 全局属性
            ncfile.Conventions = "CF-1.8"
            ncfile.title = f"HYDAT Station {station_id} - Hydrometric Data"
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
            
            if 'DRAINAGE_AREA_GROSS' in station_meta:
                ncfile.drainage_area_gross = f"{station_meta['DRAINAGE_AREA_GROSS']} km2"
            
            if 'DRAINAGE_AREA_EFFECT' in station_meta:
                ncfile.drainage_area_effective = f"{station_meta['DRAINAGE_AREA_EFFECT']} km2"
            
            # 获取经纬度
            lat = station_meta.get('LATITUDE', None)
            lon = station_meta.get('LONGITUDE', None)
            
            if lat is None or lon is None:
                print(f"    警告: 站点 {station_id} 缺少经纬度信息")
                return
            
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
            
            # 流量数据
            if flow_df is not None and len(flow_df) > 0:
                time_dim = 'time_flow'
                ncfile.createDimension(time_dim, len(flow_df))
                
                time_var = ncfile.createVariable('time_flow', 'f8', (time_dim,))
                time_var.standard_name = "time"
                time_var.long_name = "time"
                time_var.units = "days since 1850-01-01 00:00:00"
                time_var.calendar = "gregorian"
                time_var.axis = "T"
                
                reference_date = pd.Timestamp('1850-01-01')
                time_values = (flow_df['time'] - reference_date).dt.total_seconds() / 86400.0
                time_var[:] = time_values.values
                
                discharge_var = ncfile.createVariable('discharge', 'f4', 
                                                     (time_dim, 'lat', 'lon'),
                                                     fill_value=-999.0,
                                                     zlib=True, complevel=4)
                discharge_var.standard_name = "water_volume_transport_in_river_channel"
                discharge_var.long_name = "river discharge"
                discharge_var.units = "m3 s-1"
                discharge_var.coordinates = "time_flow lat lon"
                discharge_var.cell_methods = "time: mean"
                
                discharge_values = flow_df['discharge'].values
                discharge_var[:, 0, 0] = discharge_values
                
                ncfile.time_coverage_start = flow_df['time'].min().isoformat()
                ncfile.time_coverage_end = flow_df['time'].max().isoformat()
            
            # 水位数据
            if level_df is not None and len(level_df) > 0:
                time_dim = 'time_level'
                ncfile.createDimension(time_dim, len(level_df))
                
                time_var = ncfile.createVariable('time_level', 'f8', (time_dim,))
                time_var.standard_name = "time"
                time_var.long_name = "time"
                time_var.units = "days since 1850-01-01 00:00:00"
                time_var.calendar = "gregorian"
                time_var.axis = "T"
                
                reference_date = pd.Timestamp('1850-01-01')
                time_values = (level_df['time'] - reference_date).dt.total_seconds() / 86400.0
                time_var[:] = time_values.values
                
                level_var = ncfile.createVariable('water_level', 'f4',
                                                  (time_dim, 'lat', 'lon'),
                                                  fill_value=-999.0,
                                                  zlib=True, complevel=4)
                level_var.standard_name = "water_surface_height_above_reference_datum"
                level_var.long_name = "water level"
                level_var.units = "m"
                level_var.coordinates = "time_level lat lon"
                level_var.cell_methods = "time: mean"
                
                level_values = level_df['water_level'].values
                level_var[:, 0, 0] = level_values
            
            # 年度统计
            if annual_df is not None and len(annual_df) > 0:
                ncfile.createDimension('year', len(annual_df))
                
                year_var = ncfile.createVariable('year', 'i4', ('year',))
                year_var.long_name = "year"
                year_var.units = "year"
                year_var[:] = annual_df['year'].values
                
                if 'mean' in annual_df.columns:
                    mean_var = ncfile.createVariable('annual_mean_discharge', 'f4',
                                                    ('year',), fill_value=-999.0)
                    mean_var.long_name = "annual mean discharge"
                    mean_var.units = "m3 s-1"
                    mean_var[:] = annual_df['mean'].values
                
                if 'min' in annual_df.columns:
                    min_var = ncfile.createVariable('annual_min_discharge', 'f4',
                                                   ('year',), fill_value=-999.0)
                    min_var.long_name = "annual minimum discharge"
                    min_var.units = "m3 s-1"
                    min_var[:] = annual_df['min'].values
                
                if 'max' in annual_df.columns:
                    max_var = ncfile.createVariable('annual_max_discharge', 'f4',
                                                   ('year',), fill_value=-999.0)
                    max_var.long_name = "annual maximum discharge"
                    max_var.units = "m3 s-1"
                    max_var[:] = annual_df['max'].values
            
            # 站点元数据变量
            if 'DRAINAGE_AREA_GROSS' in station_meta:
                drainage_var = ncfile.createVariable('drainage_area', 'f4', ())
                drainage_var.long_name = "total drainage area"
                drainage_var.units = "km2"
                drainage_var[:] = station_meta['DRAINAGE_AREA_GROSS']
            
            if 'DRAINAGE_AREA_EFFECT' in station_meta:
                eff_drainage_var = ncfile.createVariable('effective_drainage_area', 'f4', ())
                eff_drainage_var.long_name = "effective drainage area"
                eff_drainage_var.units = "km2"
                eff_drainage_var.comment = "Contributing drainage area"
                eff_drainage_var[:] = station_meta['DRAINAGE_AREA_EFFECT']
        
        print(f"    ✓ 完成")
    
    def process_all_stations(self, max_stations=None):
        """处理站点"""
        print(f"\n{'='*80}")
        if max_stations:
            print(f"测试模式 - 处理前 {max_stations} 个站点")
        else:
            print(f"处理所有站点")
        print(f"{'='*80}\n")
        
        with nc.Dataset(self.input_nc, 'r') as ds:
            station_ids = self.load_stations(ds)
            
            if max_stations:
                station_ids = station_ids[:max_stations]
            
            print(f"\n开始提取数据...")
            print(f"输出目录: {self.output_dir}")
            print(f"{'='*80}\n")
            
            success_count = 0
            failed_count = 0
            
            for i, station_id in enumerate(station_ids, 1):
                try:
                    print(f"[{i}/{len(station_ids)}] 处理站点: {station_id}")
                    
                    flow_df = self.extract_daily_flows(ds, station_id)
                    level_df = self.extract_daily_levels(ds, station_id)
                    annual_df = self.extract_annual_statistics(ds, station_id)
                    
                    has_data = False
                    if flow_df is not None and len(flow_df) > 0:
                        print(f"    流量数据: {len(flow_df)} 条记录")
                        has_data = True
                    if level_df is not None and len(level_df) > 0:
                        print(f"    水位数据: {len(level_df)} 条记录")
                        has_data = True
                    if annual_df is not None and len(annual_df) > 0:
                        print(f"    年度统计: {len(annual_df)} 年")
                        has_data = True
                    
                    if not has_data:
                        print(f"    跳过: 无时间序列数据")
                        failed_count += 1
                        continue
                    
                    self.create_station_netcdf(station_id, flow_df, level_df, annual_df)
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
        print(f"失败: {failed_count} 个站点")
        print(f"输出目录: {self.output_dir.absolute()}")
        print(f"{'='*80}\n")

def main():
    if len(sys.argv) < 2:
        print("用法: python reorganize_by_station_fixed.py <hydat.nc> [output_dir] [max_stations]")
        print("\n参数:")
        print("  hydat.nc       - 输入的HYDAT NetCDF文件")
        print("  output_dir     - 输出目录 (默认: ./station_files/)")
        print("  max_stations   - 最多处理几个站点，不指定则处理全部")
        print("\n示例:")
        print("  python reorganize_by_station_fixed.py hydat.nc ./test/ 10")
        print("  python reorganize_by_station_fixed.py hydat.nc ./station_files/")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./station_files"
    max_stations = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    if not Path(input_file).exists():
        print(f"错误: 文件不存在: {input_file}")
        sys.exit(1)
    
    reorganizer = FixedHydatStationReorganizer(input_file, output_dir)
    reorganizer.process_all_stations(max_stations)

if __name__ == "__main__":
    main()
