#!/usr/bin/env python3
"""
HYDAT数据集全面质量控制和CF-1.8标准化处理脚本

功能包括:
1. 数据内容验证与质量标志 (Data Validation & Flagging)
2. 元数据标准化 (CF-1.8 Compliant Metadata)
3. 物理规律检查与标记
4. 时间截取和无效站点删除
5. 数据溯源信息追加

作者: Zhongwang Wei
日期: 2025-10-26
"""

import netCDF4 as nc
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


class HYDATQualityControl:
    """HYDAT数据质量控制和标准化处理类"""

    def __init__(self, input_dir, output_dir):
        """
        初始化

        Parameters:
        -----------
        input_dir : str or Path
            输入NetCDF文件目录
        output_dir : str or Path
            输出NetCDF文件目录
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 物理阈值设置
        self.Q_extreme_high = 100000.0  # m3/s - 极端高值
        self.SSC_extreme_high = 3000.0  # mg/L - 极端高值
        self.SSC_min = 0.1  # mg/L - 最小物理有效值

        # 统计信息
        self.stats = {
            'total_stations': 0,
            'processed_stations': 0,
            'removed_stations': 0,
            'stations_info': []
        }

    def check_physical_constraints(self, Q, SSC, SSL):
        """
        物理规律检查，生成质量标志

        Parameters:
        -----------
        Q : numpy.ndarray
            径流数据 (m3/s)
        SSC : numpy.ndarray
            泥沙浓度数据 (mg/L)
        SSL : numpy.ndarray
            输沙率数据 (ton/day)

        Returns:
        --------
        Q_flag, SSC_flag, SSL_flag : numpy.ndarray
            质量标志数组 (byte)
            0 = Good data
            1 = Estimated data
            2 = Suspect data (zero/extreme)
            3 = Bad data (negative)
            9 = Missing data
        """
        n = len(Q)
        Q_flag = np.full(n, 9, dtype=np.int8)  # 默认为缺失
        SSC_flag = np.full(n, 9, dtype=np.int8)
        SSL_flag = np.full(n, 9, dtype=np.int8)

        # 检查 Q (径流)
        for i in range(n):
            if Q[i] == -9999.0 or np.isnan(Q[i]):
                Q_flag[i] = 9  # Missing
            elif Q[i] < 0:
                Q_flag[i] = 3  # Bad data (negative)
            elif Q[i] == 0:
                Q_flag[i] = 2  # Suspect (zero flow - 可能是真实断流，也可能是测量问题)
            elif Q[i] > self.Q_extreme_high:
                Q_flag[i] = 2  # Suspect (extreme high)
            else:
                Q_flag[i] = 0  # Good data

        # 检查 SSC (泥沙浓度)
        for i in range(n):
            if SSC[i] == -9999.0 or np.isnan(SSC[i]):
                SSC_flag[i] = 9  # Missing
            elif SSC[i] < 0:
                SSC_flag[i] = 3  # Bad data (negative)
            elif SSC[i] < self.SSC_min:
                SSC_flag[i] = 2  # Suspect (too low)
            elif SSC[i] > self.SSC_extreme_high:
                SSC_flag[i] = 2  # Suspect (extreme high)
            else:
                SSC_flag[i] = 0  # Good data

        # 检查 SSL (输沙率)
        for i in range(n):
            if SSL[i] == -9999.0 or np.isnan(SSL[i]):
                SSL_flag[i] = 9  # Missing
            elif SSL[i] < 0:
                SSL_flag[i] = 3  # Bad data (negative)
            else:
                SSL_flag[i] = 0  # Good data

        return Q_flag, SSC_flag, SSL_flag

    def find_valid_time_range(self, time, Q, SSC, SSL, Q_flag, SSC_flag, SSL_flag):
        """
        找到有效数据的时间范围
        数据选取sediment或discharge有值的起始年份的第一个月到结束年份的12月

        Parameters:
        -----------
        time : numpy.ndarray
            时间数组 (days since 1970-01-01)
        Q, SSC, SSL : numpy.ndarray
            数据数组
        Q_flag, SSC_flag, SSL_flag : numpy.ndarray
            质量标志数组

        Returns:
        --------
        start_idx, end_idx : int
            有效数据的起止索引
        has_valid_data : bool
            是否有有效数据
        """
        # 找到所有有有效数据的索引（flag != 9，即非缺失）
        valid_Q = (Q_flag != 9)
        valid_SSC = (SSC_flag != 9)
        valid_SSL = (SSL_flag != 9)

        # 任意变量有值即可
        valid_any = valid_Q | valid_SSC | valid_SSL

        if not np.any(valid_any):
            return 0, 0, False

        # 找到第一个和最后一个有效数据的位置
        valid_indices = np.where(valid_any)[0]
        first_valid_idx = valid_indices[0]
        last_valid_idx = valid_indices[-1]

        # 转换时间到日期
        reference_date = pd.Timestamp('1970-01-01')
        first_date = reference_date + pd.Timedelta(days=float(time[first_valid_idx]))
        last_date = reference_date + pd.Timedelta(days=float(time[last_valid_idx]))

        # 起始年份第一个月的第一天
        start_date = pd.Timestamp(year=first_date.year, month=1, day=1)
        # 结束年份最后一个月的最后一天
        end_date = pd.Timestamp(year=last_date.year, month=12, day=31)

        # 找到对应的索引范围
        start_idx = 0
        end_idx = len(time) - 1

        for i in range(len(time)):
            date_i = reference_date + pd.Timedelta(days=float(time[i]))
            if date_i >= start_date:
                start_idx = i
                break

        for i in range(len(time) - 1, -1, -1):
            date_i = reference_date + pd.Timedelta(days=float(time[i]))
            if date_i <= end_date:
                end_idx = i
                break

        return start_idx, end_idx + 1, True  # end_idx+1 因为切片是左闭右开

    def calculate_completeness(self, data_array, flag_array, start_date, end_date):
        """
        计算数据完整性（Good data的百分比）

        Parameters:
        -----------
        data_array : numpy.ndarray
            数据数组
        flag_array : numpy.ndarray
            质量标志数组
        start_date, end_date : pd.Timestamp
            起止日期

        Returns:
        --------
        percent_complete : float
            完整性百分比
        """
        # 计算时间范围内的总天数
        total_days = (end_date - start_date).days + 1

        # 计算Good data的天数（flag == 0）
        good_data_count = np.sum(flag_array == 0)

        if total_days > 0:
            return (good_data_count / total_days) * 100.0
        else:
            return 0.0

    def process_station(self, input_file):
        """
        处理单个站点文件

        Parameters:
        -----------
        input_file : Path
            输入NetCDF文件路径

        Returns:
        --------
        success : bool
            是否处理成功
        station_info : dict
            站点信息字典
        """
        print(f"处理站点: {input_file.name}")

        try:
            with nc.Dataset(input_file, 'r') as ds_in:
                # 读取基本信息
                station_id = ds_in.station_id if hasattr(ds_in, 'station_id') else ''
                station_name = ds_in.station_name if hasattr(ds_in, 'station_name') else ''
                province = ds_in.province_territory if hasattr(ds_in, 'province_territory') else ''

                # 读取坐标 (兼容不同的变量名)
                if 'latitude' in ds_in.variables:
                    lat = float(ds_in.variables['latitude'][:])
                elif 'lat' in ds_in.variables:
                    lat = float(ds_in.variables['lat'][:])
                else:
                    raise ValueError("Cannot find latitude variable")

                if 'longitude' in ds_in.variables:
                    lon = float(ds_in.variables['longitude'][:])
                elif 'lon' in ds_in.variables:
                    lon = float(ds_in.variables['lon'][:])
                else:
                    raise ValueError("Cannot find longitude variable")

                # 读取其他标量
                altitude = float(ds_in.variables['altitude'][:]) if 'altitude' in ds_in.variables else -9999.0
                upstream_area = float(ds_in.variables['upstream_area'][:]) if 'upstream_area' in ds_in.variables else -9999.0

                # 读取时间序列数据
                time = ds_in.variables['time'][:]

                # 读取 Q (discharge) - 兼容不同变量名
                if 'discharge' in ds_in.variables:
                    Q = ds_in.variables['discharge'][:]
                elif 'Q' in ds_in.variables:
                    Q = ds_in.variables['Q'][:]
                else:
                    raise ValueError("Cannot find discharge variable")

                # 读取 SSC - 兼容不同变量名
                if 'ssc' in ds_in.variables:
                    SSC = ds_in.variables['ssc'][:]
                elif 'SSC' in ds_in.variables:
                    SSC = ds_in.variables['SSC'][:]
                else:
                    raise ValueError("Cannot find SSC variable")

                # 读取 SSL - 兼容不同变量名
                if 'sediment_load' in ds_in.variables:
                    SSL = ds_in.variables['sediment_load'][:]
                elif 'SSL' in ds_in.variables:
                    SSL = ds_in.variables['SSL'][:]
                else:
                    raise ValueError("Cannot find sediment load variable")

                # 物理规律检查，生成质量标志
                Q_flag, SSC_flag, SSL_flag = self.check_physical_constraints(Q, SSC, SSL)

                # 找到有效时间范围
                start_idx, end_idx, has_valid = self.find_valid_time_range(
                    time, Q, SSC, SSL, Q_flag, SSC_flag, SSL_flag
                )

                if not has_valid:
                    print(f"  ⚠ 警告: 站点 {station_id} 无有效数据，跳过")
                    self.stats['removed_stations'] += 1
                    return False, None

                # 截取有效时间范围
                time = time[start_idx:end_idx]
                Q = Q[start_idx:end_idx]
                SSC = SSC[start_idx:end_idx]
                SSL = SSL[start_idx:end_idx]
                Q_flag = Q_flag[start_idx:end_idx]
                SSC_flag = SSC_flag[start_idx:end_idx]
                SSL_flag = SSL_flag[start_idx:end_idx]

                # 计算时间范围
                reference_date = pd.Timestamp('1970-01-01')
                start_date = reference_date + pd.Timedelta(days=float(time[0]))
                end_date = reference_date + pd.Timedelta(days=float(time[-1]))

                # 计算完整性
                Q_completeness = self.calculate_completeness(Q, Q_flag, start_date, end_date)
                SSC_completeness = self.calculate_completeness(SSC, SSC_flag, start_date, end_date)
                SSL_completeness = self.calculate_completeness(SSL, SSL_flag, start_date, end_date)

                # 创建输出文件
                output_file = self.output_dir / f"HYDAT_{station_id}.nc"

                with nc.Dataset(output_file, 'w', format='NETCDF4') as ds_out:
                    # 创建维度
                    ds_out.createDimension('time', len(time))

                    # 创建时间变量
                    var_time = ds_out.createVariable('time', 'f8', ('time',))
                    var_time.standard_name = 'time'
                    var_time.long_name = 'time'
                    var_time.units = 'days since 1970-01-01 00:00:00'
                    var_time.calendar = 'gregorian'
                    var_time.axis = 'T'
                    var_time[:] = time

                    # 创建坐标变量 (标量)
                    var_lat = ds_out.createVariable('lat', 'f4')
                    var_lat.standard_name = 'latitude'
                    var_lat.long_name = 'station latitude'
                    var_lat.units = 'degrees_north'
                    var_lat.valid_range = np.array([-90.0, 90.0], dtype=np.float32)
                    var_lat[:] = lat

                    var_lon = ds_out.createVariable('lon', 'f4')
                    var_lon.standard_name = 'longitude'
                    var_lon.long_name = 'station longitude'
                    var_lon.units = 'degrees_east'
                    var_lon.valid_range = np.array([-180.0, 180.0], dtype=np.float32)
                    var_lon[:] = lon

                    # 创建其他标量变量
                    var_alt = ds_out.createVariable('altitude', 'f4', fill_value=-9999.0)
                    var_alt.standard_name = 'altitude'
                    var_alt.long_name = 'station elevation above sea level'
                    var_alt.units = 'm'
                    var_alt.positive = 'up'
                    var_alt.comment = 'Source: HYDAT database.'
                    var_alt[:] = altitude

                    var_area = ds_out.createVariable('upstream_area', 'f4', fill_value=-9999.0)
                    var_area.long_name = 'upstream drainage area'
                    var_area.units = 'km2'
                    var_area.comment = 'Source: HYDAT database.'
                    var_area[:] = upstream_area

                    # 创建数据变量 Q
                    var_Q = ds_out.createVariable('Q', 'f4', ('time',),
                                                   fill_value=-9999.0, zlib=True, complevel=4)
                    var_Q.standard_name = 'water_volume_transport_in_river_channel'
                    var_Q.long_name = 'river discharge'
                    var_Q.units = 'm3 s-1'
                    var_Q.coordinates = 'time lat lon'
                    var_Q.ancillary_variables = 'Q_flag'
                    var_Q.comment = 'Source: Original data from HYDAT database.'
                    var_Q[:] = Q

                    # Q质量标志
                    var_Q_flag = ds_out.createVariable('Q_flag', 'i1', ('time',), fill_value=np.int8(9))
                    var_Q_flag.long_name = 'quality flag for river discharge'
                    var_Q_flag.standard_name = 'status_flag'
                    var_Q_flag.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.int8)
                    var_Q_flag.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
                    var_Q_flag.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'
                    var_Q_flag[:] = Q_flag

                    # 创建数据变量 SSC
                    var_SSC = ds_out.createVariable('SSC', 'f4', ('time',),
                                                     fill_value=-9999.0, zlib=True, complevel=4)
                    var_SSC.standard_name = 'mass_concentration_of_suspended_matter_in_water'
                    var_SSC.long_name = 'suspended sediment concentration'
                    var_SSC.units = 'mg L-1'
                    var_SSC.coordinates = 'time lat lon'
                    var_SSC.ancillary_variables = 'SSC_flag'
                    var_SSC.comment = 'Source: Original data from HYDAT database.'
                    var_SSC[:] = SSC

                    # SSC质量标志
                    var_SSC_flag = ds_out.createVariable('SSC_flag', 'i1', ('time',), fill_value=np.int8(9))
                    var_SSC_flag.long_name = 'quality flag for suspended sediment concentration'
                    var_SSC_flag.standard_name = 'status_flag'
                    var_SSC_flag.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.int8)
                    var_SSC_flag.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
                    var_SSC_flag.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'
                    var_SSC_flag[:] = SSC_flag

                    # 创建数据变量 SSL
                    var_SSL = ds_out.createVariable('SSL', 'f4', ('time',),
                                                     fill_value=-9999.0, zlib=True, complevel=4)
                    var_SSL.long_name = 'suspended sediment load'
                    var_SSL.units = 'ton day-1'
                    var_SSL.coordinates = 'time lat lon'
                    var_SSL.ancillary_variables = 'SSL_flag'
                    var_SSL.comment = 'Source: Calculated. Formula: SSL (ton/day) = Q (m³/s) × SSC (mg/L) × 86.4, where 86.4 = 86400 s/day × 10⁻⁶ ton/mg × 1000 L/m³.'
                    var_SSL[:] = SSL

                    # SSL质量标志
                    var_SSL_flag = ds_out.createVariable('SSL_flag', 'i1', ('time',), fill_value=np.int8(9))
                    var_SSL_flag.long_name = 'quality flag for suspended sediment load'
                    var_SSL_flag.standard_name = 'status_flag'
                    var_SSL_flag.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.int8)
                    var_SSL_flag.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
                    var_SSL_flag.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'
                    var_SSL_flag[:] = SSL_flag

                    # 设置全局属性
                    ds_out.Conventions = 'CF-1.8, ACDD-1.3'
                    ds_out.title = 'Harmonized Global River Discharge and Sediment'
                    ds_out.summary = f'River discharge and suspended sediment data for station {station_name} (ID: {station_id}) from the HYDAT database (Water Survey of Canada). This dataset contains daily observations of discharge, suspended sediment concentration, and sediment load with quality control flags.'
                    ds_out.source = 'In-situ station data'
                    ds_out.data_source_name = 'HYDAT Dataset'
                    ds_out.station_name = station_name
                    river_name = station_name.split(' AT ')[0] if ' AT ' in station_name else station_name.split(' NEAR ')[0] if ' NEAR ' in station_name else ''
                    ds_out.river_name = river_name
                    ds_out.Source_ID = station_id
                    ds_out.Type = 'In-situ station data'
                    ds_out.temporal_resolution = 'daily'
                    ds_out.temporal_span = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                    ds_out.geographic_coverage = f"{province}, Canada"
                    ds_out.time_coverage_start = start_date.strftime('%Y-%m-%d')
                    ds_out.time_coverage_end = end_date.strftime('%Y-%m-%d')
                    ds_out.variables_provided = 'altitude, upstream_area, Q, SSC, SSL'
                    ds_out.number_of_data = '1'
                    ds_out.reference = 'HYDAT - Canadian Hydrometric Database, Water Survey of Canada'
                    ds_out.source_data_link = 'https://www.canada.ca/en/environment-climate-change/services/water-overview/quantity/monitoring/survey/data-products-services/national-archive-hydat.html'
                    ds_out.creator_name = 'Zhongwang Wei'
                    ds_out.creator_email = 'weizhw6@mail.sysu.edu.cn'
                    ds_out.creator_institution = 'Sun Yat-sen University, China'
                    ds_out.geospatial_lat_min = lat
                    ds_out.geospatial_lat_max = lat
                    ds_out.geospatial_lon_min = lon
                    ds_out.geospatial_lon_max = lon

                    # 数据溯源历史记录
                    history_entry = (
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                        f"Converted from HYDAT database to CF-1.8 compliant NetCDF format. "
                        f"Applied quality control checks including physical constraint validation "
                        f"(Q range check, SSC range check, SSL negative check). "
                        f"Trimmed data to valid time range from {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}. "
                        f"Script: process_hydat_cf18.py"
                    )
                    ds_out.history = history_entry
                    ds_out.date_created = datetime.now().strftime('%Y-%m-%d')
                    ds_out.date_modified = datetime.now().strftime('%Y-%m-%d')
                    ds_out.processing_level = 'Quality controlled and standardized'
                    ds_out.comment = (
                        f"Data quality flags indicate reliability: 0=good, 1=estimated, 2=suspect, 3=bad, 9=missing. "
                        f"Quality control applied: Q<0 flagged as bad, Q=0 flagged as suspect, Q>{self.Q_extreme_high} flagged as suspect; "
                        f"SSC<0 flagged as bad, SSC<{self.SSC_min} or SSC>{self.SSC_extreme_high} flagged as suspect; "
                        f"SSL<0 flagged as bad."
                    )

                # 收集站点信息用于CSV
                station_info = {
                    'station_name': station_name,
                    'Source_ID': station_id,
                    'river_name': river_name,
                    'longitude': lon,
                    'latitude': lat,
                    'altitude': altitude if altitude != -9999.0 else np.nan,
                    'upstream_area': upstream_area if upstream_area != -9999.0 else np.nan,
                    'Data Source Name': 'HYDAT Dataset',
                    'Type': 'In-situ',
                    'Temporal Resolution': 'daily',
                    'Temporal Span': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                    'Variables Provided': 'Q, SSC, SSL',
                    'Geographic Coverage': f"{province}, Canada",
                    'Reference/DOI': 'https://www.canada.ca/en/environment-climate-change/services/water-overview/quantity/monitoring/survey/data-products-services/national-archive-hydat.html',
                    'Q_start_date': start_date.year,
                    'Q_end_date': end_date.year,
                    'Q_percent_complete': round(Q_completeness, 2),
                    'SSC_start_date': start_date.year,
                    'SSC_end_date': end_date.year,
                    'SSC_percent_complete': round(SSC_completeness, 2),
                    'SSL_start_date': start_date.year,
                    'SSL_end_date': end_date.year,
                    'SSL_percent_complete': round(SSL_completeness, 2)
                }

                self.stats['processed_stations'] += 1
                print(f"  ✓ 成功处理")
                print(f"    时间范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
                print(f"    完整性: Q={Q_completeness:.1f}%, SSC={SSC_completeness:.1f}%, SSL={SSL_completeness:.1f}%")

                return True, station_info

        except Exception as e:
            print(f"  ✗ 错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, None



    def process_all_stations(self):
        """并行处理所有站点"""

        print(f"\n{'='*80}")
        print(f"HYDAT 数据集质量控制和CF-1.8标准化处理 (并行加速版)")
        print(f"{'='*80}\n")

        input_files = sorted(self.input_dir.glob('HYDAT_*_SEDIMENT.nc'))
        self.stats['total_stations'] = len(input_files)

        print(f"找到 {len(input_files)} 个站点文件")
        print(f"使用 CPU 核心数: {os.cpu_count()} 并行处理\n")
        print(f"输入目录: {self.input_dir}")
        print(f"输出目录: {self.output_dir}")
        print(f"{'='*80}\n")

        results = []

        # ★★★ 并行执行 ★★★
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_station = {executor.submit(self.process_station, f): f for f in input_files}

            for future in as_completed(future_to_station):
                success, station_info = future.result()
                if success and station_info:
                    results.append(station_info)

        # === 更新统计信息 ===
        self.stats['processed_stations'] = len(results)
        self.stats['removed_stations'] = self.stats['total_stations'] - len(results)
        self.stats['stations_info'] = results

        print(f"\n{'='*80}")
        print(f"处理完成! (并行)")
        print(f"{'='*80}")
        print(f"总站点数: {self.stats['total_stations']}")
        print(f"成功处理: {self.stats['processed_stations']}")
        print(f"删除站点: {self.stats['removed_stations']}")
        print(f"{'='*80}\n")

        return self.stats

    def generate_csv_summary(self, output_csv):
        """生成CSV站点摘要文件"""
        print(f"\n生成CSV摘要文件: {output_csv}")

        if not self.stats['stations_info']:
            print("  ⚠ 警告: 无站点信息可写入CSV")
            return

        df = pd.DataFrame(self.stats['stations_info'])

        # 按指定顺序排列列
        column_order = [
            'station_name', 'Source_ID', 'river_name', 'longitude', 'latitude',
            'altitude', 'upstream_area', 'Data Source Name', 'Type',
            'Temporal Resolution', 'Temporal Span', 'Variables Provided',
            'Geographic Coverage', 'Reference/DOI',
            'Q_start_date', 'Q_end_date', 'Q_percent_complete',
            'SSC_start_date', 'SSC_end_date', 'SSC_percent_complete',
            'SSL_start_date', 'SSL_end_date', 'SSL_percent_complete'
        ]

        df = df[column_order]
        df.to_csv(output_csv, index=False)

        print(f"  ✓ CSV文件已生成: {len(df)} 个站点")


def main():
    """主函数"""
    # 设置路径
    input_dir = Path('/mnt/d/sediment_data/Script/Dataset/Hydat/sediment_update/')
    output_dir = Path('/mnt/d/sediment_data/Script/Dataset/Hydat/output_update/')
    csv_file = output_dir / 'HYDAT_station_summary.csv'

    # 创建处理对象
    qc = HYDATQualityControl(input_dir, output_dir)

    # 处理所有站点
    stats = qc.process_all_stations()

    # 生成CSV摘要
    qc.generate_csv_summary(csv_file)

    print(f"\n✓ 全部完成!")
    print(f"  输出目录: {output_dir}")
    print(f"  CSV摘要: {csv_file}")


if __name__ == '__main__':
    main()
