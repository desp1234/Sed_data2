#!/usr/bin/env python3
"""
HYDAT数据质量控制和CF-1.8标准化 - 简化版本

基于已验证的update_sediment_nc.py脚本
添加质量控制检查和CF-1.8元数据标准化

作者: Zhongwang Wei
日期: 2025-10-26
"""

import netCDF4 as nc
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def add_quality_flags(Q, SSC, SSL):
    """
    生成质量标志

    Returns:
    --------
    Q_flag, SSC_flag, SSL_flag : numpy.ndarray (int8)
        0 = Good, 1 = Estimated, 2 = Suspect, 3 = Bad, 9 = Missing
    """
    n = len(Q)
    Q_flag = np.full(n, 9, dtype=np.int8)
    SSC_flag = np.full(n, 9, dtype=np.int8)
    SSL_flag = np.full(n, 9, dtype=np.int8)

    # Q物理检查
    Q_extreme = 100000.0  # m3/s
    for i in range(n):
        if Q[i] == -9999.0 or np.isnan(Q[i]):
            Q_flag[i] = 9
        elif Q[i] < 0:
            Q_flag[i] = 3
        elif Q[i] == 0:
            Q_flag[i] = 2
        elif Q[i] > Q_extreme:
            Q_flag[i] = 2
        else:
            Q_flag[i] = 0

    # SSC物理检查
    SSC_min = 0.1  # mg/L
    SSC_extreme = 3000.0  # mg/L
    for i in range(n):
        if SSC[i] == -9999.0 or np.isnan(SSC[i]):
            SSC_flag[i] = 9
        elif SSC[i] < 0:
            SSC_flag[i] = 3
        elif SSC[i] < SSC_min:
            SSC_flag[i] = 2
        elif SSC[i] > SSC_extreme:
            SSC_flag[i] = 2
        else:
            SSC_flag[i] = 0

    # SSL物理检查
    for i in range(n):
        if SSL[i] == -9999.0 or np.isnan(SSL[i]):
            SSL_flag[i] = 9
        elif SSL[i] < 0:
            SSL_flag[i] = 3
        else:
            SSL_flag[i] = 0

    return Q_flag, SSC_flag, SSL_flag


def find_valid_time_range(time, Q_flag, SSC_flag, SSL_flag):
    """找到有效数据的时间范围"""
    valid_any = (Q_flag != 9) | (SSC_flag != 9) | (SSL_flag != 9)

    if not np.any(valid_any):
        return 0, 0, False

    valid_indices = np.where(valid_any)[0]
    first_idx = valid_indices[0]
    last_idx = valid_indices[-1]

    # 转换到年月
    ref = pd.Timestamp('1970-01-01')
    first_date = ref + pd.Timedelta(days=float(time[first_idx]))
    last_date = ref + pd.Timedelta(days=float(time[last_idx]))

    # 起始年1月1日到结束年12月31日
    start_date = pd.Timestamp(year=first_date.year, month=1, day=1)
    end_date = pd.Timestamp(year=last_date.year, month=12, day=31)

    # 找到索引范围
    start_idx = 0
    end_idx = len(time) - 1
    for i in range(len(time)):
        d = ref + pd.Timedelta(days=float(time[i]))
        if d >= start_date:
            start_idx = i
            break
    for i in range(len(time)-1, -1, -1):
        d = ref + pd.Timedelta(days=float(time[i]))
        if d <= end_date:
            end_idx = i
            break

    return start_idx, end_idx+1, True


def process_hydat_file(input_file, output_dir):
    """处理单个HYDAT文件"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with nc.Dataset(input_file, 'r') as ds:
            # 读取基本信息
            station_id = ds.station_id if hasattr(ds, 'station_id') else ''
            station_name = ds.station_name if hasattr(ds, 'station_name') else ''
            province = ds.province_territory if hasattr(ds, 'province_territory') else ''

            # 读取坐标
            lat = float(ds.variables['latitude'][:])
            lon = float(ds.variables['longitude'][:])
            altitude = float(ds.variables['altitude'][:]) if 'altitude' in ds.variables else -9999.0
            upstream_area = float(ds.variables['upstream_area'][:]) if 'upstream_area' in ds.variables else -9999.0

            # 读取数据
            time = ds.variables['time'][:]
            Q = ds.variables['discharge'][:]
            SSC = ds.variables['ssc'][:]
            SSL = ds.variables['sediment_load'][:]

            # 生成质量标志
            Q_flag, SSC_flag, SSL_flag = add_quality_flags(Q, SSC, SSL)

            # 找到有效时间范围
            start_idx, end_idx, has_valid = find_valid_time_range(time, Q_flag, SSC_flag, SSL_flag)

            if not has_valid:
                print(f"  ⚠ 跳过: {station_id} - 无有效数据")
                return None

            # 截取数据
            time = time[start_idx:end_idx]
            Q = Q[start_idx:end_idx]
            SSC = SSC[start_idx:end_idx]
            SSL = SSL[start_idx:end_idx]
            Q_flag = Q_flag[start_idx:end_idx]
            SSC_flag = SSC_flag[start_idx:end_idx]
            SSL_flag = SSL_flag[start_idx:end_idx]

            # 计算时间范围
            ref = pd.Timestamp('1970-01-01')
            start_date = ref + pd.Timedelta(days=float(time[0]))
            end_date = ref + pd.Timedelta(days=float(time[-1]))

            # 计算完整性
            total_days = (end_date - start_date).days + 1
            Q_completeness = (np.sum(Q_flag == 0) / total_days * 100) if total_days > 0 else 0
            SSC_completeness = (np.sum(SSC_flag == 0) / total_days * 100) if total_days > 0 else 0
            SSL_completeness = (np.sum(SSL_flag == 0) / total_days * 100) if total_days > 0 else 0

            # 创建输出文件
            output_file = output_dir / f"HYDAT_{station_id}.nc"

            with nc.Dataset(output_file, 'w', format='NETCDF4') as ds_out:
                # 创建维度和变量
                ds_out.createDimension('time', len(time))

                # 时间
                var_time = ds_out.createVariable('time', 'f8', ('time',))
                var_time.standard_name = 'time'
                var_time.long_name = 'time'
                var_time.units = 'days since 1970-01-01 00:00:00'
                var_time.calendar = 'gregorian'
                var_time.axis = 'T'
                var_time[:] = time

                # 坐标
                var_lat = ds_out.createVariable('lat', 'f4')
                var_lat.standard_name = 'latitude'
                var_lat.long_name = 'station latitude'
                var_lat.units = 'degrees_north'
                var_lat[:] = lat

                var_lon = ds_out.createVariable('lon', 'f4')
                var_lon.standard_name = 'longitude'
                var_lon.long_name = 'station longitude'
                var_lon.units = 'degrees_east'
                var_lon[:] = lon

                # 其他标量
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

                # Q
                var_Q = ds_out.createVariable('Q', 'f4', ('time',), fill_value=-9999.0, zlib=True)
                var_Q.standard_name = 'water_volume_transport_in_river_channel'
                var_Q.long_name = 'river discharge'
                var_Q.units = 'm3 s-1'
                var_Q.coordinates = 'time lat lon'
                var_Q.ancillary_variables = 'Q_flag'
                var_Q.comment = 'Source: Original data from HYDAT database.'
                var_Q[:] = Q

                var_Q_flag = ds_out.createVariable('Q_flag', 'i1', ('time',), fill_value=np.int8(9))
                var_Q_flag.long_name = 'quality flag for river discharge'
                var_Q_flag.standard_name = 'status_flag'
                var_Q_flag.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.int8)
                var_Q_flag.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
                var_Q_flag.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing.'
                var_Q_flag[:] = Q_flag

                # SSC
                var_SSC = ds_out.createVariable('SSC', 'f4', ('time',), fill_value=-9999.0, zlib=True)
                var_SSC.standard_name = 'mass_concentration_of_suspended_matter_in_water'
                var_SSC.long_name = 'suspended sediment concentration'
                var_SSC.units = 'mg L-1'
                var_SSC.coordinates = 'time lat lon'
                var_SSC.ancillary_variables = 'SSC_flag'
                var_SSC.comment = 'Source: Original data from HYDAT database.'
                var_SSC[:] = SSC

                var_SSC_flag = ds_out.createVariable('SSC_flag', 'i1', ('time',), fill_value=np.int8(9))
                var_SSC_flag.long_name = 'quality flag for suspended sediment concentration'
                var_SSC_flag.standard_name = 'status_flag'
                var_SSC_flag.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.int8)
                var_SSC_flag.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
                var_SSC_flag.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect, 3=Bad, 9=Missing.'
                var_SSC_flag[:] = SSC_flag

                # SSL
                var_SSL = ds_out.createVariable('SSL', 'f4', ('time',), fill_value=-9999.0, zlib=True)
                var_SSL.long_name = 'suspended sediment load'
                var_SSL.units = 'ton day-1'
                var_SSL.coordinates = 'time lat lon'
                var_SSL.ancillary_variables = 'SSL_flag'
                var_SSL.comment = 'Source: Calculated. Formula: SSL (ton/day) = Q (m³/s) × SSC (mg/L) × 86.4.'
                var_SSL[:] = SSL

                var_SSL_flag = ds_out.createVariable('SSL_flag', 'i1', ('time',), fill_value=np.int8(9))
                var_SSL_flag.long_name = 'quality flag for suspended sediment load'
                var_SSL_flag.standard_name = 'status_flag'
                var_SSL_flag.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.int8)
                var_SSL_flag.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
                var_SSL_flag.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect, 3=Bad, 9=Missing.'
                var_SSL_flag[:] = SSL_flag

                # 全局属性
                river_name = station_name.split(' AT ')[0] if ' AT ' in station_name else station_name.split(' NEAR ')[0] if ' NEAR ' in station_name else ''

                ds_out.Conventions = 'CF-1.8, ACDD-1.3'
                ds_out.title = 'Harmonized Global River Discharge and Sediment'
                ds_out.summary = f'River discharge and suspended sediment data for station {station_name} (ID: {station_id}) from HYDAT database. Daily observations with quality control flags.'
                ds_out.source = 'In-situ station data'
                ds_out.data_source_name = 'HYDAT Dataset'
                ds_out.station_name = station_name
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
                ds_out.reference = 'HYDAT - Canadian Hydrometric Database'
                ds_out.source_data_link = 'https://www.canada.ca/en/environment-climate-change/services/water-overview/quantity/monitoring/survey/data-products-services/national-archive-hydat.html'
                ds_out.creator_name = 'Zhongwang Wei'
                ds_out.creator_email = 'weizhw6@mail.sysu.edu.cn'
                ds_out.creator_institution = 'Sun Yat-sen University, China'
                ds_out.geospatial_lat_min = lat
                ds_out.geospatial_lat_max = lat
                ds_out.geospatial_lon_min = lon
                ds_out.geospatial_lon_max = lon
                ds_out.history = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: CF-1.8 standardization with QC. Script: hydat_qc_standardization.py"
                ds_out.date_created = datetime.now().strftime('%Y-%m-%d')
                ds_out.date_modified = datetime.now().strftime('%Y-%m-%d')
                ds_out.processing_level = 'Quality controlled and standardized'
                ds_out.comment = 'Quality flags: 0=good, 1=estimated, 2=suspect, 3=bad, 9=missing. Physical constraints applied.'

            print(f"  ✓ {station_id}: {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')} | Q={Q_completeness:.1f}% SSC={SSC_completeness:.1f}% SSL={SSL_completeness:.1f}%")

            # 返回站点信息
            return {
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

    except Exception as e:
        print(f"  ✗ 错误 {input_file.name}: {e}")
        return None


def main():
    input_dir = Path('/Users/zhongwangwei/Downloads/Sediment/Output/daily/HYDAT')
    output_dir = Path('/Users/zhongwangwei/Downloads/Sediment/Output_r/daily/HYDAT')
    csv_file = output_dir / 'HYDAT_station_summary.csv'

    print(f"\n{'='*80}")
    print(f"HYDAT 数据质量控制和CF-1.8标准化")
    print(f"{'='*80}\n")

    input_files = sorted(input_dir.glob('HYDAT_*_SEDIMENT.nc'))
    print(f"找到 {len(input_files)} 个文件")
    print(f"输入: {input_dir}")
    print(f"输出: {output_dir}\n")

    stations_info = []
    success = 0
    skipped = 0

    for i, f in enumerate(input_files, 1):
        print(f"[{i}/{len(input_files)}] ", end='')
        info = process_hydat_file(f, output_dir)
        if info:
            stations_info.append(info)
            success += 1
        else:
            skipped += 1

    # 生成CSV
    if stations_info:
        df = pd.DataFrame(stations_info)
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
        df.to_csv(csv_file, index=False)
        print(f"\n✓ CSV生成: {csv_file} ({len(df)} 个站点)")

    print(f"\n{'='*80}")
    print(f"完成! 成功: {success} | 跳过: {skipped}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
