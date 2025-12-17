#!/usr/bin/env python3
"""
将 sediment_transport_month_yr_data.csv 转换为 NetCDF 格式

数据结构：
- 409 个河流站点
- 月度时间序列数据 (1984-2012)
- 变量：SSC (mg/L), Q (m³/s), 月泥沙通量 (ton/month)

输出：
- 每个站点一个 NetCDF 文件
- CF-1.8 兼容格式
- 单位：m³/s, ton/day, mg/L

Author: Claude Code
Date: 2025-10-19
"""

import pandas as pd
import netCDF4 as nc
import numpy as np
from datetime import datetime, timedelta
import os
import glob


def create_timeseries_netcdf(output_path, site_no, river_name, df_site, station_metadata=None):
    """
    为单个站点创建时间序列 NetCDF 文件

    Parameters:
    -----------
    output_path : str
        输出文件路径
    site_no : str
        站点编号
    river_name : str
        河流名称
    df_site : DataFrame
        该站点的时间序列数据
    station_metadata : dict, optional
        站点元数据（坐标等）
    """

    # 创建 NetCDF 文件
    dataset = nc.Dataset(output_path, 'w', format='NETCDF4')

    # 排序数据（按年月）
    df_site = df_site.sort_values(['year', 'month']).reset_index(drop=True)

    n_times = len(df_site)

    # 创建维度
    dataset.createDimension('time', n_times)
    dataset.createDimension('latitude', 1)
    dataset.createDimension('longitude', 1)

    # 获取坐标（如果有）
    if station_metadata and 'latitude' in station_metadata:
        lat_value = station_metadata['latitude']
        lon_value = station_metadata['longitude']
    else:
        # 默认值（如果没有元数据）
        lat_value = np.nan
        lon_value = np.nan

    # 创建坐标变量
    lat_var = dataset.createVariable('latitude', 'f4', ('latitude',))
    lat_var.standard_name = 'latitude'
    lat_var.long_name = 'latitude'
    lat_var.units = 'degrees_north'
    lat_var.axis = 'Y'
    lat_var[:] = lat_value

    lon_var = dataset.createVariable('longitude', 'f4', ('longitude',))
    lon_var.standard_name = 'longitude'
    lon_var.long_name = 'longitude'
    lon_var.units = 'degrees_east'
    lon_var.axis = 'X'
    lon_var[:] = lon_value

    # 创建时间变量
    time_var = dataset.createVariable('time', 'f8', ('time',))
    time_var.units = 'days since 1970-01-01 00:00:00'
    time_var.calendar = 'gregorian'
    time_var.standard_name = 'time'
    time_var.long_name = 'time'
    time_var.axis = 'T'

    # 计算时间值（每月15日作为代表日期）
    reference_date = datetime(1970, 1, 1)
    time_values = []
    for _, row in df_site.iterrows():
        date = datetime(int(row['year']), int(row['month']), 15)
        days = (date - reference_date).days
        time_values.append(days)
    time_var[:] = np.array(time_values)

    # 创建 SSC 变量 (mg/L)
    ssc_var = dataset.createVariable('SSC', 'f4', ('time', 'latitude', 'longitude'),
                                     fill_value=-9999.0)
    ssc_var.long_name = 'Suspended Sediment Concentration'
    ssc_var.standard_name = 'mass_concentration_of_suspended_matter_in_water'
    ssc_var.units = 'mg L-1'
    ssc_var.coordinates = 'time latitude longitude'

    ssc_data = np.full((n_times, 1, 1), -9999.0, dtype=np.float32)
    ssc_values = df_site['SSC_mgL'].values
    valid_mask = ~pd.isna(ssc_values)
    ssc_data[valid_mask, 0, 0] = ssc_values[valid_mask]
    ssc_var[:] = ssc_data

    # 创建 SSC 标准差变量
    ssc_sd_var = dataset.createVariable('SSC_sd', 'f4', ('time', 'latitude', 'longitude'),
                                        fill_value=-9999.0)
    ssc_sd_var.long_name = 'SSC standard deviation'
    ssc_sd_var.units = 'mg L-1'
    ssc_sd_var.coordinates = 'time latitude longitude'

    ssc_sd_data = np.full((n_times, 1, 1), -9999.0, dtype=np.float32)
    ssc_sd_values = df_site['SSC_mgL_sd'].values
    valid_mask = ~pd.isna(ssc_sd_values)
    ssc_sd_data[valid_mask, 0, 0] = ssc_sd_values[valid_mask]
    ssc_sd_var[:] = ssc_sd_data

    # 创建 Discharge 变量 (m³/s)
    discharge_var = dataset.createVariable('Discharge', 'f4', ('time', 'latitude', 'longitude'),
                                          fill_value=-9999.0)
    discharge_var.long_name = 'River discharge'
    discharge_var.standard_name = 'water_volume_transport_in_river_channel'
    discharge_var.units = 'm3 s-1'
    discharge_var.coordinates = 'time latitude longitude'

    discharge_data = np.full((n_times, 1, 1), -9999.0, dtype=np.float32)
    q_values = df_site['Q_cms'].values
    valid_mask = ~pd.isna(q_values)
    discharge_data[valid_mask, 0, 0] = q_values[valid_mask]
    discharge_var[:] = discharge_data

    # 创建月泥沙通量变量 (ton/month → ton/day)
    # 转换：ton/month → ton/day (除以当月天数的平均值 ~30.44)
    sediment_flux_var = dataset.createVariable('sediment_flux', 'f4',
                                               ('time', 'latitude', 'longitude'),
                                               fill_value=-9999.0)
    sediment_flux_var.long_name = 'Daily sediment flux'
    sediment_flux_var.standard_name = 'sediment_flux'
    sediment_flux_var.units = 'ton day-1'
    sediment_flux_var.coordinates = 'time latitude longitude'
    sediment_flux_var.description = 'Converted from monthly totals to daily average'

    sediment_data = np.full((n_times, 1, 1), -9999.0, dtype=np.float32)
    tons_values = df_site['tons_month'].values
    valid_mask = ~pd.isna(tons_values)

    # 转换为日均值：考虑每月实际天数
    for i, row in df_site.iterrows():
        if valid_mask[i]:
            year = int(row['year'])
            month = int(row['month'])
            # 计算该月的天数
            if month == 12:
                next_month_date = datetime(year + 1, 1, 1)
            else:
                next_month_date = datetime(year, month + 1, 1)
            current_month_date = datetime(year, month, 1)
            days_in_month = (next_month_date - current_month_date).days

            sediment_data[i, 0, 0] = tons_values[i] / days_in_month

    sediment_flux_var[:] = sediment_data

    # 创建月泥沙通量标准误差变量
    sediment_se_var = dataset.createVariable('sediment_flux_se', 'f4',
                                            ('time', 'latitude', 'longitude'),
                                            fill_value=-9999.0)
    sediment_se_var.long_name = 'Sediment flux standard error (daily)'
    sediment_se_var.units = 'ton day-1'
    sediment_se_var.coordinates = 'time latitude longitude'

    sediment_se_data = np.full((n_times, 1, 1), -9999.0, dtype=np.float32)
    se_values = df_site['tons_month_se'].values
    valid_mask_se = ~pd.isna(se_values)

    for i, row in df_site.iterrows():
        if valid_mask_se[i]:
            year = int(row['year'])
            month = int(row['month'])
            if month == 12:
                next_month_date = datetime(year + 1, 1, 1)
            else:
                next_month_date = datetime(year, month + 1, 1)
            current_month_date = datetime(year, month, 1)
            days_in_month = (next_month_date - current_month_date).days

            sediment_se_data[i, 0, 0] = se_values[i] / days_in_month

    sediment_se_var[:] = sediment_se_data

    # 创建数据来源标识变量
    source_var = dataset.createVariable('data_source', 'S1', ('time',))
    source_var.long_name = 'SSC data source'
    source_var.description = 'L=Landsat observation, R=Rating curve estimate'

    source_codes = []
    for source in df_site['SSC_source'].values:
        if pd.isna(source):
            code = 'U'  # Unknown
        elif 'Landsat' in str(source):
            code = 'L'  # Landsat
        elif 'Rating' in str(source):
            code = 'R'  # Rating curve
        else:
            code = 'O'  # Other
        source_codes.append(code)

    source_var[:] = np.array(source_codes, dtype='S1')

    # 全局属性
    dataset.title = f"Monthly sediment transport time series for {river_name}"
    dataset.institution = "Global sediment flux analysis"
    dataset.source = "Landsat-derived SSC and GRDC discharge data"
    dataset.history = f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    dataset.references = "Dethier et al. (2022); Yang et al. (2015); Milliman & Farnsworth (2011)"
    dataset.Conventions = 'CF-1.8'

    dataset.site_no = site_no
    dataset.river_name = river_name

    if station_metadata:
        if 'latitude' in station_metadata:
            dataset.latitude = station_metadata['latitude']
        if 'longitude' in station_metadata:
            dataset.longitude = station_metadata['longitude']
        if 'drainage_area_km2' in station_metadata:
            dataset.drainage_area_km2 = station_metadata['drainage_area_km2']

    # 时间范围
    dataset.time_coverage_start = df_site['year'].min()
    dataset.time_coverage_end = df_site['year'].max()
    dataset.time_resolution = 'monthly'

    # 数据统计
    n_landsat = (df_site['SSC_source'].str.contains('Landsat', na=False)).sum()
    n_rating = (df_site['SSC_source'].str.contains('Rating', na=False)).sum()

    dataset.n_observations = len(df_site)
    dataset.n_landsat_observations = int(n_landsat)
    dataset.n_rating_curve_estimates = int(n_rating)

    dataset.comment = f"Monthly time series with {n_landsat} Landsat observations and {n_rating} rating curve estimates"

    dataset.close()

    return {
        'site_no': site_no,
        'river_name': river_name,
        'n_times': n_times,
        'time_start': int(df_site['year'].min()),
        'time_end': int(df_site['year'].max()),
        'n_landsat': int(n_landsat),
        'n_rating': int(n_rating)
    }


def load_station_metadata():
    """
    加载站点元数据（坐标信息等）
    """
    # 尝试从相关文件加载元数据
    metadata_file = '/Users/zhongwangwei/Downloads/7808492/imports/outlet_rivers_ls_metadata.csv'

    if os.path.exists(metadata_file):
        df_meta = pd.read_csv(metadata_file, encoding='utf-8-sig')
        metadata_dict = {}

        for _, row in df_meta.iterrows():
            site_no = row['site_no']
            metadata_dict[site_no] = {
                'latitude': row['Latitude'] if 'Latitude' in row and not pd.isna(row['Latitude']) else np.nan,
                'longitude': row['Longitude'] if 'Longitude' in row and not pd.isna(row['Longitude']) else np.nan,
                'drainage_area_km2': row['drainage_area_km2'] if 'drainage_area_km2' in row and not pd.isna(row['drainage_area_km2']) else np.nan
            }

        return metadata_dict
    else:
        return {}


def main():
    """主函数：转换整个 sediment_transport_month_yr_data.csv"""

    print("=" * 70)
    print("转换 sediment_transport_month_yr_data.csv 到 NetCDF")
    print("=" * 70)

    # 输入输出路径
    input_file = '/Users/zhongwangwei/Downloads/7808492/imports/sediment_transport_month_yr_data.csv'
    output_dir = '/Users/zhongwangwei/Downloads/7808492/netcdf_output/timeseries'

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\n创建输出目录: {output_dir}")

    # 读取数据
    print(f"\n读取数据文件: {input_file}")
    df = pd.read_csv(input_file, encoding='utf-8-sig')

    print(f"  总记录数: {len(df):,}")
    print(f"  列数: {len(df.columns)}")

    # 加载站点元数据
    print("\n加载站点元数据...")
    station_metadata = load_station_metadata()
    print(f"  找到 {len(station_metadata)} 个站点的元数据")

    # 获取所有独立站点
    sites = df.groupby(['site_no', 'RiverName']).size().reset_index()[['site_no', 'RiverName']]
    print(f"\n找到 {len(sites)} 个独立站点")

    # 转换每个站点
    print("\n开始转换...")
    print("-" * 70)

    results = []

    for i, (_, row) in enumerate(sites.iterrows()):
        site_no = row['site_no']
        river_name = row['RiverName']

        # 提取该站点的数据
        df_site = df[df['site_no'] == site_no].copy()

        # 生成输出文件名
        # 清理河流名称（移除特殊字符）
        safe_river_name = "".join(c for c in river_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_river_name = safe_river_name.replace(' ', '_')

        output_filename = f"TimeSeries_{safe_river_name}_{site_no.replace('st_', 'SITE-')}.nc"
        output_path = os.path.join(output_dir, output_filename)

        # 获取元数据
        metadata = station_metadata.get(site_no, {})

        # 转换
        try:
            result = create_timeseries_netcdf(output_path, site_no, river_name, df_site, metadata)
            results.append(result)

            # 显示前10个结果
            if i < 10:
                print(f"✓ {result['river_name']:<20} {result['time_start']}-{result['time_end']:}  "
                      f"{result['n_times']:4d} months  "
                      f"(L:{result['n_landsat']:3d}, R:{result['n_rating']:3d})")

        except Exception as e:
            print(f"✗ 错误: {river_name} ({site_no}) - {str(e)}")

        # 进度显示
        if (i + 1) % 50 == 0:
            print(f"\n  进度: {i+1}/{len(sites)} 站点完成...")

    print("\n" + "-" * 70)
    print("转换完成！")
    print("=" * 70)

    # 统计
    print(f"\n统计结果:")
    print(f"  成功转换: {len(results)} 个站点")
    print(f"  总时间序列点: {sum(r['n_times'] for r in results):,}")
    print(f"  平均序列长度: {np.mean([r['n_times'] for r in results]):.1f} 个月")

    time_ranges = [r['time_end'] - r['time_start'] + 1 for r in results]
    print(f"  时间跨度: {min(time_ranges)}-{max(time_ranges)} 年")

    total_landsat = sum(r['n_landsat'] for r in results)
    total_rating = sum(r['n_rating'] for r in results)
    print(f"  Landsat 观测: {total_landsat:,}")
    print(f"  Rating curve 估算: {total_rating:,}")

    print(f"\n输出目录: {output_dir}")
    print("=" * 70)

    # 示例文件验证
    if results:
        print("\n示例文件验证 (前3个站点):")
        print("-" * 70)

        for result in results[:3]:
            pattern = os.path.join(output_dir, f"TimeSeries_{result['river_name'].replace(' ', '_')}_*.nc")
            files = glob.glob(pattern)

            if files:
                filepath = files[0]
                ds = nc.Dataset(filepath, 'r')

                print(f"\n✓ {os.path.basename(filepath)}")
                print(f"  River: {ds.river_name}")
                print(f"  Period: {ds.time_coverage_start}-{ds.time_coverage_end}")
                print(f"  Variables: {', '.join(ds.variables.keys())}")
                print(f"  Dimensions: time={len(ds.dimensions['time'])}")

                if 'Discharge' in ds.variables:
                    q_data = ds.variables['Discharge'][:]
                    valid = q_data != -9999.0
                    if np.any(valid):
                        print(f"  Discharge: {q_data[valid].min():.2f} - {q_data[valid].max():.2f} m³/s")

                if 'sediment_flux' in ds.variables:
                    sed_data = ds.variables['sediment_flux'][:]
                    valid = sed_data != -9999.0
                    if np.any(valid):
                        print(f"  Sediment flux: {sed_data[valid].min():.2f} - {sed_data[valid].max():.2f} ton/day")

                ds.close()

    print("\n" + "=" * 70)
    print("所有时间序列 NetCDF 文件已生成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
