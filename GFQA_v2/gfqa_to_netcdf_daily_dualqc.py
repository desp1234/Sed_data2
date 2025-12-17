#!/usr/bin/env python3
"""
Convert GFQA GEMStat data to NetCDF format (Observed daily data only)
with Dual Quality Control:
---------------------------------------------------------------------
功能：
- 从 Flux.csv / Water.csv / GEMStat_station_metadata.csv 读取原始数据
- 提取流量(Q-Inst)与悬浮泥沙浓度(TSS)数据
- 输出含两类质量信息：
  1. Data.Quality（来自原始CSV）
  2. QC Flags（自动判断）
- 仅保留“流量与泥沙在同一天都有观测”的日期
- 不插值、不补齐日期
- 输出 CF-1.8 兼容的 NetCDF 文件
"""

import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime
import os
from pathlib import Path


# ==========================================================
# 通用函数
# ==========================================================

def get_flag(value, thresholds, meanings):
    """根据阈值判断数据质量等级"""
    if pd.isna(value) or value == -9999.0:
        return meanings.split().index('missing_data')
    if value < thresholds.get('negative', -float('inf')):
        return meanings.split().index('bad_data')
    if value == thresholds.get('zero', -1):
        return meanings.split().index('suspect_data')
    if value > thresholds.get('extreme', float('inf')):
        return meanings.split().index('suspect_data')
    return meanings.split().index('good_data')


def clean_value(value):
    """清洗数值"""
    try:
        val = float(str(value).replace(',', '.'))
        if np.isnan(val) or val < 0:
            return -9999.0
        return val
    except Exception:
        return -9999.0


def parse_float(value):
    """解析浮点元数据"""
    if pd.isna(value):
        return -9999.0
    try:
        return float(str(value).replace(',', '.'))
    except Exception:
        return -9999.0


# ==========================================================
# 数据读取与预处理
# ==========================================================

def read_csv_files():
    """读取 CSV 文件"""
    print("Reading CSV files...")
    base_dir = Path("../../../Source/GFQA_v2/sed/")

    flux_df = pd.read_csv(f'{base_dir}/Flux.csv', delimiter=';', parse_dates=['Sample.Date'], encoding='iso-8859-1')
    water_df = pd.read_csv(f'{base_dir}/Water.csv', delimiter=';', parse_dates=['Sample.Date'], encoding='iso-8859-1')
    station_df = pd.read_excel(f'{base_dir}/GEMStat_station_metadata.xlsx')

    # print(flux_df['Sample.Date'].head())
    # print(water_df['Sample.Date'].head())
    flux_df['GEMS.Station.Number'] = flux_df['GEMS.Station.Number'].astype(str).str.strip()
    water_df['GEMS.Station.Number'] = water_df['GEMS.Station.Number'].astype(str).str.strip()
    station_df['GEMS Station Number'] = station_df['GEMS Station Number'].astype(str).str.strip()
    flux_df['Parameter.Code'] = flux_df['Parameter.Code'].astype(str).str.strip()
    water_df['Parameter.Code'] = water_df['Parameter.Code'].astype(str).str.strip()
    # print("Flux station sample:", list(flux_stations)[:5])
    # print("Water station sample:", list(water_stations)[:5])
    # print("Intersection size:", len(common_stations))


    print(f"Flux records: {len(flux_df)}")
    print(f"Water records: {len(water_df)}")
    print(f"Stations: {len(station_df)}")
    return flux_df, water_df, station_df


def extract_station_data(station_id, flux_df, water_df):
    """提取指定测站的流量与TSS数据"""
    discharge_data = flux_df[
        (flux_df['GEMS.Station.Number'] == station_id) &
        (flux_df['Parameter.Code'] == 'Q-Inst')
    ].copy()

    sediment_data = water_df[
        (water_df['GEMS.Station.Number'] == station_id) &
        (water_df['Parameter.Code'] == 'TSS')
    ].copy()

    return discharge_data, sediment_data


def find_overlapping_period(discharge_data, sediment_data):
    """找到两个数据集的重叠时间段"""
    if len(discharge_data) == 0 or len(sediment_data) == 0:
        return None, None
    start = max(discharge_data['Sample.Date'].min(), sediment_data['Sample.Date'].min())
    end = min(discharge_data['Sample.Date'].max(), sediment_data['Sample.Date'].max())
    if start > end:
        return None, None
    return start, end


def aggregate_to_daily(data, date_col='Sample.Date', value_col='Value', quality_col='Data.Quality'):
    """按日聚合（取同日平均）并附加原始Data.Quality"""
    data = data.copy()
    data['Date'] = data[date_col].dt.floor('D')
    data['Clean_Value'] = data[value_col].apply(clean_value)

    daily = (
        data.groupby('Date')
        .agg({
            'Clean_Value': 'mean',
            quality_col: lambda x: x.mode().iat[0] if not x.mode().empty else 'unknown'
        })
        .reset_index()
        .rename(columns={quality_col: 'Quality'})
    )
    return daily

def parse_lat_lon(station_row):
    lat = float(str(station_row['Latitude']).replace(',', '.'))
    lon = float(str(station_row['Longitude']).replace(',', '.'))
    return lat, lon



# ==========================================================
# 计算与文件输出
# ==========================================================

def calculate_sediment_load(q, ssc):
    """计算每日泥沙通量 (ton/day)"""
    if q == -9999.0 or ssc == -9999.0:
        return -9999.0
    return q * ssc * 0.0864


def create_netcdf_file(station_id, station_row, dates,
                       discharge, ssc, ssl,
                       q_flag, ssc_flag, ssl_flag,
                       q_quality, ssc_quality,
                       output_dir):
    """创建 NetCDF 文件（含自动QC与Data.Quality）"""

    filename = f"GFQA_{station_id}.nc"
    filepath = os.path.join(output_dir, filename)
    ds = nc.Dataset(filepath, 'w', format='NETCDF4')

    # 时间维度
    ds.createDimension('time', len(dates))
    time_var = ds.createVariable('time', 'f8', ('time',))
    time_var.units = 'days since 1970-01-01 00:00:00'
    time_var.standard_name = 'time'
    time_var.calendar = 'gregorian'
    time_var[:] = [(pd.Timestamp(d) - pd.Timestamp('1970-01-01')).days for d in dates]

    # 主变量
    q_var = ds.createVariable('Q', 'f4', ('time',), fill_value=-9999.0)
    q_var.units = 'm3 s-1'
    q_var.long_name = 'river discharge'
    q_var.ancillary_variables = 'Q_flag'
    q_var[:] = discharge

    ssc_var = ds.createVariable('SSC', 'f4', ('time',), fill_value=-9999.0)
    ssc_var.units = 'mg L-1'
    ssc_var.long_name = 'suspended sediment concentration'
    ssc_var.ancillary_variables = 'SSC_flag'
    ssc_var[:] = ssc

    ssl_var = ds.createVariable('SSL', 'f4', ('time',), fill_value=-9999.0)
    ssl_var.units = 'ton day-1'
    ssl_var.long_name = 'suspended sediment load'
    ssl_var.ancillary_variables = 'SSL_flag'
    ssl_var[:] = ssl

    q_var.coordinates = "latitude longitude"
    ssc_var.coordinates = "latitude longitude"
    ssl_var.coordinates = "latitude longitude"

    
    lat, lon = parse_lat_lon(station_row)

    # ds.latitude = lat
    # ds.longitude = lon
    # ds.altitude = parse_float(station_row.get('Elevation', -9999.0))
    # ds.upstream_area = parse_float(station_row.get('Upstream Basin Area', -9999.0))

    lat_var = ds.createVariable('latitude', 'f4')
    lat_var.units = 'degrees_north'
    lat_var.standard_name = 'latitude'
    lat_var[:] = lat

    lon_var = ds.createVariable('longitude', 'f4')
    lon_var.units = 'degrees_east'
    lon_var.standard_name = 'longitude'
    lon_var[:] = lon

    
    # 自动QC标志
    flag_meanings = "good_data suspect_data bad_data missing_data"
    for name, values, desc in zip(
        ['Q_flag', 'SSC_flag', 'SSL_flag'],
        [q_flag, ssc_flag, ssl_flag],
        ['discharge', 'SSC', 'sediment load']
    ):
        var = ds.createVariable(name, 'b', ('time',), fill_value=-127)
        var.long_name = f'quality flag for {desc}'
        var.flag_values = np.array([0, 1, 2, 3], dtype=np.byte)
        var.flag_meanings = flag_meanings
        var.comment = "0=good_data, 1=suspect_data, 2=bad_data, 3=missing_data"
        var[:] = values

    # 原始Data.Quality字符串变量
    q_quality_var = ds.createVariable('Q_quality', str, ('time',))
    q_quality_var.long_name = 'data quality label for discharge'
    q_quality_var.comment = 'Original Data.Quality from Flux.csv'
    q_quality_var[:] = np.array(q_quality, dtype='object')

    ssc_quality_var = ds.createVariable('SSC_quality', str, ('time',))
    ssc_quality_var.long_name = 'data quality label for SSC'
    ssc_quality_var.comment = 'Original Data.Quality from Water.csv'
    ssc_quality_var[:] = np.array(ssc_quality, dtype='object')

    # 元数据
    # ds.latitude = parse_float(station_row.get('Latitude', -9999.0))
    # ds.longitude = parse_float(station_row.get('Longitude', -9999.0))
    ds.altitude = parse_float(station_row.get('Elevation', -9999.0))
    ds.upstream_area = parse_float(station_row.get('Upstream Basin Area', -9999.0))

    ds.Conventions = 'CF-1.8'
    ds.title = f'GFQA Daily Observed Sediment and Discharge Data for Station {station_id}'
    ds.comment = (
        'Includes both automatic QC flags and original Data.Quality labels. '
        'Flags: 0=good_data, 1=suspect_data, 2=bad_data, 3=missing_data. '
        'Data.Quality is a text label from the GEMS/Water CSV source.'
    )
    ds.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} by gfqa_to_netcdf_daily_dualqc.py'
    ds.close()
    print(f"✅ Created file: {filename}")


# ==========================================================
# 主流程
# ==========================================================

def process_all_stations(flux_df, water_df, station_df, output_dir):
    all_records = []

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    flux_stations = set(flux_df['GEMS.Station.Number'].unique())
    water_stations = set(water_df['GEMS.Station.Number'].unique())
    common_stations = flux_stations & water_stations

    # print("Flux station sample:", list(flux_stations)[:5])
    # print("Water station sample:", list(water_stations)[:5])
    # print("Intersection size:", len(common_stations))


    for station_id in sorted(common_stations):
        print(f"\nProcessing station {station_id}")
        station_row = station_df[station_df['GEMS Station Number'] == station_id].iloc[0]


        discharge_data, sediment_data = extract_station_data(station_id, flux_df, water_df)
        start, end = find_overlapping_period(discharge_data, sediment_data)
        if start is None:
            print("  ⚠️ Skipped: no overlapping period")
            continue

        discharge_daily = aggregate_to_daily(discharge_data)
        sediment_daily = aggregate_to_daily(sediment_data)
        merged = pd.merge(discharge_daily, sediment_daily, on='Date', how='inner', suffixes=('_Q', '_SSC'))
        if merged.empty:
            print("  ⚠️ Skipped: no same-day data")
            continue

        merged['SSL'] = merged['Clean_Value_Q'] * merged['Clean_Value_SSC'] * 0.0864
        flag_meanings = "good_data suspect_data bad_data missing_data"
        q_thresholds = {'negative': 0, 'extreme': 50000}
        ssc_thresholds = {'negative': 0, 'extreme': 3000}
        ssl_thresholds = {'negative': 0}

        merged['Q_flag'] = merged['Clean_Value_Q'].apply(lambda x: get_flag(x, q_thresholds, flag_meanings))
        merged['SSC_flag'] = merged['Clean_Value_SSC'].apply(lambda x: get_flag(x, ssc_thresholds, flag_meanings))
        merged['SSL_flag'] = merged['SSL'].apply(lambda x: get_flag(x, ssl_thresholds, flag_meanings))

        # === 收集所有站点的合并数据 ===
        export_df = merged.copy()
        export_df['Station_ID'] = station_id     # 加入站点号
        all_records.append(export_df)

        create_netcdf_file(
            station_id, station_row,
            pd.to_datetime(merged['Date']).dt.to_pydatetime(),
            merged['Clean_Value_Q'].to_numpy(),
            merged['Clean_Value_SSC'].to_numpy(),
            merged['SSL'].to_numpy(),
            merged['Q_flag'].to_numpy().astype(np.byte),
            merged['SSC_flag'].to_numpy().astype(np.byte),
            merged['SSL_flag'].to_numpy().astype(np.byte),
            merged['Quality_Q'].fillna('unknown').to_numpy(),
            merged['Quality_SSC'].fillna('unknown').to_numpy(),
            output_dir
        )
    # === 所有站点合并输出 Excel ===
    if all_records:
        big_df = pd.concat(all_records, ignore_index=True)
        out_dir = Path("output_check")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "GFQA_all_stations.xlsx"
        big_df.to_excel(out_path, index=False)
        print(f"\n📘 Saved merged Excel for all stations: {out_path}")



def main():
    print("=" * 60)
    print("GFQA Observed Daily Data → NetCDF Conversion with Dual QC")
    print("=" * 60)

    flux_df, water_df, station_df = read_csv_files()
    process_all_stations(flux_df, water_df, station_df, output_dir='daily_dual_qc')

    print("\nConversion complete with Data.Quality and QC Flags!")


if __name__ == '__main__':
    main()

