#!/usr/bin/env python3
"""
快速查看泥沙站点数据的示例脚本
"""

import netCDF4 as nc
import pandas as pd
import sys
from pathlib import Path

def view_sediment_station(nc_file):
    """查看泥沙站点数据"""

    print(f"\n{'='*80}")
    print(f"泥沙数据文件: {nc_file}")
    print(f"{'='*80}\n")

    ds = nc.Dataset(nc_file, 'r')

    # 显示全局属性
    print("站点信息:")
    print(f"  站点编号: {ds.station_id}")
    if hasattr(ds, 'station_name'):
        print(f"  站点名称: {ds.station_name}")
    if hasattr(ds, 'province_territory'):
        print(f"  省/地区: {ds.province_territory}")

    # 经纬度
    if 'lat' in ds.variables and 'lon' in ds.variables:
        lat = ds.variables['lat'][0]
        lon = ds.variables['lon'][0]
        print(f"  坐标: ({lat:.4f}°N, {lon:.4f}°E)")

    print(f"\n可用的泥沙数据:")

    reference_date = pd.Timestamp('1850-01-01')

    # 每日泥沙负荷
    if 'sediment_load' in ds.variables:
        sed_load = ds.variables['sediment_load']
        time_load = ds.variables['time_sed_load']

        # 转换时间
        dates = reference_date + pd.to_timedelta(time_load[:], unit='D')

        print(f"\n  1. 每日泥沙负荷:")
        print(f"     数据点数: {len(sed_load)}")
        print(f"     时间范围: {dates.min().strftime('%Y-%m-%d')} 至 {dates.max().strftime('%Y-%m-%d')}")
        print(f"     单位: tonnes (吨)")

        # 统计信息
        if len(sed_load.shape) == 3:
            values = sed_load[:, 0, 0]
        else:
            values = sed_load[:]

        valid_values = values[values != -999.0]
        if len(valid_values) > 0:
            print(f"     最小值: {valid_values.min():.2f}")
            print(f"     最大值: {valid_values.max():.2f}")
            print(f"     平均值: {valid_values.mean():.2f}")

    # 每日悬浮泥沙浓度
    if 'suspended_sediment_concentration' in ds.variables:
        suscon = ds.variables['suspended_sediment_concentration']
        time_suscon = ds.variables['time_sed_suscon']

        dates = reference_date + pd.to_timedelta(time_suscon[:], unit='D')

        print(f"\n  2. 每日悬浮泥沙浓度:")
        print(f"     数据点数: {len(suscon)}")
        print(f"     时间范围: {dates.min().strftime('%Y-%m-%d')} 至 {dates.max().strftime('%Y-%m-%d')}")
        print(f"     单位: mg/L")

        if len(suscon.shape) == 3:
            values = suscon[:, 0, 0]
        else:
            values = suscon[:]

        valid_values = values[values != -999.0]
        if len(valid_values) > 0:
            print(f"     最小值: {valid_values.min():.2f}")
            print(f"     最大值: {valid_values.max():.2f}")
            print(f"     平均值: {valid_values.mean():.2f}")

    # 泥沙样本数据
    if 'time_sed_sample' in ds.dimensions:
        n_samples = len(ds.dimensions['time_sed_sample'])
        time_sample = ds.variables['time_sed_sample']

        dates = reference_date + pd.to_timedelta(time_sample[:], unit='D')

        print(f"\n  3. 泥沙样本数据:")
        print(f"     样本数: {n_samples}")
        print(f"     时间范围: {dates.min().strftime('%Y-%m-%d')} 至 {dates.max().strftime('%Y-%m-%d')}")

        # 列出可用的样本变量
        sample_vars = []
        for var_name in ['flow', 'temperature', 'concentration', 'dissolved_solids']:
            if var_name in ds.variables:
                sample_vars.append(var_name)

        if sample_vars:
            print(f"     可用字段: {', '.join(sample_vars)}")

    ds.close()

    print(f"\n{'='*80}\n")

def list_sediment_stations():
    """列出所有泥沙站点"""
    sediment_dir = Path('sediment')

    if not sediment_dir.exists():
        print("错误: sediment文件夹不存在")
        return

    nc_files = sorted(sediment_dir.glob('HYDAT_*_SEDIMENT.nc'))

    print(f"\n找到 {len(nc_files)} 个泥沙站点文件")
    print(f"\n前20个站点:")

    for i, file in enumerate(nc_files[:20], 1):
        station_id = file.stem.replace('HYDAT_', '').replace('_SEDIMENT', '')
        print(f"  {i:3d}. {station_id}")

    if len(nc_files) > 20:
        print(f"  ... (还有 {len(nc_files) - 20} 个)")

    print()

def main():
    if len(sys.argv) < 2:
        print("用法:")
        print("  python view_sediment_data.py list           # 列出所有泥沙站点")
        print("  python view_sediment_data.py <station_id>   # 查看指定站点的泥沙数据")
        print()
        print("示例:")
        print("  python view_sediment_data.py list")
        print("  python view_sediment_data.py 01AF006")
        sys.exit(1)

    if sys.argv[1] == 'list':
        list_sediment_stations()
    else:
        station_id = sys.argv[1]
        nc_file = f"sediment/HYDAT_{station_id}_SEDIMENT.nc"

        if not Path(nc_file).exists():
            print(f"错误: 文件不存在: {nc_file}")
            print(f"请使用 'python view_sediment_data.py list' 查看可用的站点")
            sys.exit(1)

        view_sediment_station(nc_file)

if __name__ == "__main__":
    main()
