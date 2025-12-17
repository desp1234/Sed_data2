#!/usr/bin/env python3
"""
提取所有包含泥沙数据的站点
"""

import netCDF4 as nc
import numpy as np
from pathlib import Path
import shutil

def extract_unique_stations(ds, group_name):
    """从指定的group中提取唯一的站点编号"""
    if group_name not in ds.groups:
        return set()

    group = ds.groups[group_name]
    if 'STATION_NUMBER' not in group.variables:
        return set()

    station_var = group.variables['STATION_NUMBER']
    raw_data = station_var[:]

    # 转换为字符串
    stations = set()
    try:
        # 尝试使用nc.chartostring
        station_ids = nc.chartostring(raw_data)
        for sid in station_ids:
            station_id = sid.strip()
            if station_id:
                stations.add(station_id)
    except:
        # 如果失败，使用tobytes方法
        for i in range(raw_data.shape[0]):
            try:
                station_id = raw_data[i].tobytes().decode('utf-8').strip().replace('\x00', '')
                if station_id:
                    stations.add(station_id)
            except:
                pass

    return stations

def main():
    nc_file = 'hydat.nc'

    print("正在读取NetCDF文件...")
    ds = nc.Dataset(nc_file, 'r')

    # 泥沙相关的表
    sed_groups = ['SED_DLY_LOADS', 'SED_DLY_SUSCON', 'SED_SAMPLES', 'SED_SAMPLES_PSD']

    all_sed_stations = set()

    for group_name in sed_groups:
        print(f"\n处理 {group_name}...")
        stations = extract_unique_stations(ds, group_name)
        print(f"  找到 {len(stations)} 个站点")
        all_sed_stations.update(stations)

    ds.close()

    print(f"\n总计: {len(all_sed_stations)} 个唯一的泥沙站点")

    # 保存站点列表
    output_file = 'sediment_stations.txt'
    with open(output_file, 'w') as f:
        for station in sorted(all_sed_stations):
            f.write(f"{station}\n")

    print(f"\n站点列表已保存到: {output_file}")

    # 创建sediment文件夹
    sediment_dir = Path('sediment')
    if sediment_dir.exists():
        print(f"\nsediment文件夹已存在")
    else:
        sediment_dir.mkdir()
        print(f"\n已创建sediment文件夹")

    # 复制站点文件
    station_dir = Path('station_files')
    if not station_dir.exists():
        print("错误: station_files目录不存在")
        return

    print(f"\n开始复制泥沙站点文件...")
    copied_count = 0
    not_found_count = 0

    for station_id in sorted(all_sed_stations):
        # 构建文件名
        nc_filename = f"HYDAT_{station_id}.nc"
        source_file = station_dir / nc_filename

        if source_file.exists():
            dest_file = sediment_dir / nc_filename
            shutil.copy2(source_file, dest_file)
            copied_count += 1
            if copied_count % 10 == 0:
                print(f"  已复制 {copied_count} 个文件...")
        else:
            not_found_count += 1
            if not_found_count <= 10:  # 只显示前10个未找到的
                print(f"  警告: 未找到文件 {nc_filename}")

    print(f"\n完成!")
    print(f"  成功复制: {copied_count} 个文件")
    print(f"  未找到: {not_found_count} 个文件")

    # 显示一些示例站点
    print(f"\n示例泥沙站点 (前10个):")
    for i, station in enumerate(sorted(all_sed_stations)[:10], 1):
        print(f"  {i}. {station}")

if __name__ == "__main__":
    main()
