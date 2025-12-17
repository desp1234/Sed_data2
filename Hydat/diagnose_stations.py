#!/usr/bin/env python3
"""
诊断脚本 - 检查STATIONS表的数据格式
"""

import netCDF4 as nc
import numpy as np
import sys

def diagnose_stations(nc_file):
    """诊断STATIONS表"""
    print("正在打开文件...")
    ds = nc.Dataset(nc_file, 'r')
    
    if 'STATIONS' not in ds.groups:
        print("错误: 没有找到STATIONS表")
        return
    
    stations = ds.groups['STATIONS']
    
    print(f"\n{'='*80}")
    print("STATIONS表结构")
    print(f"{'='*80}")
    
    # 维度信息
    print("\n维度:")
    for dim_name, dim in stations.dimensions.items():
        print(f"  {dim_name}: {len(dim)}")
    
    # 变量信息
    print("\n变量:")
    for var_name in stations.variables:
        var = stations.variables[var_name]
        print(f"  {var_name}: {var.dtype} {var.shape}")
    
    # 检查STATION_NUMBER
    print(f"\n{'='*80}")
    print("STATION_NUMBER 数据检查")
    print(f"{'='*80}")
    
    station_num_var = stations.variables['STATION_NUMBER']
    print(f"类型: {station_num_var.dtype}")
    print(f"形状: {station_num_var.shape}")
    
    # 读取原始数据
    raw_data = station_num_var[:]
    print(f"\n原始数据类型: {type(raw_data)}")
    print(f"原始数据形状: {raw_data.shape}")
    print(f"原始数据dtype: {raw_data.dtype}")
    
    # 显示前几个站点的原始数据
    print(f"\n前5个站点的原始数据:")
    for i in range(min(5, raw_data.shape[0])):
        print(f"  [{i}] {raw_data[i]} -> 类型: {type(raw_data[i])}")
    
    # 尝试不同的转换方法
    print(f"\n{'='*80}")
    print("尝试不同的转换方法")
    print(f"{'='*80}")
    
    # 方法1: nc.chartostring
    print("\n方法1: nc.chartostring")
    try:
        method1 = nc.chartostring(raw_data)
        print(f"  结果类型: {type(method1)}")
        print(f"  结果形状: {method1.shape}")
        print(f"  结果dtype: {method1.dtype}")
        print(f"  前5个: {method1[:5]}")
    except Exception as e:
        print(f"  错误: {e}")
    
    # 方法2: 直接转换
    print("\n方法2: 手动拼接字符")
    try:
        method2 = []
        for i in range(min(10, raw_data.shape[0])):
            chars = raw_data[i]
            # 尝试解码
            if isinstance(chars[0], bytes):
                station_id = b''.join(chars).decode('utf-8').strip()
            else:
                station_id = ''.join([c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in chars]).strip()
            method2.append(station_id)
            if i < 5:
                print(f"  [{i}] {station_id}")
        print(f"  成功转换 {len(method2)} 个站点")
    except Exception as e:
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()
    
    # 方法3: tobytes
    print("\n方法3: tobytes")
    try:
        method3 = []
        for i in range(min(10, raw_data.shape[0])):
            station_id = raw_data[i].tobytes().decode('utf-8').strip()
            method3.append(station_id)
            if i < 5:
                print(f"  [{i}] {station_id}")
        print(f"  成功转换 {len(method3)} 个站点")
    except Exception as e:
        print(f"  错误: {e}")
        import traceback
        traceback.print_exc()
    
    # 检查其他重要字段
    print(f"\n{'='*80}")
    print("其他字段示例 (前3个站点)")
    print(f"{'='*80}")
    
    for i in range(min(3, raw_data.shape[0])):
        print(f"\n站点 {i}:")
        
        if 'STATION_NAME' in stations.variables:
            name_raw = stations.variables['STATION_NAME'][i]
            try:
                name = nc.chartostring(name_raw).strip()
                print(f"  名称: {name}")
            except:
                try:
                    name = name_raw.tobytes().decode('utf-8').strip()
                    print(f"  名称: {name}")
                except:
                    print(f"  名称: (无法解码)")
        
        if 'LATITUDE' in stations.variables:
            lat = stations.variables['LATITUDE'][i]
            print(f"  纬度: {lat}")
        
        if 'LONGITUDE' in stations.variables:
            lon = stations.variables['LONGITUDE'][i]
            print(f"  经度: {lon}")
    
    ds.close()
    
    print(f"\n{'='*80}")
    print("诊断完成")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python diagnose_stations.py <hydat.nc>")
        sys.exit(1)
    
    diagnose_stations(sys.argv[1])
