#!/usr/bin/env python3
"""
深度诊断 - 检查字符数据的mask状态和原始值
"""

import netCDF4 as nc
import numpy as np
import sys

def deep_diagnose(nc_file):
    """深度诊断字符数据"""
    print("打开文件...")
    ds = nc.Dataset(nc_file, 'r')
    
    if 'STATIONS' not in ds.groups:
        print("错误: 没有STATIONS表")
        return
    
    stations = ds.groups['STATIONS']
    
    print(f"\n{'='*80}")
    print("检查 STATION_NUMBER 变量")
    print(f"{'='*80}")
    
    station_var = stations.variables['STATION_NUMBER']
    
    # 检查变量属性
    print("\n变量属性:")
    for attr in station_var.ncattrs():
        print(f"  {attr}: {station_var.getncattr(attr)}")
    
    # 读取原始数据（不自动mask）
    print("\n读取原始数据（auto_maskandscale=False）:")
    station_var.set_auto_maskandscale(False)
    raw_data = station_var[:]
    
    print(f"  类型: {type(raw_data)}")
    print(f"  形状: {raw_data.shape}")
    print(f"  dtype: {raw_data.dtype}")
    
    # 检查前几个站点的原始字节
    print(f"\n前10个站点的原始字节:")
    for i in range(min(10, raw_data.shape[0])):
        bytes_data = raw_data[i]
        print(f"  [{i}] 原始: {bytes_data}")
        print(f"      类型: {[type(b) for b in bytes_data[:3]]}")
        print(f"      值: {[b for b in bytes_data]}")
        
        # 尝试解码
        try:
            if isinstance(bytes_data[0], bytes):
                decoded = b''.join(bytes_data).decode('utf-8', errors='ignore').strip('\x00').strip()
            else:
                decoded = bytes_data.tobytes().decode('utf-8', errors='ignore').strip('\x00').strip()
            print(f"      解码: '{decoded}'")
        except Exception as e:
            print(f"      解码失败: {e}")
        print()
    
    # 检查其他字符变量
    print(f"\n{'='*80}")
    print("检查 STATION_NAME 变量")
    print(f"{'='*80}")
    
    if 'STATION_NAME' in stations.variables:
        name_var = stations.variables['STATION_NAME']
        name_var.set_auto_maskandscale(False)
        name_data = name_var[:]
        
        print(f"类型: {type(name_data)}")
        print(f"形状: {name_data.shape}")
        print(f"dtype: {name_data.dtype}")
        
        print(f"\n前5个站点名称:")
        for i in range(min(5, name_data.shape[0])):
            bytes_data = name_data[i]
            print(f"  [{i}] 原始长度: {len(bytes_data)}")
            print(f"      前10字节: {bytes_data[:10]}")
            
            # 尝试解码
            try:
                if isinstance(bytes_data[0], bytes):
                    decoded = b''.join(bytes_data).decode('utf-8', errors='ignore').strip('\x00').strip()
                else:
                    decoded = bytes_data.tobytes().decode('utf-8', errors='ignore').strip('\x00').strip()
                print(f"      解码: '{decoded}'")
            except Exception as e:
                print(f"      解码失败: {e}")
            print()
    
    # 检查数值变量（对比）
    print(f"\n{'='*80}")
    print("检查数值变量 LATITUDE (作为对比)")
    print(f"{'='*80}")
    
    if 'LATITUDE' in stations.variables:
        lat_var = stations.variables['LATITUDE']
        lat_var.set_auto_maskandscale(False)
        lat_data = lat_var[:]
        
        print(f"类型: {type(lat_data)}")
        print(f"形状: {lat_data.shape}")
        print(f"dtype: {lat_data.dtype}")
        
        print(f"\n前10个站点的纬度:")
        for i in range(min(10, lat_data.shape[0])):
            print(f"  [{i}] {lat_data[i]}")
    
    # 尝试直接从文件读取
    print(f"\n{'='*80}")
    print("尝试用h5py直接读取（如果是HDF5格式）")
    print(f"{'='*80}")
    
    try:
        import h5py
        with h5py.File(nc_file, 'r') as f:
            if 'STATIONS' in f:
                stations_h5 = f['STATIONS']
                if 'STATION_NUMBER' in stations_h5:
                    station_h5 = stations_h5['STATION_NUMBER']
                    print(f"H5PY - 类型: {station_h5.dtype}")
                    print(f"H5PY - 形状: {station_h5.shape}")
                    
                    data = station_h5[0:10]
                    print(f"\nH5PY - 前10个站点:")
                    for i, row in enumerate(data):
                        print(f"  [{i}] {row}")
                        decoded = row.tobytes().decode('utf-8', errors='ignore').strip('\x00').strip()
                        print(f"      解码: '{decoded}'")
    except ImportError:
        print("h5py未安装，跳过")
    except Exception as e:
        print(f"h5py读取失败: {e}")
    
    ds.close()
    
    print(f"\n{'='*80}")
    print("诊断完成")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python deep_diagnose.py <hydat.nc>")
        sys.exit(1)
    
    deep_diagnose(sys.argv[1])
