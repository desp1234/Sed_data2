#!/usr/bin/env python3
"""
读取和可视化HYDAT站点CF-NetCDF文件的示例脚本
"""

import netCDF4 as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import sys

def print_station_info(ncfile):
    """打印站点基本信息"""
    print("\n" + "="*80)
    print("站点信息")
    print("="*80)
    
    # 全局属性
    attrs = {
        'station_id': '站点编号',
        'station_name': '站点名称',
        'province_territory': '省份/地区',
        'drainage_area_gross': '总排水面积',
        'drainage_area_effective': '有效排水面积',
    }
    
    for attr, label in attrs.items():
        if hasattr(ncfile, attr):
            value = getattr(ncfile, attr)
            print(f"{label:20s}: {value}")
    
    # 坐标
    if 'lat' in ncfile.variables and 'lon' in ncfile.variables:
        lat = ncfile.variables['lat'][0]
        lon = ncfile.variables['lon'][0]
        print(f"{'坐标':20s}: {lat:.4f}°N, {lon:.4f}°E")
    
    # 时间范围
    if hasattr(ncfile, 'time_coverage_start'):
        print(f"{'数据起始':20s}: {ncfile.time_coverage_start}")
    if hasattr(ncfile, 'time_coverage_end'):
        print(f"{'数据结束':20s}: {ncfile.time_coverage_end}")

def print_available_variables(ncfile):
    """打印可用变量"""
    print("\n" + "="*80)
    print("可用变量")
    print("="*80)
    
    time_series_vars = []
    other_vars = []
    
    for var_name in ncfile.variables:
        var = ncfile.variables[var_name]
        
        # 跳过坐标变量
        if var_name in ['lat', 'lon', 'time_flow', 'time_level', 
                        'time_sediment_conc', 'time_sediment_load', 'year']:
            continue
        
        var_info = {
            'name': var_name,
            'dims': var.dimensions,
            'shape': var.shape,
            'long_name': var.long_name if hasattr(var, 'long_name') else '',
            'units': var.units if hasattr(var, 'units') else ''
        }
        
        # 区分时间序列和其他变量
        if len(var.dimensions) == 3 and 'time' in str(var.dimensions):
            time_series_vars.append(var_info)
        else:
            other_vars.append(var_info)
    
    if time_series_vars:
        print("\n时间序列变量:")
        for v in time_series_vars:
            print(f"  {v['name']:40s} {str(v['shape']):20s} {v['units']:15s} - {v['long_name']}")
    
    if other_vars:
        print("\n其他变量:")
        for v in other_vars:
            print(f"  {v['name']:40s} {str(v['shape']):20s} {v['units']:15s} - {v['long_name']}")

def read_timeseries_data(ncfile, var_name, time_var_name):
    """读取时间序列数据"""
    if var_name not in ncfile.variables or time_var_name not in ncfile.variables:
        return None
    
    # 读取时间
    time_var = ncfile.variables[time_var_name]
    time_values = time_var[:]
    time_units = time_var.units
    time_calendar = time_var.calendar if hasattr(time_var, 'calendar') else 'gregorian'
    
    dates = nc.num2date(time_values, units=time_units, calendar=time_calendar)
    
    # 读取数据 (time, lat, lon) -> (time,)
    data = ncfile.variables[var_name][:, 0, 0]
    
    # 移除缺失值
    valid_mask = (data != -999) & ~np.isnan(data)
    
    df = pd.DataFrame({
        'date': dates[valid_mask],
        'value': data[valid_mask]
    })
    
    return df

def plot_discharge(ncfile, output_file=None):
    """绘制流量时间序列"""
    if 'discharge' not in ncfile.variables:
        print("该站点没有流量数据")
        return
    
    df = read_timeseries_data(ncfile, 'discharge', 'time_flow')
    
    if df is None or len(df) == 0:
        print("无有效流量数据")
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df['date'], df['value'], linewidth=0.5, color='steelblue')
    
    # 设置标题
    title = "河流流量"
    if hasattr(ncfile, 'station_name'):
        title = f"{ncfile.station_name} - {title}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('流量 (m³/s)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 格式化x轴
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    
    # 统计信息
    stats_text = f"记录数: {len(df):,}\n"
    stats_text += f"均值: {df['value'].mean():.2f} m³/s\n"
    stats_text += f"最小: {df['value'].min():.2f} m³/s\n"
    stats_text += f"最大: {df['value'].max():.2f} m³/s"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {output_file}")
    else:
        plt.show()
    
    plt.close()

def plot_annual_statistics(ncfile, output_file=None):
    """绘制年度统计"""
    if 'year' not in ncfile.dimensions:
        print("该站点没有年度统计数据")
        return
    
    years = ncfile.variables['year'][:]
    
    data = {'year': years}
    labels = {}
    
    if 'annual_mean_discharge' in ncfile.variables:
        data['mean'] = ncfile.variables['annual_mean_discharge'][:]
        labels['mean'] = '年均流量'
    if 'annual_min_discharge' in ncfile.variables:
        data['min'] = ncfile.variables['annual_min_discharge'][:]
        labels['min'] = '年最小流量'
    if 'annual_max_discharge' in ncfile.variables:
        data['max'] = ncfile.variables['annual_max_discharge'][:]
        labels['max'] = '年最大流量'
    
    if len(data) <= 1:
        print("无有效年度统计数据")
        return
    
    df = pd.DataFrame(data)
    df = df[(df['mean'] != -999) if 'mean' in df.columns else df.index]
    
    if len(df) == 0:
        print("无有效年度统计数据")
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 绘制年均值
    if 'mean' in df.columns:
        ax.plot(df['year'], df['mean'], label=labels['mean'], 
                color='steelblue', linewidth=2)
    
    # 绘制范围
    if 'min' in df.columns and 'max' in df.columns:
        ax.fill_between(df['year'], df['min'], df['max'], 
                        alpha=0.3, color='lightblue', label='最小-最大范围')
    
    # 设置标题
    title = "年度流量统计"
    if hasattr(ncfile, 'station_name'):
        title = f"{ncfile.station_name} - {title}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.set_xlabel('年份', fontsize=12)
    ax.set_ylabel('流量 (m³/s)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {output_file}")
    else:
        plt.show()
    
    plt.close()

def export_to_csv(ncfile, output_dir):
    """导出数据为CSV"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    station_id = ncfile.station_id if hasattr(ncfile, 'station_id') else 'unknown'
    
    # 导出流量数据
    if 'discharge' in ncfile.variables:
        df = read_timeseries_data(ncfile, 'discharge', 'time_flow')
        if df is not None and len(df) > 0:
            output_file = output_dir / f"{station_id}_discharge.csv"
            df.to_csv(output_file, index=False)
            print(f"流量数据已导出: {output_file}")
    
    # 导出水位数据
    if 'water_level' in ncfile.variables:
        df = read_timeseries_data(ncfile, 'water_level', 'time_level')
        if df is not None and len(df) > 0:
            output_file = output_dir / f"{station_id}_water_level.csv"
            df.to_csv(output_file, index=False)
            print(f"水位数据已导出: {output_file}")
    
    # 导出年度统计
    if 'year' in ncfile.dimensions:
        years = ncfile.variables['year'][:]
        data = {'year': years}
        
        if 'annual_mean_discharge' in ncfile.variables:
            data['mean'] = ncfile.variables['annual_mean_discharge'][:]
        if 'annual_min_discharge' in ncfile.variables:
            data['min'] = ncfile.variables['annual_min_discharge'][:]
        if 'annual_max_discharge' in ncfile.variables:
            data['max'] = ncfile.variables['annual_max_discharge'][:]
        
        df = pd.DataFrame(data)
        df = df[df['mean'] != -999] if 'mean' in df.columns else df
        
        if len(df) > 0:
            output_file = output_dir / f"{station_id}_annual_stats.csv"
            df.to_csv(output_file, index=False)
            print(f"年度统计已导出: {output_file}")

def main():
    if len(sys.argv) < 2:
        print("用法: python read_station_file.py <station_netcdf_file> [options]")
        print("\n选项:")
        print("  --info              显示站点信息")
        print("  --plot-discharge    绘制流量图")
        print("  --plot-annual       绘制年度统计图")
        print("  --export-csv DIR    导出CSV到指定目录")
        print("\n示例:")
        print("  python read_station_file.py HYDAT_05BB001.nc --info")
        print("  python read_station_file.py HYDAT_05BB001.nc --plot-discharge")
        print("  python read_station_file.py HYDAT_05BB001.nc --export-csv ./csv_output/")
        sys.exit(1)
    
    station_file = sys.argv[1]
    
    if not Path(station_file).exists():
        print(f"错误: 文件不存在: {station_file}")
        sys.exit(1)
    
    # 打开文件
    ds = nc.Dataset(station_file, 'r')
    
    # 默认显示信息
    if len(sys.argv) == 2 or '--info' in sys.argv:
        print_station_info(ds)
        print_available_variables(ds)
    
    # 绘制流量图
    if '--plot-discharge' in sys.argv:
        station_id = ds.station_id if hasattr(ds, 'station_id') else 'station'
        output_file = f"{station_id}_discharge.png"
        plot_discharge(ds, output_file)
    
    # 绘制年度统计
    if '--plot-annual' in sys.argv:
        station_id = ds.station_id if hasattr(ds, 'station_id') else 'station'
        output_file = f"{station_id}_annual.png"
        plot_annual_statistics(ds, output_file)
    
    # 导出CSV
    if '--export-csv' in sys.argv:
        idx = sys.argv.index('--export-csv')
        if idx + 1 < len(sys.argv):
            output_dir = sys.argv[idx + 1]
            export_to_csv(ds, output_dir)
        else:
            print("错误: --export-csv 需要指定输出目录")
    
    ds.close()

if __name__ == "__main__":
    main()
