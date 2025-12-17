#!/usr/bin/env python3
"""
NetCDF 数据读取和可视化示例

演示如何读取和使用转换后的 NetCDF 文件
"""

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os


def read_timeseries_file(filepath):
    """
    读取时间序列 NetCDF 文件

    Returns:
        dict: 包含数据和元数据的字典
    """
    ds = nc.Dataset(filepath, 'r')

    # 提取时间并转换为年份
    time_days = ds.variables['time'][:]
    dates = [datetime(1970, 1, 1) + timedelta(days=float(t)) for t in time_days]
    years = [d.year for d in dates]

    # 提取数据 (移除空间维度)
    tss = ds.variables['TSS'][:].squeeze()
    tss = np.ma.masked_equal(tss, -9999.0)  # 处理填充值

    discharge = None
    if 'Discharge' in ds.variables:
        discharge = ds.variables['Discharge'][:].squeeze()
        discharge = np.ma.masked_equal(discharge, -9999.0)

    # 提取元数据
    metadata = {
        'river_name': ds.river_name,
        'location_id': ds.location_id,
        'latitude': ds.latitude,
        'longitude': ds.longitude,
        'country': ds.country,
        'time_period': ds.time_period,
        'references': ds.references,
        'tss_units': ds.variables['TSS'].units,
    }

    ds.close()

    return {
        'years': years,
        'tss': tss,
        'discharge': discharge,
        'metadata': metadata
    }


def plot_sediment_timeseries(data_dict, output_dir='./figures'):
    """
    绘制泥沙通量时间序列
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    river_name = data_dict['metadata']['river_name']
    years = data_dict['years']
    tss = data_dict['tss']

    plt.figure(figsize=(12, 6))
    plt.plot(years, tss, 'o-', linewidth=2, markersize=4)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel(f"Sediment Flux ({data_dict['metadata']['tss_units']})", fontsize=12)
    plt.title(f'{river_name} - Sediment Flux Time Series', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 添加趋势线
    z = np.polyfit(years, tss.compressed(), 1)
    p = np.poly1d(z)
    plt.plot(years, p(years), "r--", alpha=0.8, label=f'Trend: {z[0]:.2f} Mt/yr²')
    plt.legend()

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{river_name.replace(" ", "_")}_sediment_flux.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def compare_rivers(filepaths, output_dir='./figures'):
    """
    比较多条河流的泥沙通量
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(14, 8))

    for filepath in filepaths:
        data = read_timeseries_file(filepath)
        river_name = data['metadata']['river_name']

        # 归一化（相对于第一年）
        tss_normalized = data['tss'] / data['tss'][0] * 100

        plt.plot(data['years'], tss_normalized, 'o-',
                label=river_name, linewidth=2, markersize=3, alpha=0.7)

    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Sediment Flux (% of initial year)', fontsize=12)
    plt.title('Comparison of Sediment Flux Trends (Normalized)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=100, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'river_comparison_normalized.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def calculate_statistics(data_dict):
    """
    计算统计指标
    """
    river_name = data_dict['metadata']['river_name']
    tss = data_dict['tss']
    years = data_dict['years']

    # 基本统计
    mean_flux = np.mean(tss)
    std_flux = np.std(tss)
    min_flux = np.min(tss)
    max_flux = np.max(tss)

    # 趋势分析
    z = np.polyfit(years, tss.compressed(), 1)
    trend = z[0]  # Mt/yr per year
    total_change = (tss[-1] - tss[0]) / tss[0] * 100  # 百分比变化

    # 打印结果
    print(f"\n{'='*60}")
    print(f"Statistics for {river_name}")
    print(f"{'='*60}")
    print(f"Time Period: {data_dict['metadata']['time_period']}")
    print(f"Location: {data_dict['metadata']['latitude']:.2f}°N, {data_dict['metadata']['longitude']:.2f}°E")
    print(f"\nSediment Flux Statistics ({data_dict['metadata']['tss_units']}):")
    print(f"  Mean:    {mean_flux:.2f}")
    print(f"  Std Dev: {std_flux:.2f}")
    print(f"  Min:     {min_flux:.2f} (Year {years[np.argmin(tss)]})")
    print(f"  Max:     {max_flux:.2f} (Year {years[np.argmax(tss)]})")
    print(f"\nTrend Analysis:")
    print(f"  Linear Trend: {trend:.3f} Mt/yr² ({'decreasing' if trend < 0 else 'increasing'})")
    print(f"  Total Change: {total_change:.1f}% (from {years[0]} to {years[-1]})")
    print(f"  Average Annual Change: {total_change/(years[-1]-years[0]):.2f}% per year")
    print(f"\nReference: {data_dict['metadata']['references'][:100]}...")
    print(f"{'='*60}\n")


def main():
    """主函数"""
    print("="*70)
    print("NetCDF 数据读取和分析示例")
    print("="*70)

    # 设置文件路径
    base_dir = "/Users/zhongwangwei/Downloads/7808492/netcdf_output"

    # 时间序列文件
    files = {
        'yangtze': os.path.join(base_dir, 'TimeSeries_Yangtze_River_CHN-YANGTZE-DATONG.nc'),
        'huaihe': os.path.join(base_dir, 'TimeSeries_Huaihe_River_CHN-HUAIHE-BENGBU.nc'),
        'pearl': os.path.join(base_dir, 'TimeSeries_Pearl_River_CHN-PEARL-GAOYAO.nc'),
        'mississippi': os.path.join(base_dir, 'TimeSeries_Mississippi_River_USA-MISS-TARBERT.nc'),
    }

    # 检查文件是否存在
    available_files = {k: v for k, v in files.items() if os.path.exists(v)}

    if not available_files:
        print("错误：找不到 NetCDF 文件。请确保文件路径正确。")
        return

    print(f"\n找到 {len(available_files)} 个时间序列文件\n")

    # 1. 读取和统计分析
    print("1. 统计分析")
    print("-" * 70)
    data_collection = {}
    for key, filepath in available_files.items():
        data = read_timeseries_file(filepath)
        data_collection[key] = data
        calculate_statistics(data)

    # 2. 单个河流绘图
    print("\n2. 绘制单个河流时间序列")
    print("-" * 70)
    output_dir = os.path.join(base_dir, 'figures')
    for key, data in data_collection.items():
        plot_sediment_timeseries(data, output_dir)

    # 3. 多河流比较
    if len(available_files) > 1:
        print("\n3. 绘制多河流对比图")
        print("-" * 70)
        compare_rivers(list(available_files.values()), output_dir)

    print("\n" + "="*70)
    print("分析完成！所有图表已保存到:", output_dir)
    print("="*70)


if __name__ == "__main__":
    main()
