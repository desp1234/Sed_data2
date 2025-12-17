#!/usr/bin/env python3
"""
展示如何使用新添加的变量（SSC 和 drainage_area）

演示：
1. 直接从变量读取泥沙浓度和流域面积
2. 比较不同河流的特征
3. 绘制流域面积 vs 泥沙通量的关系

Author: Claude Code
Date: 2025-10-19
"""

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


def read_milliman_file(filepath):
    """
    读取 Milliman NetCDF 文件（使用新添加的变量）

    Returns:
        dict: 包含所有相关数据的字典
    """
    ds = nc.Dataset(filepath, 'r')

    data = {
        'river_name': ds.river_name,
        'location_id': ds.location_id,
        'latitude': ds.latitude,
        'longitude': ds.longitude,
        'country': ds.country,
        'continent': ds.continent_region,
    }

    # 读取变量（现在可以直接从变量读取，不需要从全局属性）
    data['tss_Mt_yr'] = float(ds.variables['TSS'][0,0,0])  # 泥沙通量

    # SSC - 新添加的变量！
    if 'SSC' in ds.variables:
        data['ssc_mg_L'] = float(ds.variables['SSC'][0,0,0])
    else:
        data['ssc_mg_L'] = None

    # drainage_area - 新添加的变量！
    if 'drainage_area' in ds.variables:
        data['drainage_area_km2'] = float(ds.variables['drainage_area'][:])
    else:
        data['drainage_area_km2'] = None

    # Discharge
    if 'Discharge' in ds.variables:
        discharge = ds.variables['Discharge'][0,0,0]
        if discharge != -9999.0:  # 检查填充值
            data['discharge'] = float(discharge)
            data['discharge_units'] = ds.variables['Discharge'].units
        else:
            data['discharge'] = None
            data['discharge_units'] = None
    else:
        data['discharge'] = None
        data['discharge_units'] = None

    ds.close()
    return data


def compare_rivers_example():
    """
    示例：比较几条主要河流的特征
    """
    base_dir = "/Users/zhongwangwei/Downloads/7808492/netcdf_output"

    # 选择几条著名河流
    rivers = [
        'Amazon',
        'Changjiang',
        'Mississippi',
        'Ganges',
        'Congo',
        'Nile',
        'Mekong',
        'Danube',
        'Agri'  # 小河流对比
    ]

    print("="*70)
    print("主要河流特征对比")
    print("="*70)
    print(f"\n{'河流':<15} {'流域面积':<12} {'泥沙通量':<12} {'泥沙浓度':<12} {'径流':<10}")
    print(f"{'':15} {'(km²)':<12} {'(Mt/yr)':<12} {'(mg/L)':<12} {'(m³/s)':<10}")
    print("-"*70)

    river_data = []

    for river in rivers:
        # 查找文件
        pattern = os.path.join(base_dir, f'Milliman_{river}_*.nc')
        files = glob.glob(pattern)

        if files:
            data = read_milliman_file(files[0])
            river_data.append(data)

            # 格式化输出
            area = f"{data['drainage_area_km2']:,.0f}" if data['drainage_area_km2'] else 'N/A'
            tss = f"{data['tss_Mt_yr']:,.1f}" if data['tss_Mt_yr'] else 'N/A'
            ssc = f"{data['ssc_mg_L']:,.0f}" if data['ssc_mg_L'] else 'N/A'
            discharge = f"{data['discharge']:,.0f}" if data['discharge'] else 'N/A'

            print(f"{river:<15} {area:<12} {tss:<12} {ssc:<12} {discharge:<10}")

    print("-"*70)

    return river_data


def plot_drainage_vs_sediment():
    """
    绘制流域面积 vs 泥沙通量的关系
    """
    base_dir = "/Users/zhongwangwei/Downloads/7808492/netcdf_output"

    # 读取所有 Milliman 文件
    files = glob.glob(os.path.join(base_dir, "Milliman_*.nc"))

    print(f"\n读取 {len(files)} 个河流的数据...")

    drainage_areas = []
    sediment_fluxes = []
    ssc_values = []
    river_names = []

    for filepath in files:
        data = read_milliman_file(filepath)

        # 只包含有完整数据的河流
        if (data['drainage_area_km2'] and data['tss_Mt_yr'] and
            data['drainage_area_km2'] > 0 and data['tss_Mt_yr'] > 0):

            drainage_areas.append(data['drainage_area_km2'])
            sediment_fluxes.append(data['tss_Mt_yr'])
            if data['ssc_mg_L']:
                ssc_values.append(data['ssc_mg_L'])
            else:
                ssc_values.append(np.nan)
            river_names.append(data['river_name'])

    drainage_areas = np.array(drainage_areas)
    sediment_fluxes = np.array(sediment_fluxes)
    ssc_values = np.array(ssc_values)

    print(f"有效数据: {len(drainage_areas)} 条河流")

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 图1: 流域面积 vs 泥沙通量
    scatter = ax1.scatter(drainage_areas, sediment_fluxes,
                         c=ssc_values, cmap='YlOrRd', alpha=0.6,
                         s=50, edgecolors='black', linewidth=0.5)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Drainage Area (km²)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Sediment Flux (Mt/yr)', fontsize=12, fontweight='bold')
    ax1.set_title('Drainage Area vs Sediment Flux\n(colored by SSC)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')

    # 添加颜色条
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label('SSC (mg/L)', fontsize=10)

    # 标注几条主要河流
    major_rivers = ['Amazon', 'Changjiang', 'Mississippi', 'Ganges', 'Congo']
    for i, name in enumerate(river_names):
        if name in major_rivers:
            ax1.annotate(name, (drainage_areas[i], sediment_fluxes[i]),
                        fontsize=8, alpha=0.7,
                        xytext=(5, 5), textcoords='offset points')

    # 图2: 流域面积 vs 泥沙浓度
    valid_ssc = ~np.isnan(ssc_values)
    scatter2 = ax2.scatter(drainage_areas[valid_ssc], ssc_values[valid_ssc],
                          c=sediment_fluxes[valid_ssc], cmap='viridis', alpha=0.6,
                          s=50, edgecolors='black', linewidth=0.5)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Drainage Area (km²)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('SSC (mg/L)', fontsize=12, fontweight='bold')
    ax2.set_title('Drainage Area vs Sediment Concentration\n(colored by flux)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')

    # 添加颜色条
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Sediment Flux (Mt/yr)', fontsize=10)

    plt.tight_layout()

    # 保存图表
    output_dir = os.path.join(base_dir, 'figures')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, 'drainage_area_vs_sediment.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存: {output_path}")

    plt.close()

    # 统计分析
    print("\n" + "="*70)
    print("统计分析")
    print("="*70)
    print(f"流域面积范围: {drainage_areas.min():.2f} - {drainage_areas.max():,.0f} km²")
    print(f"泥沙通量范围: {sediment_fluxes.min():.4f} - {sediment_fluxes.max():,.0f} Mt/yr")
    print(f"泥沙浓度范围: {np.nanmin(ssc_values):.2f} - {np.nanmax(ssc_values):,.0f} mg/L")
    print(f"平均泥沙浓度: {np.nanmean(ssc_values):.2f} mg/L")
    print(f"中位数泥沙浓度: {np.nanmedian(ssc_values):.2f} mg/L")


def demonstrate_easy_access():
    """
    演示：使用新变量的便捷性
    """
    base_dir = "/Users/zhongwangwei/Downloads/7808492/netcdf_output"
    filepath = os.path.join(base_dir, "Milliman_Agri_MILLIMAN-705.0.nc")

    print("\n" + "="*70)
    print("演示：直接访问变量（Agri 河示例）")
    print("="*70)

    ds = nc.Dataset(filepath, 'r')

    print("\n✓ 之前的方法（从全局属性读取）:")
    print("```python")
    print("ds = nc.Dataset('Milliman_Agri_MILLIMAN-705.0.nc', 'r')")
    print("ssc = float(ds.sediment_concentration_mg_L)  # 从全局属性")
    print("area = float(ds.drainage_area_km2)            # 从全局属性")
    print("```")
    print(f"结果: SSC = {ds.sediment_concentration_mg_L} mg/L, Area = {ds.drainage_area_km2} km²")

    print("\n✓ 现在的方法（直接从变量读取）:")
    print("```python")
    print("ds = nc.Dataset('Milliman_Agri_MILLIMAN-705.0.nc', 'r')")
    print("ssc = ds.variables['SSC'][0,0,0]         # 直接从变量")
    print("area = ds.variables['drainage_area'][:]  # 直接从变量")
    print("```")
    print(f"结果: SSC = {ds.variables['SSC'][0,0,0]} mg/L, Area = {ds.variables['drainage_area'][:]} km²")

    print("\n✓ 优势:")
    print("  1. 可以使用 xarray 自动处理")
    print("  2. 可以使用 CDO/NCO 命令行工具提取")
    print("  3. 符合 CF 标准，更易于工具识别")
    print("  4. 可以进行数组运算和批量处理")

    # xarray 示例
    print("\n✓ 使用 xarray 的示例:")
    print("```python")
    print("import xarray as xr")
    print("ds = xr.open_dataset('Milliman_Agri_MILLIMAN-705.0.nc')")
    print("print(ds['SSC'].values)")
    print("print(ds['drainage_area'].values)")
    print("```")

    ds.close()


def main():
    """主函数"""
    print("="*70)
    print("NetCDF 新变量使用示例")
    print("="*70)

    # 1. 比较河流
    compare_rivers_example()

    # 2. 演示便捷访问
    demonstrate_easy_access()

    # 3. 绘制关系图
    plot_drainage_vs_sediment()

    print("\n" + "="*70)
    print("示例完成！")
    print("="*70)


if __name__ == "__main__":
    main()
