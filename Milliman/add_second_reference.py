#!/usr/bin/env python3
"""
批量添加第二个参考文献到 NetCDF 文件
"""

import netCDF4 as nc
import glob
import os


def add_second_reference(nc_file):
    """
    为单个 NetCDF 文件添加第二个参考文献

    Args:
        nc_file: NetCDF 文件路径
    """
    # 打开文件（追加模式）
    ds = nc.Dataset(nc_file, 'a')

    # 获取当前的 references
    current_ref = ds.references if hasattr(ds, 'references') else ""

    # 第二个参考文献
    second_ref = "Dethier, E. N., Renshaw, C. E., & Magilligan, F. J. (2022). Rapid changes to global river suspended sediment flux by humans. Science, 376(6600), 1447-1452."

    # 检查是否已经包含了第二个参考文献
    if "Dethier" in current_ref:
        print(f"  已包含 Dethier 参考文献，跳过: {os.path.basename(nc_file)}")
        ds.close()
        return False

    # 合并参考文献
    new_ref = current_ref + "; " + second_ref

    # 更新 references 属性
    ds.references = new_ref

    # 更新 history 属性
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history_entry = f"{timestamp}: Added second reference (Dethier et al. 2022)"

    if hasattr(ds, 'history'):
        ds.history = ds.history + "\n" + history_entry
    else:
        ds.history = history_entry

    ds.close()
    return True


def main():
    """
    批量处理当前目录下所有的 Milliman NetCDF 文件
    """
    print("=" * 70)
    print("批量添加第二个参考文献")
    print("=" * 70)

    # 获取所有 Milliman NetCDF 文件
    nc_files = glob.glob("Milliman_*.nc")

    if not nc_files:
        print("错误：未找到 NetCDF 文件")
        return

    print(f"\n找到 {len(nc_files)} 个 NetCDF 文件\n")

    # 统计
    updated_count = 0
    skipped_count = 0
    error_count = 0

    # 处理每个文件
    for i, nc_file in enumerate(nc_files, 1):
        try:
            print(f"[{i}/{len(nc_files)}] 处理: {os.path.basename(nc_file)}")
            if add_second_reference(nc_file):
                updated_count += 1
                print(f"  ✓ 已更新")
            else:
                skipped_count += 1
        except Exception as e:
            error_count += 1
            print(f"  ✗ 错误: {str(e)}")

    # 打印总结
    print("\n" + "=" * 70)
    print("处理完成")
    print("=" * 70)
    print(f"总文件数:   {len(nc_files)}")
    print(f"已更新:     {updated_count}")
    print(f"已跳过:     {skipped_count}")
    print(f"错误:       {error_count}")
    print("=" * 70)

    # 验证一个文件
    if nc_files and updated_count > 0:
        print("\n验证示例文件:")
        print("-" * 70)
        test_file = nc_files[0]
        ds = nc.Dataset(test_file, 'r')
        print(f"文件: {os.path.basename(test_file)}")
        print(f"\nReferences:")
        print(f"{ds.references}")
        print(f"\nRecent history:")
        if hasattr(ds, 'history'):
            history_lines = ds.history.split('\n')
            for line in history_lines[-3:]:
                print(f"  {line}")
        ds.close()
        print("-" * 70)


if __name__ == "__main__":
    main()
