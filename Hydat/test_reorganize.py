#!/usr/bin/env python3
"""
测试版本 - 只处理前N个站点，用于快速验证
"""

import netCDF4 as nc
import sys
from pathlib import Path

# 导入原始脚本的类
sys.path.insert(0, str(Path(__file__).parent))
from reorganize_by_station import HydatStationReorganizer

class TestStationReorganizer(HydatStationReorganizer):
    """测试版本：只处理指定数量的站点"""
    
    def __init__(self, input_nc, output_dir, max_stations=10):
        super().__init__(input_nc, output_dir)
        self.max_stations = max_stations
    
    def process_all_stations(self):
        """处理前N个站点"""
        print(f"\n{'='*80}")
        print(f"测试模式 - 只处理前 {self.max_stations} 个站点")
        print(f"{'='*80}\n")
        
        with nc.Dataset(self.input_nc, 'r') as ds:
            # 加载站点列表
            station_ids = self.load_stations(ds)
            
            # 只处理前N个
            test_stations = station_ids[:self.max_stations]
            
            print(f"\n将处理以下 {len(test_stations)} 个站点:")
            for sid in test_stations:
                print(f"  - {sid}")
            
            print(f"\n开始提取数据...")
            print(f"输出目录: {self.output_dir}")
            print(f"{'='*80}\n")
            
            success_count = 0
            failed_count = 0
            
            for i, station_id in enumerate(test_stations, 1):
                try:
                    print(f"[{i}/{len(test_stations)}] 处理站点: {station_id}")
                    
                    # 提取各类数据
                    flow_df = self.extract_daily_flows(ds, station_id)
                    level_df = self.extract_daily_levels(ds, station_id)
                    annual_df = self.extract_annual_statistics(ds, station_id)
                    sediment_data = self.extract_sediment_data(ds, station_id)
                    
                    # 检查是否有数据
                    has_data = False
                    if flow_df is not None and len(flow_df) > 0:
                        print(f"    流量数据: {len(flow_df)} 条记录")
                        has_data = True
                    if level_df is not None and len(level_df) > 0:
                        print(f"    水位数据: {len(level_df)} 条记录")
                        has_data = True
                    if annual_df is not None and len(annual_df) > 0:
                        print(f"    年度统计: {len(annual_df)} 年")
                        has_data = True
                    if sediment_data is not None:
                        if 'concentration' in sediment_data:
                            print(f"    泥沙浓度: {len(sediment_data['concentration'])} 条记录")
                        if 'load' in sediment_data:
                            print(f"    泥沙负载: {len(sediment_data['load'])} 条记录")
                        has_data = True
                    
                    if not has_data:
                        print(f"    跳过: 无时间序列数据")
                        failed_count += 1
                        continue
                    
                    # 创建NetCDF文件
                    self.create_station_netcdf(station_id, flow_df, level_df, 
                                              annual_df, sediment_data)
                    success_count += 1
                    
                except Exception as e:
                    print(f"    ✗ 错误: {e}")
                    import traceback
                    traceback.print_exc()
                    failed_count += 1
                    continue
                
                print()
        
        # 总结
        print(f"\n{'='*80}")
        print(f"测试完成!")
        print(f"{'='*80}")
        print(f"成功: {success_count} 个站点")
        print(f"失败: {failed_count} 个站点")
        print(f"输出目录: {self.output_dir.absolute()}")
        print(f"\n检查生成的文件:")
        print(f"  ls -lh {self.output_dir}")
        print(f"\n查看站点文件:")
        print(f"  python read_station_file.py {self.output_dir}/HYDAT_*.nc --info")
        print(f"{'='*80}\n")

def main():
    if len(sys.argv) < 2:
        print("用法: python test_reorganize.py <hydat.nc> [output_dir] [max_stations]")
        print("\n测试版本 - 只处理前N个站点")
        print("\n参数:")
        print("  hydat.nc       - 输入的HYDAT NetCDF文件")
        print("  output_dir     - 输出目录 (默认: ./test_stations/)")
        print("  max_stations   - 最多处理几个站点 (默认: 10)")
        print("\n示例:")
        print("  python test_reorganize.py hydat.nc ./test_stations/ 5")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./test_stations"
    max_stations = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    if not Path(input_file).exists():
        print(f"错误: 文件不存在: {input_file}")
        sys.exit(1)
    
    reorganizer = TestStationReorganizer(input_file, output_dir, max_stations)
    reorganizer.process_all_stations()

if __name__ == "__main__":
    main()
