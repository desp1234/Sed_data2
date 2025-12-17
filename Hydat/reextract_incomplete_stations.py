#!/usr/bin/env python3
"""
重新提取不完整站点的泥沙数据
"""

import sys
sys.path.insert(0, '/Users/zhongwangwei/Downloads/Sediment/Source/Station/Hydat')

from extract_sediment_data import SedimentDataExtractor
from pathlib import Path

def main():
    input_file = '/Users/zhongwangwei/Downloads/Sediment/Source/Station/Hydat/hydat.nc'
    output_dir = '/Users/zhongwangwei/Downloads/Sediment/Source/Station/Hydat/sediment'

    # List of incomplete stations that need to be reprocessed
    incomplete_stations = [
        '01AF006', '01AF007', '01AJ006', '01AJ007',
        '01AJ010', '01AK004', '01AK005', '01AL002'
    ]

    if not Path(input_file).exists():
        print(f"Error: File not found: {input_file}")
        return

    print(f"Reextracting data for {len(incomplete_stations)} incomplete stations...")
    print("=" * 70)

    import netCDF4 as nc

    extractor = SedimentDataExtractor(input_file, output_dir)

    with nc.Dataset(input_file, 'r') as ds:
        extractor.load_stations(ds)

        success_count = 0
        failed_count = 0

        for i, station_id in enumerate(incomplete_stations, 1):
            try:
                print(f"\n[{i}/{len(incomplete_stations)}] Processing station: {station_id}")

                sed_loads_df = extractor.extract_sed_daily_loads(ds, station_id)
                sed_suscon_df = extractor.extract_sed_daily_suscon(ds, station_id)
                sed_samples_df = extractor.extract_sed_samples(ds, station_id)

                has_data = False
                if sed_loads_df is not None and len(sed_loads_df) > 0:
                    print(f"    Sediment load: {len(sed_loads_df)} records")
                    has_data = True
                if sed_suscon_df is not None and len(sed_suscon_df) > 0:
                    print(f"    SSC: {len(sed_suscon_df)} records")
                    has_data = True
                if sed_samples_df is not None and len(sed_samples_df) > 0:
                    print(f"    Samples: {len(sed_samples_df)} records")
                    has_data = True

                if not has_data:
                    print(f"    Skipping: No sediment data")
                    failed_count += 1
                    continue

                extractor.create_station_netcdf(station_id, sed_loads_df, sed_suscon_df, sed_samples_df)
                success_count += 1

            except Exception as e:
                print(f"    ✗ Error: {e}")
                import traceback
                traceback.print_exc()
                failed_count += 1
                continue

    print(f"\n{'='*70}")
    print(f"Reextraction complete!")
    print(f"{'='*70}")
    print(f"Success: {success_count} stations")
    print(f"Failed: {failed_count} stations")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
