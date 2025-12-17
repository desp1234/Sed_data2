#!/usr/bin/env python3
"""
Script to validate the processed NERC dataset.

This script reads the generated NetCDF files, calculates key statistics for
Q, SSC, and SSL, and compares them against typical real-world values to
ensure the data is of a reasonable order of magnitude.

Author: Zhongwang Wei (weizhw6@mail.sysu.edu.cn)
Date: 2025-10-26
"""

import netCDF4 as nc
import numpy as np
import os

def validate_station(nc_file):
    """
    Reads a NetCDF file, calculates statistics, and prints a summary.
    """
    try:
        with nc.Dataset(nc_file, 'r') as ds:
            station_name = ds.station_name
            
            print(f"\n--- Validation for Station: {station_name} ---")
            
            # --- Reference Values ---
            # Q: Small UK rivers are typically < 50 m3/s. Max can be a few hundred during floods.
            # SSC: Typically 10-300 mg/L. Can exceed 1000 mg/L in floods.
            # SSL: Highly variable. For a small river, a few hundred tons/day would be a high event.

            variables = ['Q', 'SSC', 'SSL']
            stats = {}

            for var in variables:
                if var in ds.variables:
                    data = ds.variables[var][:]
                    flag = ds.variables[f"{var}_flag"][:]
                    
                    # Use only good data (flag == 0)
                    good_data = data[flag == 0]
                    
                    if good_data.size > 0:
                        stats[var] = {
                            'mean': np.mean(good_data),
                            'median': np.median(good_data),
                            'min': np.min(good_data),
                            'max': np.max(good_data),
                            'count': good_data.size
                        }
                    else:
                        stats[var] = None
                else:
                    stats[var] = None

            # --- Print Statistics Table ---
            print("| Variable | Unit      | Mean    | Median  | Min     | Max      | Count   |")
            print("|----------|-----------|---------|---------|---------|----------|---------|")
            
            # Q
            if stats['Q']:
                q_stats = stats['Q']
                print(f"| Q        | m3 s-1    | {q_stats['mean']:.2f}    | {q_stats['median']:.2f}    | {q_stats['min']:.2f}    | {q_stats['max']:.2f}     | {q_stats['count']}   |")
            else:
                print("| Q        | m3 s-1    | N/A     | N/A     | N/A     | N/A      | 0       |")

            # SSC
            if stats['SSC']:
                ssc_stats = stats['SSC']
                print(f"| SSC      | mg L-1    | {ssc_stats['mean']:.2f}   | {ssc_stats['median']:.2f}   | {ssc_stats['min']:.2f}   | {ssc_stats['max']:.2f}  | {ssc_stats['count']}   |")
            else:
                print("| SSC      | mg L-1    | N/A     | N/A     | N/A     | N/A      | 0       |")

            # SSL
            if stats['SSL']:
                ssl_stats = stats['SSL']
                print(f"| SSL      | ton day-1 | {ssl_stats['mean']:.2f}    | {ssl_stats['median']:.2f}    | {ssl_stats['min']:.2f}    | {ssl_stats['max']:.2f}    | {ssl_stats['count']}   |")
            else:
                print("| SSL      | ton day-1 | N/A     | N/A     | N/A     | N/A      | 0       |")

            # --- Analysis & Conclusion ---
            print("\nAnalysis:")
            
            # Q analysis
            if stats['Q']:
                if stats['Q']['max'] < 100 and stats['Q']['mean'] < 50:
                    print("- Q values (max: {stats['Q']['max']:.2f} m3/s) are reasonable for small UK tributaries.")
                else:
                    print("- Q values seem high, requires further investigation.")
            else:
                 print("- Q: No good data to analyze.")

            # SSC analysis
            if stats['SSC']:
                if stats['SSC']['max'] < 3000 and stats['SSC']['mean'] < 500:
                    print("- SSC values (max: {stats['SSC']['max']:.2f} mg/L) are within expected range for storm events.")
                else:
                    print("- SSC values seem high, requires further investigation.")
            else:
                print("- SSC: No good data to analyze.")

            # SSL analysis
            if stats['SSL']:
                if stats['SSL']['max'] < 500:
                    print("- SSL values (max: {stats['SSL']['max']:.2f} ton/day) are reasonable for peak flow in small rivers.")
                else:
                    print("- SSL values seem high, requires further investigation.")
            else:
                print("- SSL: No good data to analyze.")

    except Exception as e:
        print(f"Error processing file {nc_file}: {e}")

def main():
    """Main validation function."""
    output_dir = '/Users/zhongwangwei/Downloads/Sediment/Output_r/daily/NERC'
    nc_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.nc')]

    if not nc_files:
        print(f"No NetCDF files found in {output_dir}")
        return

    print("="*50)
    print("NERC Dataset Validation Report")
    print("="*50)

    for nc_file in sorted(nc_files):
        validate_station(nc_file)

    print("\n" + "="*50)
    print("Validation Complete.")
    print("="*50)

if __name__ == '__main__':
    main()
