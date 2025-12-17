
import netCDF4 as nc
import pandas as pd
import os
import glob

def analyze_updated_riversed(data_path):
    print("Analyzing updated RiverSed NetCDF files...")

    nc_files = glob.glob(os.path.join(data_path, '*.nc'))

    # Analyze a few sample files
    for nc_file in nc_files[:3]:
        print(f"\n--- Analyzing file: {os.path.basename(nc_file)} ---")
        with nc.Dataset(nc_file, 'r') as ds:
            print("\nGlobal Attributes:")
            for attr in ds.ncattrs():
                print(f"  {attr}: {ds.getncattr(attr)}")

            print("\nVariable Information:")
            for var_name in ['SSC', 'SSC_flag', 'Q', 'Q_flag', 'SSL', 'SSL_flag']:
                if var_name in ds.variables:
                    var = ds.variables[var_name]
                    print(f"  Variable: {var_name}")
                    print(f"    long_name: {var.long_name}")
                    if hasattr(var, 'units'):
                        print(f"    units: {var.units}")
                    if hasattr(var, 'flag_meanings'):
                        print(f"    flag_meanings: {var.flag_meanings}")
                    # Check a few data points
                    print(f"    Sample data: {var[:5]}")
                else:
                    print(f"  Variable {var_name} not found.")

    # Display summary CSV
    summary_file = os.path.join(data_path, 'RiverSed_station_summary.csv')
    print(f"\n--- Summary CSV: {summary_file} ---")
    if os.path.exists(summary_file):
        df = pd.read_csv(summary_file)
        print(df.head())
    else:
        print("Summary CSV file not found.")

if __name__ == "__main__":
    analyze_updated_riversed('/Users/zhongwangwei/Downloads/Sediment/Output_r/daily/RiverSed')
