
import netCDF4 as nc
import numpy as np
import os

def summarize_nc_file(filepath):
    """Calculates and prints summary statistics for key variables in a NetCDF file."""
    print(f"--- Summarizing file: {os.path.basename(filepath)} ---")
    try:
        with nc.Dataset(filepath, 'r') as ds:
            summary = {}
            for var_name in ['Q', 'SSC', 'SSL']:
                if var_name in ds.variables and f"{var_name}_flag" in ds.variables:
                    var = ds.variables[var_name][:]
                    flag = ds.variables[f"{var_name}_flag"]
                    fill_value = ds.variables[var_name]._FillValue
                    
                    # Use only "good" data (flag == 0) for statistics
                    good_data_mask = (flag[:] == 0) & (var[:] != fill_value)
                    good_data = var[good_data_mask]
                    
                    if good_data.size > 0:
                        summary[var_name] = {
                            'mean': f"{np.mean(good_data):.2f}",
                            'median': f"{np.median(good_data):.2f}",
                            'min': f"{np.min(good_data):.2f}",
                            'max': f"{np.max(good_data):.2f}",
                            'count': good_data.size
                        }
                    else:
                        summary[var_name] = "No 'good' data found."
                else:
                    summary[var_name] = "Variable or its flag not found."
            
            # Print summary
            for var_name, stats in summary.items():
                print(f"Variable: {var_name}")
                if isinstance(stats, dict):
                    print(f"  Unit: {ds.variables[var_name].units}")
                    print(f"  Mean: {stats['mean']}")
                    print(f"  Median: {stats['median']}")
                    print(f"  Min: {stats['min']}")
                    print(f"  Max: {stats['max']}")
                    print(f"  Number of Good Data Points: {stats['count']}")
                else:
                    print(f"  {stats}")
                print("-" * 20)

    except Exception as e:
        print(f"  [ERROR] Could not process file: {e}")

if __name__ == "__main__":
    target_dir = "/Users/zhongwangwei/Downloads/Sediment/Output_r/daily/Mekong_Delta"
    nc_files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith('.nc')]
    
    if not nc_files:
        print("No NetCDF files found in the target directory.")
    else:
        for nc_file in sorted(nc_files):
            summarize_nc_file(nc_file)
            print("\n")
