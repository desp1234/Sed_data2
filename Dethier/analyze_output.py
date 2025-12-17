
import xarray as xr
import numpy as np
import os
import glob

def analyze_netcdf_files(directory):
    """Analyzes the NetCDF files in a directory to compute stats for Q, SSC, and SSL."""
    
    files = glob.glob(os.path.join(directory, '*.nc'))
    
    for f in files:
        try:
            ds = xr.open_dataset(f)
            print(f"\n--- Analyzing: {os.path.basename(f)} ---")

            for var in ['Q', 'SSC', 'SSL']:
                if var in ds.variables:
                    data = ds[var].values
                    # Mask fill values
                    data = np.ma.masked_equal(data, -9999.0)
                    data = data.compressed() # Get unmasked data

                    if data.size > 0:
                        mean_val = np.mean(data)
                        median_val = np.median(data)
                        min_val = np.min(data)
                        max_val = np.max(data)

                        print(f"  Variable: {var}")
                        print(f"    Mean:     {mean_val:.2f}")
                        print(f"    Median:   {median_val:.2f}")
                        print(f"    Range:    {min_val:.2f} to {max_val:.2f}")
                    else:
                        print(f"  Variable: {var} - No valid data")
                else:
                    print(f"  Variable: {var} - Not found in file")
        except Exception as e:
            print(f"Error processing file {f}: {e}")

if __name__ == '__main__':
    OUTPUT_DIR = '/Users/zhongwangwei/Downloads/Sediment/Output_r/monthly/Dethier'
    analyze_netcdf_files(OUTPUT_DIR)
