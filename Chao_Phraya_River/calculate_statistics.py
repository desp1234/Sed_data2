
import netCDF4 as nc
import numpy as np
import glob
import os

def calculate_statistics(file_path):
    """Calculates and prints statistics for Q, SSC, and SSL from a NetCDF file."""
    try:
        with nc.Dataset(file_path, 'r') as ds:
            print(f"\n--- Statistics for: {os.path.basename(file_path)} ---")
            
            for var_name in ['Q', 'SSC', 'SSL']:
                if var_name in ds.variables:
                    var = ds.variables[var_name]
                    # Mask fill values
                    data = var[:]
                    if hasattr(var, '_FillValue'):
                        data = np.ma.masked_equal(data, var._FillValue)
                    
                    if data.count() > 0: # Check if there is any valid data
                        mean_val = data.mean()
                        median_val = np.ma.median(data)
                        min_val = data.min()
                        max_val = data.max()
                        
                        print(f"  Variable: {var_name} ({var.units})")
                        print(f"    Mean:     {mean_val:.2f}")
                        print(f"    Median:   {median_val:.2f}")
                        print(f"    Range:    {min_val:.2f} to {max_val:.2f}")
                    else:
                        print(f"  Variable: {var_name} - No valid data found.")
                else:
                    print(f"  Variable: {var_name} - Not found in file.")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def main():
    """Main function to find all NetCDF files and calculate statistics."""
    output_dir = "../../../Output_r/annually_climatology/Chao_Phraya_River/"
    nc_files = glob.glob(os.path.join(output_dir, '*.nc'))
    
    if not nc_files:
        print(f"No NetCDF files found in {output_dir}")
        return
        
    print(f"Found {len(nc_files)} NetCDF files to analyze.")
    
    for nc_file in sorted(nc_files):
        calculate_statistics(nc_file)

if __name__ == "__main__":
    main()
