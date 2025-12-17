# #!/usr/bin/env python3
# """
# This script analyzes the generated NetCDF files for the Eurasian River dataset.
# It calculates and prints statistics for Q, SSC, and SSL for each station.
# """

# import netCDF4
# import numpy as np
# import glob
# import os

# OUTPUT_DIR = "/Users/zhongwangwei/Downloads/Sediment/Output_r/monthly/Eurasian_River /"

# def analyze_netcdf_files():
#     """Analyzes all NetCDF files in the output directory."""
#     nc_files = glob.glob(os.path.join(OUTPUT_DIR, "*.nc"))

#     for nc_file in nc_files:
#         with netCDF4.Dataset(nc_file, 'r') as nc:
#             print(f"--- Analyzing {os.path.basename(nc_file)} ---")
#             print(f"River: {nc.river_name}")

#             for var_name in ['Q', 'SSC', 'SSL']:
#                 if var_name in nc.variables:
#                     var = nc.variables[var_name][:]
#                     # Mask fill values and inf
#                     var = np.ma.masked_where(var == -9999.0, var)
#                     var = np.ma.masked_invalid(var)

#                     if var.count() > 0:
#                         print(f"  {var_name}:")
#                         print(f"    Mean: {var.mean():.2f}")
#                         print(f"    Median: {np.ma.median(var):.2f}")
#                         print(f"    Min: {var.min():.2f}")
#                         print(f"    Max: {var.max():.2f}")
#                     else:
#                         print(f"  {var_name}: No valid data")

# if __name__ == "__main__":
#     analyze_netcdf_files()