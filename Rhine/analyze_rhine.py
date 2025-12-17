
import netCDF4 as nc
import numpy as np
import os
import glob

def analyze_rhine_data(data_path):
    nc_files = glob.glob(os.path.join(data_path, '*.nc'))

    print("Rhine Data Analysis")
    print("=" * 40)

    for nc_file in nc_files:
        with nc.Dataset(nc_file, 'r') as ds:
            station_name = ds.station_name
            print(f"\nStation: {station_name}")

            q = ds.variables['Q'][:]
            ssc = ds.variables['SSC'][:]
            ssl = ds.variables['SSL'][:]

            # Mask missing values
            q_masked = np.ma.masked_values(q, -9999.0)
            ssc_masked = np.ma.masked_values(ssc, -9999.0)
            ssl_masked = np.ma.masked_values(ssl, -9999.0)

            print("\n--- Discharge (Q) (m3/s) ---")
            if not q_masked.mask.all():
                print(f"  Mean: {q_masked.mean():.2f}")
                print(f"  Median: {np.ma.median(q_masked):.2f}")
                print(f"  Range: {q_masked.min():.2f} - {q_masked.max():.2f}")
            else:
                print("  No valid Q data.")

            print("\n--- Suspended Sediment Concentration (SSC) (mg/L) ---")
            if not ssc_masked.mask.all():
                print(f"  Mean: {ssc_masked.mean():.2f}")
                print(f"  Median: {np.ma.median(ssc_masked):.2f}")
                print(f"  Range: {ssc_masked.min():.2f} - {ssc_masked.max():.2f}")
            else:
                print("  No valid SSC data.")

            print("\n--- Suspended Sediment Load (SSL) (ton/day) ---")
            if not ssl_masked.mask.all():
                print(f"  Mean: {ssl_masked.mean():.2f}")
                print(f"  Median: {np.ma.median(ssl_masked):.2f}")
                print(f"  Range: {ssl_masked.min():.2f} - {ssl_masked.max():.2f}")
            else:
                print("  No valid SSL data.")

if __name__ == "__main__":
    analyze_rhine_data('/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Output_r/daily/Rhine/test/')
