
import netCDF4 as nc
import numpy as np
import os

def check_nc_file(filepath):
    """Checks a single NetCDF file for compliance with physical rules and flagging."""
    print(f"--- Checking file: {os.path.basename(filepath)} ---")
    try:
        with nc.Dataset(filepath, 'r') as ds:
            # --- Check Q (River Discharge) ---
            if 'Q' in ds.variables and 'Q_flag' in ds.variables:
                q = ds.variables['Q'][:]
                q_flag = ds.variables['Q_flag'][:]
                fill_value = ds.variables['Q'].missing_value if hasattr(ds.variables['Q'], 'missing_value') else -9999.0
                
                valid_q_mask = q != fill_value

                print("Checking Q (River Discharge)...")
                
                # Rule: Q < 0 should be flagged as 3 (Bad)
                negative_q_mask = (q < 0) & valid_q_mask
                incorrectly_flagged_negative = negative_q_mask & (q_flag != 3)
                if np.any(incorrectly_flagged_negative):
                    print(f"  [FAIL] Found {np.sum(incorrectly_flagged_negative)} negative Q values not flagged as 'Bad data'.")
                else:
                    print("  [PASS] All negative Q values are correctly flagged.")

                # Rule: Q == 0 should be flagged as 2 (Suspect)
                zero_q_mask = (q == 0) & valid_q_mask
                incorrectly_flagged_zero = zero_q_mask & (q_flag != 2)
                if np.any(incorrectly_flagged_zero):
                    print(f"  [FAIL] Found {np.sum(incorrectly_flagged_zero)} zero Q values not flagged as 'Suspect data'.")
                else:
                    print("  [PASS] All zero Q values are correctly flagged.")

                # Rule: Q > 50000 should be flagged as 2 (Suspect)
                Q_EXTREME_HIGH = 50000
                extreme_q_mask = (q > Q_EXTREME_HIGH) & valid_q_mask
                incorrectly_flagged_extreme = extreme_q_mask & (q_flag != 2)
                if np.any(incorrectly_flagged_extreme):
                    print(f"  [FAIL] Found {np.sum(incorrectly_flagged_extreme)} extreme Q values (> {Q_EXTREME_HIGH}) not flagged as 'Suspect data'.")
                else:
                    print(f"  [PASS] All extreme Q values (> {Q_EXTREME_HIGH}) are correctly flagged.")
            else:
                print("  [SKIP] Q or Q_flag variable not found.")

            # --- Check SSC (Suspended Sediment Concentration) ---
            if 'SSC' in ds.variables and 'SSC_flag' in ds.variables:
                ssc = ds.variables['SSC'][:]
                ssc_flag = ds.variables['SSC_flag'][:]
                fill_value = ds.variables['SSC'].missing_value if hasattr(ds.variables['SSC'], 'missing_value') else -9999.0

                valid_ssc_mask = ssc != fill_value

                print("Checking SSC (Suspended Sediment Concentration)...")

                # Rule: SSC < 0.1 should be flagged as 3 (Bad)
                small_ssc_mask = (ssc < 0.1) & valid_ssc_mask
                incorrectly_flagged_small = small_ssc_mask & (ssc_flag != 3)
                if np.any(incorrectly_flagged_small):
                    print(f"  [FAIL] Found {np.sum(incorrectly_flagged_small)} SSC values (< 0.1) not flagged as 'Bad data'.")
                else:
                    print("  [PASS] All SSC values < 0.1 are correctly flagged.")

                # Rule: SSC > 4000 should be flagged as 2 (Suspect)
                SSC_EXTREME_HIGH = 4000
                extreme_ssc_mask = (ssc > SSC_EXTREME_HIGH) & valid_ssc_mask
                incorrectly_flagged_extreme_ssc = extreme_ssc_mask & (ssc_flag != 2)
                if np.any(incorrectly_flagged_extreme_ssc):
                    print(f"  [FAIL] Found {np.sum(incorrectly_flagged_extreme_ssc)} extreme SSC values (> {SSC_EXTREME_HIGH}) not flagged as 'Suspect data'.")
                else:
                    print(f"  [PASS] All extreme SSC values (> {SSC_EXTREME_HIGH}) are correctly flagged.")
            else:
                print("  [SKIP] SSC or SSC_flag variable not found.")

            # --- Check SSL (Suspended Sediment Load) ---
            if 'SSL' in ds.variables and 'SSL_flag' in ds.variables:
                ssl = ds.variables['SSL'][:]
                ssl_flag = ds.variables['SSL_flag'][:]
                fill_value = ds.variables['SSL'].missing_value if hasattr(ds.variables['SSL'], 'missing_value') else -9999.0

                valid_ssl_mask = ssl != fill_value

                print("Checking SSL (Suspended Sediment Load)...")
                
                # Rule: SSL < 0 should be flagged as 3 (Bad)
                negative_ssl_mask = (ssl < 0) & valid_ssl_mask
                incorrectly_flagged_negative_ssl = negative_ssl_mask & (ssl_flag != 3)
                if np.any(incorrectly_flagged_negative_ssl):
                    print(f"  [FAIL] Found {np.sum(incorrectly_flagged_negative_ssl)} negative SSL values not flagged as 'Bad data'.")
                else:
                    print("  [PASS] All negative SSL values are correctly flagged.")
            else:
                print("  [SKIP] SSL or SSL_flag variable not found.")

    except Exception as e:
        print(f"  [ERROR] Could not process file: {e}")

if __name__ == "__main__":
    target_dir = "/Users/zhongwangwei/Downloads/Sediment/Output_r/daily/Myanmar"
    nc_files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith('.nc')]

    if not nc_files:
        print("No NetCDF files found in the target directory.")
    else:
        for nc_file in sorted(nc_files):
            check_nc_file(nc_file)
            print("\n")
