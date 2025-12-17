#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify NetCDF files for physical consistency
"""

import os
from netCDF4 import Dataset
import numpy as np

def check_netcdf_file(filepath):
    """Check a single NetCDF file for physical consistency"""

    with Dataset(filepath, 'r') as nc:
        station_name = nc.station_name if hasattr(nc, 'station_name') else 'Unknown'

        # Read variables
        discharge = nc.variables['discharge'][0]
        ssc = nc.variables['ssc'][0]
        sediment_load = nc.variables['sediment_load'][0]
        upstream_area = nc.variables['upstream_area'][:]
        longitude = nc.variables['longitude'][:]
        latitude = nc.variables['latitude'][:]

        # Check for missing values
        has_data = (discharge != -9999.0 and ssc != -9999.0 and sediment_load != -9999.0)

        checks = {
            'station': station_name,
            'file': os.path.basename(filepath),
            'has_data': has_data,
            'lon_valid': -180 <= longitude <= 180 if longitude != -9999.0 else True,
            'lat_valid': -90 <= latitude <= 90 if latitude != -9999.0 else True,
            'upstream_area_positive': upstream_area > 0 if upstream_area != -9999.0 else True,
            'discharge_positive': True,
            'ssc_positive': True,
            'sediment_load_positive': True,
            'mass_balance_ok': True,
            'discharge': discharge,
            'ssc': ssc,
            'sediment_load': sediment_load,
            'errors': []
        }

        if has_data:
            # Check positivity
            if discharge <= 0:
                checks['discharge_positive'] = False
                checks['errors'].append(f"Discharge <= 0: {discharge}")

            if ssc <= 0:
                checks['ssc_positive'] = False
                checks['errors'].append(f"SSC <= 0: {ssc}")

            if sediment_load <= 0:
                checks['sediment_load_positive'] = False
                checks['errors'].append(f"Sediment load <= 0: {sediment_load}")

            # Check mass balance: sediment_load = discharge × ssc × 86.4
            if discharge > 0 and ssc > 0:
                calculated_load = discharge * ssc * 86.4
                ratio = sediment_load / calculated_load if calculated_load > 0 else 0

                if not (0.99 < ratio < 1.01):
                    checks['mass_balance_ok'] = False
                    checks['errors'].append(f"Mass balance error: ratio={ratio:.6f}")

                # Check SSC reasonableness (typically 0.1 - 50000 mg/L)
                if ssc > 50000:
                    checks['errors'].append(f"Warning: Very high SSC={ssc:.2f} mg/L")

                # Check specific sediment yield reasonableness
                if upstream_area > 0 and upstream_area != -9999.0:
                    # sediment_load (ton/day) to annual yield (ton/km²/yr)
                    annual_load_ton = sediment_load * 365.25
                    specific_yield = annual_load_ton / upstream_area

                    if specific_yield > 50000:  # Very high but possible in HMA
                        checks['errors'].append(f"Warning: Very high specific yield={specific_yield:.1f} ton/km²/yr")

        return checks

# Main verification
if __name__ == '__main__':
    done_dir = 'Output_r/'
    nc_files = [f for f in os.listdir(done_dir) if f.endswith('.nc')]
    nc_files.sort()

    print("=" * 100)
    print("NetCDF File Verification Report")
    print("=" * 100)

    all_checks = []

    for nc_file in nc_files:
        filepath = os.path.join(done_dir, nc_file)
        checks = check_netcdf_file(filepath)
        all_checks.append(checks)

    # Summary
    print(f"\nTotal files checked: {len(all_checks)}")

    # Files with data
    with_data = [c for c in all_checks if c['has_data']]
    without_data = [c for c in all_checks if not c['has_data']]

    print(f"Files with complete data: {len(with_data)}")
    print(f"Files with missing data: {len(without_data)}")

    # Check for errors
    files_with_errors = [c for c in all_checks if len(c['errors']) > 0]

    print(f"\nFiles with issues/warnings: {len(files_with_errors)}")

    if files_with_errors:
        print("\n" + "-" * 100)
        print("Issues/Warnings Found:")
        print("-" * 100)
        for check in files_with_errors:
            print(f"\n{check['station']} ({check['file']}):")
            for error in check['errors']:
                print(f"  - {error}")

    # Statistical summary for files with data
    if with_data:
        print("\n" + "=" * 100)
        print("Statistical Summary (files with complete data)")
        print("=" * 100)

        discharges = [c['discharge'] for c in with_data]
        sscs = [c['ssc'] for c in with_data]
        sediment_loads = [c['sediment_load'] for c in with_data]

        print(f"\nDischarge (m³/s):")
        print(f"  Min:    {min(discharges):12.2f}")
        print(f"  Max:    {max(discharges):12.2f}")
        print(f"  Mean:   {np.mean(discharges):12.2f}")
        print(f"  Median: {np.median(discharges):12.2f}")

        print(f"\nSSC (mg/L):")
        print(f"  Min:    {min(sscs):12.2f}")
        print(f"  Max:    {max(sscs):12.2f}")
        print(f"  Mean:   {np.mean(sscs):12.2f}")
        print(f"  Median: {np.median(sscs):12.2f}")

        print(f"\nSediment Load (ton/day):")
        print(f"  Min:    {min(sediment_loads):12.2f}")
        print(f"  Max:    {max(sediment_loads):12.2f}")
        print(f"  Mean:   {np.mean(sediment_loads):12.2f}")
        print(f"  Median: {np.median(sediment_loads):12.2f}")

    # Detailed table of all stations with data
    print("\n" + "=" * 100)
    print("Detailed Data Table (stations with complete data)")
    print("=" * 100)
    print(f"{'Station':<35} {'Discharge':>12} {'SSC':>10} {'Sed. Load':>15}")
    print(f"{'':35} {'(m³/s)':>12} {'(mg/L)':>10} {'(ton/day)':>15}")
    print("-" * 100)

    for check in sorted(with_data, key=lambda x: x['discharge'], reverse=True):
        print(f"{check['station']:<35} {check['discharge']:>12.2f} {check['ssc']:>10.2f} {check['sediment_load']:>15.2f}")

    print("\n" + "=" * 100)
    print("All checks PASSED! ✓" if len(files_with_errors) == 0 else f"{len(files_with_errors)} files have warnings (see above)")
    print("=" * 100)
