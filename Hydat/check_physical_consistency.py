#!/usr/bin/env python3
"""
Check physical consistency of sediment NetCDF files
"""

import netCDF4 as nc
import numpy as np
from pathlib import Path

def check_physical_consistency(nc_file):
    """
    Check if variables in the NetCDF file follow physical laws

    Physical relationship:
    sediment_load (ton/day) = discharge (m3/s) × ssc (mg/L) × conversion_factor

    Conversion:
    discharge: m3/s
    ssc: mg/L = g/m3
    sediment_load: ton/day

    sediment_load = discharge × ssc × 86400 s/day / 1e6 g/ton
                  = discharge × ssc × 86.4 / 1000
                  = discharge × ssc × 0.0864
    """

    issues = []
    stats = {
        'total_points': 0,
        'valid_points': 0,
        'calculation_errors': 0,
        'negative_discharge': 0,
        'negative_ssc': 0,
        'negative_sedload': 0,
        'unrealistic_ssc': 0,  # > 100000 mg/L
        'unrealistic_discharge': 0,  # > 100000 m3/s
        'unrealistic_sedload': 0  # > 1e9 ton/day
    }

    try:
        with nc.Dataset(nc_file, 'r') as ds:
            if 'discharge' not in ds.variables or 'ssc' not in ds.variables or 'sediment_load' not in ds.variables:
                return None, None

            discharge = ds.variables['discharge'][:]
            ssc = ds.variables['ssc'][:]
            sediment_load = ds.variables['sediment_load'][:]

            fill_value = -9999.0
            stats['total_points'] = len(discharge)

            # Find valid data points (all three variables have valid data)
            valid_mask = (discharge != fill_value) & (ssc != fill_value) & (sediment_load != fill_value)
            valid_mask &= ~np.isnan(discharge) & ~np.isnan(ssc) & ~np.isnan(sediment_load)

            stats['valid_points'] = np.sum(valid_mask)

            if stats['valid_points'] == 0:
                return stats, issues

            # Extract valid data
            discharge_valid = discharge[valid_mask]
            ssc_valid = ssc[valid_mask]
            sedload_valid = sediment_load[valid_mask]

            # Check for negative values (physically impossible)
            neg_discharge = discharge_valid < 0
            neg_ssc = ssc_valid < 0
            neg_sedload = sedload_valid < 0

            stats['negative_discharge'] = np.sum(neg_discharge)
            stats['negative_ssc'] = np.sum(neg_ssc)
            stats['negative_sedload'] = np.sum(neg_sedload)

            if stats['negative_discharge'] > 0:
                issues.append(f"Found {stats['negative_discharge']} negative discharge values")
            if stats['negative_ssc'] > 0:
                issues.append(f"Found {stats['negative_ssc']} negative SSC values")
            if stats['negative_sedload'] > 0:
                issues.append(f"Found {stats['negative_sedload']} negative sediment load values")

            # Check for unrealistic values
            unrealistic_ssc = ssc_valid > 100000  # > 100 g/L is very high
            unrealistic_discharge = discharge_valid > 100000  # > 100,000 m3/s is extremely high
            unrealistic_sedload = sedload_valid > 1e9  # > 1 billion ton/day is unrealistic

            stats['unrealistic_ssc'] = np.sum(unrealistic_ssc)
            stats['unrealistic_discharge'] = np.sum(unrealistic_discharge)
            stats['unrealistic_sedload'] = np.sum(unrealistic_sedload)

            if stats['unrealistic_ssc'] > 0:
                max_ssc = np.max(ssc_valid[unrealistic_ssc])
                issues.append(f"Found {stats['unrealistic_ssc']} unrealistic SSC values (max: {max_ssc:.1f} mg/L)")
            if stats['unrealistic_discharge'] > 0:
                max_discharge = np.max(discharge_valid[unrealistic_discharge])
                issues.append(f"Found {stats['unrealistic_discharge']} unrealistic discharge values (max: {max_discharge:.1f} m3/s)")
            if stats['unrealistic_sedload'] > 0:
                max_sedload = np.max(sedload_valid[unrealistic_sedload])
                issues.append(f"Found {stats['unrealistic_sedload']} unrealistic sediment load values (max: {max_sedload:.1e} ton/day)")

            # Check physical relationship: sediment_load = discharge × ssc × 0.0864
            # Allow for rounding errors and original data precision
            calculated_sedload = discharge_valid * ssc_valid * 0.0864

            # Calculate relative error
            # Avoid division by zero
            nonzero_sedload = sedload_valid > 0.01
            if np.sum(nonzero_sedload) > 0:
                rel_error = np.abs(calculated_sedload[nonzero_sedload] - sedload_valid[nonzero_sedload]) / sedload_valid[nonzero_sedload]

                # Consider errors > 5% as significant (accounting for original data precision)
                significant_errors = rel_error > 0.05
                stats['calculation_errors'] = np.sum(significant_errors)

                if stats['calculation_errors'] > 0:
                    max_error = np.max(rel_error[significant_errors]) * 100
                    # Get a sample of problematic values
                    error_indices = np.where(nonzero_sedload)[0][np.where(significant_errors)[0][:3]]

                    issues.append(f"Found {stats['calculation_errors']} calculation inconsistencies (max error: {max_error:.1f}%)")
                    issues.append("Sample of inconsistent data:")
                    for idx in error_indices:
                        actual_idx = np.where(valid_mask)[0][idx]
                        q = discharge_valid[idx]
                        s = ssc_valid[idx]
                        sed_calc = calculated_sedload[idx]
                        sed_actual = sedload_valid[idx]
                        err = rel_error[np.where(nonzero_sedload)[0] == idx][0] * 100
                        issues.append(f"  Index {actual_idx}: Q={q:.2f} m3/s, SSC={s:.2f} mg/L, "
                                    f"Sedload(calc)={sed_calc:.2f}, Sedload(actual)={sed_actual:.2f}, Error={err:.1f}%")

            return stats, issues

    except Exception as e:
        return None, [f"Error reading file: {str(e)}"]


def main():
    sediment_dir = Path('/Users/zhongwangwei/Downloads/Sediment/Source/Station/Hydat/sediment')
    sediment_files = sorted(sediment_dir.glob('HYDAT_*_SEDIMENT.nc'))

    print("Checking physical consistency of sediment data...")
    print("=" * 80)
    print("\nPhysical relationship:")
    print("  sediment_load (ton/day) = discharge (m3/s) × ssc (mg/L) × 0.0864")
    print("\nConversion factor explanation:")
    print("  0.0864 = 86400 seconds/day ÷ 1,000,000 g/ton")
    print("=" * 80)

    total_stats = {
        'files_checked': 0,
        'files_with_data': 0,
        'files_with_issues': 0,
        'total_points': 0,
        'valid_points': 0,
        'calculation_errors': 0,
        'negative_values': 0,
        'unrealistic_values': 0
    }

    files_with_issues = []

    for nc_file in sediment_files:
        total_stats['files_checked'] += 1
        stats, issues = check_physical_consistency(nc_file)

        if stats is None:
            continue

        total_stats['files_with_data'] += 1
        total_stats['total_points'] += stats['total_points']
        total_stats['valid_points'] += stats['valid_points']
        total_stats['calculation_errors'] += stats['calculation_errors']
        total_stats['negative_values'] += (stats['negative_discharge'] +
                                           stats['negative_ssc'] +
                                           stats['negative_sedload'])
        total_stats['unrealistic_values'] += (stats['unrealistic_discharge'] +
                                              stats['unrealistic_ssc'] +
                                              stats['unrealistic_sedload'])

        if issues:
            total_stats['files_with_issues'] += 1
            files_with_issues.append((nc_file.name, stats, issues))

    # Print summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files checked: {total_stats['files_checked']}")
    print(f"Files with valid data: {total_stats['files_with_data']}")
    print(f"Files with issues: {total_stats['files_with_issues']}")
    print(f"\nTotal data points: {total_stats['total_points']:,}")
    print(f"Valid data points: {total_stats['valid_points']:,}")
    print(f"  Calculation errors: {total_stats['calculation_errors']:,}")
    print(f"  Negative values: {total_stats['negative_values']:,}")
    print(f"  Unrealistic values: {total_stats['unrealistic_values']:,}")

    # Print detailed issues
    if files_with_issues:
        print(f"\n{'=' * 80}")
        print(f"FILES WITH ISSUES ({len(files_with_issues)} files)")
        print("=" * 80)

        for fname, stats, issues in files_with_issues[:20]:  # Show first 20 files with issues
            print(f"\n{fname}:")
            print(f"  Valid points: {stats['valid_points']:,}/{stats['total_points']:,}")
            for issue in issues:
                print(f"  - {issue}")

        if len(files_with_issues) > 20:
            print(f"\n... and {len(files_with_issues) - 20} more files with issues")
    else:
        print("\n✓ All files passed physical consistency checks!")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
