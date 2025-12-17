#!/usr/bin/env python3
"""
Validate GSED SSC data and compare with typical reference values

This script:
1. Calculates statistics (mean, median, min, max, std) for SSC at each station
2. Compares with typical SSC reference values from literature
3. Identifies potentially problematic stations
4. Generates validation report

Author: Zhongwang Wei
Email: weizhw6@mail.sysu.edu.cn
Date: 2025-10-26
"""

import numpy as np
import pandas as pd
import netCDF4 as nc
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Reference SSC values from literature (mg/L)
# Source: Various studies on global river sediment concentrations
SSC_REFERENCE = {
    'Amazon': {'typical': 150, 'range': (50, 500)},
    'Mississippi': {'typical': 300, 'range': (100, 800)},
    'Yangtze': {'typical': 500, 'range': (200, 1500)},
    'Yellow River': {'typical': 25000, 'range': (1000, 50000)},  # Exceptionally high
    'Ganges': {'typical': 1000, 'range': (300, 2500)},
    'Congo': {'typical': 30, 'range': (10, 100)},
    'Nile': {'typical': 2000, 'range': (500, 5000)},  # Variable, high during flood
    'Mekong': {'typical': 200, 'range': (50, 800)},
}

# General reference ranges for different river types
SSC_GENERAL_RANGES = {
    'low_sediment': (1, 50),        # Clear water rivers (tropical rainforest, stable geology)
    'moderate_sediment': (50, 300),  # Most temperate rivers
    'high_sediment': (300, 1000),    # Rivers with significant erosion
    'very_high_sediment': (1000, 3000),  # Monsoon-affected, agricultural areas
    'extreme_sediment': (3000, 50000),   # Yellow River, highly erosive basins
}

def calculate_station_statistics(nc_file):
    """
    Calculate statistics for a single station

    Args:
        nc_file: Path to NetCDF file

    Returns:
        dict: Statistics dictionary
    """
    try:
        with nc.Dataset(nc_file, 'r') as ds:
            ssc = ds.variables['SSC'][:]
            ssc_flag = ds.variables['SSC_flag'][:]

            # Get station metadata
            source_id = ds.Source_ID
            lat = float(ds.variables['lat'][:])
            lon = float(ds.variables['lon'][:])

            # Mask fill values and bad/missing data
            # Only use good (0) and suspect (2) data for statistics
            valid_mask = (ssc != -9999.0) & ~np.isnan(ssc) & \
                        ((ssc_flag == 0) | (ssc_flag == 2))

            if not np.any(valid_mask):
                return None

            ssc_valid = ssc[valid_mask]

            # Calculate statistics
            stats = {
                'Source_ID': source_id,
                'latitude': lat,
                'longitude': lon,
                'n_total': len(ssc),
                'n_valid': np.sum(valid_mask),
                'n_good': np.sum(ssc_flag == 0),
                'n_suspect': np.sum(ssc_flag == 2),
                'percent_valid': (np.sum(valid_mask) / len(ssc)) * 100,
                'mean': np.mean(ssc_valid),
                'median': np.median(ssc_valid),
                'std': np.std(ssc_valid),
                'min': np.min(ssc_valid),
                'max': np.max(ssc_valid),
                'q25': np.percentile(ssc_valid, 25),
                'q75': np.percentile(ssc_valid, 75),
            }

            return stats

    except Exception as e:
        print(f"Error processing {nc_file.name}: {e}")
        return None

def classify_ssc_range(mean_ssc):
    """
    Classify SSC into categories based on mean value

    Args:
        mean_ssc: Mean SSC value (mg/L)

    Returns:
        str: Category name
    """
    if mean_ssc < 50:
        return 'low_sediment'
    elif mean_ssc < 300:
        return 'moderate_sediment'
    elif mean_ssc < 1000:
        return 'high_sediment'
    elif mean_ssc < 3000:
        return 'very_high_sediment'
    else:
        return 'extreme_sediment'

def validate_station(stats):
    """
    Validate station data against reference ranges

    Args:
        stats: Station statistics dictionary

    Returns:
        dict: Validation results
    """
    mean_ssc = stats['mean']
    max_ssc = stats['max']

    # Classify into category
    category = classify_ssc_range(mean_ssc)
    expected_range = SSC_GENERAL_RANGES[category]

    # Check if values are reasonable
    warnings = []
    flags = []

    # Check mean against expected range
    if mean_ssc < expected_range[0] * 0.5:
        warnings.append(f"Mean SSC ({mean_ssc:.1f}) unusually low for {category} category")
        flags.append('low_mean')

    if mean_ssc > expected_range[1] * 2:
        warnings.append(f"Mean SSC ({mean_ssc:.1f}) unusually high for {category} category")
        flags.append('high_mean')

    # Check max value
    if max_ssc > 10000:
        warnings.append(f"Maximum SSC ({max_ssc:.1f}) extremely high")
        flags.append('extreme_max')

    # Check data completeness
    if stats['percent_valid'] < 20:
        warnings.append(f"Low data completeness ({stats['percent_valid']:.1f}%)")
        flags.append('low_completeness')

    # Check variability
    cv = stats['std'] / stats['mean'] if stats['mean'] > 0 else 0
    if cv > 2:
        warnings.append(f"High coefficient of variation ({cv:.2f})")
        flags.append('high_variability')

    validation = {
        'Source_ID': stats['Source_ID'],
        'category': category,
        'expected_range': f"{expected_range[0]}-{expected_range[1]}",
        'is_valid': len(warnings) == 0,
        'warnings': '; '.join(warnings) if warnings else 'OK',
        'flags': ','.join(flags) if flags else '',
    }

    return validation

def main():
    """Main validation function"""
    data_dir = Path('/Users/zhongwangwei/Downloads/Sediment/Output_r/monthly/GSED')
    output_dir = data_dir

    print("="*70)
    print("GSED SSC Data Validation")
    print("="*70)

    # Find all NetCDF files
    nc_files = sorted(data_dir.glob('GSED_*.nc'))
    print(f"\nFound {len(nc_files)} NetCDF files")

    # Calculate statistics for all stations
    print("\nCalculating statistics...")
    all_stats = []
    for i, nc_file in enumerate(nc_files):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(nc_files)} files...")

        stats = calculate_station_statistics(nc_file)
        if stats is not None:
            all_stats.append(stats)

    print(f"\nSuccessfully processed {len(all_stats)} stations")

    # Create statistics DataFrame
    stats_df = pd.DataFrame(all_stats)

    # Validate each station
    print("\nValidating data against reference ranges...")
    validations = []
    for _, row in stats_df.iterrows():
        validation = validate_station(row.to_dict())
        validations.append(validation)

    validation_df = pd.DataFrame(validations)

    # Merge statistics and validation
    result_df = pd.merge(stats_df, validation_df, on='Source_ID')

    # Print overall statistics
    print("\n" + "="*70)
    print("Overall Statistics Summary")
    print("="*70)
    print(f"\nTotal stations: {len(result_df)}")
    print(f"\nSSC Statistics (mg/L):")
    print(f"  Mean (across all stations): {stats_df['mean'].mean():.2f} ± {stats_df['mean'].std():.2f}")
    print(f"  Median (across all stations): {stats_df['median'].mean():.2f}")
    print(f"  Range: {stats_df['min'].min():.2f} - {stats_df['max'].max():.2f}")
    print(f"\nPercentiles (mg/L):")
    print(f"  25th percentile: {stats_df['mean'].quantile(0.25):.2f}")
    print(f"  50th percentile: {stats_df['mean'].quantile(0.50):.2f}")
    print(f"  75th percentile: {stats_df['mean'].quantile(0.75):.2f}")
    print(f"  95th percentile: {stats_df['mean'].quantile(0.95):.2f}")

    print(f"\nCategory Distribution:")
    category_counts = result_df['category'].value_counts()
    for cat, count in category_counts.items():
        pct = (count / len(result_df)) * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")

    print(f"\nValidation Results:")
    n_valid = result_df['is_valid'].sum()
    n_invalid = len(result_df) - n_valid
    print(f"  Valid: {n_valid} ({n_valid/len(result_df)*100:.1f}%)")
    print(f"  With warnings: {n_invalid} ({n_invalid/len(result_df)*100:.1f}%)")

    if n_invalid > 0:
        print(f"\nMost common warnings:")
        flag_counts = result_df[result_df['flags'] != '']['flags'].str.split(',').explode().value_counts()
        for flag, count in flag_counts.head(5).items():
            print(f"  {flag}: {count}")

    # Save detailed results
    output_file = output_dir / 'GSED_validation_statistics.csv'
    result_df.to_csv(output_file, index=False, float_format='%.4f')
    print(f"\nDetailed validation results saved to: {output_file}")

    # Save summary by category
    summary = stats_df.groupby(result_df['category']).agg({
        'mean': ['count', 'mean', 'std', 'min', 'max'],
        'median': 'mean',
        'percent_valid': 'mean'
    }).round(2)
    summary_file = output_dir / 'GSED_category_summary.csv'
    summary.to_csv(summary_file)
    print(f"Category summary saved to: {summary_file}")

    # Print comparison with reference values
    print("\n" + "="*70)
    print("Comparison with Reference Rivers")
    print("="*70)
    print("\nReference SSC values from literature (mg/L):")
    for river, values in SSC_REFERENCE.items():
        print(f"  {river}: {values['typical']} (range: {values['range'][0]}-{values['range'][1]})")

    print(f"\nGSED dataset mean SSC: {stats_df['mean'].mean():.1f} mg/L")
    print(f"This falls in the '{classify_ssc_range(stats_df['mean'].mean())}' category")

    print("\n" + "="*70)
    print("Validation Complete!")
    print("="*70)

if __name__ == '__main__':
    main()
