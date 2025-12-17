#!/usr/bin/env python3
"""
EUSEDcollab Data Statistics and Validation Script

This script calculates statistics for Q, SSC, and SSL from processed NetCDF files
and compares them with typical reference values to identify potential data quality issues.

Reference Ranges (from literature):
- Q (discharge):
  * Small streams: < 1 m³/s
  * Small-medium rivers: 1-100 m³/s
  * Medium rivers: 100-1000 m³/s
  * Large rivers: > 1000 m³/s

- SSC (suspended sediment concentration):
  * Clear water: < 1 mg/L
  * Lowland rivers: 20-40 mg/L
  * Highland rivers: 100 mg/L
  * Mountain rivers: 200-300 mg/L
  * Flood conditions: 800-1000 mg/L
  * Mining areas: 1000-2000 mg/L
  * Extreme events: up to 5000-100000 mg/L

- SSL (suspended sediment load):
  * Calculated from Q × SSC
  * Small rivers: few to hundreds of ton/day
  * Large rivers: thousands to tens of thousands of ton/day

Author: Zhongwang Wei (weizhw6@mail.sysu.edu.cn)
Institution: Sun Yat-sen University, China
Date: 2025-10-26
"""

import xarray as xr
import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def calculate_statistics(data, flag_data=None):
    """
    Calculate statistics for a data array, considering quality flags.

    Parameters:
    -----------
    data : numpy array
        Data values
    flag_data : numpy array, optional
        Quality flags (only use flag=0 for "good data")

    Returns:
    --------
    stats : dict
        Dictionary with mean, median, min, max, std, count
    """
    # Filter out fill values and NaN
    valid_mask = ~np.isnan(data) & (data != -9999.0)

    # If flags provided, only use good data (flag=0)
    if flag_data is not None:
        valid_mask = valid_mask & (flag_data == 0)

    valid_data = data[valid_mask]

    if len(valid_data) == 0:
        return {
            'mean': np.nan,
            'median': np.nan,
            'min': np.nan,
            'max': np.nan,
            'std': np.nan,
            'count': 0,
            'p25': np.nan,
            'p75': np.nan
        }

    stats = {
        'mean': np.mean(valid_data),
        'median': np.median(valid_data),
        'min': np.min(valid_data),
        'max': np.max(valid_data),
        'std': np.std(valid_data),
        'count': len(valid_data),
        'p25': np.percentile(valid_data, 25),
        'p75': np.percentile(valid_data, 75)
    }

    return stats


def categorize_river_size(q_mean):
    """
    Categorize river size based on mean discharge.

    Parameters:
    -----------
    q_mean : float
        Mean discharge in m³/s

    Returns:
    --------
    category : str
        River size category
    """
    if np.isnan(q_mean):
        return 'Unknown'
    elif q_mean < 1:
        return 'Small stream (< 1 m³/s)'
    elif q_mean < 100:
        return 'Small-medium river (1-100 m³/s)'
    elif q_mean < 1000:
        return 'Medium river (100-1000 m³/s)'
    else:
        return 'Large river (> 1000 m³/s)'


def assess_ssc_reasonableness(ssc_mean, ssc_max):
    """
    Assess if SSC values are within reasonable ranges.

    Parameters:
    -----------
    ssc_mean : float
        Mean SSC in mg/L
    ssc_max : float
        Maximum SSC in mg/L

    Returns:
    --------
    assessment : str
        Assessment result
    issues : list
        List of potential issues
    """
    issues = []

    if np.isnan(ssc_mean):
        return 'No data', issues

    # Check mean SSC
    if ssc_mean < 1:
        assessment = 'Very low (clear water)'
    elif ssc_mean < 40:
        assessment = 'Low (typical lowland)'
    elif ssc_mean < 100:
        assessment = 'Moderate (typical upland)'
    elif ssc_mean < 300:
        assessment = 'High (mountain/highland)'
    elif ssc_mean < 1000:
        assessment = 'Very high (flood-prone/mining)'
    else:
        assessment = 'Extremely high (potential data issue)'
        issues.append(f'Mean SSC = {ssc_mean:.1f} mg/L is extremely high')

    # Check maximum SSC
    if ssc_max > 10000:
        issues.append(f'Max SSC = {ssc_max:.1f} mg/L exceeds typical maximum (5000-10000 mg/L)')
    elif ssc_max > 5000:
        issues.append(f'Max SSC = {ssc_max:.1f} mg/L is very high (extreme event possible)')

    return assessment, issues


def check_mass_balance(q_mean, ssc_mean, ssl_mean):
    """
    Check if SSL is consistent with Q and SSC using mass balance.

    Formula: SSL (ton/day) = Q (m³/s) × SSC (mg/L) × 0.0864
    where 0.0864 = 86400 s/day × 10^-6 ton/mg × 1000 L/m³

    Parameters:
    -----------
    q_mean : float
        Mean discharge in m³/s
    ssc_mean : float
        Mean SSC in mg/L
    ssl_mean : float
        Mean SSL in ton/day

    Returns:
    --------
    is_consistent : bool
        True if mass balance is reasonable
    calculated_ssl : float
        SSL calculated from Q and SSC
    relative_error : float
        Relative error in percentage
    """
    if np.isnan(q_mean) or np.isnan(ssc_mean) or np.isnan(ssl_mean):
        return None, np.nan, np.nan

    # Calculate expected SSL
    calculated_ssl = q_mean * ssc_mean * 0.0864

    # Calculate relative error
    if ssl_mean > 0:
        relative_error = abs(calculated_ssl - ssl_mean) / ssl_mean * 100
    else:
        relative_error = np.inf

    # Consider it consistent if within ±50% (due to temporal averaging)
    is_consistent = relative_error < 50

    return is_consistent, calculated_ssl, relative_error


def process_station_file(file_path):
    """
    Process a single station NetCDF file and extract statistics.

    Parameters:
    -----------
    file_path : str or Path
        Path to NetCDF file

    Returns:
    --------
    station_stats : dict
        Dictionary with station statistics and assessments
    """
    try:
        ds = xr.open_dataset(file_path)

        # Extract metadata
        station_name = ds.attrs.get('station_name', 'N/A')
        source_id = ds.attrs.get('Source_ID', 'N/A')
        country = ds.attrs.get('country', 'N/A')

        # Initialize results
        results = {
            'Source_ID': source_id,
            'station_name': station_name,
            'country': country,
            'file_name': os.path.basename(file_path)
        }

        # Process Q, SSC, SSL
        for var in ['Q', 'SSC', 'SSL']:
            if var in ds:
                data = ds[var].values
                flag = ds[f'{var}_flag'].values if f'{var}_flag' in ds else None
                stats = calculate_statistics(data, flag)

                results[f'{var}_mean'] = stats['mean']
                results[f'{var}_median'] = stats['median']
                results[f'{var}_min'] = stats['min']
                results[f'{var}_max'] = stats['max']
                results[f'{var}_std'] = stats['std']
                results[f'{var}_count'] = stats['count']
                results[f'{var}_p25'] = stats['p25']
                results[f'{var}_p75'] = stats['p75']
            else:
                for suffix in ['mean', 'median', 'min', 'max', 'std', 'count', 'p25', 'p75']:
                    results[f'{var}_{suffix}'] = np.nan

        # River size categorization
        results['river_category'] = categorize_river_size(results['Q_mean'])

        # SSC reasonableness assessment
        ssc_assessment, ssc_issues = assess_ssc_reasonableness(
            results['SSC_mean'], results['SSC_max']
        )
        results['SSC_assessment'] = ssc_assessment
        results['SSC_issues'] = '; '.join(ssc_issues) if ssc_issues else 'None'

        # Mass balance check
        is_consistent, calc_ssl, rel_error = check_mass_balance(
            results['Q_mean'], results['SSC_mean'], results['SSL_mean']
        )
        results['SSL_calculated'] = calc_ssl
        results['SSL_relative_error_%'] = rel_error
        results['mass_balance_OK'] = 'Yes' if is_consistent else ('No' if is_consistent is not None else 'N/A')

        # Overall quality assessment
        quality_issues = []
        if results['Q_mean'] < 0:
            quality_issues.append('Negative mean Q')
        if results['SSC_mean'] < 0:
            quality_issues.append('Negative mean SSC')
        if results['SSL_mean'] < 0:
            quality_issues.append('Negative mean SSL')
        if ssc_issues:
            quality_issues.extend(ssc_issues)
        if not is_consistent and is_consistent is not None:
            quality_issues.append(f'Mass balance error: {rel_error:.1f}%')

        results['quality_issues'] = '; '.join(quality_issues) if quality_issues else 'None'
        results['overall_quality'] = 'Good' if not quality_issues else 'Needs review'

        ds.close()
        return results

    except Exception as e:
        print(f"  Error processing {os.path.basename(file_path)}: {str(e)}")
        return None


def main():
    """
    Main function to process all stations and generate statistics report.
    """
    print("="*80)
    print("EUSEDcollab Data Statistics and Validation")
    print("="*80)
    print()

    # Define paths
    input_dir = "/Users/zhongwangwei/Downloads/Sediment/Output_r/monthly/EUSEDcollab"
    output_dir = input_dir

    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        print("Please run process_eusedcollab_qc_cf18.py first.")
        return

    # Get list of NetCDF files
    nc_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.nc')])

    if len(nc_files) == 0:
        print(f"Error: No NetCDF files found in {input_dir}")
        return

    print(f"Input directory:  {input_dir}")
    print(f"Total files:      {len(nc_files)}")
    print()

    # Process all files
    print("Processing files and calculating statistics...")
    print("-" * 80)

    all_stats = []
    for i, filename in enumerate(nc_files, 1):
        file_path = os.path.join(input_dir, filename)
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(nc_files)} files processed...")

        station_stats = process_station_file(file_path)
        if station_stats:
            all_stats.append(station_stats)

    print(f"  Progress: {len(nc_files)}/{len(nc_files)} files processed.")
    print("-" * 80)
    print()

    # Create DataFrame
    df_stats = pd.DataFrame(all_stats)

    # Save detailed statistics
    stats_file = os.path.join(output_dir, 'EUSEDcollab_data_statistics.csv')
    df_stats.to_csv(stats_file, index=False, float_format='%.4f')
    print(f"✓ Detailed statistics saved: {stats_file}")

    # Generate summary report
    print()
    print("="*80)
    print("Summary Statistics")
    print("="*80)
    print()

    # River size distribution
    print("River Size Distribution:")
    print(df_stats['river_category'].value_counts())
    print()

    # SSC assessment distribution
    print("SSC Assessment Distribution:")
    print(df_stats['SSC_assessment'].value_counts())
    print()

    # Mass balance check
    print("Mass Balance Consistency:")
    print(df_stats['mass_balance_OK'].value_counts())
    print()

    # Overall quality
    print("Overall Quality Assessment:")
    print(df_stats['overall_quality'].value_counts())
    print()

    # Stations with potential issues
    issues_df = df_stats[df_stats['overall_quality'] != 'Good']
    if len(issues_df) > 0:
        print(f"Stations with Potential Issues ({len(issues_df)}):")
        print("-" * 80)
        for idx, row in issues_df.head(10).iterrows():
            print(f"  {row['Source_ID']}: {row['quality_issues']}")
        if len(issues_df) > 10:
            print(f"  ... and {len(issues_df) - 10} more. See CSV for details.")
    else:
        print("✓ All stations pass quality checks!")

    print()

    # Global statistics summary
    print("Global Statistics (all stations):")
    print("-" * 80)
    print(f"Q (discharge):")
    print(f"  Mean:   {df_stats['Q_mean'].mean():.2f} m³/s")
    print(f"  Median: {df_stats['Q_median'].median():.2f} m³/s")
    print(f"  Range:  {df_stats['Q_min'].min():.2f} - {df_stats['Q_max'].max():.2f} m³/s")
    print()
    print(f"SSC (suspended sediment concentration):")
    print(f"  Mean:   {df_stats['SSC_mean'].mean():.2f} mg/L")
    print(f"  Median: {df_stats['SSC_median'].median():.2f} mg/L")
    print(f"  Range:  {df_stats['SSC_min'].min():.2f} - {df_stats['SSC_max'].max():.2f} mg/L")
    print()
    print(f"SSL (suspended sediment load):")
    print(f"  Mean:   {df_stats['SSL_mean'].mean():.2f} ton/day")
    print(f"  Median: {df_stats['SSL_median'].median():.2f} ton/day")
    print(f"  Range:  {df_stats['SSL_min'].min():.2f} - {df_stats['SSL_max'].max():.2f} ton/day")
    print()

    print("="*80)
    print("Validation Complete")
    print("="*80)
    print()
    print("Reference: Typical values for European rivers")
    print("  Q:   Small streams (< 1 m³/s) to large rivers (> 1000 m³/s)")
    print("  SSC: Clear water (< 1 mg/L) to mining areas (1000-2000 mg/L)")
    print("  SSL: Calculated from Q × SSC × 0.0864")
    print()
    print(f"Output files:")
    print(f"  - Detailed statistics: {stats_file}")
    print("="*80)


if __name__ == '__main__':
    main()
