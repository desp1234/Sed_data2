#!/usr/bin/env python3
"""
EUSEDcollab Data Quality Control and CF-1.8 Standardization Script

This script processes EUSEDcollab monthly sediment and discharge data:
1. Data validation and quality flagging
2. CF-1.8 metadata standardization
3. Variable name standardization
4. Time range trimming based on data availability
5. Station summary CSV generation

Author: Zhongwang Wei (weizhw6@mail.sysu.edu.cn)
Institution: Sun Yat-sen University, China
Date: 2025-10-26
"""

import xarray as xr
import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')


def create_quality_flags(data_array, var_name):
    """
    Create quality flag array for a data variable.

    Parameters:
    -----------
    data_array : numpy array
        Data values to be flagged
    var_name : str
        Variable name ('Q', 'SSC', or 'SSL')

    Returns:
    --------
    flag_array : numpy array (int8)
        Quality flags: 0=Good, 1=Estimated, 2=Suspect, 3=Bad, 9=Missing
    """
    flag_array = np.full_like(data_array, 9, dtype=np.int8)  # Default: missing

    # Valid data points (not NaN and not fill value)
    valid_mask = ~np.isnan(data_array) & (data_array != -9999.0)

    if var_name == 'Q':
        # Good data: Q > 0
        flag_array[valid_mask & (data_array > 0)] = 0
        # Suspect: Q == 0 (possible zero flow)
        flag_array[valid_mask & (data_array == 0)] = 2
        # Bad: Q < 0 (physically impossible)
        flag_array[valid_mask & (data_array < 0)] = 3

    elif var_name == 'SSC':
        # Good data: 0.1 <= SSC <= 3000 mg/L
        flag_array[valid_mask & (data_array >= 0.1) & (data_array <= 3000)] = 0
        # Suspect: SSC > 3000 mg/L (extremely high)
        flag_array[valid_mask & (data_array > 3000)] = 2
        # Bad: SSC < 0.1 mg/L (too low or negative)
        flag_array[valid_mask & (data_array < 0.1)] = 3

    elif var_name == 'SSL':
        # Good data: SSL >= 0
        flag_array[valid_mask & (data_array >= 0)] = 0
        # Bad: SSL < 0 (physically impossible)
        flag_array[valid_mask & (data_array < 0)] = 3

    return flag_array


def get_data_time_range(df, var_names):
    """
    Get the time range where at least one variable has valid data.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with time index
    var_names : list
        List of variable names to check

    Returns:
    --------
    start_date, end_date : str or None
        Start and end dates in format 'YYYY-MM-DD', or None if no valid data
    """
    # Combine all variables to find any valid data
    has_data = pd.Series(False, index=df.index)
    for var in var_names:
        if var in df.columns:
            has_data |= df[var].notna() & (df[var] != -9999.0)

    if not has_data.any():
        return None, None

    valid_indices = df.index[has_data]
    start_year = valid_indices.min().year
    end_year = valid_indices.max().year

    return f'{start_year}-01-01', f'{end_year}-12-31'


def calculate_completeness(series, start_date, end_date):
    """
    Calculate data completeness percentage.

    Parameters:
    -----------
    series : pandas.Series
        Data series with datetime index
    start_date, end_date : str
        Date range in format 'YYYY-MM-DD'

    Returns:
    --------
    completeness : float
        Percentage of complete data (0-100)
    """
    if start_date is None or end_date is None:
        return 0.0

    # Create full monthly date range
    full_range = pd.date_range(start=start_date, end=end_date, freq='MS')

    # Count valid data points
    valid_data = series[~series.isna() & (series != -9999.0)]

    if len(full_range) == 0:
        return 0.0

    completeness = (len(valid_data) / len(full_range)) * 100
    return completeness


def process_single_station(input_path, output_path):
    """
    Process a single station NetCDF file.

    Parameters:
    -----------
    input_path : str
        Path to input NetCDF file
    output_path : str
        Path to output NetCDF file

    Returns:
    --------
    station_info : dict or None
        Dictionary with station metadata, or None if station should be skipped
    """
    try:
        ds = xr.open_dataset(input_path)

        # Convert to dataframe for easier processing
        df = ds.to_dataframe()

        # Rename variables to standard names
        rename_map = {
            'discharge': 'Q',
            'ssc': 'SSC',
            'sediment_load': 'SSL',
            'latitude': 'lat',
            'longitude': 'lon'
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Get time range with valid data
        var_names = ['Q', 'SSC', 'SSL']
        start_date, end_date = get_data_time_range(df, var_names)

        if start_date is None:
            print(f"  ⚠ Skipping (no valid data): {os.path.basename(input_path)}")
            return None

        # Trim to valid time range
        df = df.loc[start_date:end_date]

        # Create quality flags
        for var in var_names:
            if var in df.columns:
                df[f'{var}_flag'] = create_quality_flags(df[var].values, var)

        # Create new xarray dataset
        new_ds = xr.Dataset()

        # Add time coordinate
        new_ds['time'] = ('time', df.index.values)
        new_ds['time'].attrs = {
            'long_name': 'time',
            'standard_name': 'time',
            'units': f'days since {df.index.min().strftime("%Y-%m-%d")} 00:00:00',
            'calendar': 'gregorian',
            'axis': 'T'
        }

        # Add scalar coordinates
        if 'lat' in df.columns:
            lat_val = df['lat'].iloc[0] if not df['lat'].isna().all() else np.nan
            new_ds['lat'] = ([], lat_val)
            new_ds['lat'].attrs = {
                'long_name': 'station latitude',
                'standard_name': 'latitude',
                'units': 'degrees_north',
                'valid_range': np.array([-90.0, 90.0], dtype=np.float32)
            }

        if 'lon' in df.columns:
            lon_val = df['lon'].iloc[0] if not df['lon'].isna().all() else np.nan
            new_ds['lon'] = ([], lon_val)
            new_ds['lon'].attrs = {
                'long_name': 'station longitude',
                'standard_name': 'longitude',
                'units': 'degrees_east',
                'valid_range': np.array([-180.0, 180.0], dtype=np.float32)
            }

        if 'altitude' in df.columns:
            alt_val = df['altitude'].iloc[0] if not df['altitude'].isna().all() else np.nan
            new_ds['altitude'] = ([], alt_val)
            new_ds['altitude'].attrs = {
                'long_name': 'station elevation above sea level',
                'standard_name': 'altitude',
                'units': 'm',
                'positive': 'up',
                '_FillValue': -9999.0,
                'comment': 'Source: Original data provided by EUSEDcollab.'
            }

        if 'upstream_area' in df.columns:
            area_val = df['upstream_area'].iloc[0] if not df['upstream_area'].isna().all() else np.nan
            new_ds['upstream_area'] = ([], area_val)
            new_ds['upstream_area'].attrs = {
                'long_name': 'upstream drainage area',
                'units': 'km2',
                '_FillValue': -9999.0,
                'comment': 'Source: Original data provided by EUSEDcollab.'
            }

        # Add data variables
        for var in var_names:
            if var in df.columns:
                # Replace NaN with fill value
                data = df[var].values.copy()
                data[np.isnan(data)] = -9999.0

                new_ds[var] = ('time', data)
                new_ds[var].attrs = {
                    '_FillValue': -9999.0,
                    'missing_value': -9999.0,
                    'coordinates': 'time lat lon altitude'
                }

                if var == 'Q':
                    new_ds[var].attrs.update({
                        'long_name': 'river discharge',
                        'standard_name': 'water_volume_transport_in_river_channel',
                        'units': 'm3 s-1',
                        'ancillary_variables': 'Q_flag',
                        'comment': 'Source: Original data provided by EUSEDcollab. Unit conversion verified from source documentation.'
                    })
                elif var == 'SSC':
                    new_ds[var].attrs.update({
                        'long_name': 'suspended sediment concentration',
                        'standard_name': 'mass_concentration_of_suspended_matter_in_water',
                        'units': 'mg L-1',
                        'ancillary_variables': 'SSC_flag',
                        'comment': 'Source: Calculated from SSL and Q. Formula: SSC (mg/L) = SSL (ton/day) / (Q (m³/s) × 0.0864), where 0.0864 = 86400 s/day × 10^-6 ton/mg × 1000 L/m³.'
                    })
                elif var == 'SSL':
                    new_ds[var].attrs.update({
                        'long_name': 'suspended sediment load',
                        'units': 'ton day-1',
                        'ancillary_variables': 'SSL_flag',
                        'comment': 'Source: Original data provided by EUSEDcollab. Converted from kg to ton day-1.'
                    })

        # Add flag variables
        for var in var_names:
            if f'{var}_flag' in df.columns:
                flag_data = df[f'{var}_flag'].values.copy()
                new_ds[f'{var}_flag'] = ('time', flag_data)
                new_ds[f'{var}_flag'].attrs = {
                    'long_name': f'quality flag for {var}',
                    'standard_name': 'status_flag',
                    '_FillValue': np.int8(9),
                    'flag_values': np.array([0, 1, 2, 3, 9], dtype=np.int8),
                    'flag_meanings': 'good_data estimated_data suspect_data bad_data missing_data',
                    'comment': 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'
                }

        # Extract station information
        station_name = ds.attrs.get('station_name', 'N/A')
        country = ds.attrs.get('country', 'N/A')
        station_id = ds.attrs.get('station_id', 'N/A')

        # Determine river name (if available)
        river_name = 'N/A'  # EUSEDcollab doesn't provide river names

        # Create Source_ID
        source_id = f"EUSED_{country}_{station_name}"

        # Get reference
        reference = ds.attrs.get('references', 'https://essd.copernicus.org/articles/13/5149/2021/')

        # Global attributes
        new_ds.attrs = {
            'Conventions': 'CF-1.8, ACDD-1.3',
            'title': 'Harmonized Global River Discharge and Sediment',
            'summary': f'Monthly river discharge and suspended sediment data for station {station_name} from the EUSEDcollab project. Data has been quality controlled and standardized to CF-1.8 conventions.',
            'source': 'In-situ station data',
            'data_source_name': 'EUSEDcollab Dataset',
            'station_name': station_name,
            'river_name': river_name,
            'Source_ID': source_id,
            'country': country,
            'type': 'In-situ station data',
            'temporal_resolution': 'monthly',
            'temporal_span': f'{df.index.min().strftime("%Y-%m-%d")} to {df.index.max().strftime("%Y-%m-%d")}',
            'time_coverage_start': df.index.min().strftime('%Y-%m-%d'),
            'time_coverage_end': df.index.max().strftime('%Y-%m-%d'),
            'geospatial_lat_min': float(new_ds['lat'].values) if 'lat' in new_ds else np.nan,
            'geospatial_lat_max': float(new_ds['lat'].values) if 'lat' in new_ds else np.nan,
            'geospatial_lon_min': float(new_ds['lon'].values) if 'lon' in new_ds else np.nan,
            'geospatial_lon_max': float(new_ds['lon'].values) if 'lon' in new_ds else np.nan,
            'geospatial_vertical_min': float(new_ds['altitude'].values) if 'altitude' in new_ds else np.nan,
            'geospatial_vertical_max': float(new_ds['altitude'].values) if 'altitude' in new_ds else np.nan,
            'geographic_coverage': f'{country}',
            'variables_provided': 'altitude, upstream_area, Q, SSC, SSL',
            'reference': reference,
            'source_data_link': 'https://essd.copernicus.org/articles/13/5149/2021/',
            'creator_name': 'Zhongwang Wei',
            'creator_email': 'weizhw6@mail.sysu.edu.cn',
            'creator_institution': 'Sun Yat-sen University, China',
            'date_created': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            'date_modified': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            'processing_level': 'Quality controlled and standardized',
            'history': f'{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}: Quality control, CF-1.8 standardization, and metadata enhancement applied. Script: process_eusedcollab_qc_cf18.py',
            'comment': 'Data represents monthly average values. Quality flags indicate data reliability: 0=good, 1=estimated, 2=suspect, 3=bad, 9=missing. Discharge values for Danish stations corrected by 1000× factor as documented in conversion notes.'
        }

        # Save to NetCDF
        # Prepare encoding without duplicating units in time attrs
        encoding = {
            'time': {'dtype': 'float64', 'zlib': True, 'complevel': 4,
                    'units': f'days since {df.index.min().strftime("%Y-%m-%d")} 00:00:00',
                    'calendar': 'gregorian'}
        }

        # Add encoding for data variables if they exist
        if 'Q' in new_ds:
            encoding['Q'] = {'dtype': 'float32', 'zlib': True, 'complevel': 4}
            encoding['Q_flag'] = {'dtype': 'int8', 'zlib': True, 'complevel': 4}
        if 'SSC' in new_ds:
            encoding['SSC'] = {'dtype': 'float32', 'zlib': True, 'complevel': 4}
            encoding['SSC_flag'] = {'dtype': 'int8', 'zlib': True, 'complevel': 4}
        if 'SSL' in new_ds:
            encoding['SSL'] = {'dtype': 'float32', 'zlib': True, 'complevel': 4}
            encoding['SSL_flag'] = {'dtype': 'int8', 'zlib': True, 'complevel': 4}

        # Remove units and calendar from time attrs to avoid conflict with encoding
        time_units = new_ds['time'].attrs.pop('units', None)
        time_calendar = new_ds['time'].attrs.pop('calendar', None)

        new_ds.to_netcdf(output_path, format='NETCDF4', encoding=encoding, unlimited_dims=['time'])

        # Restore attrs for metadata (not needed since they're in encoding)
        # if time_units:
        #     new_ds['time'].attrs['units'] = time_units
        # if time_calendar:
        #     new_ds['time'].attrs['calendar'] = time_calendar

        # Prepare station summary info
        q_comp = calculate_completeness(df['Q'] if 'Q' in df.columns else pd.Series(), start_date, end_date)
        ssc_comp = calculate_completeness(df['SSC'] if 'SSC' in df.columns else pd.Series(), start_date, end_date)
        ssl_comp = calculate_completeness(df['SSL'] if 'SSL' in df.columns else pd.Series(), start_date, end_date)

        station_info = {
            'Source_ID': source_id,
            'station_name': station_name,
            'river_name': river_name,
            'longitude': float(new_ds['lon'].values) if 'lon' in new_ds else np.nan,
            'latitude': float(new_ds['lat'].values) if 'lat' in new_ds else np.nan,
            'altitude': float(new_ds['altitude'].values) if 'altitude' in new_ds else np.nan,
            'upstream_area': float(new_ds['upstream_area'].values) if 'upstream_area' in new_ds else np.nan,
            'Q_start_date': df.index.min().strftime('%Y-%m'),
            'Q_end_date': df.index.max().strftime('%Y-%m'),
            'Q_percent_complete': round(q_comp, 2),
            'SSC_start_date': df.index.min().strftime('%Y-%m'),
            'SSC_end_date': df.index.max().strftime('%Y-%m'),
            'SSC_percent_complete': round(ssc_comp, 2),
            'SSL_start_date': df.index.min().strftime('%Y-%m'),
            'SSL_end_date': df.index.max().strftime('%Y-%m'),
            'SSL_percent_complete': round(ssl_comp, 2),
            'Data Source Name': 'EUSEDcollab Dataset',
            'Type': 'In-situ',
            'Temporal Resolution': 'monthly',
            'Temporal Span': f'{df.index.min().strftime("%Y-%m")} to {df.index.max().strftime("%Y-%m")}',
            'Variables Provided': 'Q, SSC, SSL',
            'Geographic Coverage': country,
            'Reference/DOI': reference
        }

        ds.close()
        new_ds.close()

        print(f"  ✓ Processed: {os.path.basename(output_path)}")
        return station_info

    except Exception as e:
        print(f"  ✗ Error processing {os.path.basename(input_path)}: {str(e)}")
        return None


def main():
    """
    Main processing function.
    """
    print("="*80)
    print("EUSEDcollab Data Quality Control and CF-1.8 Standardization")
    print("="*80)
    print()

    # Define paths
    input_dir = "/Users/zhongwangwei/Downloads/Sediment/Output/monthly/EUSEDcollab"
    output_dir = "/Users/zhongwangwei/Downloads/Sediment/Output_r/monthly/EUSEDcollab"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get list of input files
    nc_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.nc')])

    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Total files:      {len(nc_files)}")
    print()

    # Process each file
    station_summaries = []
    success_count = 0
    skip_count = 0
    error_count = 0

    print("Processing files...")
    print("-" * 80)

    for filename in nc_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        station_info = process_single_station(input_path, output_path)

        if station_info is not None:
            station_summaries.append(station_info)
            success_count += 1
        elif os.path.exists(output_path):
            error_count += 1
        else:
            skip_count += 1

    print("-" * 80)
    print()

    # Generate station summary CSV
    if station_summaries:
        summary_df = pd.DataFrame(station_summaries)
        csv_path = os.path.join(output_dir, 'EUSEDcollab_station_summary.csv')
        summary_df.to_csv(csv_path, index=False)
        print(f"✓ Station summary CSV saved: {csv_path}")
        print()

    # Print summary
    print("="*80)
    print("Processing Summary")
    print("="*80)
    print(f"Total files:      {len(nc_files)}")
    print(f"Successfully processed: {success_count}")
    print(f"Skipped (no data):      {skip_count}")
    print(f"Errors:                 {error_count}")
    print()
    print(f"Output directory: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
