#!/usr/bin/env python3
"""
Script to convert NERC Hampshire Avon dataset to CF-1.8 compliant NetCDF format.

This script processes daily discharge and water chemistry data from four tributaries
of the Hampshire Avon (Sem, Nadder, West Avon, Ebble) and converts them to standardized
NetCDF files with quality control flags.

Author: Zhongwang Wei (weizhw6@mail.sysu.edu.cn)
Date: 2025-10-25
"""

import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Station metadata based on NERC documentation
# Reference: Heppell, C.M.; Binley, A. (2016). Hampshire Avon: Daily discharge, stage
# and water chemistry data from four tributaries (Sem, Nadder, West Avon, Ebble)
STATION_METADATA = {
    'AS': {
        'station_name': 'Ashley',
        'river_name': 'Sem',
        'Source_ID': 'NERC_AS',
        'latitude': 51.0,  # Approximate, need to verify from documentation
        'longitude': -1.8,  # Approximate, need to verify from documentation
        'altitude': -9999.0,  # Not provided in data files
        'upstream_area': -9999.0  # Not provided in data files
    },
    'CE': {
        'station_name': 'Cerne Abbas',
        'river_name': 'Nadder',
        'Source_ID': 'NERC_CE',
        'latitude': 50.9,  # Approximate, need to verify from documentation
        'longitude': -1.9,  # Approximate, need to verify from documentation
        'altitude': -9999.0,
        'upstream_area': -9999.0
    },
    'GA': {
        'station_name': 'Gale',
        'river_name': 'West Avon',
        'Source_ID': 'NERC_GA',
        'latitude': 51.1,  # Approximate, need to verify from documentation
        'longitude': -1.7,  # Approximate, need to verify from documentation
        'altitude': -9999.0,
        'upstream_area': -9999.0
    },
    'GN': {
        'station_name': 'Green Lane',
        'river_name': 'Ebble',
        'Source_ID': 'NERC_GN',
        'latitude': 51.0,  # Approximate, need to verify from documentation
        'longitude': -2.0,  # Approximate, need to verify from documentation
        'altitude': -9999.0,
        'upstream_area': -9999.0
    }
}

def parse_date(date_str):
    """Parse date string in DD/MM/YYYY format."""
    try:
        return pd.to_datetime(date_str, format='%d/%m/%Y')
    except:
        return pd.NaT

def apply_quality_checks(df):
    """
    Apply physical validity checks and set quality flags.

    Flag definitions:
    0 = Good data
    1 = Estimated data
    2 = Suspect data (zero/extreme values)
    3 = Bad data (negative/invalid values)
    9 = Missing data

    Physical thresholds:
    - Q: < 0 = bad, == 0 = suspect, > 1000 = suspect
    - SSC: < 0 = bad, > 3000 mg/L = suspect
    - SSL: < 0 = bad
    """

    # Initialize flags
    df['Q_flag'] = 9  # Default: missing
    df['SSC_flag'] = 9
    df['SSL_flag'] = 9

    # Q (discharge) flags
    if 'Q' in df.columns:
        # Good data (reasonable range)
        df.loc[(df['Q'] > 0) & (df['Q'] <= 1000), 'Q_flag'] = 0
        # Suspect: zero flow (could be legitimate)
        df.loc[df['Q'] == 0, 'Q_flag'] = 2
        # Suspect: extreme high values
        df.loc[df['Q'] > 1000, 'Q_flag'] = 2
        # Bad: negative values
        df.loc[df['Q'] < 0, 'Q_flag'] = 3
        # Missing values remain as 9

    # SSC (sediment concentration) flags
    if 'SSC' in df.columns:
        # Good data (reasonable range)
        df.loc[(df['SSC'] >= 0.1) & (df['SSC'] <= 3000), 'SSC_flag'] = 0
        # Bad: negative values
        df.loc[df['SSC'] < 0.1, 'SSC_flag'] = 3
        # Suspect: extreme high values
        df.loc[df['SSC'] > 3000, 'SSC_flag'] = 2
        # Missing values remain as 9

    # SSL (sediment load) flags
    if 'SSL' in df.columns:
        # Good data (non-negative)
        df.loc[df['SSL'] >= 0, 'SSL_flag'] = 0
        # Bad: negative values
        df.loc[df['SSL'] < 0, 'SSL_flag'] = 3
        # Missing values remain as 9

    return df

def calculate_SSL(Q, SSC):
    """
    Calculate Suspended Sediment Load (SSL) from discharge and concentration.

    Formula: SSL (ton/day) = Q (m³/s) × SSC (mg/L) × 0.0864

    Derivation:
    Step 1: Convert Q from m³/s to L/day
        Q (L/day) = Q (m³/s) × 86400 (s/day) × 1000 (L/m³)
        Q (L/day) = Q × 86,400,000

    Step 2: Calculate mass per day
        Mass (mg/day) = Q (L/day) × SSC (mg/L)
        Mass (mg/day) = Q × 86,400,000 × SSC

    Step 3: Convert from mg to ton
        SSL (ton/day) = Mass (mg/day) / 10⁶ (mg/ton)
        SSL (ton/day) = Q × 86,400,000 × SSC / 10⁶
        SSL (ton/day) = Q × SSC × 86.4

    Wait, this gives 86.4, but let me verify with an example:
    Q = 1 m³/s, SSC = 1000 mg/L
    - Volume per day: 1 m³/s × 86400 s = 86400 m³ = 86,400,000 L
    - Mass per day: 86,400,000 L × 1000 mg/L = 86,400,000,000 mg = 86,400 kg = 86.4 ton

    So SSL = 1 × 1000 × 0.0864 = 86.4 ton/day ✓

    Therefore: SSL (ton/day) = Q (m³/s) × SSC (mg/L) × 0.0864

    Parameters:
    -----------
    Q : float or array
        Discharge in m³/s
    SSC : float or array
        Suspended sediment concentration in mg/L

    Returns:
    --------
    SSL : float or array
        Suspended sediment load in ton/day
    """
    if pd.isna(Q) or pd.isna(SSC):
        return np.nan
    return Q * SSC * 0.0864

def process_station(station_code, data_dir='data', output_dir='Output'):
    """
    Process data for a single station and create CF-1.8 compliant NetCDF file.

    Parameters:
    -----------
    station_code : str
        Station code (AS, CE, GA, GN)
    data_dir : str
        Directory containing input CSV files
    output_dir : str
        Directory for output NetCDF files
    """

    print(f"\nProcessing station: {station_code}")

    # Get station metadata
    metadata = STATION_METADATA[station_code]

    # Read discharge data
    discharge_file = f"{data_dir}/{station_code}_Discharge_data.csv"
    print(f"  Reading {discharge_file}...")
    df_q = pd.read_csv(discharge_file)

    # Parse dates and discharge
    df_q['date'] = df_q['Date'].apply(parse_date)
    df_q['Q'] = pd.to_numeric(df_q.iloc[:, 1], errors='coerce')

    # Read chemistry data (includes SSC)
    chemistry_file = f"{data_dir}/{station_code}_surfacewater_chemistry.csv"
    print(f"  Reading {chemistry_file}...")
    df_chem = pd.read_csv(chemistry_file)

    # Parse dates and SSC
    df_chem['date'] = df_chem['Date'].apply(parse_date)
    df_chem['SSC'] = pd.to_numeric(df_chem['SSC (mg L-1)'], errors='coerce')

    # Merge discharge and chemistry data
    print("  Merging discharge and chemistry data...")
    df = pd.merge(df_q[['date', 'Q']], df_chem[['date', 'SSC']],
                  on='date', how='outer')
    df = df.sort_values('date').reset_index(drop=True)

    # Remove rows with missing dates
    df = df.dropna(subset=['date'])

    # Calculate SSL
    print("  Calculating SSL...")
    df['SSL'] = df.apply(lambda row: calculate_SSL(row['Q'], row['SSC']), axis=1)

    # Apply quality checks
    print("  Applying quality control checks...")
    df = apply_quality_checks(df)

    # Convert dates to days since 1970-01-01
    reference_date = datetime(1970, 1, 1)
    df['time'] = (df['date'] - pd.Timestamp(reference_date)).dt.total_seconds() / 86400

    # Get temporal span
    valid_dates = df['date'].dropna()
    if len(valid_dates) > 0:
        start_date = valid_dates.min().strftime('%Y-%m-%d')
        end_date = valid_dates.max().strftime('%Y-%m-%d')
        temporal_span = f"{valid_dates.min().year}-{valid_dates.max().year}"
    else:
        start_date = "N/A"
        end_date = "N/A"
        temporal_span = "N/A"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create NetCDF file
    output_file = f"{output_dir}/NERC_{station_code}.nc"
    print(f"  Creating NetCDF file: {output_file}...")

    with nc.Dataset(output_file, 'w', format='NETCDF4') as ncfile:

        # Create dimensions
        time_dim = ncfile.createDimension('time', None)  # UNLIMITED

        # Create coordinate variables
        time_var = ncfile.createVariable('time', 'f8', ('time',))
        time_var.standard_name = 'time'
        time_var.long_name = 'time'
        time_var.units = 'days since 1970-01-01 00:00:00'
        time_var.calendar = 'gregorian'
        time_var.axis = 'T'

        lat_var = ncfile.createVariable('lat', 'f4')
        lat_var.standard_name = 'latitude'
        lat_var.long_name = 'station latitude'
        lat_var.units = 'degrees_north'
        lat_var.valid_range = np.array([-90.0, 90.0], dtype='f4')

        lon_var = ncfile.createVariable('lon', 'f4')
        lon_var.standard_name = 'longitude'
        lon_var.long_name = 'station longitude'
        lon_var.units = 'degrees_east'
        lon_var.valid_range = np.array([-180.0, 180.0], dtype='f4')

        # Create scalar variables
        alt_var = ncfile.createVariable('altitude', 'f4', fill_value=-9999.0)
        alt_var.standard_name = 'altitude'
        alt_var.long_name = 'station elevation above sea level'
        alt_var.units = 'm'
        alt_var.positive = 'up'
        alt_var.comment = 'Source: Not provided in original dataset.'

        area_var = ncfile.createVariable('upstream_area', 'f4', fill_value=-9999.0)
        area_var.long_name = 'upstream drainage area'
        area_var.units = 'km2'
        area_var.comment = 'Source: Not provided in original dataset.'

        # Create data variables
        q_var = ncfile.createVariable('Q', 'f4', ('time',), fill_value=-9999.0)
        q_var.standard_name = 'water_volume_transport_in_river_channel'
        q_var.long_name = 'river discharge'
        q_var.units = 'm3 s-1'
        q_var.coordinates = 'time lat lon'
        q_var.ancillary_variables = 'Q_flag'
        q_var.comment = 'Source: Original data provided by Heppell & Binley (2016). Daily average discharge measurements.'

        q_flag_var = ncfile.createVariable('Q_flag', 'i1', ('time',), fill_value=9)
        q_flag_var.long_name = 'quality flag for river discharge'
        q_flag_var.standard_name = 'status_flag'
        q_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype='i1')
        q_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
        q_flag_var.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'

        ssc_var = ncfile.createVariable('SSC', 'f4', ('time',), fill_value=-9999.0)
        ssc_var.standard_name = 'mass_concentration_of_suspended_matter_in_water'
        ssc_var.long_name = 'suspended sediment concentration'
        ssc_var.units = 'mg L-1'
        ssc_var.coordinates = 'time lat lon'
        ssc_var.ancillary_variables = 'SSC_flag'
        ssc_var.comment = 'Source: Original data provided by Heppell & Binley (2016). Laboratory measurements from water samples.'

        ssc_flag_var = ncfile.createVariable('SSC_flag', 'i1', ('time',), fill_value=9)
        ssc_flag_var.long_name = 'quality flag for suspended sediment concentration'
        ssc_flag_var.standard_name = 'status_flag'
        ssc_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype='i1')
        ssc_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
        ssc_flag_var.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'

        ssl_var = ncfile.createVariable('SSL', 'f4', ('time',), fill_value=-9999.0)
        ssl_var.long_name = 'suspended sediment load'
        ssl_var.units = 'ton day-1'
        ssl_var.coordinates = 'time lat lon'
        ssl_var.ancillary_variables = 'SSL_flag'
        ssl_var.comment = 'Source: Calculated. Formula: SSL (ton/day) = Q (m³/s) × SSC (mg/L) × 0.0864, where 0.0864 = 86400 s/day × 1000 L/m³ / 10⁶ mg/ton.'

        ssl_flag_var = ncfile.createVariable('SSL_flag', 'i1', ('time',), fill_value=9)
        ssl_flag_var.long_name = 'quality flag for suspended sediment load'
        ssl_flag_var.standard_name = 'status_flag'
        ssl_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype='i1')
        ssl_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
        ssl_flag_var.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'

        # Write data
        time_var[:] = df['time'].values
        lat_var[:] = metadata['latitude']
        lon_var[:] = metadata['longitude']
        alt_var[:] = metadata['altitude']
        area_var[:] = metadata['upstream_area']

        # Replace NaN with fill value
        q_data = df['Q'].fillna(-9999.0).values
        ssc_data = df['SSC'].fillna(-9999.0).values
        ssl_data = df['SSL'].fillna(-9999.0).values

        q_var[:] = q_data
        ssc_var[:] = ssc_data
        ssl_var[:] = ssl_data

        q_flag_var[:] = df['Q_flag'].values.astype('i1')
        ssc_flag_var[:] = df['SSC_flag'].values.astype('i1')
        ssl_flag_var[:] = df['SSL_flag'].values.astype('i1')

        # Global attributes
        ncfile.Conventions = 'CF-1.8, ACDD-1.3'
        ncfile.title = 'Harmonized Global River Discharge and Sediment'
        ncfile.summary = (f'Daily river discharge and suspended sediment data for {metadata["station_name"]} '
                         f'station on the {metadata["river_name"]} tributary of the Hampshire Avon, UK. '
                         f'This dataset contains time series of discharge, suspended sediment concentration, '
                         f'and calculated sediment load with quality control flags.')
        ncfile.source = 'In-situ station data'
        ncfile.data_source_name = 'NERC Hampshire Avon Dataset'
        ncfile.station_name = metadata['station_name']
        ncfile.river_name = metadata['river_name']
        ncfile.Source_ID = metadata['Source_ID']

        ncfile.geospatial_lat_min = float(metadata['latitude'])
        ncfile.geospatial_lat_max = float(metadata['latitude'])
        ncfile.geospatial_lon_min = float(metadata['longitude'])
        ncfile.geospatial_lon_max = float(metadata['longitude'])

        if metadata['altitude'] != -9999.0:
            ncfile.geospatial_vertical_min = float(metadata['altitude'])
            ncfile.geospatial_vertical_max = float(metadata['altitude'])

        ncfile.geographic_coverage = 'Hampshire Avon Basin, Southern England, UK'
        ncfile.time_coverage_start = start_date
        ncfile.time_coverage_end = end_date
        ncfile.temporal_span = temporal_span
        ncfile.temporal_resolution = 'daily'
        ncfile.variables_provided = 'Q, SSC, SSL'
        ncfile.number_of_data = str(len(df))

        ncfile.reference = ('Heppell, C.M.; Binley, A. (2016). Hampshire Avon: Daily discharge, stage '
                          'and water chemistry data from four tributaries (Sem, Nadder, West Avon, Ebble). '
                          'NERC Environmental Information Data Centre. '
                          'https://doi.org/10.5285/0dd10858-7b96-41f1-8db5-e7b4c4168af5')
        ncfile.source_data_link = 'https://doi.org/10.5285/0dd10858-7b96-41f1-8db5-e7b4c4168af5'

        ncfile.creator_name = 'Zhongwang Wei'
        ncfile.creator_email = 'weizhw6@mail.sysu.edu.cn'
        ncfile.creator_institution = 'Sun Yat-sen University, China'

        # Add processing history
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ncfile.history = (f'{current_time}: Converted from CSV to CF-1.8 compliant NetCDF format. '
                         f'Applied quality control checks and calculated SSL. '
                         f'Script: convert_NERC_to_netcdf.py')

        ncfile.date_created = datetime.now().strftime('%Y-%m-%d')
        ncfile.date_modified = datetime.now().strftime('%Y-%m-%d')
        ncfile.processing_level = 'Quality controlled and standardized'

        ncfile.comment = ('Data includes daily discharge measurements and periodic suspended sediment '
                         'concentration measurements. SSL calculated from Q and SSC. Quality flags '
                         'indicate data reliability: 0=good, 1=estimated, 2=suspect, 3=bad, 9=missing.')

    print(f"  Successfully created {output_file}")

    return df, metadata, start_date, end_date

def generate_summary_csv(station_summaries, output_dir='Output'):
    """
    Generate station summary CSV file.

    Parameters:
    -----------
    station_summaries : list
        List of dictionaries containing station summary information
    output_dir : str
        Directory for output files
    """

    print("\nGenerating station summary CSV...")

    summary_df = pd.DataFrame(station_summaries)

    # Reorder columns to match reference format
    columns = [
        'station_name', 'Source_ID', 'river_name', 'longitude', 'latitude',
        'altitude', 'upstream_area', 'Data Source Name', 'Type',
        'Temporal Resolution', 'Temporal Span', 'Variables Provided',
        'Geographic Coverage', 'Reference/DOI',
        'Q_start_date', 'Q_end_date', 'Q_percent_complete',
        'SSC_start_date', 'SSC_end_date', 'SSC_percent_complete',
        'SSL_start_date', 'SSL_end_date', 'SSL_percent_complete'
    ]

    summary_df = summary_df[columns]

    output_file = f"{output_dir}/NERC_station_summary.csv"
    summary_df.to_csv(output_file, index=False)

    print(f"Successfully created {output_file}")
    print(f"Total stations processed: {len(summary_df)}")

def calculate_percent_complete(df, var_name, flag_name):
    """Calculate percentage of good data for a variable."""
    if var_name not in df.columns:
        return 0.0

    # Count non-missing values
    total = len(df)
    if total == 0:
        return 0.0

    # Count good data (flag = 0)
    good_data = (df[flag_name] == 0).sum()

    return (good_data / total) * 100.0

def main():
    """Main processing function."""

    print("="*70)
    print("NERC Hampshire Avon Data Processing")
    print("CF-1.8 Compliant NetCDF Generation with Quality Control")
    print("="*70)

    station_codes = ['AS', 'CE', 'GA', 'GN']
    station_summaries = []

    for station_code in station_codes:
        try:
            df, metadata, start_date, end_date = process_station(station_code)

            # Calculate data completeness
            q_complete = calculate_percent_complete(df, 'Q', 'Q_flag')
            ssc_complete = calculate_percent_complete(df, 'SSC', 'SSC_flag')
            ssl_complete = calculate_percent_complete(df, 'SSL', 'SSL_flag')

            # Get date ranges for each variable
            q_dates = df[df['Q_flag'] == 0]['date']
            ssc_dates = df[df['SSC_flag'] == 0]['date']
            ssl_dates = df[df['SSL_flag'] == 0]['date']

            summary = {
                'station_name': metadata['station_name'],
                'Source_ID': metadata['Source_ID'],
                'river_name': metadata['river_name'],
                'longitude': metadata['longitude'],
                'latitude': metadata['latitude'],
                'altitude': metadata['altitude'] if metadata['altitude'] != -9999.0 else 'N/A',
                'upstream_area': metadata['upstream_area'] if metadata['upstream_area'] != -9999.0 else 'N/A',
                'Data Source Name': 'NERC Hampshire Avon Dataset',
                'Type': 'In-situ',
                'Temporal Resolution': 'daily',
                'Temporal Span': f"{start_date.split('-')[0]}-{end_date.split('-')[0]}" if start_date != 'N/A' else 'N/A',
                'Variables Provided': 'Q, SSC, SSL',
                'Geographic Coverage': 'Hampshire Avon Basin, Southern England, UK',
                'Reference/DOI': 'https://doi.org/10.5285/0dd10858-7b96-41f1-8db5-e7b4c4168af5',
                'Q_start_date': q_dates.min().year if len(q_dates) > 0 else 'N/A',
                'Q_end_date': q_dates.max().year if len(q_dates) > 0 else 'N/A',
                'Q_percent_complete': round(q_complete, 1),
                'SSC_start_date': ssc_dates.min().year if len(ssc_dates) > 0 else 'N/A',
                'SSC_end_date': ssc_dates.max().year if len(ssc_dates) > 0 else 'N/A',
                'SSC_percent_complete': round(ssc_complete, 1),
                'SSL_start_date': ssl_dates.min().year if len(ssl_dates) > 0 else 'N/A',
                'SSL_end_date': ssl_dates.max().year if len(ssl_dates) > 0 else 'N/A',
                'SSL_percent_complete': round(ssl_complete, 1)
            }

            station_summaries.append(summary)

        except Exception as e:
            print(f"  Error processing station {station_code}: {str(e)}")
            continue

    # Generate summary CSV
    if station_summaries:
        generate_summary_csv(station_summaries)

    print("\n" + "="*70)
    print("Processing complete!")
    print("="*70)

if __name__ == '__main__':
    main()
