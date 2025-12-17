#!/usr/bin/env python3
"""
Process GFQA_v2 Dataset to CF-1.8 Compliant NetCDF Format

This script processes GFQA_v2 (GEMStat Flux and water quality data) into
CF-1.8 compliant NetCDF files with quality control flags and comprehensive metadata.

Data Processing Steps:
1. Read existing NetCDF files from Output/daily/GFQA_v2
2. Rename variables to standard names (discharge→Q, ssc→SSC, sediment_load→SSL)
3. Create quality control flag variables based on physical constraints
4. Trim time series to valid data range
5. Update metadata to CF-1.8 standard with ACDD-1.3 compliance
6. Write standardized NetCDF files to Output_r/daily/GFQA_v2
7. Generate station summary CSV

Author: Zhongwang Wei
Institution: Sun Yat-sen University, China
Email: weizhw6@mail.sysu.edu.cn
Date: 2025-10-25
"""

import os
import sys
import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

# Data paths
INPUT_DIR = '/Users/zhongwangwei/Downloads/Sediment/Output/daily/GFQA_v2'
METADATA_FILE = '/Users/zhongwangwei/Downloads/Sediment/Source/GFQA_v2/sed/GEMStat_station_metadata.csv'
OUTPUT_DIR = '/Users/zhongwangwei/Downloads/Sediment/Output_r/daily/GFQA_v2'

# Quality flag definitions
FLAG_GOOD = 0        # Good data
FLAG_ESTIMATED = 1   # Estimated data
FLAG_SUSPECT = 2     # Suspect data (e.g., zero/extreme values)
FLAG_BAD = 3         # Bad data (e.g., negative values)
FLAG_MISSING = 9     # Missing in source

# Physical thresholds for quality control
Q_MIN = 0.0           # m³/s, values < 0 are bad
Q_EXTREME = 10000.0   # m³/s, values > this are suspect
SSC_MIN = 0.0         # mg/L, values < 0 are bad
SSC_EXTREME = 3000.0  # mg/L, values > this are suspect (3 g/ml as per requirements)
SSL_MIN = 0.0         # ton/day, values < 0 are bad

FILL_VALUE = -9999.0

# =============================================================================
# Helper Functions
# =============================================================================

def create_quality_flag(data, variable_name):
    """
    Create quality flag array based on physical constraints

    Parameters:
    -----------
    data : array-like
        Data values
    variable_name : str
        Variable name ('Q', 'SSC', or 'SSL')

    Returns:
    --------
    flags : numpy array (byte)
        Quality flags (0=good, 1=estimated, 2=suspect, 3=bad, 9=missing)
    """
    flags = np.full(len(data), FLAG_MISSING, dtype=np.byte)

    for i, val in enumerate(data):
        if pd.isna(val) or val == FILL_VALUE:
            flags[i] = FLAG_MISSING
            continue

        if variable_name == 'Q':
            if val < Q_MIN:
                flags[i] = FLAG_BAD  # Negative discharge
            elif val == 0.0:
                flags[i] = FLAG_SUSPECT  # Zero flow (may be real, but suspect)
            elif val > Q_EXTREME:
                flags[i] = FLAG_SUSPECT  # Extremely high flow
            else:
                flags[i] = FLAG_GOOD

        elif variable_name == 'SSC':
            if val < SSC_MIN:
                flags[i] = FLAG_BAD  # Negative concentration
            elif val > SSC_EXTREME:
                flags[i] = FLAG_SUSPECT  # Extremely high concentration
            else:
                flags[i] = FLAG_GOOD

        elif variable_name == 'SSL':
            if val < SSL_MIN:
                flags[i] = FLAG_BAD  # Negative load
            else:
                flags[i] = FLAG_GOOD

    return flags


def extract_station_id_from_filename(filename):
    """Extract station ID from filename like GFQA_ARG00014.nc"""
    return filename.replace('GFQA_', '').replace('.nc', '')


def read_gfqa_metadata(station_id):
    """Read metadata for a specific GFQA station from CSV"""
    try:
        # Try different encoding options
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
            try:
                meta_df = pd.read_csv(METADATA_FILE, encoding=encoding, sep=';', decimal=',')
                break
            except:
                continue

        # Find station by GEMS Station Number (which should match our ID)
        station_row = meta_df[meta_df['GEMS Station Number'] == station_id]

        if len(station_row) == 0:
            print(f"  Warning: Station ID {station_id} not found in metadata")
            return None

        station_row = station_row.iloc[0]

        # Calculate upstream area from string if needed
        upstream_area = station_row.get('Upstream Basin Area', np.nan)
        if pd.isna(upstream_area) or upstream_area == '':
            upstream_area = np.nan
        else:
            try:
                upstream_area = float(upstream_area)
            except:
                upstream_area = np.nan

        # Convert latitude and longitude properly (handle both . and , as decimal separator)
        def parse_coord(val):
            if pd.isna(val):
                return 0.0
            try:
                if isinstance(val, str):
                    val = val.replace(',', '.')
                return float(val)
            except:
                return 0.0

        metadata = {
            'station_id': station_id,
            'station_name': station_row.get('Station Identifier', f'Station {station_id}'),
            'water_type': station_row.get('Water Type', ''),
            'river_name': station_row.get('Water Body Name', ''),
            'latitude': parse_coord(station_row.get('Latitude', 0)),
            'longitude': parse_coord(station_row.get('Longitude', 0)),
            'country': station_row.get('Country Name', ''),
            'altitude': station_row.get('Elevation', np.nan),
            'upstream_area': upstream_area,
            'monitoring_type': station_row.get('Monitoring Type', ''),
            'main_basin': station_row.get('Main Basin', ''),
        }

        # Handle altitude
        if pd.notna(metadata['altitude']):
            try:
                metadata['altitude'] = float(metadata['altitude'])
            except:
                metadata['altitude'] = np.nan

        return metadata
    except Exception as e:
        print(f"  Warning: Error reading metadata for station {station_id}: {str(e)}")
        return None


def calculate_data_completeness(data, flags):
    """Calculate percentage of good data (flag=0)"""
    total_count = len(data)
    if total_count == 0:
        return 0.0

    good_count = np.sum(flags == FLAG_GOOD)
    return (good_count / total_count) * 100.0


def trim_to_valid_data(df, date_col='date'):
    """
    Trim dataframe to period with valid data
    Keeps data from first valid Q or SSL value to last valid value
    """
    # Find valid data (not NaN and not missing)
    valid_q = df['Q'].notna() & (df['Q'] != FILL_VALUE)
    valid_ssl = df['SSL'].notna() & (df['SSL'] != FILL_VALUE)
    valid_data = valid_q | valid_ssl

    if not valid_data.any():
        return None  # No valid data

    # Find first and last valid indices
    valid_indices = valid_data[valid_data].index
    first_valid = valid_indices[0]
    last_valid = valid_indices[-1]

    # Trim to valid range
    df_trimmed = df.loc[first_valid:last_valid].copy()

    return df_trimmed


# =============================================================================
# Main Processing Functions
# =============================================================================

def read_existing_netcdf(nc_file):
    """
    Read existing NetCDF file and extract data as DataFrame
    Returns: (date_array, Q, SSC, SSL, time_values)
    """
    with nc.Dataset(nc_file, 'r') as ds:
        # Read time
        time_var = ds.variables['time'][:]
        time_units = ds.variables['time'].units

        # Convert time values to datetime
        # units format: "days since 1970-01-01 00:00:00"
        try:
            reference_date_str = time_units.split('since')[1].strip()
            reference_date = datetime.strptime(reference_date_str, '%Y-%m-%d %H:%M:%S')
        except:
            reference_date = datetime(1970, 1, 1)

        dates = [reference_date + timedelta(days=float(t)) for t in time_var]

        # Read data variables
        q = ds.variables['discharge'][:] if 'discharge' in ds.variables else np.full(len(dates), FILL_VALUE)
        ssc = ds.variables['ssc'][:] if 'ssc' in ds.variables else np.full(len(dates), FILL_VALUE)
        ssl = ds.variables['sediment_load'][:] if 'sediment_load' in ds.variables else np.full(len(dates), FILL_VALUE)

        # Handle masked arrays
        if isinstance(q, np.ma.MaskedArray):
            q = q.filled(FILL_VALUE)
        if isinstance(ssc, np.ma.MaskedArray):
            ssc = ssc.filled(FILL_VALUE)
        if isinstance(ssl, np.ma.MaskedArray):
            ssl = ssl.filled(FILL_VALUE)

        # Read scalar variables
        lat = float(ds.variables['latitude'][:]) if 'latitude' in ds.variables else np.nan
        lon = float(ds.variables['longitude'][:]) if 'longitude' in ds.variables else np.nan
        alt = float(ds.variables['altitude'][:]) if 'altitude' in ds.variables else np.nan
        area = float(ds.variables['upstream_area'][:]) if 'upstream_area' in ds.variables else np.nan

        return {
            'dates': dates,
            'time_values': time_var,
            'Q': q,
            'SSC': ssc,
            'SSL': ssl,
            'latitude': lat,
            'longitude': lon,
            'altitude': alt,
            'upstream_area': area,
        }


def process_station(nc_file):
    """
    Process a single station: read data, apply QC, write new NetCDF

    Returns:
    --------
    station_info : dict or None
        Dictionary with station summary information, or None if processing failed
    """

    # Extract station ID from filename
    filename = os.path.basename(nc_file)
    station_id = extract_station_id_from_filename(filename)

    print(f"\nProcessing station {station_id}...")

    # Read metadata
    metadata = read_gfqa_metadata(station_id)
    if metadata is None:
        print(f"  Skipping: Metadata not found")
        return None

    # Read existing NetCDF
    try:
        data = read_existing_netcdf(nc_file)
    except Exception as e:
        print(f"  Error reading file: {str(e)}")
        return None

    if data is None or len(data['dates']) == 0:
        print(f"  Skipping: No data available")
        return None

    # Create DataFrame for easier manipulation
    df = pd.DataFrame({
        'date': data['dates'],
        'time_values': data['time_values'],
        'Q': data['Q'],
        'SSC': data['SSC'],
        'SSL': data['SSL'],
    })

    # Trim to valid data range
    df = trim_to_valid_data(df)
    if df is None or len(df) == 0:
        print(f"  Skipping: No valid data after trimming")
        return None

    # Create quality flags
    q_flag = create_quality_flag(df['Q'].values, 'Q')
    ssc_flag = create_quality_flag(df['SSC'].values, 'SSC')
    ssl_flag = create_quality_flag(df['SSL'].values, 'SSL')

    # Calculate data completeness
    q_completeness = calculate_data_completeness(df['Q'].values, q_flag)
    ssc_completeness = calculate_data_completeness(df['SSC'].values, ssc_flag)
    ssl_completeness = calculate_data_completeness(df['SSL'].values, ssl_flag)

    # Get date range
    start_date = df['date'].min()
    end_date = df['date'].max()

    print(f"  Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"  Data points: {len(df)}")
    print(f"  Q completeness: {q_completeness:.1f}%")
    print(f"  SSC completeness: {ssc_completeness:.1f}%")
    print(f"  SSL completeness: {ssl_completeness:.1f}%")

    # Create output NetCDF file
    output_file = os.path.join(OUTPUT_DIR, f'GFQA_{station_id}.nc')
    write_netcdf_cf18(df, metadata, data, q_flag, ssc_flag, ssl_flag, output_file)

    # Return station summary
    station_info = {
        'station_name': metadata.get('station_name', f'Station {station_id}'),
        'Source_ID': f'GFQA_{station_id}',
        'river_name': metadata.get('river_name', ''),
        'longitude': metadata.get('longitude', np.nan),
        'latitude': metadata.get('latitude', np.nan),
        'altitude': metadata.get('altitude', np.nan),
        'upstream_area': metadata.get('upstream_area', np.nan),
        'Data Source Name': 'GEMStat GFQA Dataset',
        'Type': 'In-situ',
        'Temporal Resolution': 'daily',
        'Temporal Span': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        'Variables Provided': 'Q, SSC, SSL',
        'Geographic Coverage': f"{metadata.get('country', '')}",
        'Reference/DOI': 'https://www.unep.org/resources/report/global-environment-monitoring-system-water-gemswater',
        'Q_start_date': start_date.strftime('%Y'),
        'Q_end_date': end_date.strftime('%Y'),
        'Q_percent_complete': q_completeness,
        'SSC_start_date': start_date.strftime('%Y'),
        'SSC_end_date': end_date.strftime('%Y'),
        'SSC_percent_complete': ssc_completeness,
        'SSL_start_date': start_date.strftime('%Y'),
        'SSL_end_date': end_date.strftime('%Y'),
        'SSL_percent_complete': ssl_completeness,
    }

    return station_info


def write_netcdf_cf18(df, metadata, data, q_flag, ssc_flag, ssl_flag, output_file):
    """
    Write CF-1.8 and ACDD-1.3 compliant NetCDF file
    """

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Create NetCDF file
    with nc.Dataset(output_file, 'w', format='NETCDF4') as ds:

        # =================================================================
        # Dimensions
        # =================================================================
        time_dim = ds.createDimension('time', None)  # UNLIMITED

        # =================================================================
        # Coordinate Variables
        # =================================================================

        # Time
        time_var = ds.createVariable('time', 'f8', ('time',))
        time_var.standard_name = 'time'
        time_var.long_name = 'time'
        time_var.units = 'days since 1970-01-01 00:00:00'
        time_var.calendar = 'gregorian'
        time_var.axis = 'T'
        time_var[:] = df['time_values'].values

        # Latitude (scalar)
        lat_var = ds.createVariable('lat', 'f4')
        lat_var.standard_name = 'latitude'
        lat_var.long_name = 'station latitude'
        lat_var.units = 'degrees_north'
        lat_var.valid_range = np.array([-90.0, 90.0], dtype=np.float32)
        lat_var[:] = metadata['latitude']

        # Longitude (scalar)
        lon_var = ds.createVariable('lon', 'f4')
        lon_var.standard_name = 'longitude'
        lon_var.long_name = 'station longitude'
        lon_var.units = 'degrees_east'
        lon_var.valid_range = np.array([-180.0, 180.0], dtype=np.float32)
        lon_var[:] = metadata['longitude']

        # Altitude (scalar)
        alt_var = ds.createVariable('altitude', 'f4', fill_value=FILL_VALUE)
        alt_var.standard_name = 'altitude'
        alt_var.long_name = 'station elevation above sea level'
        alt_var.units = 'm'
        alt_var.positive = 'up'
        if pd.notna(metadata.get('altitude')):
            alt_var[:] = metadata['altitude']
            alt_var.comment = 'Source: Original data provided by GEMStat/GEMS-Water Programme.'
        else:
            alt_var[:] = FILL_VALUE
            alt_var.comment = 'Source: Not available in GEMStat metadata.'

        # Upstream area (scalar)
        area_var = ds.createVariable('upstream_area', 'f4', fill_value=FILL_VALUE)
        area_var.long_name = 'upstream drainage area'
        area_var.units = 'km2'
        if pd.notna(metadata.get('upstream_area')):
            area_var[:] = metadata['upstream_area']
            area_var.comment = 'Source: Original data provided by GEMStat/GEMS-Water Programme.'
        else:
            area_var[:] = FILL_VALUE
            area_var.comment = 'Source: Not available in GEMStat metadata.'

        # =================================================================
        # Data Variables
        # =================================================================

        # Q (Discharge)
        q_var = ds.createVariable('Q', 'f4', ('time',), fill_value=FILL_VALUE)
        q_var.standard_name = 'water_volume_transport_in_river_channel'
        q_var.long_name = 'river discharge'
        q_var.units = 'm3 s-1'
        q_var.coordinates = 'time lat lon'
        q_var.ancillary_variables = 'Q_flag'
        q_var.comment = 'Source: Original data provided by GEMStat/GEMS-Water Programme. Units: m³/s'
        q_var[:] = df['Q'].values

        # Q flag
        q_flag_var = ds.createVariable('Q_flag', 'b', ('time',), fill_value=FLAG_MISSING)
        q_flag_var.long_name = 'quality flag for river discharge'
        q_flag_var.standard_name = 'status_flag'
        q_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.byte)
        q_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
        q_flag_var.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'
        q_flag_var[:] = q_flag

        # SSC (Suspended Sediment Concentration)
        ssc_var = ds.createVariable('SSC', 'f4', ('time',), fill_value=FILL_VALUE)
        ssc_var.standard_name = 'mass_concentration_of_suspended_matter_in_water'
        ssc_var.long_name = 'suspended sediment concentration'
        ssc_var.units = 'mg L-1'
        ssc_var.coordinates = 'time lat lon'
        ssc_var.ancillary_variables = 'SSC_flag'
        ssc_var.comment = 'Source: Original data provided by GEMStat/GEMS-Water Programme. Units converted to mg/L.'
        ssc_var[:] = df['SSC'].values

        # SSC flag
        ssc_flag_var = ds.createVariable('SSC_flag', 'b', ('time',), fill_value=FLAG_MISSING)
        ssc_flag_var.long_name = 'quality flag for suspended sediment concentration'
        ssc_flag_var.standard_name = 'status_flag'
        ssc_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.byte)
        ssc_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
        ssc_flag_var.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'
        ssc_flag_var[:] = ssc_flag

        # SSL (Suspended Sediment Load)
        ssl_var = ds.createVariable('SSL', 'f4', ('time',), fill_value=FILL_VALUE)
        ssl_var.standard_name = 'suspended_sediment_load'
        ssl_var.long_name = 'suspended sediment load'
        ssl_var.units = 'ton day-1'
        ssl_var.coordinates = 'time lat lon'
        ssl_var.ancillary_variables = 'SSL_flag'
        ssl_var.comment = 'Source: Original data provided by GEMStat/GEMS-Water Programme. Units: ton/day'
        ssl_var[:] = df['SSL'].values

        # SSL flag
        ssl_flag_var = ds.createVariable('SSL_flag', 'b', ('time',), fill_value=FLAG_MISSING)
        ssl_flag_var.long_name = 'quality flag for suspended sediment load'
        ssl_flag_var.standard_name = 'status_flag'
        ssl_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.byte)
        ssl_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
        ssl_flag_var.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'
        ssl_flag_var[:] = ssl_flag

        # =================================================================
        # Global Attributes - CF-1.8 and ACDD-1.3
        # =================================================================

        # Conventions
        ds.Conventions = 'CF-1.8, ACDD-1.3'

        # Title and Summary
        ds.title = 'Harmonized Global River Discharge and Sediment'
        ds.summary = f'River discharge and suspended sediment data for {metadata.get("station_name", "")} station from the GEMStat GFQA (Flux monitoring) database. This dataset contains daily measurements of river discharge, suspended sediment concentration, and sediment load with quality control flags.'

        # Data Source
        ds.source = 'In-situ station data'
        ds.data_source_name = 'GEMStat GFQA Dataset'

        # Station Information
        ds.station_name = metadata.get('station_name', f'Station {metadata.get("station_id")}')
        ds.river_name = metadata.get('river_name', '')
        ds.Source_ID = f'GFQA_{metadata.get("station_id")}'

        # Temporal Information
        start_date = df['date'].min()
        end_date = df['date'].max()
        ds.temporal_resolution = 'daily'
        ds.temporal_span = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        ds.time_coverage_start = start_date.strftime('%Y-%m-%d')
        ds.time_coverage_end = end_date.strftime('%Y-%m-%d')

        # Spatial Information
        ds.geospatial_lat_min = float(metadata.get('latitude', 0))
        ds.geospatial_lat_max = float(metadata.get('latitude', 0))
        ds.geospatial_lon_min = float(metadata.get('longitude', 0))
        ds.geospatial_lon_max = float(metadata.get('longitude', 0))

        if pd.notna(metadata.get('altitude')):
            ds.geospatial_vertical_min = float(metadata.get('altitude'))
            ds.geospatial_vertical_max = float(metadata.get('altitude'))

        geographic_coverage = f"{metadata.get('main_basin', '')}".strip()
        if not geographic_coverage and metadata.get('country'):
            geographic_coverage = metadata.get('country')
        ds.geographic_coverage = geographic_coverage

        # Variables
        ds.variables_provided = 'altitude, upstream_area, Q, SSC, SSL'
        ds.number_of_data = '1'

        # References
        ds.reference = 'https://www.unep.org/resources/report/global-environment-monitoring-system-water-gemswater'
        ds.source_data_link = 'https://gemstat.bafg.de/'

        # Creator Information
        ds.creator_name = 'Zhongwang Wei'
        ds.creator_email = 'weizhw6@mail.sysu.edu.cn'
        ds.creator_institution = 'Sun Yat-sen University, China'

        # Processing Information
        now = datetime.now()
        ds.date_created = now.strftime('%Y-%m-%d')
        ds.date_modified = now.strftime('%Y-%m-%d')
        ds.processing_level = 'Quality controlled and CF-1.8 standardized'

        # History (Data Provenance)
        history_text = f"{now.strftime('%Y-%m-%d %H:%M:%S')}: "
        history_text += "Processed from GEMStat GFQA database (GEMS/Water Programme). "
        history_text += "Applied comprehensive quality control checks with physical threshold validation. "
        history_text += "Standardized to CF-1.8 compliant format with ACDD-1.3 metadata. "
        history_text += "Added quality flag variables for discharge, sediment concentration, and sediment load. "
        history_text += f"Data trimmed to valid range ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}). "
        history_text += "Script: process_gfqa_v2_to_cf18.py"
        ds.history = history_text

        # Comment
        monitoring_type = metadata.get('monitoring_type', 'Unknown')
        water_type = metadata.get('water_type', 'River')
        ds.comment = f'Monitoring Type: {monitoring_type}. Water Type: {water_type}. Quality flags: 0=good, 1=estimated, 2=suspect, 3=bad, 9=missing.'

    print(f"  Written: {output_file}")


def generate_summary_csv(station_list, output_dir):
    """Generate station summary CSV file"""

    if len(station_list) == 0:
        print("\nNo stations processed, skipping summary CSV generation")
        return

    # Create DataFrame
    summary_df = pd.DataFrame(station_list)

    # Reorder columns
    column_order = [
        'station_name', 'Source_ID', 'river_name', 'longitude', 'latitude',
        'altitude', 'upstream_area', 'Data Source Name', 'Type',
        'Temporal Resolution', 'Temporal Span', 'Variables Provided',
        'Geographic Coverage', 'Reference/DOI',
        'Q_start_date', 'Q_end_date', 'Q_percent_complete',
        'SSC_start_date', 'SSC_end_date', 'SSC_percent_complete',
        'SSL_start_date', 'SSL_end_date', 'SSL_percent_complete'
    ]

    # Only include columns that exist
    available_cols = [col for col in column_order if col in summary_df.columns]
    summary_df = summary_df[available_cols]

    # Write CSV
    csv_file = os.path.join(output_dir, 'GFQA_v2_station_summary.csv')
    summary_df.to_csv(csv_file, index=False)

    print(f"\nStation summary CSV written: {csv_file}")
    print(f"Total stations processed: {len(station_list)}")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main processing function"""

    print("="*80)
    print("GFQA_v2 Dataset Processing to CF-1.8 Format")
    print("="*80)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nInput directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Find all GFQA NetCDF files
    if not os.path.exists(INPUT_DIR):
        print(f"ERROR: Input directory not found: {INPUT_DIR}")
        return

    nc_files = [f for f in os.listdir(INPUT_DIR) if f.startswith('GFQA_') and f.endswith('.nc')]

    if len(nc_files) == 0:
        print(f"ERROR: No GFQA NetCDF files found in {INPUT_DIR}")
        return

    print(f"\nFound {len(nc_files)} GFQA NetCDF files to process")

    # Process each station
    station_list = []

    for idx, nc_file in enumerate(sorted(nc_files), 1):
        nc_path = os.path.join(INPUT_DIR, nc_file)

        try:
            station_info = process_station(nc_path)
            if station_info is not None:
                station_list.append(station_info)
        except Exception as e:
            print(f"  ERROR processing {nc_file}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Generate summary CSV
    generate_summary_csv(station_list, OUTPUT_DIR)

    print("\n" + "="*80)
    print(f"Processing complete! {len(station_list)} stations processed successfully.")
    print("="*80)


if __name__ == '__main__':
    main()
