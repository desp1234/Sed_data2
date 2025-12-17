#!/usr/bin/env python3
"""
Process EUSEDcollab Dataset to CF-1.8 Compliant NetCDF Format

This script processes the EUSEDcollab (European Sediment Collaboration) dataset
into CF-1.8 compliant NetCDF files with quality control flags and comprehensive metadata.

Data Processing Steps:
1. Read original CSV data (Q_SSL and METADATA files)
2. Convert units:
   - Q: m³/day → m³/s (÷ 86400)
   - SSC: kg/m³ → mg/L (× 1,000,000)
   - SSL: kg/day → ton/day (÷ 1000)
3. Apply quality control checks and create quality flags
4. Trim time series to valid data range
5. Write CF-1.8 compliant NetCDF files
6. Generate station summary CSV

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
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

# Data paths
SOURCE_DIR = '/Users/guizijin/DATA/sediment_data/EUSEDcollab/'
OUTPUT_DIR = '/Users/guizijin/DATA/sediment_data/EUSEDcollab/output/'
METADATA_FILE = os.path.join(SOURCE_DIR, 'ALL_METADATA.csv')

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

# =============================================================================
# NEW FUNCTION ADDED HERE (Replace old behaviors)
# =============================================================================

def detect_and_convert_columns(df):
    """
    Detect and convert Q, SSC, SSL to:
        Q   -> m3/s
        SSC -> mg/L
        SSL -> ton/day
    Handles: daily, monthly, instantaneous, event-based data
    """

    df = df.copy()

    if 'date' in df.columns:
        days_in_month = df['date'].dt.days_in_month
    else:
        days_in_month = None

    # ---- Q ----
    q_col = next((c for c in df.columns if c.lower().startswith('q') and '(' in c.lower()), None)

    if q_col:
        col = q_col.lower()

        if 'event' in col:
            df['Q_event'] = pd.to_numeric(df[q_col], errors='coerce')

        elif 'm-1' in col and days_in_month is not None:
            df['Q'] = pd.to_numeric(df[q_col], errors='coerce') / (days_in_month * 86400.0)

        elif 'd-1' in col:
            df['Q'] = pd.to_numeric(df[q_col], errors='coerce') / 86400.0

        elif 's-1' in col or 'ts-1' in col or '/s' in col:
            df['Q'] = pd.to_numeric(df[q_col], errors='coerce')

        else:
            df['Q'] = pd.to_numeric(df[q_col], errors='coerce')
    else:
        df['Q'] = np.nan

    # ---- SSC ----
    ssc_col = next((c for c in df.columns if 'ssc' in c.lower() or 'turbidity' in c.lower()), None)

    if ssc_col:
        col = ssc_col.lower()

        if 'kg' in col and 'm-3' in col:
            df['SSC'] = pd.to_numeric(df[ssc_col], errors='coerce') * 1e6

        elif 'g' in col and 'm-3' in col:
            df['SSC'] = pd.to_numeric(df[ssc_col], errors='coerce') * 1e3

        elif 'turbidity' in col:
            df['SSC'] = pd.to_numeric(df[ssc_col], errors='coerce')
            df['SSC_flag'] = FLAG_ESTIMATED

        else:
            df['SSC'] = np.nan
    else:
        df['SSC'] = np.nan

    # ---- SSL ----
    ssl_col = next((c for c in df.columns if 'ssl' in c.lower()), None)

    if ssl_col:
        col = ssl_col.lower()
        if 'kg' in col and 'm-1' in col and days_in_month is not None:
                    df['SSL'] = pd.to_numeric(df[ssl_col], errors='coerce') / days_in_month / 1000.0

        elif 'kg' in col and 'd-1' in col:
            df['SSL'] = pd.to_numeric(df[ssl_col], errors='coerce') / 1000.0

        elif 'kg' in col and 'event' in col:
            df['SSL_event'] = pd.to_numeric(df[ssl_col], errors='coerce')

        elif ('t' in col or 'ton' in col) and 'event' in col:
            df['SSL_event'] = pd.to_numeric(df[ssl_col], errors='coerce')

        else:
            df['SSL'] = np.nan

    else:
        df['SSL'] = np.nan

    return df



def parse_date_flexible(date_str):
    """Parse date from various formats"""
    if pd.isna(date_str):
        return None

    date_str = str(date_str).strip()

    # Try different date formats
    for fmt in ['%d/%m/%Y %H:%M:%S', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d', '%m/%d/%Y', '%m/%d/%Y %H:%M:%S']:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    # Try pandas parsing as last resort
    try:
        return pd.to_datetime(date_str)
    except:
        print(f"Warning: Could not parse date: {date_str}")
        return None


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


def calculate_data_completeness(data, flags):
    """Calculate percentage of good data (flag=0)"""
    total_count = len(data)
    if total_count == 0:
        return 0.0

    good_count = np.sum(flags == FLAG_GOOD)
    return (good_count / total_count) * 100.0


# =============================================================================
# Main Processing Functions
# =============================================================================

def read_station_metadata(station_id):
    """Read metadata for a specific station"""

    # Read ALL_METADATA.csv
    meta_df = pd.read_csv(METADATA_FILE, encoding='utf-8-sig')

    # Find station by ID
    station_row = meta_df[meta_df['Catchment ID'] == station_id]

    if len(station_row) == 0:
        print(f"Warning: Station ID {station_id} not found in metadata")
        return None

    station_row = station_row.iloc[0]

    metadata = {
        'catchment_id': station_id,
        'station_name': station_row['Catchment name'],
        'latitude': station_row['Latitude (4 decimal places)'],
        'longitude': station_row['Longitude (4 decimal places)'],
        'country': station_row['Country'],
        'drainage_area': station_row['Drainage area (ha)'] / 100.0,  # Convert ha to km²
        'stream_type': station_row['Stream type'],
        'data_type': station_row['Data type'],
        'land_use_agriculture': station_row.get('Land use: % agriculture', np.nan),
        'land_use_forest': station_row.get('Land use: % forest', np.nan),
        'start_date': parse_date_flexible(station_row['Measurement start date (DD/MM/YYYY)']),
        'end_date': parse_date_flexible(station_row['Measurement end date (DD/MM/YYYY)']),
        'references': station_row['Relevant references with full details'],
        'contact_name': station_row['Contact name'],
        'contact_email': station_row['Contact email'],
    }

    return metadata


def read_station_data(station_id, country):
    """
    Read Q_SSL data for a specific station, automatically detect column names,
    convert units to:
        Q   -> m3/s
        SSC -> mg/L
        SSL -> ton/day
    Supports daily, monthly, and event-type data formats.
    """

    q_ssl_file = os.path.join(SOURCE_DIR, 'Q_SSL', f'ID_{station_id}_Q_SSL_{country}.csv')

    if not os.path.exists(q_ssl_file):
        print(f"Warning: Data file not found: {q_ssl_file}")
        return None

    df = pd.read_csv(q_ssl_file)

    # ----------------------------------------------------
    # 1) Flexible Date Parsing
    # ----------------------------------------------------
    date_cols = [c for c in df.columns if 'date' in c.lower()]
    if len(date_cols) == 0:
        print(f"  Warning: No date column found in: {q_ssl_file}")
        return None
    
    df['date'] = df[date_cols[0]].apply(parse_date_flexible)
    df = df[df['date'].notna()].copy()

    if len(df) == 0:
        print(f"  Warning: No valid dates found")
        return None

    df = df.sort_values('date').reset_index(drop=True)

    # ----------------------------------------------------
    # 2) Auto column recognition + unit normalization
    # ----------------------------------------------------
    df = detect_and_convert_columns(df)

    # ----------------------------------------------------
    # 3) EVENT DATA HANDLING (IF EVENT COLUMNS FOUND)
    # ----------------------------------------------------
    if 'Q_event' in df.columns or 'SSL_event' in df.columns:

        print(f"  Event-data detected → converting to daily values")

        # Event start/end date detection
        if 'Start date (DD/MM/YYYY)' in df.columns and 'End date (DD/MM/YYYY)' in df.columns:
            df['start_date'] = pd.to_datetime(df['Start date (DD/MM/YYYY)'], errors='coerce')
            df['end_date']   = pd.to_datetime(df['End date (DD/MM/YYYY)'], errors='coerce')
        else:
            df['start_date'] = df['date']
            df['end_date']   = df['date']

        df['duration_days'] = (df['end_date'] - df['start_date']).dt.total_seconds() / 86400.0
        df['duration_days'] = df['duration_days'].replace(0, np.nan).fillna(1.0)

        # Q_event → m³/event → m³/s
        if 'Q_event' in df.columns:
            df['Q'] = df['Q_event'] / (df['duration_days'] * 86400.0)

        # SSL_event → ton/event → ton/day
        if 'SSL_event' in df.columns:
            df['SSL'] = df['SSL_event'] / df['duration_days']
            # kg/event case: convert kg → ton
            if df['SSL'].max() > 100:  # heuristic: large numbers mean kg not ton
                df['SSL'] = df['SSL'] / 1000.0


    # ----------------------------------------------------
    # NEW STEP: Fill missing Q / SSC / SSL using SSL = Q * SSC * 0.0864
    # ----------------------------------------------------
    # Only operate on clean numeric values
    df['Q'] = pd.to_numeric(df['Q'], errors='coerce')
    df['SSC'] = pd.to_numeric(df['SSC'], errors='coerce')
    df['SSL'] = pd.to_numeric(df['SSL'], errors='coerce')

    factor = 0.0864  # given coefficient

    # Case 1: SSL missing → compute from Q and SSC
    mask = df['SSL'].isna() & df['Q'].notna() & df['SSC'].notna()
    df.loc[mask, 'SSL'] = df.loc[mask, 'Q'] * df.loc[mask, 'SSC'] * factor

    # Case 2: SSC missing → compute from Q and SSL
    mask = df['SSC'].isna() & df['Q'].notna() & df['SSL'].notna()
    df.loc[mask, 'SSC'] = df.loc[mask, 'SSL'] / (df.loc[mask, 'Q'] * factor)

    # Case 3: Q missing → compute from SSC and SSL
    mask = df['Q'].isna() & df['SSC'].notna() & df['SSL'].notna()
    df.loc[mask, 'Q'] = df.loc[mask, 'SSL'] / (df.loc[mask, 'SSC'] * factor)

    # ----------------------------------------------------
    # 4) Replace NaN with Fill Value
    # ----------------------------------------------------
    for col in ['Q', 'SSC', 'SSL']:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(FILL_VALUE)

    return df[['date', 'Q', 'SSC', 'SSL']]


def process_station(station_id, country):
    """
    Process a single station: read data, apply QC, write NetCDF

    Returns:
    --------
    station_info : dict or None
        Dictionary with station summary information, or None if processing failed
    """

    print(f"\nProcessing station ID_{station_id}_{country}...")

    # Read metadata
    metadata = read_station_metadata(station_id)
    if metadata is None:
        return None

    # Read data
    df = read_station_data(station_id, country)
    if df is None or len(df) == 0:
        print(f"  Skipping: No data available")
        return None

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

    # Create NetCDF file
    output_file = os.path.join(OUTPUT_DIR, f'EUSEDcollab_{country}-{metadata["station_name"]}-ID{station_id}.nc')
    write_netcdf(df, metadata, q_flag, ssc_flag, ssl_flag, output_file)

    # Return station summary
    station_info = {
        'station_name': metadata['station_name'],
        'Source_ID': f'EUSED_{station_id}',
        'river_name': '',  # Not available in metadata
        'longitude': metadata['longitude'],
        'latitude': metadata['latitude'],
        'altitude': np.nan,  # Not available in metadata
        'upstream_area': metadata['drainage_area'],
        'Data Source Name': 'EUSEDcollab Dataset',
        'Type': 'In-situ',
        'Temporal Resolution': 'daily',
        'Temporal Span': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
        'Variables Provided': 'Q, SSC, SSL',
        'Geographic Coverage': f"{metadata['country']}",
        'Reference/DOI': metadata['references'],
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


def write_netcdf(df, metadata, q_flag, ssc_flag, ssl_flag, output_file):
    """
    Write CF-1.8 compliant NetCDF file
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

        # Convert dates to days since epoch
        epoch = datetime(1970, 1, 1)
        time_values = np.array([(d - epoch).total_seconds() / 86400.0 for d in df['date']])
        time_var[:] = time_values

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

        # Altitude (scalar) - not available in metadata
        alt_var = ds.createVariable('altitude', 'f4', fill_value=FILL_VALUE)
        alt_var.standard_name = 'altitude'
        alt_var.long_name = 'station elevation above sea level'
        alt_var.units = 'm'
        alt_var.positive = 'up'
        alt_var.comment = 'Source: Not available in EUSEDcollab metadata.'
        alt_var[:] = FILL_VALUE

        # Upstream area (scalar)
        area_var = ds.createVariable('upstream_area', 'f4', fill_value=FILL_VALUE)
        area_var.long_name = 'upstream drainage area'
        area_var.units = 'km2'
        area_var.comment = 'Source: Original data provided by EUSEDcollab. Converted from hectares.'
        if pd.notna(metadata['drainage_area']):
            area_var[:] = metadata['drainage_area']
        else:
            area_var[:] = FILL_VALUE

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

        # Set comment based on data type
        if metadata.get('data_type', '').lower().startswith('event'):
            q_var.comment = 'Source: Original data provided by EUSEDcollab. Event data converted to daily rates: Q (m³/event) divided by event duration (days) and 86400 s/day.'
        else:
            q_var.comment = 'Source: Original data provided by EUSEDcollab. Units converted from m³/day to m³/s (÷ 86400).'
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

        # Comment is same for both event and daily data (concentration doesn't need temporal conversion)
        ssc_var.comment = 'Source: Original data provided by EUSEDcollab. Units converted from kg/m³ to mg/L (× 1,000,000).'
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
        ssl_var.long_name = 'suspended sediment load'
        ssl_var.units = 'ton day-1'
        ssl_var.coordinates = 'time lat lon'
        ssl_var.ancillary_variables = 'SSL_flag'

        # Set comment based on data type
        if metadata.get('data_type', '').lower().startswith('event'):
            ssl_var.comment = 'Source: Original data provided by EUSEDcollab. Event data converted to daily rates: SSL (kg/event) divided by event duration (days) and 1000 kg/ton.'
        else:
            ssl_var.comment = 'Source: Original data provided by EUSEDcollab. Units converted from kg/day to ton/day (÷ 1000).'
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
        # Global Attributes
        # =================================================================

        ds.Conventions = 'CF-1.8, ACDD-1.3'
        ds.title = 'Harmonized Global River Discharge and Sediment'
        ds.summary = f'River discharge and suspended sediment data for {metadata["station_name"]} station from the EUSEDcollab (European Sediment Collaboration) database. This dataset contains daily measurements of discharge, suspended sediment concentration, and sediment load with quality control flags.'

        ds.source = 'In-situ station data'
        ds.data_source_name = 'EUSEDcollab Dataset'
        ds.station_name = metadata['station_name']
        ds.river_name = ''  # Not available
        ds.Source_ID = f'EUSED_{metadata["catchment_id"]}'

        # Temporal information
        start_date = df['date'].min()
        end_date = df['date'].max()
        ds.temporal_resolution = 'daily'
        ds.temporal_span = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        ds.time_coverage_start = start_date.strftime('%Y-%m-%d')
        ds.time_coverage_end = end_date.strftime('%Y-%m-%d')

        # Spatial information
        ds.geospatial_lat_min = float(metadata['latitude'])
        ds.geospatial_lat_max = float(metadata['latitude'])
        ds.geospatial_lon_min = float(metadata['longitude'])
        ds.geospatial_lon_max = float(metadata['longitude'])
        ds.geographic_coverage = f"{metadata['country']}, {metadata['stream_type']} stream"

        # Variables
        ds.variables_provided = 'altitude, upstream_area, Q, SSC, SSL'
        ds.number_of_data = '1'

        # References
        ds.reference = metadata['references']
        ds.source_data_link = 'https://esdac.jrc.ec.europa.eu/content/european-sediment-collaboration-eusedcollab-database'

        # Creator information
        ds.creator_name = 'Zhongwang Wei'
        ds.creator_email = 'weizhw6@mail.sysu.edu.cn'
        ds.creator_institution = 'Sun Yat-sen University, China'

        # Contact information (original data provider)
        if pd.notna(metadata.get('contact_name')):
            ds.contributor_name = metadata['contact_name']
        if pd.notna(metadata.get('contact_email')):
            ds.contributor_email = metadata['contact_email']

        # Processing information
        now = datetime.now()
        ds.date_created = now.strftime('%Y-%m-%d')
        ds.date_modified = now.strftime('%Y-%m-%d')
        ds.processing_level = 'Quality controlled and standardized'

        # History (data provenance)
        history_text = f"{now.strftime('%Y-%m-%d %H:%M:%S')}: Converted from EUSEDcollab CSV format to CF-1.8 compliant NetCDF format. "
        history_text += "Applied quality control checks and standardized units. "

        # Add data-type specific conversion notes
        if metadata.get('data_type', '').lower().startswith('event'):
            history_text += "Event data converted to daily: used event start date as record date, "
            history_text += "converted event totals to daily rates by dividing by event duration. "
            history_text += "Unit conversions: Q (m³/event → m³/s), SSC (kg/m³ → mg/L, ×1,000,000), SSL (kg/event → ton/day). "
        else:
            history_text += "Unit conversions: Q (m³/day → m³/s, ÷86400), SSC (kg/m³ → mg/L, ×1,000,000), SSL (kg/day → ton/day, ÷1000). "

        history_text += "Script: process_eusedcollab_to_cf18.py"
        ds.history = history_text

        ds.comment = f'Data type: {metadata["data_type"]}. Stream type: {metadata["stream_type"]}. Quality flags indicate data reliability: 0=good, 1=estimated, 2=suspect, 3=bad, 9=missing.'
        
        # =================================================================
        # Optional: Store Comment field as metadata
        # =================================================================
        if 'Comment' in df.columns:
            # 去重 + 去空白 + 合并成一段文字
            comments = df['Comment'].dropna().astype(str).unique()
            if len(comments) > 0:
                ds.comment_source = "; ".join(comments)
            else:
                ds.comment_source = "No comment text provided in original dataset."
        else:
            ds.comment_source = "No comment column in source dataset."

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

    summary_df = summary_df[column_order]

    # Write CSV
    csv_file = os.path.join(output_dir, 'EUSEDcollab_station_summary.csv')
    summary_df.to_csv(csv_file, index=False)

    print(f"\nStation summary CSV written: {csv_file}")
    print(f"Total stations processed: {len(station_list)}")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main processing function"""

    print("="*80)
    print("EUSEDcollab Dataset Processing to CF-1.8 Format")
    print("="*80)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read metadata to get list of stations
    meta_df = pd.read_csv(METADATA_FILE, encoding='utf-8-sig')

    print(f"\nFound {len(meta_df)} stations in metadata")
    print(f"Output directory: {OUTPUT_DIR}")

    # Process each station
    station_list = []

    for idx, row in meta_df.iterrows():
        station_id = row['Catchment ID']
        country = row['Country']

        try:
            station_info = process_station(station_id, country)
            if station_info is not None:
                station_list.append(station_info)
        except Exception as e:
            print(f"  ERROR processing station ID_{station_id}_{country}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Generate summary CSV
    generate_summary_csv(station_list, OUTPUT_DIR)

    print("\n" + "="*80)
    print("Processing complete!")
    print("="*80)


if __name__ == '__main__':
    main()
