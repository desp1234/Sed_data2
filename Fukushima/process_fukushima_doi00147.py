#!/usr/bin/env python3
"""
Enhanced processing script for Fukushima Niida River dataset (DOI 10.34355/CRiED.U.Tsukuba.00147)

This script converts high-frequency discharge and suspended sediment data to daily averages
with CF-1.8 compliant metadata, quality flags, and complete data provenance tracking.

Data source: Feng, B., Onda, Y., Wakiyama, Y., Taniguchi, K., Hashimoto, A., & Zhang, Y. (2022).
Dataset of water discharge and suspended sediment at Niida river basin downstream (Haramachi)
during 2013 to 2018 and upstream (Notegami) during 2015 to 2018.
CRiED, University of Tsukuba. https://doi.org/10.34355/CRiED.U.Tsukuba.00147

Unit Conversions:
- Discharge (Q): m³/s (no conversion needed)
- SSC: g/L → mg/L (multiply by 1000)
- Sediment Load (SSL): Q (m³/s) × SSC (g/L) × 86.4 = ton/day
  Formula derivation:
  Q (m³/s) × 86400 (s/day) = m³/day
  m³/day × 1000 (L/m³) = 1000 L/day
  1000 L/day × SSC (g/L) = 1,000 × SSC (g/day)
  1,000 × SSC (g/day) / 1,000,000 (g/ton) = SSC/1000 (ton/day)
  Therefore: SSL = Q × SSC × 86400 / 1,000,000 = Q × SSC × 0.0864... ≈ Q × SSC × 86.4
"""

import os
import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def read_fukushima_excel_data(filepath):
    """
    Read Fukushima Excel data file and extract discharge and SSC data.

    Parameters:
    -----------
    filepath : str
        Path to Excel file containing raw data

    Returns:
    --------
    dict
        Dictionary with station names as keys and DataFrames as values
    """
    # Read all sheets from workbook
    all_sheets = pd.read_excel(filepath, sheet_name=None)
    station_data = {}

    for sheet_name, df in all_sheets.items():
        print(f"Processing sheet: {sheet_name}")

        # Column mapping for DOI00147 data:
        # Col 0: DOI, 1: DID, 2: station_name, 3-7: date/time (yyyy,mm,dd,hh,min),
        # 8: xyear, 9: LatDir, 10: Nsflag, 11: latitude, 12: LonDir,
        # 13: Ewflag, 14: longitude, 15: altdepflag, 16: depth (m below water surface),
        # 17: sample_type, 18: discharge (m³/s), 19: SSC (g/L), 20: SSC_uncertainty (g/L)

        # Skip first 2 header rows, then skip units row
        data = pd.read_excel(filepath, sheet_name=sheet_name, skiprows=2, header=None)
        data = data.iloc[1:].reset_index(drop=True)

        # Extract station name
        station_name = data.iloc[0, 2]

        # Create datetime from components
        data['datetime'] = pd.to_datetime({
            'year': data[3].astype(int),
            'month': data[4].astype(int),
            'day': data[5].astype(int),
            'hour': data[6].astype(int),
            'minute': data[7].astype(int)
        })

        # Extract and organize relevant columns
        df_clean = pd.DataFrame({
            'datetime': data['datetime'],
            'latitude': data[11].astype(float),
            'longitude': data[14].astype(float),
            'depth': data[16].astype(float),  # Sampling depth (m below water surface)
            'discharge': data[18].astype(float),  # m³/s
            'ssc': data[19].astype(float),  # g/L
            'ssc_uncertainty': data[20].astype(float)  # g/L
        })

        # Unit conversion: SSC from g/L to mg/L
        df_clean['ssc_mg_L'] = df_clean['ssc'] * 1000
        df_clean['ssc_uncertainty_mg_L'] = df_clean['ssc_uncertainty'] * 1000

        # Calculate sediment load: Q × SSC × 86.4 = ton/day
        df_clean['sediment_load'] = df_clean['discharge'] * df_clean['ssc'] * 86.4

        # Store station data
        if station_name not in station_data:
            station_data[station_name] = []
        station_data[station_name].append(df_clean)

    # Combine multiple records per station and sort chronologically
    combined_data = {}
    for station, data_list in station_data.items():
        combined = pd.concat(data_list, ignore_index=True)
        combined = combined.sort_values('datetime').reset_index(drop=True)
        combined_data[station] = combined

    return combined_data


def aggregate_to_daily(df):
    """
    Aggregate high-frequency data to daily averages.

    Parameters:
    -----------
    df : pd.DataFrame
        Input data with datetime column

    Returns:
    --------
    pd.DataFrame
        Daily aggregated data
    """
    df_indexed = df.set_index('datetime')

    # Resample to daily frequency, taking mean
    daily = df_indexed.resample('D').mean()

    # Recalculate sediment load from daily averages
    daily['sediment_load'] = daily['discharge'] * daily['ssc'] * 86.4

    return daily.reset_index()


def perform_quality_control(daily_df):
    """
    Apply quality control checks and create flag variables.

    Flag definitions:
    - 0: Good data (passed all QC checks)
    - 1: Estimated data
    - 2: Suspect data (extreme or unusual values)
    - 3: Bad data (physically impossible)
    - 9: Missing data (NaN in source)

    Parameters:
    -----------
    daily_df : pd.DataFrame
        Daily aggregated data

    Returns:
    --------
    pd.DataFrame
        Data with added flag columns
    """
    qc_df = daily_df.copy()

    # Initialize flag variables (default = 0 = good)
    qc_df['Q_flag'] = np.zeros(len(qc_df), dtype=np.int8)
    qc_df['SSC_flag'] = np.zeros(len(qc_df), dtype=np.int8)
    qc_df['SSL_flag'] = np.zeros(len(qc_df), dtype=np.int8)

    # Q (discharge) quality checks
    qc_df.loc[qc_df['discharge'].isna(), 'Q_flag'] = 9  # Missing
    qc_df.loc[(qc_df['discharge'] < 0) & (qc_df['Q_flag'] != 9), 'Q_flag'] = 3  # Bad (negative)
    qc_df.loc[(qc_df['discharge'] == 0) & (qc_df['Q_flag'] == 0), 'Q_flag'] = 2  # Suspect (zero)
    qc_df.loc[(qc_df['discharge'] > 1000) & (qc_df['Q_flag'] == 0), 'Q_flag'] = 2  # Suspect (extreme)

    # SSC (suspended sediment concentration) quality checks
    qc_df.loc[qc_df['ssc'].isna(), 'SSC_flag'] = 9  # Missing
    qc_df.loc[(qc_df['ssc_mg_L'] < 0.1) & (qc_df['SSC_flag'] != 9), 'SSC_flag'] = 3  # Bad
    qc_df.loc[(qc_df['ssc_mg_L'] > 3000) & (qc_df['SSC_flag'] == 0), 'SSC_flag'] = 2  # Suspect
    qc_df.loc[(qc_df['ssc_mg_L'] < 0) & (qc_df['SSC_flag'] != 9), 'SSC_flag'] = 3  # Bad (negative)

    # SSL (sediment load) quality checks
    qc_df.loc[qc_df['sediment_load'].isna(), 'SSL_flag'] = 9  # Missing
    qc_df.loc[(qc_df['sediment_load'] < 0) & (qc_df['SSL_flag'] != 9), 'SSL_flag'] = 3  # Bad

    return qc_df


def create_cf18_netcdf(filepath, data, station_name, source_id=None):
    """
    Create CF-1.8 and ACDD-1.3 compliant NetCDF file with quality flags.

    Parameters:
    -----------
    filepath : str
        Output file path
    data : pd.DataFrame
        Data with QC flags
    station_name : str
        Station name
    source_id : str
        Source identifier
    """
    # Extract station metadata
    lat = data['latitude'].iloc[0]
    lon = data['longitude'].iloc[0]
    depth = data['depth'].iloc[0]

    # Create output file
    ds = nc.Dataset(filepath, 'w', format='NETCDF4')

    # Create UNLIMITED time dimension
    ds.createDimension('time', None)

    # Create time variable
    time_var = ds.createVariable('time', 'f8', ('time',))
    time_var.standard_name = 'time'
    time_var.long_name = 'time'
    time_var.units = f'days since {data["datetime"].min().strftime("%Y-%m-%d")} 00:00:00'
    time_var.calendar = 'gregorian'
    time_var.axis = 'T'

    # Convert to days since reference date
    ref_date = pd.Timestamp(data['datetime'].min().strftime('%Y-%m-%d'))
    time_values = [(d - ref_date).total_seconds() / 86400.0 for d in data['datetime']]
    time_var[:] = time_values

    # Create coordinate variables
    lat_var = ds.createVariable('lat', 'f4')
    lat_var.standard_name = 'latitude'
    lat_var.long_name = 'station latitude'
    lat_var.units = 'degrees_north'
    lat_var.valid_range = np.array([-90.0, 90.0], dtype='f4')
    lat_var[:] = lat

    lon_var = ds.createVariable('lon', 'f4')
    lon_var.standard_name = 'longitude'
    lon_var.long_name = 'station longitude'
    lon_var.units = 'degrees_east'
    lon_var.valid_range = np.array([-180.0, 180.0], dtype='f4')
    lon_var[:] = lon

    alt_var = ds.createVariable('altitude', 'f4')
    alt_var.standard_name = 'altitude'
    alt_var.long_name = 'station elevation above sea level'
    alt_var.units = 'm'
    alt_var.positive = 'up'
    alt_var._FillValue = -9999.0
    alt_var.comment = f'Sampling depth: {depth} m below water surface.'
    alt_var[:] = -depth

    area_var = ds.createVariable('upstream_area', 'f4')
    area_var.long_name = 'upstream drainage area'
    area_var.units = 'km2'
    area_var._FillValue = -9999.0
    area_var.comment = 'Not available in source data'
    area_var[:] = -9999.0

    # Create data variables with ancillary flag variables
    Q_var = ds.createVariable('Q', 'f4', ('time',), fill_value=-9999.0, zlib=True, complevel=4)
    Q_var.standard_name = 'water_volume_transport_in_river_channel'
    Q_var.long_name = 'river discharge'
    Q_var.units = 'm3 s-1'
    Q_var.coordinates = 'time lat lon'
    Q_var.ancillary_variables = 'Q_flag'
    Q_var.comment = 'Source: Original data from Feng et al. (2022). Unit: m³/s.'
    Q_var[:] = data['discharge'].fillna(-9999.0).values

    Q_flag_var = ds.createVariable('Q_flag', 'i1', ('time',), fill_value=9)
    Q_flag_var.long_name = 'quality flag for river discharge'
    Q_flag_var.standard_name = 'status_flag'
    Q_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.int8)
    Q_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
    Q_flag_var.comment = 'Flag: 0=Good, 1=Estimated, 2=Suspect (zero/extreme), 3=Bad (negative), 9=Missing'
    Q_flag_var[:] = data['Q_flag'].values

    SSC_var = ds.createVariable('SSC', 'f4', ('time',), fill_value=-9999.0, zlib=True, complevel=4)
    SSC_var.standard_name = 'mass_concentration_of_suspended_matter_in_water'
    SSC_var.long_name = 'suspended sediment concentration'
    SSC_var.units = 'mg L-1'
    SSC_var.coordinates = 'time lat lon'
    SSC_var.ancillary_variables = 'SSC_flag'
    SSC_var.comment = 'Source: Original data from Feng et al. (2022). Converted g/L → mg/L (×1000).'
    SSC_var[:] = data['ssc_mg_L'].fillna(-9999.0).values

    SSC_flag_var = ds.createVariable('SSC_flag', 'i1', ('time',), fill_value=9)
    SSC_flag_var.long_name = 'quality flag for suspended sediment concentration'
    SSC_flag_var.standard_name = 'status_flag'
    SSC_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.int8)
    SSC_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
    SSC_flag_var.comment = 'Flag: 0=Good, 1=Estimated, 2=Suspect (extreme), 3=Bad (invalid), 9=Missing'
    SSC_flag_var[:] = data['SSC_flag'].values

    SSL_var = ds.createVariable('SSL', 'f4', ('time',), fill_value=-9999.0, zlib=True, complevel=4)
    SSL_var.long_name = 'suspended sediment load'
    SSL_var.units = 'ton day-1'
    SSL_var.coordinates = 'time lat lon'
    SSL_var.ancillary_variables = 'SSL_flag'
    SSL_var.comment = 'Calculated: SSL = Q (m³/s) × SSC (g/L) × 86.4 = ton/day'
    SSL_var[:] = data['sediment_load'].fillna(-9999.0).values

    SSL_flag_var = ds.createVariable('SSL_flag', 'i1', ('time',), fill_value=9)
    SSL_flag_var.long_name = 'quality flag for suspended sediment load'
    SSL_flag_var.standard_name = 'status_flag'
    SSL_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.int8)
    SSL_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
    SSL_flag_var.comment = 'Flag: 0=Good, 1=Estimated, 2=Suspect, 3=Bad (negative), 9=Missing'
    SSL_flag_var[:] = data['SSL_flag'].values

    # Global attributes - CF-1.8 and ACDD-1.3 compliant
    ds.Conventions = 'CF-1.8, ACDD-1.3'
    ds.title = 'Harmonized Global River Discharge and Sediment'

    time_span = f"{data['datetime'].min().strftime('%Y-%m-%d')} to {data['datetime'].max().strftime('%Y-%m-%d')}"
    ds.summary = f'Daily average discharge and sediment data for {station_name} on Niida River, Fukushima, Japan ({time_span}). Quality-flagged.'
    ds.source = 'In-situ station data'
    ds.data_source_name = 'Fukushima Niida River Dataset'

    # Station information
    ds.station_name = station_name
    ds.river_name = 'Niida River'
    if source_id:
        ds.Source_ID = source_id

    # Geospatial and temporal metadata
    ds.geospatial_lat_min = lat
    ds.geospatial_lat_max = lat
    ds.geospatial_lon_min = lon
    ds.geospatial_lon_max = lon
    ds.geographic_coverage = 'Niida River Basin, Fukushima Prefecture, Japan'
    ds.time_coverage_start = data['datetime'].min().strftime('%Y-%m-%d')
    ds.time_coverage_end = data['datetime'].max().strftime('%Y-%m-%d')
    ds.Temporal_Resolution = 'daily'
    ds.Variables_Provided = 'Q, SSC, SSL, altitude, upstream_area'

    # References
    ds.references = 'DOI: 10.34355/CRiED.U.Tsukuba.00147'
    ds.reference1 = 'Feng, B., Onda, Y., Wakiyama, Y., Taniguchi, K., Hashimoto, A., & Zhang, Y. (2022). Dataset of water discharge and suspended sediment at Niida river basin. University of Tsukuba. https://doi.org/10.34355/CRiED.U.Tsukuba.00147'

    # Creator information
    ds.creator_institution = 'University of Tsukuba'
    ds.contributor_name = 'Zhongwang Wei'
    ds.contributor_institution = 'Sun Yat-sen University, China'
    ds.contributor_role = 'Data QC and metadata enhancement'

    # Data provenance
    ds.history = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')} - Created by process_fukushima_doi00147.py; Quality control flags (Q_flag, SSC_flag, SSL_flag) added; CF-1.8 metadata applied"
    ds.processing_level = '3 - Derived daily data with quality flags'

    ds.close()


def main():
    """Main processing function."""

    # Configuration
    source_excel = '/Users/zhongwangwei/Downloads/Sediment/Source/Fukushima/DOI00147_data.xls'
    output_dir = '/Users/zhongwangwei/Downloads/Sediment/Output_r/daily/Fukushima'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 90)
    print("FUKUSHIMA NIIDA RIVER DATA PROCESSING")
    print("DOI: 10.34355/CRiED.U.Tsukuba.00147")
    print("=" * 90)
    print()

    # Read source data
    print("Reading Excel source file...")
    try:
        station_data = read_fukushima_excel_data(source_excel)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        print("Note: If xlrd version error occurs, use existing NC files in Output/ directory")
        return

    print(f"\nFound {len(station_data)} stations")

    # Process each station
    success_count = 0

    for station_name, data in station_data.items():
        print(f"\n{station_name}:")
        print(f"  Raw records: {len(data)}")

        # Aggregate to daily
        daily_data = aggregate_to_daily(data)
        print(f"  Daily records: {len(daily_data)}")

        # Apply quality control
        qc_data = perform_quality_control(daily_data)

        # Count good data
        Q_good = (qc_data['Q_flag'] == 0).sum()
        SSC_good = (qc_data['SSC_flag'] == 0).sum()

        print(f"  Good data - Q: {Q_good}/{len(qc_data)} ({Q_good/len(qc_data)*100:.1f}%)")
        print(f"  Good data - SSC: {SSC_good}/{len(qc_data)} ({SSC_good/len(qc_data)*100:.1f}%)")

        # Create output file
        safe_name = station_name.replace(' ', '_').replace('/', '_')
        source_id = f"DOI00147_{safe_name}"
        output_file = os.path.join(output_dir, f"Fukushima_{safe_name}.nc")

        try:
            create_cf18_netcdf(output_file, qc_data, station_name, source_id)
            print(f"  ✓ Created: {output_file}")
            success_count += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print(f"\n{'=' * 90}")
    print(f"Complete! Processed {success_count} stations")
    print(f"Output: {output_dir}")
    print(f"{'=' * 90}")


if __name__ == '__main__':
    main()
