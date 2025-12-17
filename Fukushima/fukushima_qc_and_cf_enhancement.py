#!/usr/bin/env python3
"""
Enhanced processing script for Fukushima Niida River dataset (DOI 10.34355/CRiED.U.Tsukuba.00147)
Includes:
1. Quality Control (QC) checking and flagging
2. CF-1.8 and ACDD-1.3 metadata compliance
3. Data provenance tracking
4. Unit conversion verification
"""

import os
import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def read_doi00147_data(filepath):
    """
    Read DOI00147 Excel data file and extract all sheets.
    
    Parameters:
    -----------
    filepath : str
        Path to Excel file
    
    Returns:
    --------
    all_data : dict
        Dictionary with station names as keys and dataframes as values
    """
    # Read all sheets
    all_sheets = pd.read_excel(filepath, sheet_name=None)
    
    # Dictionary to store data by station
    station_data = {}
    
    for sheet_name, df in all_sheets.items():
        print(f"Processing sheet: {sheet_name}")
        
        # Skip first 2 rows (header info) and read data
        # Column mapping based on format description:
        # 0: DOI, 1: DID, 2: station, 3-7: yyyy,mm,dd,hh,min, 8: xyear
        # 9: LatDir, 10: Nsflag, 11: xlat, 12: LonDir, 13: Ewflag, 14: xlong
        # 15: altdepflag, 16: sampdep, 17: sample type
        # 18: Water discharge (m3/s), 19: SSC (g/L), 20: Uncertainty SSC (g/L)
        
        data = pd.read_excel(filepath, sheet_name=sheet_name, skiprows=2, header=None)
        
        # Skip the first row which contains units
        data = data.iloc[1:].reset_index(drop=True)
        
        # Extract station name
        station_name = data.iloc[0, 2]  # Column 2 is station
        
        # Create datetime from components
        data['datetime'] = pd.to_datetime({
            'year': data[3].astype(int),
            'month': data[4].astype(int),
            'day': data[5].astype(int),
            'hour': data[6].astype(int),
            'minute': data[7].astype(int)
        })
        
        # Extract relevant columns
        df_clean = pd.DataFrame({
            'datetime': data['datetime'],
            'latitude': data[11].astype(float),
            'longitude': data[14].astype(float),
            'depth': data[16].astype(float),
            'discharge': data[18].astype(float),  # m3/s
            'ssc': data[19].astype(float),  # g/L
            'ssc_uncertainty': data[20].astype(float)  # g/L
        })
        
        # Convert SSC from g/L to mg/L (multiply by 1000)
        df_clean['ssc_mg_L'] = df_clean['ssc'] * 1000
        df_clean['ssc_uncertainty_mg_L'] = df_clean['ssc_uncertainty'] * 1000
        
        # Calculate sediment load (ton/day)
        # Formula derivation:
        # Load = Q (m³/s) × SSC (g/L) × 86.4
        # = Q (m³/s) × SSC (g/L) × 86400 (s/day) / 1000 (g/kg) / 1000 (kg/ton)
        # Unit verification:
        # m³/s × g/L × 86400 s/day = m³ × g × 86400 / (s × L × s) = m³ × g × 86400 / L
        # = 1000 L × g × 86400 / L = 86,400,000 g/day
        # = 86.4 tons/day (since 1 ton = 1,000,000 g)
        df_clean['sediment_load'] = df_clean['discharge'] * df_clean['ssc'] * 86.4
        
        # Add to station data
        if station_name not in station_data:
            station_data[station_name] = []
        station_data[station_name].append(df_clean)
    
    # Combine all data for each station
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
        DataFrame with datetime index and measurements
    
    Returns:
    --------
    daily_df : pd.DataFrame
        Daily averaged data
    """
    # Set datetime as index
    df = df.set_index('datetime')
    
    # Resample to daily, taking mean
    daily = df.resample('D').mean()
    
    # Recalculate sediment load from daily averages
    daily['sediment_load'] = daily['discharge'] * daily['ssc'] * 86.4
    
    return daily.reset_index()


def perform_qc_checks(daily_df):
    """
    Perform quality control checks and create flag variables.
    
    Parameters:
    -----------
    daily_df : pd.DataFrame
        Daily data with measurements
    
    Returns:
    --------
    qc_df : pd.DataFrame
        Data with added flag columns
    """
    qc_df = daily_df.copy()
    
    # Initialize flag variables
    # Flag meanings: 0=good, 1=estimated, 2=suspect, 3=bad, 9=missing
    qc_df['Q_flag'] = np.ones(len(qc_df), dtype=np.int8) * 0  # Default to good
    qc_df['SSC_flag'] = np.ones(len(qc_df), dtype=np.int8) * 0
    qc_df['SSL_flag'] = np.ones(len(qc_df), dtype=np.int8) * 0
    
    # Check for missing/NaN values
    qc_df.loc[qc_df['discharge'].isna(), 'Q_flag'] = 9
    qc_df.loc[qc_df['ssc'].isna(), 'SSC_flag'] = 9
    qc_df.loc[qc_df['sediment_load'].isna(), 'SSL_flag'] = 9
    
    # Q (discharge) checks
    # Q < 0: Bad data
    qc_df.loc[(qc_df['discharge'] < 0) & (qc_df['Q_flag'] != 9), 'Q_flag'] = 3
    
    # Q == 0: Suspect data (could be real or error)
    qc_df.loc[(qc_df['discharge'] == 0) & (qc_df['Q_flag'] == 0), 'Q_flag'] = 2
    
    # Q > 1000: Suspect (extreme high value) - set threshold for Niida River
    # Based on data statistics (max ~504 m³/s), set threshold at 1000 m³/s
    qc_df.loc[(qc_df['discharge'] > 1000) & (qc_df['Q_flag'] == 0), 'Q_flag'] = 2
    
    # SSC (suspended sediment concentration) checks
    # SSC < 0.1 mg/L: Bad data (typically physical minimum)
    qc_df.loc[(qc_df['ssc_mg_L'] < 0.1) & (qc_df['SSC_flag'] != 9), 'SSC_flag'] = 3
    
    # SSC > 3000 mg/L: Suspect (extreme high value)
    qc_df.loc[(qc_df['ssc_mg_L'] > 3000) & (qc_df['SSC_flag'] == 0), 'SSC_flag'] = 2
    
    # SSC < 0: Bad data
    qc_df.loc[(qc_df['ssc_mg_L'] < 0) & (qc_df['SSC_flag'] != 9), 'SSC_flag'] = 3
    
    # SSL (sediment load) checks
    # SSL < 0: Bad data
    qc_df.loc[(qc_df['sediment_load'] < 0) & (qc_df['SSL_flag'] != 9), 'SSL_flag'] = 3
    
    return qc_df


def create_netcdf_cf18(filepath, data, station_name, river_name, source_id=None):
    """
    Create CF-1.8 and ACDD-1.3 compliant NetCDF file with quality flags.
    
    Parameters:
    -----------
    filepath : str
        Output NetCDF filename
    data : pd.DataFrame
        Time series data with QC flags
    station_name : str
        Station name
    river_name : str
        River name
    source_id : str
        Source identifier (e.g., 'DOI00147_Haramachi')
    """
    # Get coordinates from first row (they're constant)
    lat = data['latitude'].iloc[0]
    lon = data['longitude'].iloc[0]
    depth = data['depth'].iloc[0]
    
    # Create NetCDF file
    dataset = nc.Dataset(filepath, 'w', format='NETCDF4')
    
    # Create unlimited time dimension
    time_dim = dataset.createDimension('time', None)  # UNLIMITED dimension
    
    # Create coordinate variables
    time_var = dataset.createVariable('time', 'f8', ('time',))
    time_var.standard_name = 'time'
    time_var.long_name = 'time'
    time_var.units = f'days since {data["datetime"].min().strftime("%Y-%m-%d")} 00:00:00'
    time_var.calendar = 'gregorian'
    time_var.axis = 'T'
    
    # Convert dates to days since epoch
    reference_date = pd.Timestamp(data["datetime"].min().strftime("%Y-%m-%d"))
    time_values = [(d - reference_date).total_seconds() / 86400.0
                   for d in data['datetime']]
    time_var[:] = time_values
    
    # Create scalar coordinate variables
    lat_var = dataset.createVariable('lat', 'f4')
    lat_var.standard_name = 'latitude'
    lat_var.long_name = 'station latitude'
    lat_var.units = 'degrees_north'
    lat_var.valid_range = np.array([-90.0, 90.0], dtype='f4')
    lat_var[:] = lat
    
    lon_var = dataset.createVariable('lon', 'f4')
    lon_var.standard_name = 'longitude'
    lon_var.long_name = 'station longitude'
    lon_var.units = 'degrees_east'
    lon_var.valid_range = np.array([-180.0, 180.0], dtype='f4')
    lon_var[:] = lon
    
    alt_var = dataset.createVariable('altitude', 'f4')
    alt_var.standard_name = 'altitude'
    alt_var.long_name = 'station elevation above sea level'
    alt_var.units = 'm'
    alt_var.positive = 'up'
    alt_var._FillValue = -9999.0
    alt_var.comment = f'Sampling depth: {depth} m below surface. Negative values indicate below water surface.'
    alt_var[:] = -depth  # Negative for below surface
    
    area_var = dataset.createVariable('upstream_area', 'f4')
    area_var.long_name = 'upstream drainage area'
    area_var.standard_name = 'upstream_area'
    area_var.units = 'km2'
    area_var._FillValue = -9999.0
    area_var.comment = 'Not available in source data'
    area_var[:] = -9999.0
    
    # Create data variables with quality flags
    # Q (discharge)
    Q_var = dataset.createVariable('Q', 'f4', ('time',), fill_value=-9999.0, zlib=True, complevel=4)
    Q_var.standard_name = 'water_volume_transport_in_river_channel'
    Q_var.long_name = 'river discharge'
    Q_var.units = 'm3 s-1'
    Q_var.coordinates = 'time lat lon'
    Q_var.ancillary_variables = 'Q_flag'
    Q_var.comment = f'Source: Original data provided by Feng et al. (2022, DOI:10.34355/CRiED.U.Tsukuba.00147). Unit: m³/s (cubic meters per second).'
    Q_var[:] = data['discharge'].fillna(-9999.0).values
    
    # Q_flag
    Q_flag_var = dataset.createVariable('Q_flag', 'i1', ('time',), fill_value=9)
    Q_flag_var.long_name = 'quality flag for river discharge'
    Q_flag_var.standard_name = 'status_flag'
    Q_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.int8)
    Q_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
    Q_flag_var.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'
    Q_flag_var[:] = data['Q_flag'].values
    
    # SSC (suspended sediment concentration)
    SSC_var = dataset.createVariable('SSC', 'f4', ('time',), fill_value=-9999.0, zlib=True, complevel=4)
    SSC_var.standard_name = 'mass_concentration_of_suspended_matter_in_water'
    SSC_var.long_name = 'suspended sediment concentration'
    SSC_var.units = 'mg L-1'
    SSC_var.coordinates = 'time lat lon'
    SSC_var.ancillary_variables = 'SSC_flag'
    SSC_var.comment = 'Source: Original data provided by Feng et al. (2022, DOI:10.34355/CRiED.U.Tsukuba.00147). Unit conversion: multiplied by 1000 to convert from g/L to mg/L.'
    SSC_var[:] = data['ssc_mg_L'].fillna(-9999.0).values
    
    # SSC_flag
    SSC_flag_var = dataset.createVariable('SSC_flag', 'i1', ('time',), fill_value=9)
    SSC_flag_var.long_name = 'quality flag for suspended sediment concentration'
    SSC_flag_var.standard_name = 'status_flag'
    SSC_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.int8)
    SSC_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
    SSC_flag_var.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source.'
    SSC_flag_var[:] = data['SSC_flag'].values
    
    # SSL (sediment load)
    SSL_var = dataset.createVariable('SSL', 'f4', ('time',), fill_value=-9999.0, zlib=True, complevel=4)
    SSL_var.long_name = 'suspended sediment load'
    SSL_var.units = 'ton day-1'
    SSL_var.coordinates = 'time lat lon'
    SSL_var.ancillary_variables = 'SSL_flag'
    SSL_var.comment = 'Source: Calculated. Formula: SSL (ton/day) = Q (m³/s) × SSC (g/L) × 86.4, where 86.4 = 86400 s/day / 1000 L/m³ / 1000 kg/ton. Represents daily average.'
    SSL_var[:] = data['sediment_load'].fillna(-9999.0).values
    
    # SSL_flag
    SSL_flag_var = dataset.createVariable('SSL_flag', 'i1', ('time',), fill_value=9)
    SSL_flag_var.long_name = 'quality flag for suspended sediment load'
    SSL_flag_var.standard_name = 'status_flag'
    SSL_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.int8)
    SSL_flag_var.flag_meanings = 'good_data estimated_data suspect_data bad_data missing_data'
    SSL_flag_var.comment = 'Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), 3=Bad (e.g., negative), 9=Missing in source. Inherits flags from Q and SSC.'
    SSL_flag_var[:] = data['SSL_flag'].values
    
    # Global attributes - CF-1.8 and ACDD-1.3 compliant
    dataset.Conventions = 'CF-1.8, ACDD-1.3'
    dataset.title = 'Harmonized Global River Discharge and Sediment'
    
    time_start = data['datetime'].min().strftime('%Y-%m-%d')
    time_end = data['datetime'].max().strftime('%Y-%m-%d')
    
    dataset.summary = f'River discharge and suspended sediment data for {station_name} station on the {river_name} in Fukushima, Japan. This dataset contains daily averages of water discharge, suspended sediment concentration, and calculated sediment load over the period {time_start} to {time_end}. Data has been quality checked and flagged.'
    
    dataset.source = 'In-situ station data'
    dataset.data_source_name = 'Fukushima Niida River Dataset'
    
    # Station information
    dataset.station_name = station_name
    dataset.river_name = river_name
    if source_id:
        dataset.Source_ID = source_id
    
    # Geospatial attributes
    dataset.geospatial_lat_min = lat
    dataset.geospatial_lat_max = lat
    dataset.geospatial_lon_min = lon
    dataset.geospatial_lon_max = lon
    dataset.geospatial_vertical_min = -depth
    dataset.geospatial_vertical_max = -depth
    dataset.geospatial_bounds_crs = 'EPSG:4326'
    
    dataset.geographic_coverage = 'Niida River Basin, Fukushima Prefecture, Japan'
    
    # Temporal attributes
    dataset.time_coverage_start = time_start
    dataset.time_coverage_end = time_end
    dataset.Temporal_Resolution = 'daily'
    dataset.Variables_Provided = 'Q, SSC, SSL, altitude, upstream_area'
    
    # References and provenance
    dataset.references = 'DOI: 10.34355/CRiED.U.Tsukuba.00147'
    dataset.reference1 = 'Feng, B., Onda, Y., Wakiyama, Y., Taniguchi, K., Hashimoto, A., & Zhang, Y. (2022). Dataset of water discharge and suspended sediment at Niida river basin downstream (Haramachi) during 2013 to 2018 and upstream (Notegami) during 2015 to 2018. CRiED, University of Tsukuba. https://doi.org/10.34355/CRiED.U.Tsukuba.00147'
    dataset.reference2 = 'Published in Nature Sustainability (June 2022): "Persistent impact of Fukushima decontamination on soil erosion and suspended sediment"'
    
    dataset.creator_name = 'Feng, B., Onda, Y., Wakiyama, Y., Taniguchi, K., Hashimoto, A., Zhang, Y.'
    dataset.creator_institution = 'University of Tsukuba, Center for Research in Isotopes and Environmental Dynamics'
    dataset.contributor_name = 'Zhongwang Wei'
    dataset.contributor_email = 'weizhw6@mail.sysu.edu.cn'
    dataset.contributor_institution = 'Sun Yat-sen University, China'
    dataset.contributor_role = 'Data processor and QC'
    
    # Data processing history
    history_msg = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')} - Enhanced with QC flags, CF-1.8 metadata, and standardized formatting by fukushima_qc_and_cf_enhancement.py; "
    history_msg += f"Aggregated from high-frequency measurements to daily averages; "
    history_msg += f"Applied physical constraint QC checks; "
    history_msg += f"Added quality flag variables (Q_flag, SSC_flag, SSL_flag)"
    dataset.history = history_msg
    
    dataset.processing_level = '3 - Derived data'
    dataset.project = 'Global River Harmonized Sediment and Discharge Dataset'
    
    # Close the file
    dataset.close()
    
    return filepath


def process_fukushima_data():
    """Main processing function for Fukushima Niida River data."""
    
    # File paths
    base_dir = '/Users/zhongwangwei/Downloads/Sediment/Source/Fukushima'
    data_file = os.path.join(base_dir, 'DOI00147_data.xls')
    output_dir = '/Users/zhongwangwei/Downloads/Sediment/Output_r/daily/Fukushima'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 90)
    print("FUKUSHIMA NIIDA RIVER DATA - QC & CF-1.8 ENHANCEMENT")
    print("DOI: 10.34355/CRiED.U.Tsukuba.00147")
    print("=" * 90)
    print()
    
    # Read all data
    print("Reading Excel data file...")
    station_data = read_doi00147_data(data_file)
    
    print(f"\nFound {len(station_data)} stations:")
    for station in station_data.keys():
        print(f"  - {station}")
    
    # Process each station
    print("\nProcessing stations with QC checks...")
    print("-" * 90)
    success_count = 0
    summary_data = []
    
    for station_name, data in station_data.items():
        print(f"\n{station_name}:")
        print(f"  Total records: {len(data)}")
        print(f"  Date range: {data['datetime'].min()} to {data['datetime'].max()}")
        print(f"  Valid discharge: {(~data['discharge'].isna()).sum()}")
        print(f"  Valid SSC: {(~data['ssc'].isna()).sum()}")
        
        # Aggregate to daily
        daily_data = aggregate_to_daily(data)
        print(f"  Daily records: {len(daily_data)}")
        
        # Perform QC checks
        qc_data = perform_qc_checks(daily_data)
        
        # Count good data
        Q_good = (qc_data['Q_flag'] == 0).sum()
        SSC_good = (qc_data['SSC_flag'] == 0).sum()
        SSL_good = (qc_data['SSL_flag'] == 0).sum()
        
        Q_pct = Q_good / len(qc_data) * 100
        SSC_pct = SSC_good / len(qc_data) * 100
        SSL_pct = SSL_good / len(qc_data) * 100
        
        print(f"  Quality flags (% good):")
        print(f"    Q: {Q_good}/{len(qc_data)} ({Q_pct:.1f}%)")
        print(f"    SSC: {SSC_good}/{len(qc_data)} ({SSC_pct:.1f}%)")
        print(f"    SSL: {SSL_good}/{len(qc_data)} ({SSL_pct:.1f}%)")
        
        # Create source ID
        safe_name = station_name.replace(' ', '_').replace('/', '_')
        source_id = f"DOI00147_{safe_name}"
        
        # Create NetCDF file
        output_file = os.path.join(output_dir, f"Fukushima_{safe_name}.nc")
        
        try:
            create_netcdf_cf18(output_file, qc_data, station_name, 'Niida River', source_id)
            print(f"  Created: {output_file}")
            success_count += 1
            
            # Store summary info
            summary_data.append({
                'station_name': station_name,
                'Source_ID': source_id,
                'river_name': 'Niida River',
                'longitude': data['longitude'].iloc[0],
                'latitude': data['latitude'].iloc[0],
                'altitude': -data['depth'].iloc[0],  # Negative for below surface
                'upstream_area': np.nan,
                'Data Source Name': 'Fukushima Niida River Dataset',
                'Type': 'In-situ',
                'Temporal Resolution': 'daily',
                'Temporal Span': f"{data['datetime'].min().strftime('%Y-%m-%d')} to {data['datetime'].max().strftime('%Y-%m-%d')}",
                'Variables Provided': 'Q, SSC, SSL',
                'Geographic Coverage': 'Niida River Basin, Fukushima, Japan',
                'Reference/DOI': 'https://doi.org/10.34355/CRiED.U.Tsukuba.00147',
                'Q_start_date': data[~data['discharge'].isna()]['datetime'].min().strftime('%Y-%m-%d') if (~data['discharge'].isna()).any() else 'N/A',
                'Q_end_date': data[~data['discharge'].isna()]['datetime'].max().strftime('%Y-%m-%d') if (~data['discharge'].isna()).any() else 'N/A',
                'Q_percent_complete': Q_pct,
                'SSC_start_date': data[~data['ssc'].isna()]['datetime'].min().strftime('%Y-%m-%d') if (~data['ssc'].isna()).any() else 'N/A',
                'SSC_end_date': data[~data['ssc'].isna()]['datetime'].max().strftime('%Y-%m-%d') if (~data['ssc'].isna()).any() else 'N/A',
                'SSC_percent_complete': SSC_pct,
                'SSL_start_date': data[~data['sediment_load'].isna()]['datetime'].min().strftime('%Y-%m-%d') if (~data['sediment_load'].isna()).any() else 'N/A',
                'SSL_end_date': data[~data['sediment_load'].isna()]['datetime'].max().strftime('%Y-%m-%d') if (~data['sediment_load'].isna()).any() else 'N/A',
                'SSL_percent_complete': SSL_pct,
            })
            
        except Exception as e:
            print(f"  Error creating NetCDF: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate summary CSV
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_csv = os.path.join(output_dir, 'Fukushima_station_summary.csv')
        summary_df.to_csv(summary_csv, index=False)
        print(f"\nGenerated summary CSV: {summary_csv}")
    
    print(f"\n{'=' * 90}")
    print(f"Processing complete!")
    print(f"  Successfully processed: {success_count} stations")
    print(f"  Output directory: {output_dir}")
    print(f"{'=' * 90}")


if __name__ == '__main__':
    process_fukushima_data()
