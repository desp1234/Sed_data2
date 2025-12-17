#!/usr/bin/env python3
"""
Convert Rhine River sediment data from txt to NetCDF format.
"""

import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime
import os
from pathlib import Path

def read_spm_data(file_path):
    """Read the SPM (suspended particulate matter) data."""
    df = pd.read_csv(file_path, sep=' ', quotechar='"')
    df['date'] = pd.to_datetime(df['date'])
    df.columns = df.columns.str.strip()
    return df

def read_mpm_data(file_path):
    """Read the MPM (moving point measurement) data."""
    df = pd.read_csv(file_path, sep=' ', quotechar='"')
    df['date'] = pd.to_datetime(df['date'])
    df.columns = df.columns.str.strip()
    return df

def process_station_data(station_name, spm_df, mpm_df):
    """
    Process data for a single station.

    Returns:
        dict: Processed data dictionary or None if station should be excluded
    """
    # Get station data from both sources
    spm_station = spm_df[spm_df['station'] == station_name].copy()
    mpm_station = mpm_df[mpm_df['station'] == station_name].copy()

    # Combine data sources
    # For SPM data, we already have SSC and discharge
    spm_station_clean = spm_station[['date', 'suspended_sediment_concentration [mg/L]',
                                       'discharge [m3/s]', 'latitude [WGS84]',
                                       'longitude [WGS84]']].copy()
    spm_station_clean.columns = ['date', 'ssc', 'discharge', 'latitude', 'longitude']

    # For MPM data, average by date to get daily values
    if len(mpm_station) > 0:
        mpm_grouped = mpm_station.groupby('date').agg({
            'discharge [m3/s]': 'first',  # Discharge is same for all measurements on same day
            'suspended_sediment_concentration [mg/l]': 'mean',  # Average SSC across depths
            'latitude [WGS84]': 'first',
            'longitude [WGS84]': 'first'
        }).reset_index()
        mpm_grouped.columns = ['date', 'discharge', 'ssc', 'latitude', 'longitude']

        # Combine SPM and MPM data
        combined = pd.concat([spm_station_clean, mpm_grouped], ignore_index=True)
    else:
        combined = spm_station_clean

    # Remove duplicates and handle NA values
    combined = combined.drop_duplicates(subset=['date'])
    combined = combined.replace('NA', np.nan)
    combined['ssc'] = pd.to_numeric(combined['ssc'], errors='coerce')
    combined['discharge'] = pd.to_numeric(combined['discharge'], errors='coerce')

    # Check if we have valid data
    if len(combined) == 0:
        print(f"  No data for station {station_name}")
        return None

    # Remove rows where both sediment and discharge are NaN
    combined = combined.dropna(subset=['ssc', 'discharge'], how='all')

    if len(combined) == 0:
        print(f"  All data is NaN for station {station_name}")
        return None

    # Find years with both sediment and discharge data
    combined['year'] = combined['date'].dt.year
    combined['month'] = combined['date'].dt.month

    # Get years with sediment data (not all NaN)
    sediment_years = set(combined[combined['ssc'].notna()]['year'].unique())
    # Get years with discharge data (not all NaN)
    discharge_years = set(combined[combined['discharge'].notna()]['year'].unique())

    # Find overlapping years
    overlap_years = sediment_years.intersection(discharge_years)

    if len(overlap_years) == 0:
        print(f"  No overlapping years for station {station_name}")
        return None

    # Get start and end years
    start_year = min(overlap_years)
    end_year = max(overlap_years)

    print(f"  {station_name}: {start_year}-{end_year} ({len(overlap_years)} years with overlap)")

    # Create monthly time series from start_year-01 to end_year-12
    start_date = pd.Timestamp(f'{start_year}-01-01')
    end_date = pd.Timestamp(f'{end_year}-12-31')
    monthly_dates = pd.date_range(start=start_date, end=end_date, freq='MS')  # Month start

    # Create empty dataframe with monthly dates
    monthly_data = pd.DataFrame({'date': monthly_dates})
    monthly_data['year'] = monthly_data['date'].dt.year
    monthly_data['month'] = monthly_data['date'].dt.month

    # For each month, average the available data
    monthly_ssc = []
    monthly_discharge = []

    for idx, row in monthly_data.iterrows():
        month_data = combined[(combined['year'] == row['year']) &
                             (combined['month'] == row['month'])]

        if len(month_data) > 0:
            # Average SSC and discharge for the month
            ssc_val = month_data['ssc'].mean() if month_data['ssc'].notna().any() else np.nan
            discharge_val = month_data['discharge'].mean() if month_data['discharge'].notna().any() else np.nan
        else:
            ssc_val = np.nan
            discharge_val = np.nan

        monthly_ssc.append(ssc_val)
        monthly_discharge.append(discharge_val)

    monthly_data['ssc'] = monthly_ssc
    monthly_data['discharge'] = monthly_discharge

    # Check if all sediment or all discharge is NaN
    if monthly_data['ssc'].isna().all() or monthly_data['discharge'].isna().all():
        print(f"  All monthly values are NaN for station {station_name}")
        return None

    # Calculate sediment load: Load (ton/day) = Q (m³/s) × SSC (mg/L) × 86.4
    # Note: 86.4 = 86400 seconds/day × 1 ton/1e6 mg × 1000 L/m³ / 1000
    monthly_data['sediment_load'] = monthly_data['discharge'] * monthly_data['ssc'] * 86.4

    # Get station metadata
    station_info = combined.iloc[0]

    return {
        'station_name': station_name,
        'data': monthly_data,
        'latitude': float(station_info['latitude']),
        'longitude': float(station_info['longitude']),
        'start_year': start_year,
        'end_year': end_year
    }

def create_netcdf(station_data, output_dir):
    """Create NetCDF file for a station."""
    station_name = station_data['station_name']
    data = station_data['data']

    # Create output filename
    output_file = os.path.join(output_dir, f'Rhine_{station_name}.nc')

    # Create NetCDF file
    with nc.Dataset(output_file, 'w', format='NETCDF4') as ncfile:
        # Create dimensions
        time_dim = ncfile.createDimension('time', len(data))

        # Create coordinate variables
        time_var = ncfile.createVariable('time', 'f8', ('time',))
        time_var.standard_name = 'time'
        time_var.long_name = 'time of measurement'
        time_var.units = 'days since 1970-01-01 00:00:00'
        time_var.calendar = 'gregorian'
        time_var.axis = 'T'

        # Convert dates to days since 1970-01-01
        reference_date = pd.Timestamp('1970-01-01')
        time_var[:] = (data['date'] - reference_date).dt.total_seconds() / 86400.0

        # Create spatial coordinate variables (scalars)
        lat_var = ncfile.createVariable('latitude', 'f4')
        lat_var.standard_name = 'latitude'
        lat_var.long_name = 'station latitude'
        lat_var.units = 'degrees_north'
        lat_var.valid_range = np.array([-90.0, 90.0], dtype='f4')
        lat_var[:] = station_data['latitude']

        lon_var = ncfile.createVariable('longitude', 'f4')
        lon_var.standard_name = 'longitude'
        lon_var.long_name = 'station longitude'
        lon_var.units = 'degrees_east'
        lon_var.valid_range = np.array([-180.0, 180.0], dtype='f4')
        lon_var[:] = station_data['longitude']

        # Create altitude variable (unknown, set to NaN)
        alt_var = ncfile.createVariable('altitude', 'f4')
        alt_var.standard_name = 'altitude'
        alt_var.long_name = 'station altitude above sea level'
        alt_var.units = 'm'
        alt_var[:] = np.nan

        # Create upstream area variable (unknown, set to NaN)
        area_var = ncfile.createVariable('upstream_area', 'f4')
        area_var.long_name = 'upstream drainage area'
        area_var.units = 'km2'
        area_var.comment = 'Not available for Rhine stations'
        area_var[:] = np.nan

        # Create data variables
        discharge_var = ncfile.createVariable('discharge', 'f4', ('time',),
                                              fill_value=-9999.0,
                                              chunksizes=(len(data),))
        discharge_var.standard_name = 'water_volume_transport_in_river_channel'
        discharge_var.long_name = 'river discharge'
        discharge_var.units = 'm3 s-1'
        discharge_var.coordinates = 'time latitude longitude'
        discharge_var[:] = data['discharge'].fillna(-9999.0).values

        ssc_var = ncfile.createVariable('ssc', 'f4', ('time',),
                                       fill_value=-9999.0,
                                       chunksizes=(len(data),))
        ssc_var.standard_name = 'mass_concentration_of_suspended_matter_in_water'
        ssc_var.long_name = 'suspended sediment concentration'
        ssc_var.units = 'mg L-1'
        ssc_var.coordinates = 'time latitude longitude'
        ssc_var[:] = data['ssc'].fillna(-9999.0).values

        sediment_load_var = ncfile.createVariable('sediment_load', 'f4', ('time',),
                                                  fill_value=-9999.0,
                                                  chunksizes=(len(data),))
        sediment_load_var.long_name = 'suspended sediment load'
        sediment_load_var.units = 'ton day-1'
        sediment_load_var.coordinates = 'time latitude longitude'
        sediment_load_var[:] = data['sediment_load'].fillna(-9999.0).values

        # Global attributes
        ncfile.Conventions = 'CF-1.8'
        ncfile.title = f'Rhine River Sediment and Discharge Data for Station {station_name}'
        ncfile.institution = 'Rhine River Monitoring Network'
        ncfile.source = 'In-situ observations from Rhine River monitoring stations'
        ncfile.history = f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} by convert_to_netcdf.py'
        ncfile.references = 'Slabon, A., Terweh, S. and Hoffmann, T.O. (2025), Vertical and Lateral Variability of Suspended Sediment Transport in the Rhine River. Hydrological Processes, 39: e70070. https://doi.org/10.1002/hyp.70070'
        ncfile.comment = 'Sediment load calculated as: Load = Q × SSC × 86.4 (Q in m³/s, SSC in mg/L, Load in ton/day). Monthly averaged values from original measurements.'
        ncfile.station_id = station_name
        ncfile.station_name = station_name
        ncfile.river_name = 'Rhine'
        ncfile.data_period = f'{station_data["start_year"]}-{station_data["end_year"]}'

    print(f"  Created {output_file}")

def main():
    """Main function to process all data."""
    print("Rhine River Data Conversion to NetCDF")
    print("=" * 60)

    # Read input files
    print("\nReading input files...")
    spm_file = '/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Source/Rhine/data_spm.txt'
    mpm_file = '/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Source/Rhine/data_mpm.txt'

    spm_df = read_spm_data(spm_file)
    mpm_df = read_mpm_data(mpm_file)

    print(f"  SPM data: {len(spm_df)} records")
    print(f"  MPM data: {len(mpm_df)} records")

    # Get all unique stations
    stations = set(spm_df['station'].unique()) | set(mpm_df['station'].unique())
    print(f"\nFound {len(stations)} unique stations:")
    for station in sorted(stations):
        print(f"  - {station}")

    # Create output directory
    output_dir = '/share/home/dq134/wzx/sed_data/sediment_wzx_1111/Source/Rhine/done'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Process each station
    print("\nProcessing stations...")
    successful = 0
    skipped = 0

    for station in sorted(stations):
        print(f"\nProcessing {station}...")
        station_data = process_station_data(station, spm_df, mpm_df)

        if station_data is not None:
            create_netcdf(station_data, output_dir)
            successful += 1
        else:
            print(f"  Skipped {station}")
            skipped += 1

    print("\n" + "=" * 60)
    print(f"Conversion complete!")
    print(f"  Successfully converted: {successful} stations")
    print(f"  Skipped: {skipped} stations")
    print(f"  Output location: {output_dir}")

if __name__ == '__main__':
    main()
