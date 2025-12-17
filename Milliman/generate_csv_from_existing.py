#!/usr/bin/env python3
"""
Generate CSV summary from existing Milliman NetCDF files.

This script reads the already processed NetCDF files and creates the station summary CSV.

Author: Zhongwang Wei
Date: 2025-10-25
"""

import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime
import os
import glob
import sys

def extract_station_info(input_file):
    """
    Extract station information from a NetCDF file for CSV summary.

    Parameters:
    -----------
    input_file : str
        Path to NetCDF file

    Returns:
    --------
    station_info : dict or None
        Dictionary containing station metadata
    """

    try:
        ds = nc.Dataset(input_file, 'r')

        # Read coordinates
        if 'lat' in ds.variables:
            lat = float(ds.variables['lat'][:])
            lon = float(ds.variables['lon'][:])
        elif 'latitude' in ds.variables:
            lat = float(ds.variables['latitude'][:])
            lon = float(ds.variables['longitude'][:])
        else:
            ds.close()
            return None

        # Read upstream area
        if 'upstream_area' in ds.variables:
            upstream_area = float(ds.variables['upstream_area'][:])
        elif 'drainage_area' in ds.variables:
            upstream_area = float(ds.variables['drainage_area'][:])
        else:
            upstream_area = np.nan

        # Read altitude
        alt = float(ds.variables['altitude'][:]) if 'altitude' in ds.variables else np.nan

        # Try to read data variables
        if 'Discharge' in ds.variables:
            q_val = float(ds.variables['Discharge'][0, 0, 0])
        elif 'Q' in ds.variables:
            q_val = float(ds.variables['Q'][0]) if len(ds.variables['Q'].shape) == 1 else float(ds.variables['Q'][0, 0, 0])
        else:
            q_val = np.nan

        if 'SSC' in ds.variables:
            ssc_val = float(ds.variables['SSC'][0]) if len(ds.variables['SSC'].shape) == 1 else float(ds.variables['SSC'][0, 0, 0])
        else:
            ssc_val = np.nan

        if 'TSS' in ds.variables:
            ssl_val = float(ds.variables['TSS'][0, 0, 0])
        elif 'SSL' in ds.variables:
            ssl_val = float(ds.variables['SSL'][0]) if len(ds.variables['SSL'].shape) == 1 else float(ds.variables['SSL'][0, 0, 0])
        else:
            ssl_val = np.nan

        # Read metadata
        location_id = ds.location_id if hasattr(ds, 'location_id') else os.path.basename(input_file).replace('.nc', '')
        river_name = ds.river_name if hasattr(ds, 'river_name') else ""
        country = ds.country if hasattr(ds, 'country') else ""
        continent = ds.continent_region if hasattr(ds, 'continent_region') else ""

        # Get temporal info
        if 'time' in ds.variables:
            time_vals = ds.variables['time'][:]
            time_units = ds.variables['time'].units
            time_calendar = ds.variables['time'].calendar
            dates = nc.num2date(time_vals, units=time_units, calendar=time_calendar)
            if len(dates) > 0:
                representative_year = dates[0].year
            else:
                representative_year = 2000
        else:
            representative_year = 2000

        ds.close()

        # Check for valid data
        q_valid = not (np.isnan(q_val) or q_val == -9999.0)
        ssc_valid = not (np.isnan(ssc_val) or ssc_val == -9999.0)
        ssl_valid = not (np.isnan(ssl_val) or ssl_val == -9999.0)

        # Determine variables provided
        vars_provided = []
        if q_valid:
            vars_provided.append("Q")
        if ssc_valid:
            vars_provided.append("SSC")
        if ssl_valid:
            vars_provided.append("SSL")
        vars_provided_str = ", ".join(vars_provided) if vars_provided else "none"

        # Calculate percentages (simplified - assume all good if data exists)
        q_percent = 100.0 if q_valid else 'N/A'
        ssc_percent = 100.0 if ssc_valid else 'N/A'
        ssl_percent = 100.0 if ssl_valid else 'N/A'

        # Prepare station info
        station_info = {
            'station_name': river_name,
            'Source_ID': location_id,
            'river_name': river_name,
            'longitude': lon,
            'latitude': lat,
            'altitude': alt if not np.isnan(alt) and alt != -9999.0 else 'N/A',
            'upstream_area': upstream_area if not np.isnan(upstream_area) else 'N/A',
            'Data Source Name': 'Milliman & Farnsworth Global River Sediment Database',
            'Type': 'In-situ',
            'Temporal Resolution': 'climatology',
            'Temporal Span': 'various (pre-2012)',
            'Variables Provided': vars_provided_str,
            'Geographic Coverage': f"{country}, {continent}" if country else continent,
            'Reference/DOI': 'https://doi.org/10.1126/science.abn7980',
            'Q_start_date': representative_year if q_valid else 'N/A',
            'Q_end_date': representative_year if q_valid else 'N/A',
            'Q_percent_complete': q_percent,
            'SSC_start_date': representative_year if ssc_valid else 'N/A',
            'SSC_end_date': representative_year if ssc_valid else 'N/A',
            'SSC_percent_complete': ssc_percent,
            'SSL_start_date': representative_year if ssl_valid else 'N/A',
            'SSL_end_date': representative_year if ssl_valid else 'N/A',
            'SSL_percent_complete': ssl_percent
        }

        return station_info

    except Exception as e:
        print(f"  ERROR reading {os.path.basename(input_file)}: {e}")
        try:
            ds.close()
        except:
            pass
        return None


def main():
    """Main processing function."""

    print("="*80)
    print("Generating Milliman Station Summary CSV from Existing NetCDF Files")
    print("="*80)
    print()

    # Path - check both output locations
    input_dir1 = '/Users/zhongwangwei/Downloads/Sediment/Output_r/annually_climatology/Milliman'
    input_dir2 = '/Users/zhongwangwei/Downloads/Sediment/Output/annually_climatology/Milliman'

    # Use whichever has more files
    files1 = glob.glob(os.path.join(input_dir1, 'Milliman_*.nc'))
    files2 = glob.glob(os.path.join(input_dir2, 'Milliman_*.nc'))

    if len(files1) >= len(files2):
        input_dir = input_dir1
        input_files = sorted(files1)
        print(f"Using Output_r directory: {len(files1)} files")
    else:
        input_dir = input_dir2
        input_files = sorted(files2)
        print(f"Using Output directory: {len(files2)} files")

    output_dir = '/Users/zhongwangwei/Downloads/Sediment/Output_r/annually_climatology/Milliman'
    os.makedirs(output_dir, exist_ok=True)

    if len(input_files) == 0:
        print(f"ERROR: No NetCDF files found")
        sys.exit(1)

    print(f"Found {len(input_files)} NetCDF files to process")
    print()

    # Process each file
    station_info_list = []
    processed_count = 0
    error_count = 0

    for i, input_file in enumerate(input_files):
        if (i + 1) % 100 == 0:
            print(f"Progress: {i+1}/{len(input_files)} files processed...")

        station_info = extract_station_info(input_file)
        if station_info:
            station_info_list.append(station_info)
            processed_count += 1
        else:
            error_count += 1

    print()
    print("="*80)
    print("Generating Station Summary CSV")
    print("="*80)
    print()

    # Create DataFrame and save to CSV
    if len(station_info_list) > 0:
        df = pd.DataFrame(station_info_list)

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

        df = df[column_order]

        csv_file = os.path.join(output_dir, 'Milliman_station_summary.csv')
        df.to_csv(csv_file, index=False)

        print(f"Station summary saved to: {csv_file}")
        print(f"Total stations: {len(df)}")

        # Print some statistics
        print()
        print("Variable availability:")
        q_count = sum(1 for x in df['Q_percent_complete'] if x != 'N/A')
        ssc_count = sum(1 for x in df['SSC_percent_complete'] if x != 'N/A')
        ssl_count = sum(1 for x in df['SSL_percent_complete'] if x != 'N/A')
        print(f"  Q (Discharge): {q_count} stations")
        print(f"  SSC (Concentration): {ssc_count} stations")
        print(f"  SSL (Load): {ssl_count} stations")
    else:
        print("WARNING: No successful stations processed, CSV not created")

    print()
    print("="*80)
    print("Processing Summary")
    print("="*80)
    print(f"Total files found: {len(input_files)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Errors encountered: {error_count}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    print()


if __name__ == '__main__':
    main()
