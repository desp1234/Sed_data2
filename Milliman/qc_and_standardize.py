#!/usr/bin/env python3
"""
Quality Control and CF-1.8 Standardization for Milliman Global River Sediment Database.

This script:
1. Reads existing NetCDF files from Output/annually_climatology/Milliman
2. Restructures dimensions from (time, lat, lon) to scalar coordinates
3. Renames TSS → SSL for consistency
4. Performs quality control checks and adds quality flags
5. Standardizes metadata to CF-1.8 compliance (following ALi_De_Boer reference)
6. Generates station summary CSV
7. Saves standardized files to Output_r/annually_climatology/Milliman

Unit Conversions (already done in input files):
- Discharge: km³/yr → m³/s: Q (m³/s) = Q (km³/yr) × 10⁹ / 31,557,600
- TSS: Mt/yr → ton/day: TSS (ton/day) = TSS (Mt/yr) × 10⁶ / 365.25
- SSC: mg/L (calculated from TSS and Q)

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

def perform_qc_checks(value, var_type='Q'):
    """
    Perform quality control checks and return quality flag.

    Quality flags:
    0 = Good data
    1 = Estimated data
    2 = Suspect data (extreme or zero values)
    3 = Bad data (negative or invalid)
    9 = Missing data

    Parameters:
    -----------
    value : float
        Value to check
    var_type : str
        Type of variable: 'Q', 'SSC', or 'SSL'

    Returns:
    --------
    flag : int
        Quality flag (0, 1, 2, 3, or 9)
    """
    if np.isnan(value) or value == -9999.0:
        return 9  # Missing data

    if var_type == 'Q':  # Discharge
        if value < 0:
            return 3  # Bad data (negative)
        elif value == 0:
            return 2  # Suspect (zero flow - unusual for climatology)
        elif value > 500000:  # Extremely high (> Amazon peak)
            return 2  # Suspect
        else:
            return 0  # Good data

    elif var_type == 'SSC':  # Suspended Sediment Concentration
        if value < 0:
            return 3  # Bad data (negative)
        elif value > 3000:  # mg/L
            return 2  # Suspect (extremely high)
        else:
            return 0  # Good data

    elif var_type == 'SSL':  # Suspended Sediment Load
        if value < 0:
            return 3  # Bad data (negative)
        else:
            return 0  # Good data (no upper limit for SSL)

    return 0


def standardize_netcdf_file(input_file, output_dir):
    """
    Standardize a single NetCDF file to CF-1.8 compliance with QC flags.

    Parameters:
    -----------
    input_file : str
        Path to input NetCDF file
    output_dir : str
        Path to output directory

    Returns:
    --------
    station_info : dict or None
        Dictionary containing station metadata for CSV summary
    """

    print(f"\nProcessing: {os.path.basename(input_file)}")

    # Read input file
    try:
        ds_in = nc.Dataset(input_file, 'r')

        # Read coordinates (currently arrays, need to extract scalar values)
        lat = float(ds_in.variables['latitude'][0])
        lon = float(ds_in.variables['longitude'][0])
        time_val = ds_in.variables['time'][0]

        # Read data variables (currently 3D arrays [time, lat, lon])
        q_val = float(ds_in.variables['Discharge'][0, 0, 0]) if 'Discharge' in ds_in.variables else np.nan
        ssc_val = float(ds_in.variables['SSC'][0, 0, 0]) if 'SSC' in ds_in.variables else np.nan
        tss_val = float(ds_in.variables['TSS'][0, 0, 0]) if 'TSS' in ds_in.variables else np.nan

        # Read scalar drainage area
        drainage_area = float(ds_in.variables['drainage_area'][:]) if 'drainage_area' in ds_in.variables else np.nan

        # Read metadata from global attributes
        location_id = ds_in.location_id if hasattr(ds_in, 'location_id') else ""
        river_name = ds_in.river_name if hasattr(ds_in, 'river_name') else ""
        country = ds_in.country if hasattr(ds_in, 'country') else ""
        continent = ds_in.continent_region if hasattr(ds_in, 'continent_region') else ""

        # Get time units for metadata
        time_units = ds_in.variables['time'].units
        time_calendar = ds_in.variables['time'].calendar

        # Close input file before creating output
        ds_in.close()

    except Exception as e:
        print(f"  ERROR reading {os.path.basename(input_file)}: {e}")
        try:
            ds_in.close()
        except:
            pass
        return None

    # Skip if no valid data
    if (np.isnan(q_val) or q_val == -9999.0) and (np.isnan(ssc_val) or ssc_val == -9999.0):
        print(f"  SKIPPED: No valid Q or SSC data")
        return None

    # Perform QC checks
    q_flag = perform_qc_checks(q_val, 'Q')
    ssc_flag = perform_qc_checks(ssc_val, 'SSC')
    ssl_flag = perform_qc_checks(tss_val, 'SSL')

    # Calculate statistics for CSV
    q_percent = 100.0 if q_flag == 0 else 0.0
    ssc_percent = 100.0 if ssc_flag == 0 else 0.0
    ssl_percent = 100.0 if ssl_flag == 0 else 0.0

    # Convert time to year for temporal span
    dates = nc.num2date([time_val], units=time_units, calendar=time_calendar)
    representative_year = dates[0].year if len(dates) > 0 else 2000

    # Create output filename (keep original naming convention)
    output_file = os.path.join(output_dir, os.path.basename(input_file))

    print(f"  River: {river_name} ({country})")
    print(f"  Location: {lat:.3f}°, {lon:.3f}°")
    print(f"  Q: {q_val:.2f} m³/s (flag={q_flag}), SSC: {ssc_val:.2f} mg/L (flag={ssc_flag}), SSL: {tss_val:.2f} ton/day (flag={ssl_flag})")

    # Create standardized NetCDF file
    with nc.Dataset(output_file, 'w', format='NETCDF4') as ds:

        # Create dimensions - CF-1.8 compliant
        time_dim = ds.createDimension('time', None)  # UNLIMITED

        # Create coordinate variables
        # Time
        time_var = ds.createVariable('time', 'f8', ('time',))
        time_var.long_name = "time"
        time_var.standard_name = "time"
        time_var.units = "days since 1970-01-01 00:00:00"
        time_var.calendar = "gregorian"
        time_var.axis = "T"
        time_var[:] = [time_val]

        # Latitude (scalar)
        lat_var = ds.createVariable('lat', 'f4')
        lat_var.long_name = "station latitude"
        lat_var.standard_name = "latitude"
        lat_var.units = "degrees_north"
        lat_var[:] = lat

        # Longitude (scalar)
        lon_var = ds.createVariable('lon', 'f4')
        lon_var.long_name = "station longitude"
        lon_var.standard_name = "longitude"
        lon_var.units = "degrees_east"
        lon_var[:] = lon

        # Altitude (not available in Milliman data)
        alt_var = ds.createVariable('altitude', 'f4', fill_value=-9999.0)
        alt_var.long_name = "station elevation above sea level"
        alt_var.standard_name = "altitude"
        alt_var.units = "m"
        alt_var.positive = "up"
        alt_var.comment = "Source: Not available in Milliman database."
        alt_var[:] = -9999.0

        # Upstream drainage area
        area_var = ds.createVariable('upstream_area', 'f4', fill_value=-9999.0)
        area_var.long_name = "upstream drainage area"
        area_var.units = "km2"
        area_var.comment = "Source: Original data from Milliman & Farnsworth (2011)."
        if not np.isnan(drainage_area):
            area_var[:] = drainage_area
        else:
            area_var[:] = -9999.0

        # Q - River Discharge
        q_var = ds.createVariable('Q', 'f4', ('time',), fill_value=-9999.0, zlib=True, complevel=4)
        q_var.long_name = "river discharge"
        q_var.standard_name = "water_volume_transport_in_river_channel"
        q_var.units = "m3 s-1"
        q_var.coordinates = "time lat lon altitude"
        q_var.ancillary_variables = "Q_flag"
        q_var.comment = "Source: Original data from Milliman & Farnsworth (2011). " \
                        "Unit conversion: Original unit km³/yr converted to m³/s using formula: " \
                        "Q (m³/s) = Q (km³/yr) × 10⁹ / 31,557,600. " \
                        "Represents long-term average discharge."
        q_var[:] = [q_val if not np.isnan(q_val) else -9999.0]

        # Q quality flag
        q_flag_var = ds.createVariable('Q_flag', 'b', ('time',), fill_value=9, zlib=True, complevel=4)
        q_flag_var.long_name = "quality flag for river discharge"
        q_flag_var.standard_name = "status_flag"
        q_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.byte)
        q_flag_var.flag_meanings = "good_data estimated_data suspect_data bad_data missing_data"
        q_flag_var.comment = "Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), " \
                             "3=Bad (e.g., negative), 9=Missing in source."
        q_flag_var[:] = [q_flag]

        # SSC - Suspended Sediment Concentration
        ssc_var = ds.createVariable('SSC', 'f4', ('time',), fill_value=-9999.0, zlib=True, complevel=4)
        ssc_var.long_name = "suspended sediment concentration"
        ssc_var.standard_name = "mass_concentration_of_suspended_matter_in_water"
        ssc_var.units = "mg L-1"
        ssc_var.coordinates = "time lat lon altitude"
        ssc_var.ancillary_variables = "SSC_flag"
        ssc_var.comment = "Source: Calculated from SSL and Q. Formula: SSC (mg/L) = SSL (ton/day) / (Q (m³/s) × 86.4). " \
                          "Represents long-term average suspended sediment concentration."
        ssc_var[:] = [ssc_val if not np.isnan(ssc_val) else -9999.0]

        # SSC quality flag
        ssc_flag_var = ds.createVariable('SSC_flag', 'b', ('time',), fill_value=9, zlib=True, complevel=4)
        ssc_flag_var.long_name = "quality flag for suspended sediment concentration"
        ssc_flag_var.standard_name = "status_flag"
        ssc_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.byte)
        ssc_flag_var.flag_meanings = "good_data estimated_data suspect_data bad_data missing_data"
        ssc_flag_var.comment = "Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), " \
                               "3=Bad (e.g., negative), 9=Missing in source."
        ssc_flag_var[:] = [ssc_flag]

        # SSL - Suspended Sediment Load (renamed from TSS)
        ssl_var = ds.createVariable('SSL', 'f4', ('time',), fill_value=-9999.0, zlib=True, complevel=4)
        ssl_var.long_name = "suspended sediment load"
        ssl_var.units = "ton day-1"
        ssl_var.coordinates = "time lat lon altitude"
        ssl_var.ancillary_variables = "SSL_flag"
        ssl_var.comment = "Source: Original data from Milliman & Farnsworth (2011). " \
                          "Unit conversion: Original unit Mt/yr converted to ton/day using formula: " \
                          "SSL (ton/day) = SSL (Mt/yr) × 10⁶ / 365.25. " \
                          "Represents long-term average suspended sediment load."
        ssl_var[:] = [tss_val if not np.isnan(tss_val) else -9999.0]

        # SSL quality flag
        ssl_flag_var = ds.createVariable('SSL_flag', 'b', ('time',), fill_value=9, zlib=True, complevel=4)
        ssl_flag_var.long_name = "quality flag for suspended sediment load"
        ssl_flag_var.standard_name = "status_flag"
        ssl_flag_var.flag_values = np.array([0, 1, 2, 3, 9], dtype=np.byte)
        ssl_flag_var.flag_meanings = "good_data estimated_data suspect_data bad_data missing_data"
        ssl_flag_var.comment = "Flag definitions: 0=Good, 1=Estimated, 2=Suspect (e.g., zero/extreme), " \
                               "3=Bad (e.g., negative), 9=Missing in source."
        ssl_flag_var[:] = [ssl_flag]

        # Global attributes - CF-1.8 and ACDD-1.3 compliant
        ds.Conventions = "CF-1.8, ACDD-1.3"
        ds.title = "Harmonized Global River Discharge and Sediment"
        ds.summary = f"Long-term average discharge and suspended sediment data for {river_name} " \
                     f"({country}). Data from the Milliman & Farnsworth (2011) Global River Sediment Database, " \
                     f"representing climatological averages compiled from various observation periods (pre-2012). " \
                     f"Supplemented with satellite-derived sediment data from Dethier et al. (2022)."

        # Source and data information
        ds.source = "In-situ station data"
        ds.data_source_name = "Milliman & Farnsworth Global River Sediment Database"
        ds.station_name = river_name
        ds.river_name = river_name
        ds.Source_ID = location_id

        # Type and resolution
        ds.Type = "In-situ"
        ds.Temporal_Resolution = "climatology"
        ds.Temporal_Span = "various (pre-2012)"
        ds.Geographic_Coverage = f"{country}, {continent}"

        # Variables provided
        vars_provided = []
        if not np.isnan(q_val) and q_val != -9999.0:
            vars_provided.append("Q")
        if not np.isnan(ssc_val) and ssc_val != -9999.0:
            vars_provided.append("SSC")
        if not np.isnan(tss_val) and tss_val != -9999.0:
            vars_provided.append("SSL")
        vars_provided_str = ", ".join(vars_provided) if vars_provided else "none"
        ds.Variables_Provided = vars_provided_str
        ds.Number_of_data = "1"

        # References
        ds.Reference = "Milliman, J.D., and Farnsworth, K.L. (2011). River Discharge to the Coastal Ocean: " \
                       "A Global Synthesis. Cambridge University Press, 392 pp.; " \
                       "Dethier, E. N., Renshaw, C. E., & Magilligan, F. J. (2022). Rapid changes to global " \
                       "river suspended sediment flux by humans. Science, 376(6600), 1447-1452."
        ds.source_data_link = "https://doi.org/10.1126/science.abn7980"

        # Creator information
        ds.creator_name = "Zhongwang Wei"
        ds.creator_email = "weizhw6@mail.sysu.edu.cn"
        ds.creator_institution = "Sun Yat-sen University, China"

        # Temporal coverage
        ds.time_coverage_start = f"{representative_year}-01-01"
        ds.time_coverage_end = f"{representative_year}-12-31"
        ds.temporal_span = "various (pre-2012)"
        ds.temporal_resolution = "climatology"

        # Spatial coverage
        ds.geospatial_lat_min = float(lat)
        ds.geospatial_lat_max = float(lat)
        ds.geospatial_lon_min = float(lon)
        ds.geospatial_lon_max = float(lon)
        ds.geographic_coverage = f"{country}, {continent}"

        # Processing history
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ds.history = f"{current_time}: Quality controlled and standardized to CF-1.8 compliance. " \
                     f"Added quality flags, restructured dimensions from (time,lat,lon) to scalar coordinates, " \
                     f"renamed TSS to SSL. Script: qc_and_standardize.py"

        ds.date_created = datetime.now().strftime("%Y-%m-%d")
        ds.date_modified = datetime.now().strftime("%Y-%m-%d")
        ds.processing_level = "Quality controlled and standardized"

        # Additional comments
        ds.comment = f"Data represents long-term average (climatological) values compiled from various " \
                     f"observation periods before 2012. Quality flags indicate data reliability: " \
                     f"0=good, 1=estimated, 2=suspect, 3=bad, 9=missing. " \
                     f"Unit conversions documented in variable comments."

        ds.data_limitations = "Climatological averages only; specific observation periods vary by station. " \
                              "Altitude data not available in original database."

        # Store country and continent
        ds.country = country
        ds.continent_region = continent

    # Prepare station info for CSV
    station_info = {
        'station_name': river_name,
        'Source_ID': location_id,
        'river_name': river_name,
        'longitude': lon,
        'latitude': lat,
        'altitude': 'N/A',
        'upstream_area': drainage_area if not np.isnan(drainage_area) else 'N/A',
        'Data Source Name': 'Milliman & Farnsworth Global River Sediment Database',
        'Type': 'In-situ',
        'Temporal Resolution': 'climatology',
        'Temporal Span': 'various (pre-2012)',
        'Variables Provided': vars_provided_str,
        'Geographic Coverage': f"{country}, {continent}",
        'Reference/DOI': 'https://doi.org/10.1126/science.abn7980',
        'Q_start_date': representative_year if not np.isnan(q_val) and q_val != -9999.0 else 'N/A',
        'Q_end_date': representative_year if not np.isnan(q_val) and q_val != -9999.0 else 'N/A',
        'Q_percent_complete': q_percent if not np.isnan(q_val) and q_val != -9999.0 else 'N/A',
        'SSC_start_date': representative_year if not np.isnan(ssc_val) and ssc_val != -9999.0 else 'N/A',
        'SSC_end_date': representative_year if not np.isnan(ssc_val) and ssc_val != -9999.0 else 'N/A',
        'SSC_percent_complete': ssc_percent if not np.isnan(ssc_val) and ssc_val != -9999.0 else 'N/A',
        'SSL_start_date': representative_year if not np.isnan(tss_val) and tss_val != -9999.0 else 'N/A',
        'SSL_end_date': representative_year if not np.isnan(tss_val) and tss_val != -9999.0 else 'N/A',
        'SSL_percent_complete': ssl_percent if not np.isnan(tss_val) and tss_val != -9999.0 else 'N/A'
    }

    return station_info


def main():
    """Main processing function."""

    print("="*80)
    print("Milliman Global River Sediment Database - QC and CF-1.8 Standardization")
    print("="*80)
    print()

    # Paths
    input_dir = '/Users/zhongwangwei/Downloads/Sediment/Output/annually_climatology/Milliman'
    output_dir = '/Users/zhongwangwei/Downloads/Sediment/Output_r/annually_climatology/Milliman'

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all NetCDF files
    input_files = sorted(glob.glob(os.path.join(input_dir, 'Milliman_*.nc')))

    if len(input_files) == 0:
        print(f"ERROR: No NetCDF files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(input_files)} NetCDF files to process")
    print()

    # Process each file
    station_info_list = []
    processed_count = 0
    skipped_count = 0
    error_count = 0

    for i, input_file in enumerate(input_files):
        if (i + 1) % 50 == 0:
            print(f"\n--- Progress: {i+1}/{len(input_files)} files processed ---\n")

        try:
            station_info = standardize_netcdf_file(input_file, output_dir)
            if station_info:
                station_info_list.append(station_info)
                processed_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            print(f"  ERROR processing {os.path.basename(input_file)}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1

    print()
    print("="*80)
    print("Generating Station Summary CSV")
    print("="*80)
    print()

    # Create DataFrame and save to CSV
    if len(station_info_list) > 0:
        df = pd.DataFrame(station_info_list)

        # Reorder columns to match reference format
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
    else:
        print("WARNING: No successful stations processed, CSV not created")
    print()

    # Print summary
    print("="*80)
    print("Processing Summary")
    print("="*80)
    print(f"Total files found: {len(input_files)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped (no valid data): {skipped_count}")
    print(f"Errors encountered: {error_count}")
    print(f"Output directory: {output_dir}")
    print()
    print("="*80)
    print("Quality Control Summary")
    print("="*80)
    print("Quality Checks Applied:")
    print("  Q (Discharge):")
    print("    - Q < 0: Flagged as BAD (flag=3)")
    print("    - Q == 0: Flagged as SUSPECT (flag=2)")
    print("    - Q > 500,000 m³/s: Flagged as SUSPECT (flag=2)")
    print("  SSC (Suspended Sediment Concentration):")
    print("    - SSC < 0: Flagged as BAD (flag=3)")
    print("    - SSC > 3000 mg/L: Flagged as SUSPECT (flag=2)")
    print("  SSL (Suspended Sediment Load):")
    print("    - SSL < 0: Flagged as BAD (flag=3)")
    print("    - Valid SSL: Flagged as GOOD (flag=0)")
    print("="*80)
    print()


if __name__ == '__main__':
    main()
