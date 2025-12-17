#!/usr/bin/env python3
"""
Update sediment NetCDF files to match the required specifications.
This script:
1. Merges sediment and discharge data
2. Standardizes variable names and attributes
3. Converts lat/lon from dimensions to scalar variables
4. Unifies time dimensions
5. Calculates missing variables if possible
"""

import netCDF4 as nc
import numpy as np
from pathlib import Path
import sys

def update_sediment_file(sediment_file, discharge_file, output_file=None):
    """
    Update a sediment NetCDF file to match specifications.

    Parameters:
    -----------
    sediment_file : str or Path
        Path to the input sediment NetCDF file
    discharge_file : str or Path
        Path to the corresponding discharge NetCDF file
    output_file : str or Path, optional
        Path to the output file. If None, overwrites the input file.
    """

    if output_file is None:
        output_file = sediment_file

    print(f"Processing: {Path(sediment_file).name}")

    # Read input files
    with nc.Dataset(sediment_file, 'r') as ds_sed, nc.Dataset(discharge_file, 'r') as ds_dis:

        # Extract scalar coordinates
        latitude = float(ds_sed['lat'][0])
        longitude = float(ds_sed['lon'][0])

        # Extract drainage area (upstream_area)
        upstream_area = float(ds_dis['drainage_area'][:]) if 'drainage_area' in ds_dis.variables else np.nan

        # Extract altitude if available (set to NaN if not found)
        altitude = np.nan  # Not available in current data

        # Get sediment data
        time_sed_load = ds_sed['time_sed_load'][:] if 'time_sed_load' in ds_sed.variables else None
        sediment_load_raw = ds_sed['sediment_load'][:, 0, 0] if 'sediment_load' in ds_sed.variables else None

        time_sed_suscon = ds_sed['time_sed_suscon'][:] if 'time_sed_suscon' in ds_sed.variables else None
        ssc_raw = ds_sed['suspended_sediment_concentration'][:, 0, 0] if 'suspended_sediment_concentration' in ds_sed.variables else None

        # Get discharge data
        time_flow = ds_dis['time_flow'][:]
        discharge_raw = ds_dis['discharge'][:, 0, 0]

        # Determine unified time axis (use the union of all time points)
        all_times = []
        if time_sed_load is not None:
            all_times.append(time_sed_load)
        if time_sed_suscon is not None:
            all_times.append(time_sed_suscon)
        all_times.append(time_flow)

        # Use the time dimension with most coverage or merge all unique times
        time_combined = np.unique(np.concatenate(all_times))
        time_combined.sort()

        # Initialize arrays with fill values
        fill_value = -9999.0
        n_time = len(time_combined)

        sediment_load = np.full(n_time, fill_value, dtype=np.float32)
        ssc = np.full(n_time, fill_value, dtype=np.float32)
        discharge = np.full(n_time, fill_value, dtype=np.float32)

        # Map sediment_load to unified time
        if time_sed_load is not None and sediment_load_raw is not None:
            for i, t in enumerate(time_sed_load):
                idx = np.where(time_combined == t)[0]
                if len(idx) > 0:
                    val = sediment_load_raw[i]
                    if val != -999.0:  # Original fill value
                        # Convert from "tonnes" to "ton day-1" (assuming original is already per day)
                        sediment_load[idx[0]] = val

        # Map ssc to unified time
        if time_sed_suscon is not None and ssc_raw is not None:
            for i, t in enumerate(time_sed_suscon):
                idx = np.where(time_combined == t)[0]
                if len(idx) > 0:
                    val = ssc_raw[i]
                    if val != -999.0:  # Original fill value
                        ssc[idx[0]] = val

        # Map discharge to unified time
        for i, t in enumerate(time_flow):
            idx = np.where(time_combined == t)[0]
            if len(idx) > 0:
                val = discharge_raw[i]
                if val != -999.0:  # Original fill value
                    discharge[idx[0]] = val

        # Calculate missing values if possible
        # If sediment_load and discharge exist but ssc is missing, calculate ssc
        # Formula: sediment_load = discharge × ssc × 86.4
        # Therefore: ssc = sediment_load / (discharge × 86.4)
        for i in range(n_time):
            if (ssc[i] == fill_value and
                sediment_load[i] != fill_value and
                discharge[i] != fill_value and
                discharge[i] > 0):
                # sediment_load is in ton/day, discharge in m3/s
                # ssc = (sediment_load * 1000 kg) / (discharge * 86400 s/day) * 1000 mg/kg / 1000 L/m3
                # ssc = sediment_load * 1000 / (discharge * 86.4)
                ssc[i] = (sediment_load[i] * 1000.0) / (discharge[i] * 86.4)

            # If ssc and discharge exist but sediment_load is missing, calculate it
            elif (sediment_load[i] == fill_value and
                  ssc[i] != fill_value and
                  discharge[i] != fill_value):
                # sediment_load = discharge × ssc × 86.4 / 1000
                sediment_load[i] = discharge[i] * ssc[i] * 86.4 / 1000.0

        # Get global attributes from original file
        station_id = ds_sed.station_id if hasattr(ds_sed, 'station_id') else ''
        station_name = ds_sed.station_name if hasattr(ds_sed, 'station_name') else ''
        province = ds_sed.province_territory if hasattr(ds_sed, 'province_territory') else ''

    # Create output file
    with nc.Dataset(output_file, 'w', format='NETCDF4') as ds_out:

        # Create dimension
        ds_out.createDimension('time', n_time)

        # Create variables
        # Time
        var_time = ds_out.createVariable('time', 'f8', ('time',))
        var_time.standard_name = 'time'
        var_time.long_name = 'time of measurement'
        var_time.units = 'days since 1970-01-01 00:00:00'
        var_time.calendar = 'gregorian'
        var_time.axis = 'T'
        # Convert from "days since 1850-01-01" to "days since 1970-01-01"
        # Difference: 120 years * 365.25 days/year = 43830 days
        days_offset = 43830.0
        var_time[:] = time_combined - days_offset

        # Latitude (scalar)
        var_lat = ds_out.createVariable('latitude', 'f4')
        var_lat.standard_name = 'latitude'
        var_lat.long_name = 'station latitude'
        var_lat.units = 'degrees_north'
        var_lat.valid_range = np.array([-90.0, 90.0], dtype=np.float32)
        var_lat[:] = latitude

        # Longitude (scalar)
        var_lon = ds_out.createVariable('longitude', 'f4')
        var_lon.standard_name = 'longitude'
        var_lon.long_name = 'station longitude'
        var_lon.units = 'degrees_east'
        var_lon.valid_range = np.array([-180.0, 180.0], dtype=np.float32)
        var_lon[:] = longitude

        # Altitude (scalar)
        var_alt = ds_out.createVariable('altitude', 'f4', fill_value=fill_value)
        var_alt.standard_name = 'altitude'
        var_alt.long_name = 'station altitude above sea level'
        var_alt.units = 'm'
        var_alt[:] = altitude if not np.isnan(altitude) else fill_value

        # Upstream area (scalar)
        var_area = ds_out.createVariable('upstream_area', 'f4', fill_value=fill_value)
        var_area.long_name = 'upstream drainage area'
        var_area.units = 'km2'
        var_area[:] = upstream_area if not np.isnan(upstream_area) else fill_value

        # Discharge
        var_dis = ds_out.createVariable('discharge', 'f4', ('time',),
                                        fill_value=fill_value, chunksizes=[n_time])
        var_dis.standard_name = 'water_volume_transport_in_river_channel'
        var_dis.long_name = 'river discharge'
        var_dis.units = 'm3 s-1'
        var_dis.coordinates = 'time latitude longitude'
        var_dis[:] = discharge

        # SSC
        var_ssc = ds_out.createVariable('ssc', 'f4', ('time',),
                                        fill_value=fill_value, chunksizes=[n_time])
        var_ssc.standard_name = 'mass_concentration_of_suspended_matter_in_water'
        var_ssc.long_name = 'suspended sediment concentration'
        var_ssc.units = 'mg L-1'
        var_ssc.coordinates = 'time latitude longitude'
        var_ssc[:] = ssc

        # Sediment load
        var_load = ds_out.createVariable('sediment_load', 'f4', ('time',),
                                         fill_value=fill_value, chunksizes=[n_time])
        var_load.long_name = 'suspended sediment load'
        var_load.units = 'ton day-1'
        var_load.coordinates = 'time latitude longitude'
        var_load.comment = 'Calculated as: sediment_load = discharge × ssc × 86.4'
        var_load[:] = sediment_load

        # Global attributes
        ds_out.Conventions = 'CF-1.8'
        ds_out.title = f'HYDAT Station {station_id} - Sediment and Discharge Data'
        ds_out.institution = 'Water Survey of Canada / Environment and Climate Change Canada'
        ds_out.source = 'HYDAT - Canadian Hydrometric Database'
        ds_out.history = f'Updated to standardized format on {np.datetime64("today")}'
        ds_out.references = 'https://www.canada.ca/en/environment-climate-change/services/water-overview/quantity/monitoring/survey/data-products-services/national-archive-hydat.html'
        ds_out.station_id = station_id
        ds_out.station_name = station_name
        ds_out.province_territory = province
        ds_out.geospatial_lat_min = float(latitude)
        ds_out.geospatial_lat_max = float(latitude)
        ds_out.geospatial_lon_min = float(longitude)
        ds_out.geospatial_lon_max = float(longitude)

    print(f"  ✓ Successfully updated: {Path(output_file).name}")
    return True


def main():
    """Process all sediment files in the sediment directory."""

    base_dir = Path('/Users/zhongwangwei/Downloads/Sediment/Source/Station/Hydat')
    sediment_dir = base_dir / 'sediment'
    discharge_dir = base_dir / 'discharge_waterlevel'

    # Get all sediment files
    sediment_files = sorted(sediment_dir.glob('HYDAT_*_SEDIMENT.nc'))

    print(f"Found {len(sediment_files)} sediment files to process")
    print("=" * 70)

    success_count = 0
    error_count = 0

    for sed_file in sediment_files:
        # Extract station ID from filename
        # HYDAT_01AF006_SEDIMENT.nc -> 01AF006
        station_id = sed_file.stem.replace('HYDAT_', '').replace('_SEDIMENT', '')

        # Find corresponding discharge file
        dis_file = discharge_dir / f'HYDAT_{station_id}.nc'

        if not dis_file.exists():
            print(f"Warning: Discharge file not found for {station_id}, skipping...")
            error_count += 1
            continue

        try:
            update_sediment_file(sed_file, dis_file)
            success_count += 1
        except Exception as e:
            print(f"  ✗ Error processing {sed_file.name}: {str(e)}")
            error_count += 1
            import traceback
            traceback.print_exc()

    print("=" * 70)
    print(f"Processing complete:")
    print(f"  Success: {success_count}")
    print(f"  Errors:  {error_count}")
    print(f"  Total:   {len(sediment_files)}")


if __name__ == '__main__':
    main()
