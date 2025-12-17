
import pandas as pd
import numpy as np
import netCDF4 as nc
from datetime import datetime, timedelta
import os
import glob

def update_riversed_netcdf(input_dir, output_dir):
    nc_files = glob.glob(os.path.join(input_dir, '*.nc'))
    station_summary_data = []

    for nc_file in nc_files:
        with nc.Dataset(nc_file, 'r') as src:
            # Create a new file in the output directory
            output_file = os.path.join(output_dir, os.path.basename(nc_file))
            with nc.Dataset(output_file, 'w', format='NETCDF4') as dst:
                # Copy dimensions
                for name, dimension in src.dimensions.items():
                    dst.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))

                # Copy variables and apply modifications
                for name, variable in src.variables.items():
                    if name not in ['Q', 'SSC', 'SSL', 'Q_flag', 'SSC_flag', 'SSL_flag', 'time', 'lat', 'lon', 'altitude', 'upstream_area', 'sediment_load', 'discharge', 'latitude', 'longitude', 'ssc']:
                        # Just copy other variables
                        out_var = dst.createVariable(name, variable.datatype, variable.dimensions)
                        out_var.setncatts({k: variable.getncattr(k) for k in variable.ncattrs()})
                        out_var[:] = variable[:]
                        continue

                    # Standardize variable names and attributes
                    if name in ['ssc', 'SSC']:
                        new_name = 'SSC'
                        ssc_data = src.variables[name][:]
                        
                        # QC Checks for SSC
                        ssc_flag_data = np.zeros_like(ssc_data, dtype=np.byte)
                        ssc_flag_data[ssc_data < 0] = 3  # Bad data
                        ssc_flag_data[ssc_data > 3000] = 2 # Suspect data
                        ssc_flag_data[np.ma.getmask(ssc_data)] = 9 # Missing data

                        # Create SSC variable
                        ssc_var = dst.createVariable(new_name, 'f4', ('time',), fill_value=-9999.0)
                        ssc_var.units = 'mg L-1'
                        ssc_var.standard_name = 'suspended_sediment_concentration'
                        ssc_var.long_name = 'Suspended Sediment Concentration'
                        ssc_var.ancillary_variables = 'SSC_flag'
                        ssc_var[:] = ssc_data

                        # Create SSC_flag variable
                        ssc_flag_var = dst.createVariable('SSC_flag', 'b', ('time',))
                        ssc_flag_var.long_name = "Quality flag for Suspended Sediment Concentration"
                        ssc_flag_var.flag_values = np.array([0, 2, 3, 9], dtype=np.byte)
                        ssc_flag_var.flag_meanings = "good_data suspect_data bad_data missing_data"
                        ssc_flag_var[:] = ssc_flag_data
                    
                    elif name in ['time']:
                        time_var = dst.createVariable('time', src.variables[name].datatype, src.variables[name].dimensions)
                        time_var.long_name = "time"
                        time_var.standard_name = "time"
                        time_var.units = src.variables[name].units
                        time_var.calendar = "gregorian"
                        time_var[:] = src.variables[name][:]

                    elif name in ['lat', 'latitude']:
                        lat_var = dst.createVariable('lat', 'f4')
                        lat_var.long_name = "station latitude"
                        lat_var.standard_name = "latitude"
                        lat_var.units = "degrees_north"
                        lat_var[:] = src.variables[name][:]
                    
                    elif name in ['lon', 'longitude']:
                        lon_var = dst.createVariable('lon', 'f4')
                        lon_var.long_name = "station longitude"
                        lon_var.standard_name = "longitude"
                        lon_var.units = "degrees_east"
                        lon_var[:] = src.variables[name][:]

                # Add Q and SSL as NaN
                time_len = len(dst.dimensions['time'])
                
                q_var = dst.createVariable('Q', 'f4', ('time',), fill_value=-9999.0)
                q_var.units = 'm3 s-1'
                q_var.standard_name = 'river_discharge'
                q_var.long_name = 'River Discharge'
                q_var.ancillary_variables = 'Q_flag'
                q_var[:] = np.full(time_len, -9999.0)

                q_flag_var = dst.createVariable('Q_flag', 'b', ('time',))
                q_flag_var.long_name = "Quality flag for River Discharge"
                q_flag_var.flag_values = np.array([9], dtype=np.byte)
                q_flag_var.flag_meanings = "missing_data"
                q_flag_var[:] = np.full(time_len, 9, dtype=np.byte)

                ssl_var = dst.createVariable('SSL', 'f4', ('time',), fill_value=-9999.0)
                ssl_var.units = 'ton day-1'
                ssl_var.long_name = 'Suspended Sediment Load'
                ssl_var.ancillary_variables = 'SSL_flag'
                ssl_var[:] = np.full(time_len, -9999.0)

                ssl_flag_var = dst.createVariable('SSL_flag', 'b', ('time',))
                ssl_flag_var.long_name = "Quality flag for Suspended Sediment Load"
                ssl_flag_var.flag_values = np.array([9], dtype=np.byte)
                ssl_flag_var.flag_meanings = "missing_data"
                ssl_flag_var[:] = np.full(time_len, 9, dtype=np.byte)

                # Update Global Attributes
                dst.title = 'Harmonized Global River Discharge and Sediment'
                dst.data_source_name = 'RiverSed Dataset'
                dst.station_name = src.getncattr('station_id') if 'station_id' in src.ncattrs() else ''
                dst.river_name = '' # Not available
                dst.source_id = src.getncattr('station_id') if 'station_id' in src.ncattrs() else ''
                dst.type = 'satellite station'
                dst.temporal_resolution = 'daily'
                
                time_units = dst.variables['time'].units
                start_date_num = dst.variables['time'][0]
                end_date_num = dst.variables['time'][-1]
                start_date = nc.num2date(start_date_num, time_units)
                end_date = nc.num2date(end_date_num, time_units)

                dst.temporal_span = f'{start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}'
                dst.geographic_coverage = 'USA'
                dst.variables_provided = "SSC"
                dst.reference = 'Gardner et al. (2023), Human activities change suspended sediment concentration along rivers, Environmental Research Letters.'
                dst.history = f'Updated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} by update_riversed_netcdf.py. Original history: {src.history if "history" in src.ncattrs() else ""}'
                dst.summary = 'This dataset contains daily suspended sediment concentration data for rivers in the USA, derived from satellite imagery. Discharge and sediment load are not available.'
                dst.creator_name = 'Zhongwang Wei'
                dst.creator_email = 'weizhw6@mail.sysu.edu.cn'
                dst.creator_institution = 'Sun Yat-sen University, China'
                dst.conventions = 'CF-1.8, ACDD-1.3'

                # Station summary data
                ssc_good_count = np.sum(ssc_flag_data == 0)
                total_count = len(ssc_flag_data)
                
                station_summary_data.append({
                    'Source_ID': dst.source_id,
                    'station_name': dst.station_name,
                    'river_name': dst.river_name,
                    'longitude': lon_var[...],
                    'latitude': lat_var[...],
                    'altitude': '',
                    'upstream_area': '',
                    'Q_start_date': '',
                    'Q_end_date': '',
                    'Q_percent_complete': 0,
                    'SSC_start_date': start_date.strftime('%Y-%m-%d') if ssc_good_count > 0 else '',
                    'SSC_end_date': end_date.strftime('%Y-%m-%d') if ssc_good_count > 0 else '',
                    'SSC_percent_complete': (ssc_good_count / total_count) * 100 if total_count > 0 else 0,
                    'SSL_start_date': '',
                    'SSL_end_date': '',
                    'SSL_percent_complete': 0,
                })

    # Create CSV
    summary_df = pd.DataFrame(station_summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'RiverSed_station_summary.csv'), index=False)

if __name__ == "__main__":
    update_riversed_netcdf('/Users/zhongwangwei/Downloads/Sediment/Output/daily/RiverSed', '/Users/zhongwangwei/Downloads/Sediment/Output_r/daily/RiverSed')
