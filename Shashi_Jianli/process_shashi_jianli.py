
import pandas as pd
import numpy as np
import xarray as xr
import json
from datetime import datetime
from pathlib import Path

def get_flag(value, thresholds, meanings):
    if pd.isna(value):
        return meanings.split().index('missing_data')
    if value < thresholds['negative']:
        return meanings.split().index('bad_data')
    if value == thresholds['zero']:
        return meanings.split().index('suspect_data')
    if value > thresholds['extreme']:
        return meanings.split().index('suspect_data')
    return meanings.split().index('good_data')

def process_shashi_jianli():
    # Define paths
    source_dir = Path("/Users/zhongwangwei/Downloads/Sediment/Source/Shashi_Jianli")
    output_dir = Path("/Users/zhongwangwei/Downloads/Sediment/Output_r/daily/Shashi_Jianli")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(source_dir / 'station_data.csv')
    with open(source_dir / 'station_coords.json', 'r') as f:
        coords = json.load(f)

    # Station information
    stations = {
        'SS': {'name': 'Shashi', 'id': 'SS', 'river_name': 'Yangtze River'},
        'JL': {'name': 'Jianli', 'id': 'JL', 'river_name': 'Yangtze River'}
    }

    summary_data = []

    for station_id, station_info in stations.items():
        df_station = df[['Date', f'{station_id}_discharge', f'{station_id}_SSC']].copy()
        df_station.columns = ['Date', 'Q', 'SSC_kg_m3']
        df_station['Date'] = pd.to_datetime(df_station['Date'])

        # Convert to numeric, coercing errors
        df_station['Q'] = pd.to_numeric(df_station['Q'], errors='coerce')
        df_station['SSC_kg_m3'] = pd.to_numeric(df_station['SSC_kg_m3'], errors='coerce')
        
        # Drop rows where both Q and SSC are NaN
        df_station.dropna(subset=['Q', 'SSC_kg_m3'], how='all', inplace=True)

        if df_station.empty:
            continue

        # Unit conversions
        df_station['SSC'] = df_station['SSC_kg_m3'] * 1000  # kg/m3 to mg/L
        df_station['SSL'] = df_station['Q'] * df_station['SSC_kg_m3'] * 86.4  # ton/day

        # QC Flags
        q_thresholds = {'negative': 0, 'zero': 0, 'extreme': 120000}
        ssc_thresholds = {'negative': 0, 'zero': -1, 'extreme': 10000} # No zero check for ssc
        ssl_thresholds = {'negative': 0, 'zero': -1, 'extreme': float('inf')} # No zero/extreme check for ssl

        flag_meanings = "good_data suspect_data bad_data missing_data"
        
        df_station['Q_flag'] = df_station['Q'].apply(lambda x: get_flag(x, q_thresholds, flag_meanings))
        df_station['SSC_flag'] = df_station['SSC'].apply(lambda x: get_flag(x, ssc_thresholds, flag_meanings))
        df_station['SSL_flag'] = df_station['SSL'].apply(lambda x: get_flag(x, ssl_thresholds, flag_meanings))

        # Time cropping
        valid_data = df_station.dropna(subset=['Q', 'SSC_kg_m3'], how='all')
        if valid_data.empty:
            continue
        
        start_date = valid_data['Date'].min()
        end_date = valid_data['Date'].max()
        
        date_index = pd.date_range(start=f"{start_date.year}-01-01", end=f"{end_date.year}-12-31", freq='D')
        df_station.set_index('Date', inplace=True)
        df_station = df_station.reindex(date_index)
        df_station.index.name = 'time'


        # Create xarray Dataset
        ds = xr.Dataset()
        ds['time'] = ('time', df_station.index)
        
        # Add variables to dataset
        variables = {
            'Q': {'units': 'm3 s-1', 'long_name': 'River Discharge', 'standard_name': 'river_discharge'},
            'SSC': {'units': 'mg L-1', 'long_name': 'Suspended Sediment Concentration', 'standard_name': 'mass_concentration_of_suspended_matter_in_water'},
            'SSL': {'units': 'ton day-1', 'long_name': 'Suspended Sediment Load', 'standard_name': 'load_of_suspended_matter'},
        }

        for var, attrs in variables.items():
            ds[var] = ('time', df_station[var].values.astype(np.float32))
            ds[var].attrs = {
                'long_name': attrs['long_name'],
                'standard_name': attrs['standard_name'],
                'units': attrs['units'],
                '_FillValue': -9999.0,
                'ancillary_variables': f'{var}_flag',
                'comment': f"Source: Original data from reference. Calculated if applicable."
            }
            
            # Add flag variables
            ds[f'{var}_flag'] = ('time', df_station[f'{var}_flag'].values.astype(np.byte))
            ds[f'{var}_flag'].attrs = {
                'long_name': f'Quality flag for {attrs["long_name"]}',
                '_FillValue': -127,
                'flag_values': np.array([0, 1, 2, 3], dtype=np.byte),
                'flag_meanings': flag_meanings,
                'comment': "Flag definitions: 0=good_data, 1=suspect_data, 2=bad_data, 3=missing_data"
            }

        # Add coordinate variables
        ds['lat'] = ((), coords[station_id]['lat'], {'long_name': 'station latitude', 'standard_name': 'latitude', 'units': 'degrees_north'})
        ds['lon'] = ((), coords[station_id]['lon'], {'long_name': 'station longitude', 'standard_name': 'longitude', 'units': 'degrees_east'})
        ds['altitude'] = ((), np.nan, {'long_name': 'station altitude', 'standard_name': 'altitude', 'units': 'm', 'comment': 'Not available in source data'})
        ds['upstream_area'] = ((), np.nan, {'long_name': 'upstream drainage area', 'units': 'km2', 'comment': 'Not available in source data'})

        # Global attributes
        ds.attrs = {
            'title': 'Harmonized Global River Discharge and Sediment',
            'Data_Source_Name': 'Shashi_Jianli Dataset',
            'station_name': station_info['name'],
            'river_name': station_info['river_name'],
            'Source_ID': station_id,
            'Type': 'In-situ station data',
            'Temporal_Resolution': 'daily',
            'Temporal_Span': f'{start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}',
            'Geographic_Coverage': 'Yangtze River Basin, China',
            'Variables_Provided': 'Q, SSC, SSL',
            'Reference': 'Nones, M., Guo, C. (2025). Remote sensing as a support tool to map suspended sediment concentration over extended river reaches. Acta Geophysica, 73:4655-4668. https://doi.org/10.1007/s11600-025-01638-x',
            'summary': 'This dataset contains daily river discharge and suspended sediment data for the Shashi and Jianli stations on the Yangtze River.',
            'creator_name': 'Zhongwang Wei',
            'creator_email': 'weizhw6@mail.sysu.edu.cn',
            'creator_institution': 'Sun Yat-sen University, China',
            'Conventions': 'CF-1.8, ACDD-1.3',
            'history': f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        }

        # Save to NetCDF
        output_file = output_dir / f'Shashi_Jianli_{station_id}.nc'
        ds.to_netcdf(output_file, format='NETCDF4', encoding={'time': {'units': f'days since {start_date.year}-01-01'}})

        # Summary data for CSV
        for var in ['Q', 'SSC', 'SSL']:
            good_data = df_station[df_station[f'{var}_flag'] == 0]
            if not good_data.empty:
                summary_data.append({
                    'Source_ID': station_id,
                    'station_name': station_info['name'],
                    'river_name': station_info['river_name'],
                    'longitude': coords[station_id]['lon'],
                    'latitude': coords[station_id]['lat'],
                    'altitude': np.nan,
                    'upstream_area': np.nan,
                    'Variable': var,
                    'Start_Date': good_data.index.min().strftime('%Y-%m-%d'),
                    'End_Date': good_data.index.max().strftime('%Y-%m-%d'),
                    'Percent_Complete': 100 * len(good_data) / len(df_station.loc[good_data.index.min():good_data.index.max()]),
                    'Mean': good_data[var].mean(),
                    'Median': good_data[var].median(),
                    'Range': f"{good_data[var].min()} - {good_data[var].max()}"
                })

    # Create and save summary CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'Shashi_Jianli_station_summary.csv', index=False)
    
    print("Processing complete.")

if __name__ == '__main__':
    process_shashi_jianli()
