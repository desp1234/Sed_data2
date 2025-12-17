
import pandas as pd
import numpy as np
import xarray as xr
import json
from datetime import datetime
from pathlib import Path
import glob

# Unit conversion constants
CFS_TO_CMS = 0.028316846592
FEET_TO_METERS = 0.3048
MILES_TO_KM = 1.60934

def get_flag(value, thresholds, meanings):
    if pd.isna(value):
        return meanings.split().index('missing_data')
    if value < thresholds.get('negative', -float('inf')):
        return meanings.split().index('bad_data')
    if value == thresholds.get('zero', -1):
        return meanings.split().index('suspect_data')
    if value > thresholds.get('extreme', float('inf')):
        return meanings.split().index('suspect_data')
    return meanings.split().index('good_data')

def process_usgs():
    source_dir = Path("/Users/zhongwangwei/Downloads/Sediment/Source/USGS/usgs_data_by_station")
    output_dir = Path("/Users/zhongwangwei/Downloads/Sediment/Output_r/daily/USGS")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = Path("/Users/zhongwangwei/Downloads/Sediment/Script/Dataset/USGS/common_sites_info.csv")

    all_station_summary_data = []
    station_dirs = sorted(list(source_dir.glob('station_*')))

    # Load common sites info
    sites_info_df = pd.read_csv(metadata_file, low_memory=False)

    for station_dir in station_dirs:
        station_id = station_dir.name.split('_')[1]
        print(f"Processing station {station_id}...")

        # Read data
        discharge_df = None
        sediment_df = None
        if (station_dir / f'{station_id}_discharge.csv').exists():
            discharge_df = pd.read_csv(station_dir / f'{station_id}_discharge.csv', comment='#')
        if (station_dir / f'{station_id}_sediment.csv').exists():
            sediment_df = pd.read_csv(station_dir / f'{station_id}_sediment.csv', comment='#')

        if discharge_df is None and sediment_df is None:
            print(f"  Skipping station {station_id}: no data files found.")
            continue


        # Get metadata
        station_info = sites_info_df[sites_info_df['site_no'] == int(station_id)]
        if station_info.empty:
            print(f"  Skipping station {station_id}: metadata not found.")
            continue
        station_info = station_info.iloc[0]

        if discharge_df is not None:
            discharge_df['datetime'] = pd.to_datetime(discharge_df['datetime'])
            q_col = next((col for col in discharge_df.columns if '00060' in col), None)
            if q_col:
                discharge_df = discharge_df[['datetime', q_col]].rename(columns={q_col: 'Q'})
                discharge_df.set_index('datetime', inplace=True)

        if sediment_df is not None:
            sediment_df['datetime'] = pd.to_datetime(sediment_df['datetime'])
            ssc_col = next((col for col in sediment_df.columns if '80154' in col), None)
            if ssc_col:
                sediment_df = sediment_df[['datetime', ssc_col]].rename(columns={ssc_col: 'SSC'})
                sediment_df.set_index('datetime', inplace=True)

        if discharge_df is not None and sediment_df is not None:
            df = pd.merge(discharge_df, sediment_df, on='datetime', how='outer')
        elif discharge_df is not None:
            df = discharge_df
        elif sediment_df is not None:
            df = sediment_df
        else:
            print(f"  Skipping station {station_id}: No valid data columns found.")
            continue

        if 'Q' in df.columns:
            df['Q'] = pd.to_numeric(df['Q'], errors='coerce') * CFS_TO_CMS
        if 'SSC' in df.columns:
            df['SSC'] = pd.to_numeric(df['SSC'], errors='coerce')
        subset_cols = [col for col in ['Q', 'SSC'] if col in df.columns]
        valid_data = df.dropna(subset=subset_cols, how='all')
        if valid_data.empty:
            print(f"  Skipping station {station_id}: No valid data after dropping NaNs.")
            continue

        start_date = valid_data.index.min()
        end_date = valid_data.index.max()
        date_index = pd.date_range(start=f"{start_date.year}-01-01", end=f"{end_date.year}-12-31", freq='D')
        df = df.reindex(date_index)

        # QC Flags
        q_thresholds = {'negative': 0, 'zero': 0, 'extreme': 50000}
        ssc_thresholds = {'negative': 0, 'extreme': 3000}
        flag_meanings = "good_data suspect_data bad_data missing_data"

        if 'Q' in df.columns:
            df['Q_flag'] = df['Q'].apply(lambda x: get_flag(x, q_thresholds, flag_meanings))
        if 'SSC' in df.columns:
            df['SSC_flag'] = df['SSC'].apply(lambda x: get_flag(x, ssc_thresholds, flag_meanings))
        if 'Q' in df.columns and 'SSC' in df.columns:
            df['SSL'] = df['Q'] * df['SSC'] * 0.0864 # Corrected formula
            ssl_thresholds = {'negative': 0}
            df['SSL_flag'] = df['SSL'].apply(lambda x: get_flag(x, ssl_thresholds, flag_meanings))



        # Create xarray Dataset
        ds = xr.Dataset()
        ds['time'] = ('time', df.index)

        variables = {
            'Q': {'units': 'm3 s-1', 'long_name': 'River Discharge', 'standard_name': 'river_discharge'},
            'SSC': {'units': 'mg L-1', 'long_name': 'Suspended Sediment Concentration', 'standard_name': 'mass_concentration_of_suspended_matter_in_water'},
            'SSL': {'units': 'ton day-1', 'long_name': 'Suspended Sediment Load', 'standard_name': 'load_of_suspended_matter'},
        }

        for var, attrs in variables.items():
            if var in df.columns:
                ds[var] = ('time', df[var].astype(np.float32).values)
                ds[var].attrs = {
                    'long_name': attrs['long_name'], 'standard_name': attrs['standard_name'], 'units': attrs['units'],
                    '_FillValue': -9999.0, 'ancillary_variables': f'{var}_flag',
                    'comment': "Source: Original data from USGS. Calculated if applicable."
                }
                if f'{var}_flag' in df.columns:
                    ds[f'{var}_flag'] = ('time', df[f'{var}_flag'].astype(np.byte).values)
                    ds[f'{var}_flag'].attrs = {
                        'long_name': f'Quality flag for {attrs["long_name"]}', '_FillValue': -127,
                        'flag_values': np.array([0, 1, 2, 3], dtype=np.byte), 'flag_meanings': flag_meanings,
                        'comment': "Flag definitions: 0=good_data, 1=suspect_data, 2=bad_data, 3=missing_data"
                    }

        # Coordinates and other metadata
        ds['lat'] = ((), station_info['dec_lat_va'])
        ds['lon'] = ((), station_info['dec_long_va'])
        ds['altitude'] = ((), station_info['alt_va'] * FEET_TO_METERS if pd.notna(station_info['alt_va']) else np.nan)
        ds['upstream_area'] = ((), station_info['drain_area_va'] * MILES_TO_KM**2 if pd.notna(station_info['drain_area_va']) else np.nan)

        # Global attributes
        ds.attrs = {
            'title': 'Harmonized Global River Discharge and Sediment',
            'Data_Source_Name': 'USGS NWIS',
            'station_name': station_info['station_nm'],
            'river_name': station_info.get('river_nm', 'N/A'), # river name not in metadata
            'Source_ID': station_id,
            'Type': 'In-situ station data',
            'Temporal_Resolution': 'daily',
            'Temporal_Span': f'{start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}',
            'Geographic_Coverage': f'{station_info["county_cd"]}, {station_info["state_cd"]}, USA',
            'Variables_Provided': 'Q, SSC, SSL',
            'Reference': 'https://waterdata.usgs.gov/nwis',
            'summary': 'This dataset contains daily river discharge and suspended sediment data from the USGS National Water Information System (NWIS).',
            'creator_name': 'Zhongwang Wei',
            'creator_email': 'weizhw6@mail.sysu.edu.cn',
            'creator_institution': 'Sun Yat-sen University, China',
            'Conventions': 'CF-1.8, ACDD-1.3',
            'history': f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        }

        # Save to NetCDF
        output_file = output_dir / f'USGS_{station_id}.nc'
        ds.to_netcdf(output_file, format='NETCDF4', encoding={'time': {'units': f'days since {start_date.year}-01-01'}})

        # Summary for CSV
        for var in ['Q', 'SSC', 'SSL']:
            if f'{var}_flag' in df.columns:
                good_data = df[df[f'{var}_flag'] == 0]
                if not good_data.empty:
                    all_station_summary_data.append({
                        'Source_ID': station_id,
                        'station_name': station_info['station_nm'],
                        'river_name': station_info.get('river_nm', 'N/A'),
                        'longitude': station_info['dec_long_va'],
                        'latitude': station_info['dec_lat_va'],
                        'altitude': ds["altitude"].item(),
                        'upstream_area': ds["upstream_area"].item(),
                        'Variable': var,
                        'Start_Date': good_data.index.min().strftime('%Y-%m-%d'),
                        'End_Date': good_data.index.max().strftime('%Y-%m-%d'),
                        'Percent_Complete': 100 * len(good_data) / len(df.loc[good_data.index.min():good_data.index.max()]),
                        'Mean': good_data[var].mean(),
                        'Median': good_data[var].median(),
                        'Range': f"{good_data[var].min()} - {good_data[var].max()}"
                    })

    # Create and save summary CSV
    summary_df = pd.DataFrame(all_station_summary_data)
    summary_df.to_csv(output_dir / 'USGS_station_summary.csv', index=False)

    print("Processing complete.")

if __name__ == '__main__':
    process_usgs()
