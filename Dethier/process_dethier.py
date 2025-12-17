
import pandas as pd
import xarray as xr
import numpy as np
import os
from datetime import datetime
import glob

def get_days_in_month(year, month):
    """Returns the number of days in a given month and year."""
    if month == 2:
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            return 29
        else:
            return 28
    elif month in [4, 6, 9, 11]:
        return 30
    else:
        return 31

def process_dethier_data_from_nc(input_nc_dir, output_dir, summary_csv_path):
    """
    Processes the Dethier dataset, performs QC, and saves as CF-compliant NetCDF files.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
 # 收集所有 nc 文件路径
    nc_files = sorted(glob.glob(os.path.join(input_nc_dir, "*.nc")))

    station_summary_list = []

    for nc_file in nc_files:
        print(f"Processing: {os.path.basename(nc_file)}")

        # 读取 NetCDF
        ds_raw = xr.open_dataset(nc_file)

        # 提取站点元信息
        station_id = ds_raw.attrs.get("site_no", os.path.splitext(os.path.basename(nc_file))[0])
        river_name = ds_raw.attrs.get("river_name", "UnknownRiver")
        latitude = float(ds_raw.attrs.get("latitude", ds_raw.latitude.values))
        longitude = float(ds_raw.attrs.get("longitude", ds_raw.longitude.values))

        # 统一变量名以兼容后续部分
        # Q = Discharge, SSL = sediment_flux, SSC = SSC
        ds = xr.Dataset()
        ds["Q"] = ds_raw["Discharge"].squeeze()
        ds["SSL"] = ds_raw["sediment_flux"].squeeze()
        ds["SSC"] = ds_raw["SSC"].squeeze()

        # 添加坐标
        ds = ds.assign_coords({
            "latitude": latitude,
            "longitude": longitude
        })

        # 将 dataset 转为 DataFrame（和你原来一样继续处理）
        station_df_ts = ds.to_dataframe().reset_index().set_index("time")
        # --- QC Flags ---
        # Initialize flags
        station_df_ts["Q_flag"] = 9     # missing by default
        station_df_ts["SSC_flag"] = 9
        station_df_ts["SSL_flag"] = 9

        # ----- Discharge Q -----
        # missing
        station_df_ts.loc[~station_df_ts["Q"].isna(), "Q_flag"] = 0
        # bad
        station_df_ts.loc[station_df_ts["Q"] < 0, "Q_flag"] = 3
        # suspect
        station_df_ts.loc[station_df_ts["Q"] == 0, "Q_flag"] = 2
        station_df_ts.loc[station_df_ts["Q"] > 100000, "Q_flag"] = 2   # Extreme discharge

        # ----- SSC -----
        # valid SSC
        station_df_ts.loc[~station_df_ts["SSC"].isna(), "SSC_flag"] = 0
        # bad
        station_df_ts.loc[station_df_ts["SSC"] < 0, "SSC_flag"] = 3
        # suspect
        station_df_ts.loc[station_df_ts["SSC"] > 3000, "SSC_flag"] = 2   # extreme SSC

        # ----- SSL -----
        station_df_ts.loc[~station_df_ts["SSL"].isna(), "SSL_flag"] = 0
        # bad
        station_df_ts.loc[station_df_ts["SSL"] < 0, "SSL_flag"] = 3
        # suspect
station_df_ts.loc[station_df_ts["SSL"] > 1_000_000, "SSL_flag"] = 2

        # --- Time Slicing ---
        valid_data = station_df_ts[(station_df_ts['Q'] > 0) | (station_df_ts['SSL'] > 0)]
        if valid_data.empty:
            print(f"Skipping station {station_id}: No valid data.")
            continue
            
        start_date = valid_data.index.min()
        end_date = valid_data.index.max()
        
        station_df_ts = station_df_ts.loc[start_date:end_date]

        # --- Create NetCDF file ---
        safe_river_name = "".join(c if c.isalnum() or c in ['_', '-'] else "_" for c in river_name)
        output_filename = os.path.join(output_dir, f"Dethier_{safe_river_name}_{station_id}.nc")
        
        ds = xr.Dataset()

        # Coordinates
        ds.coords['time'] = ('time', station_df_ts.index)
        ds.coords['latitude'] = ('latitude', [station_df_ts['latitude'].iloc[0]])
        ds.coords['longitude'] = ('longitude', [station_df_ts['longitude'].iloc[0]])

        # Variables
        ds['Q'] = ('time', station_df_ts['Q'].values, {
            'long_name': 'River Discharge',
            'standard_name': 'river_discharge',
            'units': 'm3 s-1',
            '_FillValue': -9999.0,
            'ancillary_variables': 'Q_flag'
        })
        ds['SSC'] = ('time', station_df_ts['SSC'].values, {
            'long_name': 'Suspended Sediment Concentration',
            'standard_name': 'mass_concentration_of_suspended_matter_in_water',
            'units': 'mg L-1',
            '_FillValue': -9999.0,
            'ancillary_variables': 'SSC_flag'
        })
        ds['SSL'] = ('time', station_df_ts['SSL'].values, {
            'long_name': 'Suspended Sediment Load',
            'standard_name': 'sediment_transport_in_river',
            'units': 'ton day-1',
            '_FillValue': -9999.0,
            'ancillary_variables': 'SSL_flag'
        })
        
        # Flag variables
        ds['Q_flag'] = ('time', station_df_ts['Q_flag'].values.astype(np.byte), {
            'long_name': 'Quality flag for River Discharge',
            '_FillValue': -127,
            'flag_values': np.array([0, 1, 2, 3, 9], dtype=np.byte),
            'flag_meanings': 'good_data estimated_data suspect_data bad_data missing_data'
        })
        ds['SSC_flag'] = ('time', station_df_ts['SSC_flag'].values.astype(np.byte), {
            'long_name': 'Quality flag for Suspended Sediment Concentration',
            '_FillValue': -127,
            'flag_values': np.array([0, 1, 2, 3, 9], dtype=np.byte),
            'flag_meanings': 'good_data estimated_data suspect_data bad_data missing_data'
        })
        ds['SSL_flag'] = ('time', station_df_ts['SSL_flag'].values.astype(np.byte), {
            'long_name': 'Quality flag for Suspended Sediment Load',
            '_FillValue': -127,
            'flag_values': np.array([0, 1, 2, 3, 9], dtype=np.byte),
            'flag_meanings': 'good_data estimated_data suspect_data bad_data missing_data'
        })
        raw_attrs = ds_raw.attrs  # 原始全局属性

        ds.attrs.update({
            'title': raw_attrs.get('title', 'Harmonized Global River Discharge and Sediment'),
            'Data_Source_Name': raw_attrs.get('Data_Source_Name', 'Dethier Dataset'),
            'station_name': raw_attrs.get('station_name', station_id),
            'river_name': raw_attrs.get('river_name', river_name),
            'Source_ID': raw_attrs.get('site_no', station_id),
            'Type': raw_attrs.get('Type', 'Satellite station'),
            'Temporal_Resolution': raw_attrs.get('Temporal_Resolution', 'monthly'),
            'Temporal_Span': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            'Geographic_Coverage': raw_attrs.get('Geographic_Coverage', 'Unknown'),
            'Variables_Provided': 'Q, SSC, SSL',
            'Reference1': 'Dethier, E. N., et al. (2022), Science, DOI:10.1126/science.abn7980',
            'summary': 'This dataset provides monthly river discharge and sediment data, processed and QC-filtered.',
            'creator_name': 'Zhongwang Wei',
            'creator_email': 'weizhw6@mail.sysu.edu.cn',
            'creator_institution': 'Sun Yat-sen University, China',
            'Conventions': 'CF-1.8, ACDD-1.3',
            'history': f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} from {os.path.basename(INPUT_NC_DIR)}'
        })
        # Save to NetCDF
        ds.to_netcdf(output_filename)
        print(f"Saved to {output_filename}")

        # --- Station Summary ---
        q_good = station_df_ts[station_df_ts['Q_flag'] == 0]['Q']
        ssc_good = station_df_ts[station_df_ts['SSC_flag'] == 0]['SSC']
        ssl_good = station_df_ts[station_df_ts['SSL_flag'] == 0]['SSL']
        
        station_summary = {
            'Source_ID': station_id,
            'station_name': ds.attrs['station_name'],
            'river_name': ds.attrs['river_name'],
            'longitude': ds.coords['longitude'].values[0],
            'latitude': ds.coords['latitude'].values[0],
            'Q_start_date': q_good.index.min().strftime('%Y-%m-%d') if not q_good.empty else 'N/A',
            'Q_end_date': q_good.index.max().strftime('%Y-%m-%d') if not q_good.empty else 'N/A',
            'Q_percent_complete': (len(q_good) / len(station_df_ts)) * 100 if not station_df_ts.empty else 0,
            'SSC_start_date': ssc_good.index.min().strftime('%Y-%m-%d') if not ssc_good.empty else 'N/A',
            'SSC_end_date': ssc_good.index.max().strftime('%Y-%m-%d') if not ssc_good.empty else 'N/A',
            'SSC_percent_complete': (len(ssc_good) / len(station_df_ts)) * 100 if not station_df_ts.empty else 0,
            'SSL_start_date': ssl_good.index.min().strftime('%Y-%m-%d') if not ssl_good.empty else 'N/A',
            'SSL_end_date': ssl_good.index.max().strftime('%Y-%m-%d') if not ssl_good.empty else 'N/A',
            'SSL_percent_complete': (len(ssl_good) / len(station_df_ts)) * 100 if not station_df_ts.empty else 0,
        }
        station_summary_list.append(station_summary)

    # Save summary CSV
    summary_df = pd.DataFrame(station_summary_list)
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Summary saved to {summary_csv_path}")

if __name__ == '__main__':
    INPUT_NC_DIR  = '/mnt/d/sediment_data/Source/Dethier/nc_convert'
    OUTPUT_DIR = '/mnt/d/sediment_data/Script/Dataset/Dethier/Output_r'
    SUMMARY_CSV = '/mnt/d/sediment_data/Script/Dataset/Dethier/Output_r/Dethier_station_summary.csv'
    process_dethier_data_from_nc(INPUT_NC_DIR, OUTPUT_DIR, SUMMARY_CSV)
