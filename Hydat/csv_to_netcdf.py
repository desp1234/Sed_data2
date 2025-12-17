#!/usr/bin/env python3
"""
Alternative HYDAT to NetCDF Converter - Works with CSV exports

If you can't use mdbtools, you can export the MDB tables to CSV files first,
then use this script to convert them to NetCDF.

To export from Access on Windows:
1. Open Hydat.mdb in Microsoft Access
2. For each table: Right-click > Export > Text File (CSV)
3. Save all CSVs to a folder
4. Run this script on that folder
"""

import pandas as pd
import netCDF4 as nc
import numpy as np
from datetime import datetime
import os
import sys
import glob

class CSVtoNetCDFConverter:
    def __init__(self, csv_folder, output_path):
        self.csv_folder = csv_folder
        self.output_path = output_path
        self.tables = {}
        
    def load_csv_files(self):
        """Load all CSV files from the folder"""
        csv_files = glob.glob(os.path.join(self.csv_folder, "*.csv"))
        
        if not csv_files:
            print(f"No CSV files found in {self.csv_folder}")
            return False
        
        print(f"Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            table_name = os.path.splitext(os.path.basename(csv_file))[0]
            print(f"Loading: {table_name}...")
            
            try:
                df = pd.read_csv(csv_file, encoding='utf-8', low_memory=False)
                self.tables[table_name] = df
                print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                print(f"  Error loading {csv_file}: {e}")
                continue
        
        return len(self.tables) > 0
    
    def write_to_netcdf(self):
        """Write the data to NetCDF format"""
        print(f"\nWriting to NetCDF: {self.output_path}")
        
        with nc.Dataset(self.output_path, 'w', format='NETCDF4') as ncfile:
            # Add global attributes
            ncfile.title = "HYDAT - Canadian Hydrometric Database"
            ncfile.institution = "Water Survey of Canada"
            ncfile.source = f"Converted from CSV exports in {self.csv_folder}"
            ncfile.history = f"Created on {datetime.now().isoformat()}"
            ncfile.Conventions = "CF-1.8"
            
            # Process each table
            for table_name, df in self.tables.items():
                print(f"\nProcessing table: {table_name}")
                
                if len(df) == 0:
                    print(f"  Skipping empty table")
                    continue
                
                # Create a group for each table
                grp = ncfile.createGroup(table_name)
                
                # Create dimension for number of records
                dim_name = f"n_records"
                grp.createDimension(dim_name, len(df))
                
                # Add table metadata
                grp.description = f"Table: {table_name}"
                grp.n_records = len(df)
                grp.n_columns = len(df.columns)
                
                # Process each column
                for col in df.columns:
                    try:
                        col_data = df[col]
                        
                        # Try to infer date columns
                        if any(keyword in col.upper() for keyword in ['DATE', 'YEAR', 'MONTH', 'DAY']):
                            if col.upper() == 'YEAR' and col_data.dtype in ['int64', 'float64']:
                                # Year column - keep as integer
                                var = grp.createVariable(col, 'i4', (dim_name,), fill_value=-999)
                                var[:] = col_data.fillna(-999).astype(np.int32)
                                var.long_name = col
                                var.description = "Year"
                                
                            elif col.upper() in ['MONTH', 'DAY']:
                                # Month/Day column
                                var = grp.createVariable(col, 'i4', (dim_name,), fill_value=-999)
                                var[:] = col_data.fillna(-999).astype(np.int32)
                                var.long_name = col
                                
                            else:
                                # Try to parse as date
                                try:
                                    dates = pd.to_datetime(col_data, errors='coerce')
                                    if dates.notna().sum() > len(dates) * 0.5:  # If >50% are valid dates
                                        var = grp.createVariable(col, 'f8', (dim_name,), fill_value=-999.0)
                                        var.units = "days since 1970-01-01"
                                        var.calendar = "gregorian"
                                        numeric_dates = (dates - pd.Timestamp("1970-01-01")).dt.total_seconds() / 86400.0
                                        var[:] = numeric_dates.fillna(-999.0)
                                        var.long_name = col
                                        continue
                                except:
                                    pass
                        
                        # Handle by data type
                        if col_data.dtype == 'object' or col_data.dtype == 'string':
                            # String data
                            max_len = col_data.astype(str).str.len().max()
                            if pd.isna(max_len) or max_len == 0:
                                max_len = 1
                            max_len = min(int(max_len) + 1, 1000)  # Cap at 1000 chars
                            
                            str_dim = f"strlen_{col}"
                            grp.createDimension(str_dim, max_len)
                            
                            var = grp.createVariable(col, 'S1', (dim_name, str_dim))
                            
                            # Convert to character array
                            str_data = col_data.fillna('').astype(str)
                            char_array = np.zeros((len(str_data), max_len), dtype='S1')
                            for i, s in enumerate(str_data):
                                s_bytes = s.encode('utf-8')[:max_len]
                                char_array[i, :len(s_bytes)] = list(s_bytes)
                            var[:] = char_array
                            
                        elif col_data.dtype in ['int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8']:
                            # Integer data
                            var = grp.createVariable(col, 'i4', (dim_name,), fill_value=-999)
                            var[:] = col_data.fillna(-999).astype(np.int32)
                            
                        elif col_data.dtype in ['float64', 'float32']:
                            # Float data
                            var = grp.createVariable(col, 'f4', (dim_name,), fill_value=-999.0)
                            var[:] = col_data.fillna(-999.0).astype(np.float32)
                            
                        elif col_data.dtype == 'bool':
                            # Boolean data
                            var = grp.createVariable(col, 'i1', (dim_name,), fill_value=-1)
                            var[:] = col_data.fillna(False).astype(np.int8)
                            
                        else:
                            # Unknown type - convert to string
                            max_len = 100
                            str_dim = f"strlen_{col}"
                            if str_dim not in grp.dimensions:
                                grp.createDimension(str_dim, max_len)
                            
                            var = grp.createVariable(col, 'S1', (dim_name, str_dim))
                            str_data = col_data.fillna('').astype(str)
                            char_array = np.zeros((len(str_data), max_len), dtype='S1')
                            for i, s in enumerate(str_data):
                                s_bytes = s.encode('utf-8')[:max_len]
                                char_array[i, :len(s_bytes)] = list(s_bytes)
                            var[:] = char_array
                        
                        # Add variable attributes
                        var.long_name = col
                        
                        # Add column statistics for numeric data
                        if col_data.dtype in ['int64', 'int32', 'int16', 'float64', 'float32']:
                            try:
                                var.min_value = float(col_data.min())
                                var.max_value = float(col_data.max())
                                var.mean_value = float(col_data.mean())
                            except:
                                pass
                        
                    except Exception as e:
                        print(f"  Warning: Could not write column {col}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                print(f"  Written {len(df)} records with {len(df.columns)} columns")
        
        print(f"\n✓ Conversion complete!")
        print(f"  Output file: {self.output_path}")
        print(f"  File size: {os.path.getsize(self.output_path) / 1024 / 1024:.2f} MB")

def main():
    if len(sys.argv) < 2:
        print("Usage: python csv_to_netcdf.py <csv_folder> [output.nc]")
        print("\nThis script converts CSV exports from HYDAT to NetCDF format.")
        print("\nThe CSV folder should contain CSV files exported from the HYDAT database.")
        print("Each CSV file should be named after the table (e.g., STATIONS.csv, DLY_FLOWS.csv)")
        sys.exit(1)
    
    csv_folder = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "hydat_from_csv.nc"
    
    if not os.path.isdir(csv_folder):
        print(f"Error: Folder not found: {csv_folder}")
        sys.exit(1)
    
    converter = CSVtoNetCDFConverter(csv_folder, output_path)
    
    # Load CSV files
    if not converter.load_csv_files():
        print("No CSV files could be loaded")
        sys.exit(1)
    
    # Write to NetCDF
    converter.write_to_netcdf()

if __name__ == "__main__":
    main()
