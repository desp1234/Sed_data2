#!/usr/bin/env python3
"""
Convert Hydat.mdb (Canadian Hydrometric Database) to NetCDF format

This script converts the Microsoft Access database format to NetCDF,
preserving the relational structure as much as possible.
"""

import subprocess
import pandas as pd
import netCDF4 as nc
import numpy as np
from datetime import datetime
import os
import sys

class HydatConverter:
    def __init__(self, mdb_path, output_path):
        self.mdb_path = mdb_path
        self.output_path = output_path
        self.tables = {}
        
    def check_mdbtools(self):
        """Check if mdbtools is installed"""
        try:
            subprocess.run(['mdb-ver', self.mdb_path], 
                          capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("ERROR: mdbtools not found. Please install it:")
            print("  Ubuntu/Debian: sudo apt-get install mdbtools")
            print("  MacOS: brew install mdbtools")
            return False
    
    def get_table_names(self):
        """Get list of tables in the database"""
        try:
            result = subprocess.run(['mdb-tables', '-1', self.mdb_path],
                                  capture_output=True, text=True, check=True)
            tables = [t.strip() for t in result.stdout.strip().split('\n') if t.strip()]
            return tables
        except subprocess.CalledProcessError as e:
            print(f"Error reading tables: {e}")
            return []
    
    def export_table_to_dataframe(self, table_name):
        """Export a table from MDB to pandas DataFrame"""
        try:
            result = subprocess.run(['mdb-export', self.mdb_path, table_name],
                                  capture_output=True, text=True, check=True)
            
            # Parse CSV output
            from io import StringIO
            df = pd.read_csv(StringIO(result.stdout))
            return df
        except subprocess.CalledProcessError as e:
            print(f"Error exporting table {table_name}: {e}")
            return None
        except Exception as e:
            print(f"Error parsing table {table_name}: {e}")
            return None
    
    def load_all_tables(self):
        """Load all tables from the database"""
        print("Getting table names...")
        table_names = self.get_table_names()
        print(f"Found {len(table_names)} tables")
        
        for table_name in table_names:
            print(f"Loading table: {table_name}...")
            df = self.export_table_to_dataframe(table_name)
            if df is not None:
                self.tables[table_name] = df
                print(f"  Loaded {len(df)} rows")
    
    def write_to_netcdf(self):
        """Write the data to NetCDF format"""
        print(f"\nWriting to NetCDF: {self.output_path}")
        
        # Create NetCDF file
        with nc.Dataset(self.output_path, 'w', format='NETCDF4') as ncfile:
            # Add global attributes
            ncfile.title = "HYDAT - Canadian Hydrometric Database"
            ncfile.institution = "Water Survey of Canada"
            ncfile.source = f"Converted from {os.path.basename(self.mdb_path)}"
            ncfile.history = f"Created on {datetime.now().isoformat()}"
            ncfile.Conventions = "CF-1.8"
            
            # Process each table
            for table_name, df in self.tables.items():
                print(f"Processing table: {table_name}")
                
                if len(df) == 0:
                    print(f"  Skipping empty table: {table_name}")
                    continue
                
                # Create a group for each table
                grp = ncfile.createGroup(table_name)
                
                # Add dimension for number of records
                dim_name = f"{table_name}_records"
                grp.createDimension(dim_name, len(df))
                
                # Process each column
                for col in df.columns:
                    try:
                        # Handle different data types
                        if df[col].dtype == 'object':
                            # String data
                            max_len = df[col].astype(str).str.len().max()
                            if pd.isna(max_len) or max_len == 0:
                                max_len = 1
                            
                            # Create string dimension if needed
                            str_dim = f"{col}_strlen"
                            if str_dim not in grp.dimensions:
                                grp.createDimension(str_dim, int(max_len))
                            
                            var = grp.createVariable(col, 'S1', (dim_name, str_dim))
                            
                            # Convert to character array
                            str_data = df[col].fillna('').astype(str)
                            char_array = np.array([list(s.ljust(int(max_len))) 
                                                  for s in str_data])
                            var[:] = nc.stringtochar(char_array)
                            
                        elif 'datetime' in str(df[col].dtype):
                            # Date/time data
                            var = grp.createVariable(col, 'f8', (dim_name,))
                            var.units = "days since 1970-01-01"
                            
                            # Convert to numeric (days since epoch)
                            dates = pd.to_datetime(df[col])
                            var[:] = (dates - pd.Timestamp("1970-01-01")).dt.total_seconds() / 86400.0
                            
                        elif df[col].dtype in ['int64', 'int32', 'int16', 'int8']:
                            # Integer data
                            var = grp.createVariable(col, 'i4', (dim_name,), 
                                                   fill_value=-999)
                            var[:] = df[col].fillna(-999).astype(np.int32)
                            
                        elif df[col].dtype in ['float64', 'float32']:
                            # Float data
                            var = grp.createVariable(col, 'f4', (dim_name,),
                                                   fill_value=-999.0)
                            var[:] = df[col].fillna(-999.0).astype(np.float32)
                            
                        else:
                            # Default: convert to string
                            max_len = 50
                            str_dim = f"{col}_strlen"
                            if str_dim not in grp.dimensions:
                                grp.createDimension(str_dim, max_len)
                            
                            var = grp.createVariable(col, 'S1', (dim_name, str_dim))
                            str_data = df[col].fillna('').astype(str)
                            char_array = np.array([list(s.ljust(max_len)) 
                                                  for s in str_data])
                            var[:] = nc.stringtochar(char_array)
                        
                        # Add attributes
                        var.long_name = col
                        
                    except Exception as e:
                        print(f"  Warning: Could not write column {col}: {e}")
                        continue
                
                print(f"  Written {len(df)} records")
        
        print(f"\nConversion complete! Output: {self.output_path}")
        print(f"File size: {os.path.getsize(self.output_path) / 1024 / 1024:.2f} MB")

def main():
    if len(sys.argv) < 2:
        print("Usage: python hydat_to_netcdf.py <path_to_hydat.mdb> [output.nc]")
        print("\nThis script converts a HYDAT database (Hydat.mdb) to NetCDF format.")
        print("Requires: mdbtools to be installed on the system")
        sys.exit(1)
    
    mdb_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "hydat.nc"
    
    if not os.path.exists(mdb_path):
        print(f"Error: File not found: {mdb_path}")
        sys.exit(1)
    
    converter = HydatConverter(mdb_path, output_path)
    
    # Check for mdbtools
    if not converter.check_mdbtools():
        sys.exit(1)
    
    # Load all tables
    converter.load_all_tables()
    
    # Write to NetCDF
    converter.write_to_netcdf()

if __name__ == "__main__":
    main()
