#!/usr/bin/env python3
"""
Inspect and summarize a HYDAT NetCDF file
"""

import netCDF4 as nc
import sys
import numpy as np

def inspect_netcdf(filepath):
    """Inspect the contents of a NetCDF file"""
    
    print(f"\n{'='*80}")
    print(f"Inspecting: {filepath}")
    print(f"{'='*80}\n")
    
    try:
        ds = nc.Dataset(filepath, 'r')
        
        # Global attributes
        print("GLOBAL ATTRIBUTES:")
        print("-" * 80)
        for attr in ds.ncattrs():
            print(f"  {attr}: {getattr(ds, attr)}")
        
        # Groups (tables)
        print(f"\n\nTABLES (GROUPS): {len(ds.groups)}")
        print("-" * 80)
        
        for group_name in sorted(ds.groups.keys()):
            group = ds.groups[group_name]
            n_records = len(group.dimensions[list(group.dimensions.keys())[0]])
            n_vars = len(group.variables)
            
            print(f"\n  {group_name}:")
            print(f"    Records: {n_records:,}")
            print(f"    Variables: {n_vars}")
            
            # Show first few variable names
            var_names = list(group.variables.keys())[:10]
            print(f"    Columns: {', '.join(var_names)}", end="")
            if len(group.variables) > 10:
                print(f" ... (+{len(group.variables)-10} more)")
            else:
                print()
        
        # Detailed view of selected tables
        important_tables = ['STATIONS', 'DLY_FLOWS', 'DLY_LEVELS', 'ANNUAL_STATISTICS']
        available_important = [t for t in important_tables if t in ds.groups]
        
        if available_important:
            print(f"\n\nDETAILED VIEW OF KEY TABLES:")
            print("-" * 80)
            
            for table_name in available_important:
                group = ds.groups[table_name]
                print(f"\n  {table_name}:")
                
                for var_name in sorted(group.variables.keys())[:15]:
                    var = group.variables[var_name]
                    
                    # Get variable info
                    dtype = var.dtype
                    shape = var.shape
                    
                    # Get value range for numeric types
                    info_str = f"    {var_name:30s} {str(dtype):10s} {str(shape):20s}"
                    
                    if dtype in [np.float32, np.float64, np.int32, np.int64]:
                        try:
                            data = var[:]
                            valid_data = data[data != -999]
                            if len(valid_data) > 0:
                                info_str += f" range: [{valid_data.min():.2f}, {valid_data.max():.2f}]"
                        except:
                            pass
                    
                    print(info_str)
                
                if len(group.variables) > 15:
                    print(f"    ... ({len(group.variables)-15} more variables)")
        
        # Size information
        import os
        file_size = os.path.getsize(filepath)
        print(f"\n\nFILE SIZE: {file_size / 1024 / 1024:.2f} MB")
        
        ds.close()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def extract_stations_summary(filepath):
    """Extract and display station summary"""
    
    try:
        ds = nc.Dataset(filepath, 'r')
        
        if 'STATIONS' not in ds.groups:
            print("\nNo STATIONS table found")
            return
        
        stations = ds.groups['STATIONS']
        
        print(f"\n\nSTATIONS SUMMARY:")
        print("-" * 80)
        
        # Get key variables
        n_stations = len(stations.dimensions[list(stations.dimensions.keys())[0]])
        print(f"Total stations: {n_stations:,}")
        
        # Try to get some statistics
        if 'PROV_TERR_STATE_LOC' in stations.variables:
            print("\nStations by Province/Territory:")
            prov_data = nc.chartostring(stations.variables['PROV_TERR_STATE_LOC'][:])
            unique, counts = np.unique(prov_data, return_counts=True)
            for prov, count in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
                print(f"  {prov.decode('utf-8') if isinstance(prov, bytes) else prov}: {count}")
        
        if 'HYD_STATUS' in stations.variables:
            print("\nHydrometric Status:")
            status_data = nc.chartostring(stations.variables['HYD_STATUS'][:])
            unique, counts = np.unique(status_data, return_counts=True)
            for status, count in zip(unique, counts):
                status_str = status.decode('utf-8') if isinstance(status, bytes) else status
                print(f"  {status_str}: {count}")
        
        ds.close()
        
    except Exception as e:
        print(f"Error extracting stations: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_netcdf.py <hydat.nc>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    inspect_netcdf(filepath)
    extract_stations_summary(filepath)

if __name__ == "__main__":
    main()
