#!/usr/bin/env python3
"""
Remove the comment attribute from upstream_area variable in all sediment NetCDF files
"""

import netCDF4 as nc
import numpy as np
from pathlib import Path
import tempfile
import shutil

def remove_upstream_area_comment(nc_file):
    """
    Remove the comment attribute from upstream_area variable
    """
    # Create a temporary file
    temp_file = nc_file.parent / f"{nc_file.stem}_temp.nc"

    try:
        # Open original file for reading
        with nc.Dataset(nc_file, 'r') as src:
            # Check if upstream_area exists
            if 'upstream_area' not in src.variables:
                return False

            # Check if it has a comment attribute
            if not hasattr(src.variables['upstream_area'], 'comment'):
                return False

            # Create new file
            with nc.Dataset(temp_file, 'w', format='NETCDF4') as dst:
                # Copy dimensions
                for name, dimension in src.dimensions.items():
                    dst.createDimension(name, len(dimension) if not dimension.isunlimited() else None)

                # Copy variables
                for name, variable in src.variables.items():
                    # Determine chunksizes
                    chunksizes = None
                    if variable.chunking() != 'contiguous':
                        chunksizes = variable.chunking()

                    # Create variable
                    fill_value = variable._FillValue if hasattr(variable, '_FillValue') else None

                    dst_var = dst.createVariable(
                        name,
                        variable.datatype,
                        variable.dimensions,
                        fill_value=fill_value,
                        chunksizes=chunksizes
                    )

                    # Copy variable attributes, excluding comment for upstream_area
                    for attr_name in variable.ncattrs():
                        if attr_name == '_FillValue':
                            continue  # Already set
                        if name == 'upstream_area' and attr_name == 'comment':
                            continue  # Skip this attribute
                        dst_var.setncattr(attr_name, variable.getncattr(attr_name))

                    # Copy data
                    dst_var[:] = variable[:]

                # Copy global attributes
                for attr_name in src.ncattrs():
                    dst.setncattr(attr_name, src.getncattr(attr_name))

        # Replace original file with updated file
        shutil.move(str(temp_file), str(nc_file))
        return True

    except Exception as e:
        # Clean up temp file if it exists
        if temp_file.exists():
            temp_file.unlink()
        raise e


def main():
    sediment_dir = Path('/Users/zhongwangwei/Downloads/Sediment/Source/Station/Hydat/sediment')
    sediment_files = sorted(sediment_dir.glob('HYDAT_*_SEDIMENT.nc'))

    print(f"Removing comment attribute from upstream_area in {len(sediment_files)} files...")
    print("=" * 70)

    success_count = 0
    skipped_count = 0
    error_count = 0

    for i, nc_file in enumerate(sediment_files, 1):
        try:
            result = remove_upstream_area_comment(nc_file)
            if result:
                print(f"[{i}/{len(sediment_files)}] ✓ Updated: {nc_file.name}")
                success_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            print(f"[{i}/{len(sediment_files)}] ✗ Error: {nc_file.name}: {e}")
            error_count += 1

    print("=" * 70)
    print(f"Processing complete:")
    print(f"  Updated: {success_count}")
    print(f"  Skipped (no comment): {skipped_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total: {len(sediment_files)}")
    print("=" * 70)


if __name__ == '__main__':
    main()
