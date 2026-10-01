"""
Run this ONCE against one of your original training GRIB files to get the
exact pressure levels (and their order) the model was trained on.

Usage:
    python extract_pressure_levels.py path/to/pressure_temperature_2020-01.grib

This prints a Python list you can paste directly into config/pressure_levels.json
"""

import sys
import json
import xarray as xr

if len(sys.argv) != 2:
    print("Usage: python extract_pressure_levels.py <path_to_a_pressure_level_grib_file>")
    sys.exit(1)

path = sys.argv[1]
ds = xr.open_dataset(path, engine="cfgrib")

level_coord_name = None
for candidate in ["isobaricInhPa", "level", "pressure_level", "plev"]:
    if candidate in ds.coords:
        level_coord_name = candidate
        break

if level_coord_name is None:
    print("Could not find a pressure-level coordinate. Available coords:")
    print(list(ds.coords))
    sys.exit(1)

levels = [float(v) for v in ds.coords[level_coord_name].values]

print(f"\nFound coordinate '{level_coord_name}' with {len(levels)} levels:")
print(levels)

print(f"\nSanity check: {len(levels)} levels x 4 pressure variables = {len(levels) * 4} channels")
print(f"Plus 4 surface channels = {len(levels) * 4 + 4} total (should be 124)")

with open("pressure_levels_extracted.json", "w") as f:
    json.dump(levels, f, indent=2)
print("\nSaved to pressure_levels_extracted.json -- copy this into config/pressure_levels.json")
