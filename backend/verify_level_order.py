"""
Determines whether your pressure levels are stored ascending (30->1000 hPa)
or descending (1000->30 hPa) using a physical fact that doesn't depend on
having the original GRIB files: geopotential height increases monotonically
as pressure decreases (i.e. as you go up in altitude). So the geopotential
block in your saved pl_mean.npy must be either strictly increasing or
strictly decreasing -- whichever it is tells us the true stored order.

Run this from the backend/ folder:
    python verify_level_order.py

It auto-searches common locations for pl_mean.npy. To point at a specific
file instead, pass the path explicitly:
    python verify_level_order.py "model\\pl_mean.npy"
"""

import sys
import os
import json
import numpy as np

PRESSURE_VARIABLES = ["temperature", "u_component_of_wind", "v_component_of_wind", "geopotential"]
KNOWN_LEVELS_SET = [
    30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500,
    550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000,
]  # from download.py -- the SET is certain, the order is what we're checking


def find_pl_mean():
    """Checks the common places this file might be, so you don't have to
    remember your exact folder layout."""
    candidates = [
        "pl_mean.npy",
        "model/pl_mean.npy",
        "model\\pl_mean.npy",
        "config/pl_mean.npy",
        "../model/pl_mean.npy",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


if len(sys.argv) > 1:
    pl_mean_path = sys.argv[1]
    if not os.path.exists(pl_mean_path):
        print(f"ERROR: {pl_mean_path} not found.")
        sys.exit(1)
else:
    pl_mean_path = find_pl_mean()
    if pl_mean_path is None:
        print("Could not find pl_mean.npy in any of the usual locations "
              "(., model/, config/, ../model/).")
        print("Run this instead, giving the exact path:")
        print(r'    python verify_level_order.py "model\pl_mean.npy"')
        sys.exit(1)

print(f"Using: {os.path.abspath(pl_mean_path)}")
pl_mean = np.load(pl_mean_path)

n_levels = len(KNOWN_LEVELS_SET)
if len(pl_mean) != n_levels * len(PRESSURE_VARIABLES):
    print(f"WARNING: pl_mean.npy has {len(pl_mean)} channels, expected {n_levels * len(PRESSURE_VARIABLES)} "
          f"({n_levels} levels x {len(PRESSURE_VARIABLES)} variables). Something else may be wrong -- "
          f"double check this is the right file before trusting the result below.")

# geopotential is the 4th pressure variable -> last block of n_levels channels
geo_start = n_levels * (len(PRESSURE_VARIABLES) - 1)
geopotential_means = pl_mean[geo_start : geo_start + n_levels]

print("\nGeopotential mean per channel index (in the order stored in your training data):")
for i, val in enumerate(geopotential_means):
    print(f"  index {i:2d}: {val:10.2f}")

is_increasing = all(geopotential_means[i] < geopotential_means[i + 1] for i in range(len(geopotential_means) - 1))
is_decreasing = all(geopotential_means[i] > geopotential_means[i + 1] for i in range(len(geopotential_means) - 1))

if is_increasing:
    true_order = sorted(KNOWN_LEVELS_SET, reverse=True)  # 1000 -> 30
    direction = "DESCENDING pressure (1000 hPa first, 30 hPa last) -- ascending altitude"
elif is_decreasing:
    true_order = sorted(KNOWN_LEVELS_SET)  # 30 -> 1000
    direction = "ASCENDING pressure (30 hPa first, 1000 hPa last) -- descending altitude"
else:
    print("\nERROR: geopotential means are NOT monotonic in either direction. This means either:")
    print("  - pl_mean.npy doesn't actually correspond to this variable/level layout, or")
    print("  - something is wrong with the saved stats.")
    print("Do not guess an order from this -- paste these values back for a closer look.")
    sys.exit(1)

print(f"\nDetected order: {direction}")
print(f"Confirmed pressure level order: {true_order}")

os.makedirs("config", exist_ok=True)
with open("config/pressure_levels.json", "w") as f:
    json.dump(true_order, f, indent=2)
print("\nWrote config/pressure_levels.json with the verified order.")