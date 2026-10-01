"""
Fetches the most recently available ERA5 data from the Copernicus Climate
Data Store (CDS) and builds the (INPUT_LEN, C, H, W) input window the model
expects.

IMPORTANT -- ERA5 latency: ERA5T (near-real-time preliminary data) typically
lags 5 days behind the actual current date. This is NOT live weather -- it's
the most recent reanalysis data that actually exists. The API returns the
actual datetime used so the frontend can display it honestly rather than
implying this is real-time.

Setup required before this works:
1. Register for a free CDS account: https://cds.climate.copernicus.eu
2. Create ~/.cdsapirc with your URL and API key (CDS gives you this after
   registering) -- see https://cds.climate.copernicus.eu/api-how-to
3. pip install cdsapi
"""

import os
import tempfile
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import cdsapi

from model_utils import (
    PRESSURE_VARIABLES, PRESSURE_LEVELS_HPA, SURFACE_VARIABLES,
    N_LEVELS, N_SURFACE, C, ORIG_H, ORIG_W, PAD_H, PAD_W,
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, INPUT_LEN, STEP_HOURS,
)

ERA5T_LATENCY_DAYS = 5


def get_target_timesteps():
    """Returns INPUT_LEN datetimes, 6h apart, ending at the most recent
    plausibly-available ERA5T synoptic hour (00/06/12/18 UTC)."""
    now = datetime.utcnow()
    latest_possible = now - timedelta(days=ERA5T_LATENCY_DAYS)
    floored_hour = (latest_possible.hour // STEP_HOURS) * STEP_HOURS
    end_time = latest_possible.replace(hour=floored_hour, minute=0, second=0, microsecond=0)

    timesteps = [end_time - timedelta(hours=STEP_HOURS * i) for i in range(INPUT_LEN - 1, -1, -1)]
    return timesteps  # oldest -> newest


def _pad_to_multiple(arr, pad_h_target, pad_w_target):
    h, w = arr.shape[-2], arr.shape[-1]
    pad_h, pad_w = pad_h_target - h, pad_w_target - w
    if pad_h < 0 or pad_w < 0:
        raise RuntimeError(f"Fetched grid ({h}x{w}) is larger than expected padded size "
                            f"({pad_h_target}x{pad_w_target}) -- check CDS area/grid params.")
    pad_width = [(0, 0)] * (arr.ndim - 2) + [(0, pad_h), (0, pad_w)]
    return np.pad(arr, pad_width, mode="edge")


def _cds_request_dates_times(timesteps):
    dates = sorted({t.strftime("%Y-%m-%d") for t in timesteps})
    times = sorted({t.strftime("%H:00") for t in timesteps})
    return dates, times


def fetch_input_window(cds_client: cdsapi.Client = None, work_dir: str = None):
    """
    Returns:
        window: (INPUT_LEN, C, PAD_H, PAD_W) float32, physical units, padded
        timesteps: list of datetime objects actually used (oldest -> newest)
    """
    if C != 124:
        raise RuntimeError("model_utils.PRESSURE_LEVELS_HPA is not filled in correctly -- "
                            "fix that before fetching data (see model_utils.py).")

    client = cds_client or cdsapi.Client()
    work_dir = work_dir or tempfile.mkdtemp(prefix="era5_live_")
    timesteps = get_target_timesteps()
    dates, times = _cds_request_dates_times(timesteps)

    area = [LAT_MAX, LON_MIN, LAT_MIN, LON_MAX]  # [North, West, South, East]
    grid = [0.25, 0.25]

    # --- pressure-level variables ---
    pl_arrays = []
    for variable in PRESSURE_VARIABLES:
        target = os.path.join(work_dir, f"pressure_{variable}.grib")
        client.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis",
                "format": "grib",
                "variable": variable,
                "pressure_level": [str(p) for p in PRESSURE_LEVELS_HPA],
                "date": dates,
                "time": times,
                "area": area,
                "grid": grid,
            },
            target,
        )
        ds = xr.open_dataset(target, engine="cfgrib")
        var_name = list(ds.data_vars)[0]
        # select exactly the requested timesteps, in order (Cartesian product from CDS may include extras)
        ds_times = [t for t in ds.time.values]
        selected = ds.sel(time=[np.datetime64(t) for t in timesteps])
        pl_arrays.append(selected[var_name].values)  # (INPUT_LEN, N_LEVELS, H, W)

    pl_arr = np.concatenate(pl_arrays, axis=1)  # (INPUT_LEN, N_PRESSURE_CHANNELS, H, W)

    # --- surface variables ---
    target = os.path.join(work_dir, "surface.grib")
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "grib",
            "variable": SURFACE_VARIABLES,
            "date": dates,
            "time": times,
            "area": area,
            "grid": grid,
        },
        target,
    )
    sfc_ds = xr.open_dataset(target, engine="cfgrib")
    sfc_vars = list(sfc_ds.data_vars)
    sfc_selected = sfc_ds.sel(time=[np.datetime64(t) for t in timesteps])
    sfc_arr = np.stack([sfc_selected[v].values for v in sfc_vars], axis=1)  # (INPUT_LEN, N_SURFACE, H, W)

    full = np.concatenate([pl_arr, sfc_arr], axis=1).astype(np.float32)  # (INPUT_LEN, C, ORIG_H, ORIG_W)

    if full.shape[1] != C:
        raise RuntimeError(f"Fetched {full.shape[1]} channels but expected {C} -- "
                            f"check PRESSURE_LEVELS_HPA and variable lists match training exactly.")

    full_padded = _pad_to_multiple(full, PAD_H, PAD_W)
    return full_padded, timesteps
