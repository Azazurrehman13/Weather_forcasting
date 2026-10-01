"""
Manual-entry MET forecasting core (numpy only -- no torch import here, so this
logic can be unit-tested without the model).

Flow:  user-entered profile  ->  SI units  ->  fill gaps  ->  normalize
       ->  broadcast to (INPUT_LEN, C, H, W)  ->  model  ->  read the
       user's pixel  ->  de-normalize  ->  friendly units.
"""
import math
import os
from datetime import datetime, timedelta

import numpy as np

# ---------------------------------------------------------------- constants --
PRESSURE_VARIABLES = ["temperature", "u_component_of_wind", "v_component_of_wind", "geopotential"]
SFC_VARIABLES = ["t2m", "u10", "v10", "sp"]          # order = training channel order
# Level SET from the training download script; the ORDER is derived from pl_mean.npy.
KNOWN_LEVELS = [30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500,
                550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
N_LEVELS = len(KNOWN_LEVELS)
N_PL = N_LEVELS * len(PRESSURE_VARIABLES)            # 120
N_SFC = len(SFC_VARIABLES)                           # 4
C = N_PL + N_SFC                                     # 124

INPUT_LEN, HORIZON, STEP_HOURS = 4, 12, 6
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, RES = 23.5, 37.25, 60.75, 78.0, 0.25
ORIG_H, ORIG_W, PAD_H, PAD_W = 56, 70, 56, 72

G = 9.80665
WIND_TO_MS = {"ms": 1.0, "kt": 0.514444, "kmh": 1 / 3.6}

# From the notebook's evaluation cells (test set, real gridded inputs).
SKILL = {
    "leads": [6, 24, 48, 72],
    "r2_normalized": {
        "temperature": [0.837, 0.810, 0.764, 0.726], "u_component_of_wind": [0.756, 0.667, 0.575, 0.518],
        "v_component_of_wind": [0.572, 0.340, 0.129, 0.056], "geopotential": [0.894, 0.860, 0.799, 0.762],
        "t2m": [0.667, 0.667, 0.651, 0.635], "u10": [0.305, 0.236, 0.173, 0.148],
        "v10": [0.310, 0.234, 0.169, 0.140], "sp": [0.742, 0.690, 0.635, 0.596]},
    "rmse": {
        "temperature": [2.199, 2.357, 2.643, 2.857], "u_component_of_wind": [4.658, 5.388, 6.245, 6.737],
        "v_component_of_wind": [3.629, 5.076, 6.185, 6.585], "geopotential": [324.7, 361.5, 439.7, 486.2],
        "t2m": [4.984, 4.955, 5.092, 5.197], "u10": [1.528, 1.684, 1.783, 1.807],
        "v10": [1.652, 1.832, 1.919, 1.944], "sp": [260.0, 282.5, 297.2, 303.6]},
    "rmse_unit": {"temperature": "K", "u_component_of_wind": "m/s", "v_component_of_wind": "m/s",
                  "geopotential": "m2/s2", "t2m": "K", "u10": "m/s", "v10": "m/s", "sp": "Pa"},
}


# ------------------------------------------------------------------ geometry --
def nearest_grid_index(lat, lon):
    """(row, col) in the 56x70 native grid. ERA5 GRIB is stored NORTH -> SOUTH,
    so row 0 is LAT_MAX (verified against the t2m map in the training notebook)."""
    if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
        raise ValueError(f"lat/lon ({lat}, {lon}) is outside the model domain "
                         f"(lat {LAT_MIN}-{LAT_MAX}, lon {LON_MIN}-{LON_MAX}).")
    row = int(math.floor((LAT_MAX - lat) / RES + 0.5))
    col = int(math.floor((lon - LON_MIN) / RES + 0.5))
    return min(max(row, 0), ORIG_H - 1), min(max(col, 0), ORIG_W - 1)


# ------------------------------------------------------------- unit helpers --
def wind_to_uv(speed, direction_deg):
    """Meteorological convention: direction = where the wind blows FROM."""
    r = math.radians(direction_deg)
    return -speed * math.sin(r), -speed * math.cos(r)


def uv_to_wind(u, v):
    s = math.hypot(u, v)
    return s, (math.degrees(math.atan2(-u, -v)) % 360.0 if s > 1e-6 else 0.0)


# -------------------------------------------------------------------- stats --
class Stats:
    def __init__(self, pl_mean, pl_std, sfc_mean, sfc_std):
        if pl_mean.shape != (N_PL,) or pl_std.shape != (N_PL,):
            raise RuntimeError(f"pl_mean/pl_std must have shape ({N_PL},); got {pl_mean.shape}/{pl_std.shape}")
        for name, a in (("sfc_pixel_mean", sfc_mean), ("sfc_pixel_std", sfc_std)):
            if a.shape != (N_SFC, PAD_H, PAD_W):
                raise RuntimeError(f"{name} must have shape ({N_SFC},{PAD_H},{PAD_W}); got {a.shape}")
        self.pl_mean, self.pl_std = pl_mean.astype(np.float64), pl_std.astype(np.float64)
        self.sfc_mean, self.sfc_std = sfc_mean.astype(np.float64), sfc_std.astype(np.float64)
        self.levels = resolve_level_order(self.pl_mean)   # pressure (hPa) per level-index

    @classmethod
    def load(cls, stats_dir):
        f = lambda n: np.load(os.path.join(stats_dir, n))
        return cls(f("pl_mean.npy"), f("pl_std.npy"), f("sfc_pixel_mean.npy"), f("sfc_pixel_std.npy"))


def resolve_level_order(pl_mean):
    """Geopotential rises monotonically with altitude, so the geopotential block of
    pl_mean reveals the stored level order without needing the original GRIBs."""
    g = pl_mean[N_LEVELS * 3: N_LEVELS * 4]
    if np.all(np.diff(g) > 0):
        return sorted(KNOWN_LEVELS, reverse=True)   # 1000 -> 30 hPa
    if np.all(np.diff(g) < 0):
        return sorted(KNOWN_LEVELS)                 # 30 -> 1000 hPa
    raise RuntimeError("Geopotential means in pl_mean.npy are not monotonic -- the stats file "
                       "does not match the expected 4 variables x 30 levels layout.")


# ------------------------------------------------------------ input parsing --
def _num(x, name, lo, hi, unit):
    if not (lo <= x <= hi):
        raise ValueError(f"{name} = {x:g} is outside the plausible range {lo:g}..{hi:g} {unit}. Check your units.")
    return x


def _temp_k(v, unit, ctx):
    k = v + 273.15 if unit == "C" else v
    return _num(k, f"{ctx} temperature", 150, 340, "K (-123..67 C)")


def _wind(speed, direction, u, v, unit, ctx):
    sd, uv = (speed is not None or direction is not None), (u is not None or v is not None)
    if sd and uv:
        raise ValueError(f"{ctx}: give wind as speed+direction OR u+v, not both.")
    f = WIND_TO_MS[unit]
    if sd:
        if speed is None or direction is None:
            raise ValueError(f"{ctx}: wind needs BOTH speed and direction.")
        _num(speed * f, f"{ctx} wind speed", 0, 150, "m/s")
        _num(direction, f"{ctx} wind direction", 0, 360, "deg")
        return wind_to_uv(speed * f, direction)
    if uv:
        if u is None or v is None:
            raise ValueError(f"{ctx}: wind needs BOTH u and v.")
        _num(u * f, f"{ctx} u", -150, 150, "m/s")
        _num(v * f, f"{ctx} v", -150, 150, "m/s")
        return u * f, v * f
    return None


HARD_SIGMA = 8   # beyond this many training std-devs, a value is essentially impossible
                 # for this location -- we refuse to run the model on it instead of
                 # silently returning garbage (networks extrapolate very badly this far
                 # outside anything they were trained on).

EXTRAP_LENGTH_SCALE = 0.75   # ln(hPa); a rough guess at how far a single level's
                             # anomaly should carry -- NOT fit from real vertical
                             # correlations (the stats files only give per-level
                             # mean/std, no cross-level covariance to fit it from).


def _sfc_pretty(i, val_phys, mean_phys, std_phys):
    """SFC_VARIABLES physical units -> friendly display units for error/warning text."""
    if SFC_VARIABLES[i] == "sp":
        return val_phys / 100, mean_phys / 100, std_phys / 100, "hPa"
    if SFC_VARIABLES[i] == "t2m":
        return val_phys - 273.15, mean_phys - 273.15, std_phys, "\u00b0C"
    return val_phys, mean_phys, std_phys, "m/s"       # u10 / v10


def _fill(levels, known, clim):
    """Build a full profile from as few as ONE entered level. Entered levels are
    returned exactly (status 0). Every other level gets climatology PLUS a
    Gaussian-weighted blend of the anomalies (entered value - climatology) at
    the entered levels, in ln(pressure) distance -- so a level near an entered
    one is nudged away from the flat average instead of jumping straight to it,
    and the influence fades back to pure climatology (status 2) further away.
    This is a smoothing heuristic to avoid a visible step change next to your
    real data, not a model-based reconstruction of the missing levels."""
    n = len(levels)
    vals, status = clim.copy(), np.full(n, 2, dtype=int)
    if not known:
        return vals, status
    kp = np.array(sorted(known))
    klog = np.log(kp)
    kclim = np.array([clim[levels.index(p)] for p in kp])
    kanom = np.array([known[p] for p in kp]) - kclim
    for i, p in enumerate(levels):
        if p in known:
            vals[i], status[i] = known[p], 0
            continue
        w = np.exp(-((math.log(p) - klog) / EXTRAP_LENGTH_SCALE) ** 2)
        s = w.sum()
        if s > 1e-3:
            w_capped = w * min(1.0, 1.0 / s)          # never overshoot the entered anomalies
            vals[i] = clim[i] + float(np.dot(w_capped, kanom))
            if s > 0.05:
                status[i] = 1
    return vals, status


# ---------------------------------------------------------------- forecaster --
class Forecaster:
    """forward_fn: (INPUT_LEN, C, PAD_H, PAD_W) normalized float32 -> (HORIZON, C, ORIG_H, ORIG_W) normalized."""

    def __init__(self, stats, forward_fn, model_info=None):
        self.s, self.forward, self.model_info = stats, forward_fn, model_info or {}

    # -- typical values (training climatology) to scaffold the form ----------------
    def climatology(self, lat, lon):
        row, col = nearest_grid_index(lat, lon)
        s, n = self.s, N_LEVELS
        rows = []
        for i, p in enumerate(self.s.levels):
            u, v = s.pl_mean[n + i], s.pl_mean[2 * n + i]
            sp, d = uv_to_wind(u, v)
            rows.append({"pressure_hpa": p, "temperature_c": s.pl_mean[i] - 273.15, "wind_speed_ms": sp,
                         "wind_dir_deg": d, "u": u, "v": v, "height_m": s.pl_mean[3 * n + i] / G})
        rows.sort(key=lambda r: -r["pressure_hpa"])
        m, sd = s.sfc_mean[:, row, col], s.sfc_std[:, row, col]
        sp10, d10 = uv_to_wind(m[1], m[2])
        return {"grid_point": {"row": row, "col": col}, "levels": rows,
                "surface": {"t2m_c": m[0] - 273.15, "wind_speed_ms": sp10, "wind_dir_deg": d10,
                            "u10": m[1], "v10": m[2], "sp_hpa": m[3] / 100,
                            "t2m_c_std": sd[0], "u10_std": sd[1], "v10_std": sd[2], "sp_hpa_std": sd[3] / 100}}

    # -- main entry -----------------------------------------------------------------
    def predict(self, req):
        s, lv = self.s, self.s.levels
        row, col = nearest_grid_index(req["lat"], req["lon"])
        units = req.get("units") or {}
        tu, wu = units.get("temperature", "C"), units.get("wind", "ms")
        warnings = []

        # 1. levels -> SI dicts {pressure: value}
        known = {k: {} for k in PRESSURE_VARIABLES}
        for e in req["levels"]:
            p = int(e["pressure_hpa"])
            if p not in lv:
                raise ValueError(f"{p} hPa is not a model level. Valid levels: {sorted(lv, reverse=True)}")
            ctx = f"{p} hPa"
            if any(p in d for d in known.values()):
                raise ValueError(f"{ctx} appears more than once.")
            if e.get("temperature") is not None:
                known["temperature"][p] = _temp_k(e["temperature"], tu, ctx)
            w = _wind(e.get("wind_speed"), e.get("wind_dir"), e.get("u"), e.get("v"), wu, ctx)
            if w:
                known["u_component_of_wind"][p], known["v_component_of_wind"][p] = w
            if e.get("height") is not None:
                known["geopotential"][p] = _num(e["height"], f"{ctx} height", -500, 50000, "m") * G
        n_lv = len({p for d in known.values() for p in d})
        if n_lv < 1:
            raise ValueError("Enter values for at least 1 pressure level.")
        if n_lv == 1:
            warnings.append("Only 1 level entered. Every other level is a smoothed extrapolation "
                            "toward the training average, not measured or model-inferred data -- "
                            "treat this forecast as low confidence, especially far from that level.")
        elif n_lv < 10:
            warnings.append(f"Only {n_lv} levels entered; the rest are extrapolated/climatology "
                            f"fill, so the model sees mostly generic data away from your levels.")
        hs = sorted(known["geopotential"].items(), reverse=True)
        if any(b[1] <= a[1] for a, b in zip(hs, hs[1:])):
            warnings.append("Heights do not increase with altitude (falling pressure) -- check the height column.")

        # 2. fill gaps + normalize pressure-level channels (global per-channel stats)
        pl_phys, status = np.zeros(N_PL), {}
        for vi, var in enumerate(PRESSURE_VARIABLES):
            sl = slice(vi * N_LEVELS, (vi + 1) * N_LEVELS)
            pl_phys[sl], status[var] = _fill(lv, known[var], s.pl_mean[sl])
        pl_norm = (pl_phys - s.pl_mean) / s.pl_std
        entered = np.concatenate([status[v] == 0 for v in PRESSURE_VARIABLES])
        pl_bad = np.where(entered & (np.abs(pl_norm) > HARD_SIGMA))[0]
        if len(pl_bad):
            c = pl_bad[0]
            var, plevel = PRESSURE_VARIABLES[c // N_LEVELS], lv[c % N_LEVELS]
            mean, std = s.pl_mean[c], s.pl_std[c]
            raise ValueError(f"{var} at {plevel} hPa is {pl_norm[c]:+.0f} std from the training mean "
                             f"(typical there: {mean:.0f} \u00b1 {std:.0f}) -- far beyond anything the model "
                             f"was trained on, so I can't produce a meaningful forecast from it. Check units/values.")
        for c in np.where(entered & (np.abs(pl_norm) > 5) & (np.abs(pl_norm) <= HARD_SIGMA))[0][:5]:
            warnings.append(f"{PRESSURE_VARIABLES[c // N_LEVELS]} at {lv[c % N_LEVELS]} hPa is {pl_norm[c]:+.1f} "
                            f"std from the training mean -- unusual, check units/values.")

        # 3. surface -> anomaly at the user's pixel (NOT raw broadcast; see README)
        sf = req["surface"]
        wsfc = _wind(sf.get("wind_speed"), sf.get("wind_dir"), sf.get("u10"), sf.get("v10"), wu, "surface")
        if wsfc is None or sf.get("t2m") is None or sf.get("sp_hpa") is None:
            raise ValueError("Surface temperature, wind and pressure are all required.")
        sfc_phys = np.array([_temp_k(sf["t2m"], tu, "surface"), wsfc[0], wsfc[1],
                             _num(sf["sp_hpa"], "surface pressure", 300, 1100, "hPa") * 100])
        sfc_norm = (sfc_phys - s.sfc_mean[:, row, col]) / s.sfc_std[:, row, col]
        sfc_bad = np.where(np.abs(sfc_norm) > HARD_SIGMA)[0]
        if len(sfc_bad):
            i = sfc_bad[0]
            val, mean, std, unit = _sfc_pretty(i, sfc_phys[i], s.sfc_mean[i, row, col], s.sfc_std[i, row, col])
            raise ValueError(f"Surface {SFC_VARIABLES[i]} = {val:.1f} {unit} is {sfc_norm[i]:+.0f} std from typical "
                             f"at this exact location (typical here: {mean:.0f} \u00b1 {std:.0f} {unit}). That's far "
                             f"beyond anything the model was trained on -- I can't produce a meaningful forecast "
                             f"from it. Try \"Fill with typical values\" to see a realistic starting point, or "
                             f"check your units/decimal point/location.")
        for i in np.where((np.abs(sfc_norm) > 4) & (np.abs(sfc_norm) <= HARD_SIGMA))[0]:
            warnings.append(f"Surface {SFC_VARIABLES[i]} is {sfc_norm[i]:+.1f} std from the typical value at this "
                            f"lat/lon -- check the location, station height and units.")

        # 4. steady-state input window, uniform over the grid
        x = np.zeros((INPUT_LEN, C, PAD_H, PAD_W), dtype=np.float32)
        x[:, :N_PL] = pl_norm[None, :, None, None]
        x[:, N_PL:] = sfc_norm[None, :, None, None]

        # 5. run model, read the user's pixel, de-normalize
        pred = np.asarray(self.forward(x))[:, :, row, col].astype(np.float64)   # (HORIZON, C)
        phys = np.concatenate([pred[:, :N_PL] * s.pl_std + s.pl_mean,
                               pred[:, N_PL:] * s.sfc_std[:, row, col] + s.sfc_mean[:, row, col]], axis=1)

        # 6. friendly output
        t0 = None
        if req.get("obs_time"):
            try:
                t0 = datetime.fromisoformat(req["obs_time"].replace("Z", ""))
            except ValueError:
                raise ValueError("obs_time must be ISO format, e.g. 2026-09-24T06:00")
        n = N_LEVELS

        def level_rows(vec, st=None):
            out = []
            for i, p in enumerate(lv):
                u, v = vec[n + i], vec[2 * n + i]
                sp, d = uv_to_wind(u, v)
                r = {"pressure_hpa": p, "temperature_k": vec[i], "temperature_c": vec[i] - 273.15, "u": u, "v": v,
                     "wind_speed_ms": sp, "wind_dir_deg": d, "height_m": vec[3 * n + i] / G}
                if st is not None:
                    r["source"] = {"temperature": int(st["temperature"][i]), "wind": int(st["u_component_of_wind"][i]),
                                   "height": int(st["geopotential"][i])}
                out.append(r)
            return sorted(out, key=lambda r: -r["pressure_hpa"])

        def sfc_row(vec):
            sp, d = uv_to_wind(vec[1], vec[2])
            return {"t2m_k": vec[0], "t2m_c": vec[0] - 273.15, "u10": vec[1], "v10": vec[2],
                    "wind_speed_ms": sp, "wind_dir_deg": d, "sp_hpa": vec[3] / 100}

        forecast = [{"lead_hours": (k + 1) * STEP_HOURS,
                     "valid_time": (t0 + timedelta(hours=(k + 1) * STEP_HOURS)).isoformat() if t0 else None,
                     "surface": sfc_row(phys[k, N_PL:]), "levels": level_rows(phys[k])} for k in range(HORIZON)]
        return {"grid_point": {"row": row, "col": col}, "warnings": warnings,
                "input_used": {"surface": sfc_row(sfc_phys), "levels": level_rows(pl_phys, status)},
                "forecast": forecast}