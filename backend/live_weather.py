"""
Fetches real current conditions + Open-Meteo's own forecast from the free,
keyless Open-Meteo API, and compares our model's forecast against it.

HONESTY NOTE, read before trusting any number this produces:
Open-Meteo's historical archive only has SURFACE variables for past dates --
it does NOT provide pressure-level (upper-air) data for anything except its
own forecast (current + future). So there is no free way to get true
"already-happened, ground-truth" upper-air data to backtest against. What
this module actually does is:
  1. Fetch Open-Meteo's CURRENT conditions -> feed into our model as input
     (a live analogue of "Fill with typical values", but real instead of
     climatology).
  2. Fetch Open-Meteo's OWN forecast for the same future times our model
     predicts (+6h..+72h) -> use that as the comparison target.
This is a MODEL-VS-MODEL comparison (ours vs. Open-Meteo's ECMWF/GFS blend),
not a comparison against observed ground truth. It's still a genuinely useful
sanity check -- Open-Meteo's underlying models are operational, real-world
systems -- but it is not the same claim as "accuracy against reality", and
the API response says so explicitly so nothing downstream can quietly imply
otherwise.

LEVEL COVERAGE: Open-Meteo's standard pressure levels are a coarser set than
ours. Only levels present in BOTH lists are compared -- levels are never
interpolated to fill gaps here, because presenting an interpolated value as
"the actual" would be misleading in a tool whose whole point is telling the
truth about accuracy.
"""

import urllib.request
import urllib.parse
import json
from datetime import datetime, timezone

import numpy as np

import met_core as mc

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo's standard pressure-level set (generic Weather Forecast API).
OPEN_METEO_LEVELS = [1000, 975, 950, 925, 900, 850, 800, 700, 600, 500,
                     400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10]

# Only compare at levels present in both the model's levels and Open-Meteo's.
COMPARABLE_LEVELS = sorted(set(OPEN_METEO_LEVELS) & set(mc.KNOWN_LEVELS), reverse=True)


def _fetch_json(url: str, timeout: float = 15.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _build_hourly_var_list():
    surface_vars = ["temperature_2m", "wind_speed_10m", "wind_direction_10m", "surface_pressure"]
    level_vars = []
    for p in COMPARABLE_LEVELS:
        level_vars += [f"temperature_{p}hPa", f"wind_speed_{p}hPa",
                       f"wind_direction_{p}hPa", f"geopotential_height_{p}hPa"]
    return surface_vars + level_vars


def fetch_open_meteo(lat: float, lon: float, forecast_days: int = 4) -> dict:
    """Raw fetch -- returns Open-Meteo's hourly response as-is (times + arrays)."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(_build_hourly_var_list()),
        "forecast_days": forecast_days,
        "timezone": "UTC",
        "wind_speed_unit": "ms",
        "temperature_unit": "celsius",
    }
    url = OPEN_METEO_URL + "?" + urllib.parse.urlencode(params)
    try:
        data = _fetch_json(url)
    except Exception as e:
        raise RuntimeError(f"Could not reach Open-Meteo ({e}). Check your internet connection.")
    if "hourly" not in data:
        raise RuntimeError(f"Open-Meteo response missing 'hourly' data: {data.get('reason', data)}")
    return data


def _nearest_hour_index(times: list, target: datetime) -> int:
    parsed = [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in times]
    diffs = [abs((t - target).total_seconds()) for t in parsed]
    return int(np.argmin(diffs))


def build_predict_request_from_live(lat: float, lon: float, om: dict) -> tuple:
    """Turns Open-Meteo's CURRENT hour into a met_core-shaped predict() request.
    Returns (request_dict, now_index, hourly_data, missing_levels)."""
    h = om["hourly"]
    now = datetime.now(timezone.utc)
    now_idx = _nearest_hour_index(h["time"], now)

    def g(key, idx):
        v = h.get(key, [None] * len(h["time"]))[idx]
        return None if v is None else float(v)

    surface = {
        "t2m": g("temperature_2m", now_idx),
        "sp_hpa": g("surface_pressure", now_idx),
        "wind_speed": g("wind_speed_10m", now_idx),
        "wind_dir": g("wind_direction_10m", now_idx),
    }
    if None in surface.values():
        raise RuntimeError("Open-Meteo did not return complete surface data for the current hour.")

    levels, missing = [], []
    for p in mc.KNOWN_LEVELS:
        if p not in COMPARABLE_LEVELS:
            continue  # not in Open-Meteo's set at all -- left for the model to extrapolate/climatology-fill
        t = g(f"temperature_{p}hPa", now_idx)
        ws = g(f"wind_speed_{p}hPa", now_idx)
        wd = g(f"wind_direction_{p}hPa", now_idx)
        hgt = g(f"geopotential_height_{p}hPa", now_idx)
        if None in (t, ws, wd, hgt):
            missing.append(p)
            continue
        levels.append({"pressure_hpa": p, "temperature": t, "wind_speed": ws, "wind_dir": wd, "height": hgt})

    req = {
        "lat": lat, "lon": lon,
        "obs_time": h["time"][now_idx],
        "units": {"temperature": "C", "wind": "ms"},
        "surface": surface,
        "levels": levels,
    }
    return req, now_idx, h, missing


def _safe(x):
    """Converts NaN -> None. Required before anything goes into a JSON
    response: Python's float('nan') is not valid JSON, and Starlette's
    default JSONResponse enforces that strictly (allow_nan=False) -- a NaN
    anywhere in the response body crashes the whole endpoint with a 500
    ('Out of range float values are not JSON compliant'), not just that one
    field. None/null is valid JSON and matches the intended 'n/a' meaning."""
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    return x


def _skill_relative_pct(abs_error: float, rmse_scale: float) -> float:
    """'vs. model's typical error' -- how many test-set RMSEs off this single
    live miss is. A STRICTER, statistics-facing number: 100% only if the miss
    is much smaller than the model's own known typical error; ~37% if the
    miss equals that typical error; drops further as it gets worse than
    typical. NOT the same thing as 'how close is this to the true value' --
    see _value_relative_pct for that. Smooth exponential decay (never a hard
    cliff to 0) so a 1.7x-typical miss and a 20x-typical miss stay
    distinguishable instead of both showing an identical floor value."""
    if rmse_scale <= 0:
        return float("nan")
    return 100.0 * np.exp(-abs_error / rmse_scale)


def _value_relative_pct(abs_error: float, reference_value: float) -> float:
    """The intuitive 'accuracy %' most people mean: how close the prediction
    is to the actual value, as a plain percentage of that value. Uses smooth
    exponential decay for the same reason as _skill_relative_pct (no hard
    cliff), but scaled by |reference_value| instead of the model's RMSE.

    IMPORTANT: reference_value must never be a Celsius temperature -- Celsius
    crosses zero (0C is a valid, common value, not 'no temperature'), so
    dividing by it is undefined at 0C and meaningless near it. Pass the
    Kelvin value for temperature instead (always positive, no zero-crossing)."""
    ref = abs(reference_value)
    if ref < 1e-6:
        return float("nan")  # reference itself is ~zero -- any % of it is meaningless, don't fabricate a number
    return 100.0 * np.exp(-abs_error / ref)


VS_TYPICAL_WARNING_THRESHOLD = 20.0  # below this avg %, flag the variable as unreliable for this run


def _check_model_output_sanity(model_result: dict) -> list:
    """Checks the model's OWN forecast (independent of Open-Meteo) for
    physically impossible output -- the kind of thing that indicates the
    uniform-broadcast manual-entry simplification has broken down for this
    input, not just 'the forecast is somewhat off'."""
    warnings = []
    seen_negative_height_levels = set()
    non_monotonic_leads = set()

    for f in model_result["forecast"]:
        levels_sorted = sorted(f["levels"], key=lambda l: l["pressure_hpa"], reverse=True)  # high pressure (low alt) first
        prev_height = None
        for lvl in levels_sorted:
            if lvl["height_m"] < 0:
                seen_negative_height_levels.add(lvl["pressure_hpa"])
            if prev_height is not None and lvl["height_m"] <= prev_height:
                # height must strictly increase as pressure decreases (going up in altitude)
                non_monotonic_leads.add(f["lead_hours"])
            prev_height = lvl["height_m"]

    if seen_negative_height_levels:
        levels_str = ", ".join(str(p) for p in sorted(seen_negative_height_levels, reverse=True))
        warnings.append(
            f"PHYSICALLY IMPOSSIBLE OUTPUT: negative height at {levels_str} hPa in at least one lead time. "
            f"Height cannot be negative at these levels for this region. This is a known failure mode of "
            f"manual-entry mode's uniform-broadcast simplification, not a display bug -- treat this "
            f"forecast's low-level fields as unreliable."
        )
    if non_monotonic_leads:
        warnings.append(
            f"Height is not monotonically increasing with altitude in {len(non_monotonic_leads)} lead time(s) "
            f"(e.g. +{min(non_monotonic_leads)}h) -- another sign the input profile is producing an "
            f"unstable/unphysical forecast, likely for the same reason as above."
        )
    return warnings


def _check_comparison_sanity(lead_results: list) -> list:
    """Flags any variable whose average 'vs. typical error' is so low that
    the headline 'pct_of_value' number (which excludes temperature) could be
    hiding a serious problem a quick glance wouldn't catch."""
    warnings = []
    groups = {"temperature": [], "wind_speed": [], "height": [], "surface_t2m": [], "surface_wind": [], "sp": []}
    for lead in lead_results:
        for sv in lead["surface"]:
            if sv["variable"] == "t2m":
                groups["surface_t2m"].append(sv["vs_typical_error_pct"])
            elif sv["variable"] == "surface_wind_speed":
                groups["surface_wind"].append(sv["vs_typical_error_pct"])
            elif sv["variable"] == "sp":
                groups["sp"].append(sv["vs_typical_error_pct"])
        for r in lead["levels"]:
            groups["temperature"].append(r["temperature"]["vs_typical_error_pct"])
            if r["wind_speed"]:
                groups["wind_speed"].append(r["wind_speed"]["vs_typical_error_pct"])
            if r["height"]:
                groups["height"].append(r["height"]["vs_typical_error_pct"])

    labels = {"temperature": "Pressure-level temperature", "wind_speed": "Pressure-level wind speed",
              "height": "Height", "surface_t2m": "Surface temperature (t2m)",
              "surface_wind": "Surface wind speed", "sp": "Surface pressure"}
    for key, vals in groups.items():
        vals = [v for v in vals if v is not None]
        if vals and (avg := sum(vals) / len(vals)) < VS_TYPICAL_WARNING_THRESHOLD:
            warnings.append(
                f"{labels[key]} is averaging only {avg:.0f}% vs. the model's own typical error across "
                f"this run -- this variable's forecast is far outside what the model normally produces "
                f"and should not be trusted here, even though it may not show up in the headline % above."
            )
    return warnings


def compare_forecast_to_open_meteo(model_result: dict, om_hourly: dict, obs_time_iso: str) -> dict:
    """model_result: the dict returned by met_core.Forecaster.predict().
    Returns per-lead, per-variable comparisons plus overall accuracy %."""
    t0 = datetime.fromisoformat(obs_time_iso).replace(tzinfo=timezone.utc)
    times = om_hourly["time"]

    def g(key, idx):
        v = om_hourly.get(key, [None] * len(times))[idx]
        return None if v is None else float(v)

    lead_results = []
    all_pct = []

    for f in model_result["forecast"]:
        lead_h = f["lead_hours"]
        target_time = t0 + __import__("datetime").timedelta(hours=lead_h)
        idx = _nearest_hour_index(times, target_time)
        actual_time = times[idx]

        skill_idx = min(range(len(mc.SKILL["leads"])),
                         key=lambda i: abs(mc.SKILL["leads"][i] - lead_h))

        def entry(name, model_val, om_val, unit, rmse_scale, is_temperature=False):
            """'% of value' is skipped (NaN) for temperature.
            It's not just a divide-by-zero
            risk (Celsius crosses zero); even the Kelvin-based version is
            misleading, since Kelvin's ~200-300K baseline is so large that
            almost ANY realistic forecast error (even 5-10C, a genuinely bad
            miss) looks like ~97-100% 'accurate' next to it. There's no
            physically meaningful non-arbitrary zero for temperature the way
            there is for wind speed (0 = no wind) or height, so a ratio-based
            percentage doesn't mean anything real for this variable --
            vs_typical_error_pct is the correct, honest number to use here."""
            err = abs(model_val - om_val)
            pct_of_value = None if is_temperature else _safe(_value_relative_pct(err, abs(om_val)))
            return {
                "variable": name, "model": model_val, "open_meteo": om_val,
                "abs_error": err, "unit": unit,
                "pct_of_value": pct_of_value,                                    # None, never NaN
                "vs_typical_error_pct": _safe(_skill_relative_pct(err, rmse_scale)),  # None, never NaN
            }

        surface_vars = []
        om_t2m, om_ws, om_wd, om_sp = g("temperature_2m", idx), g("wind_speed_10m", idx), g("wind_direction_10m", idx), g("surface_pressure", idx)
        if om_t2m is not None:
            surface_vars.append(entry("t2m", f["surface"]["t2m_c"], om_t2m, "C",
                                       mc.SKILL["rmse"]["t2m"][skill_idx], is_temperature=True))
        if om_ws is not None:
            surface_vars.append(entry("surface_wind_speed", f["surface"]["wind_speed_ms"], om_ws, "m/s",
                                       mc.SKILL["rmse"]["u10"][skill_idx]))
        if om_sp is not None:
            sp_err_hpa = abs(f["surface"]["sp_hpa"] - om_sp)
            surface_vars.append({
                "variable": "sp", "model": f["surface"]["sp_hpa"], "open_meteo": om_sp,
                "abs_error": sp_err_hpa, "unit": "hPa",
                "pct_of_value": _safe(_value_relative_pct(sp_err_hpa, om_sp)),
                # SKILL's sp RMSE is stored in Pa (matches training's physical units); convert error to Pa to match
                "vs_typical_error_pct": _safe(_skill_relative_pct(sp_err_hpa * 100, mc.SKILL["rmse"]["sp"][skill_idx])),
            })

        level_rows = []
        for lvl in f["levels"]:
            p = lvl["pressure_hpa"]
            if p not in COMPARABLE_LEVELS:
                continue
            om_t = g(f"temperature_{p}hPa", idx)
            om_w = g(f"wind_speed_{p}hPa", idx)
            om_h = g(f"geopotential_height_{p}hPa", idx)
            if om_t is None:
                continue
            temp_entry = entry("temperature", lvl["temperature_c"], om_t, "C",
                                mc.SKILL["rmse"]["temperature"][skill_idx], is_temperature=True)
            wind_entry = None if om_w is None else entry(
                "wind_speed", lvl["wind_speed_ms"], om_w, "m/s", mc.SKILL["rmse"]["u_component_of_wind"][skill_idx])
            height_entry = None if om_h is None else entry(
                "height", lvl["height_m"], om_h, "m", mc.SKILL["rmse"]["geopotential"][skill_idx] / 9.80665)
            level_rows.append({"pressure_hpa": p, "temperature": temp_entry,
                                "wind_speed": wind_entry, "height": height_entry})

        # headline % uses pct_of_value (the intuitive "how close to actual" number) -- not the
        # stricter skill-relative one, since that's what most people mean by "accuracy %".
        # pct_of_value is None (not NaN) for anything not applicable (temperature, or a rare
        # near-zero reference) -- `is not None` is the correct check here, not np.isnan, which
        # would raise on None rather than filter it out.
        pcts = [v["pct_of_value"] for v in surface_vars if v["pct_of_value"] is not None]
        for r in level_rows:
            for key in ("temperature", "wind_speed", "height"):
                if r[key] is not None and r[key]["pct_of_value"] is not None:
                    pcts.append(r[key]["pct_of_value"])
        lead_avg = float(np.mean(pcts)) if pcts else None
        all_pct.extend(pcts)

        lead_results.append({
            "lead_hours": lead_h,
            "target_time_requested": target_time.isoformat(),
            "target_time_actual_open_meteo": actual_time,
            "surface": surface_vars,
            "levels": level_rows,
            "lead_accuracy_pct": lead_avg,
        })

    overall_pct = float(np.mean(all_pct)) if all_pct else None

    warnings = _check_model_output_sanity(model_result) + _check_comparison_sanity(lead_results)

    return {
        "overall_accuracy_pct": overall_pct,  # = average "% of value" (the intuitive number) across everything compared; None if nothing was comparable
        "warnings": warnings,  # non-empty means: do NOT trust this run at face value, read these before the headline %
        "metric_explainer": (
            "Each variable shows TWO percentages: 'pct_of_value' is the intuitive one -- how close "
            "the prediction is to Open-Meteo's number, as a plain percentage of that value. This is "
            "NOT shown for temperature -- there's no physically meaningful non-arbitrary zero for "
            "temperature (unlike wind speed, where 0 m/s means no wind), so a ratio-based percentage "
            "doesn't mean anything real for it (Kelvin's huge baseline would make almost any error, "
            "even a bad 5-10C miss, falsely look like ~97-100%). Use vs_typical_error_pct for "
            "temperature instead. 'vs_typical_error_pct' (shown for every variable) is stricter -- how "
            "this single miss compares to the MODEL'S OWN known typical error from real evaluation "
            "(100% only if this miss is much smaller than that typical error). The headline "
            "overall_accuracy_pct above averages pct_of_value across wind/pressure/height only -- "
            "temperature is excluded from it for the reason above, not omitted by accident."
        ),
        "compared_levels": COMPARABLE_LEVELS,
        "uncompared_levels": sorted(set(mc.KNOWN_LEVELS) - set(COMPARABLE_LEVELS), reverse=True),
        "by_lead": lead_results,
        "disclaimer": (
            "This compares our model's forecast against Open-Meteo's OWN forecast "
            "(ECMWF/GFS-based) for the same future times -- it is a model-vs-model "
            "sanity check, not a comparison against observed ground truth. Open-Meteo's "
            "historical archive does not provide free upper-air data for past dates, so "
            "true backtesting against observations isn't available without a paid/CDS source."
        ),
    }