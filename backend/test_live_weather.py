"""Tests live_weather.py's parsing/comparison logic with a synthetic Open-Meteo
response (no real network call) -- run: python -m pytest tests -q"""
import os, sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import met_core as mc
import live_weather as lw


def fake_hourly(base_time, n_hours=97):
    """Builds a synthetic Open-Meteo 'hourly' block: constant values everywhere
    except an offset per hour so we can check the right hour gets picked."""
    times = [(base_time + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(n_hours)]
    h = {"time": times}
    h["temperature_2m"] = [10.0 + 0.01 * i for i in range(n_hours)]
    h["wind_speed_10m"] = [5.0] * n_hours
    h["wind_direction_10m"] = [270.0] * n_hours
    h["surface_pressure"] = [950.0] * n_hours
    for p in lw.COMPARABLE_LEVELS:
        h[f"temperature_{p}hPa"] = [(-40.0 + 0.001 * p) + 0.01 * i for i in range(n_hours)]
        h[f"wind_speed_{p}hPa"] = [20.0] * n_hours
        h[f"wind_direction_{p}hPa"] = [250.0] * n_hours
        h[f"geopotential_height_{p}hPa"] = [float(p) * 10 for i in range(n_hours)] if False else [5000.0 + p for i in range(n_hours)]
    return h


def test_comparable_levels_is_real_intersection():
    assert set(lw.COMPARABLE_LEVELS).issubset(set(mc.KNOWN_LEVELS))
    assert set(lw.COMPARABLE_LEVELS).issubset(set(lw.OPEN_METEO_LEVELS))
    assert len(lw.COMPARABLE_LEVELS) > 10   # sanity: should be a substantial overlap, not near-empty


def test_build_request_picks_nearest_hour_and_uses_comparable_levels_only():
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    h = fake_hourly(now - timedelta(hours=1))  # base time slightly in the past so "now" falls inside the series
    om = {"hourly": h}
    req, idx, hourly, missing = lw.build_predict_request_from_live(30.5, 69.5, om)

    assert req["surface"]["t2m"] == h["temperature_2m"][idx]
    assert req["surface"]["sp_hpa"] == h["surface_pressure"][idx]
    got_levels = {l["pressure_hpa"] for l in req["levels"]}
    assert got_levels == set(lw.COMPARABLE_LEVELS)      # only comparable levels are sent to the model
    assert missing == []                                 # fake data has no gaps


def test_perfect_match_gives_100pct_accuracy():
    """If the model's forecast exactly equals Open-Meteo's forecast at every
    lead, accuracy should be (very close to) 100% everywhere."""
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    base = now - timedelta(hours=1)
    h = fake_hourly(base)
    obs_time_iso = base.isoformat()

    # Build a model_result whose forecast values equal Open-Meteo's at each lead's nearest hour.
    forecast = []
    for lead_h in [(k + 1) * mc.STEP_HOURS for k in range(mc.HORIZON)]:
        target = datetime.fromisoformat(obs_time_iso) + timedelta(hours=lead_h)
        idx = lw._nearest_hour_index(h["time"], target)
        levels = [{"pressure_hpa": p, "temperature_c": h[f"temperature_{p}hPa"][idx],
                   "wind_speed_ms": h[f"wind_speed_{p}hPa"][idx],
                   "height_m": h[f"geopotential_height_{p}hPa"][idx]} for p in lw.COMPARABLE_LEVELS]
        forecast.append({"lead_hours": lead_h,
                          "surface": {"t2m_c": h["temperature_2m"][idx], "wind_speed_ms": h["wind_speed_10m"][idx],
                                      "sp_hpa": h["surface_pressure"][idx]},
                          "levels": levels})
    model_result = {"forecast": forecast}

    cmp = lw.compare_forecast_to_open_meteo(model_result, h, obs_time_iso)
    assert cmp["overall_accuracy_pct"] == 100.0 or abs(cmp["overall_accuracy_pct"] - 100.0) < 0.01
    for lead in cmp["by_lead"]:
        for sv in lead["surface"]:
            assert sv["abs_error"] < 1e-9


def test_large_error_gives_low_accuracy_not_negative_or_nan():
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    base = now - timedelta(hours=1)
    h = fake_hourly(base)
    obs_time_iso = base.isoformat()

    forecast = [{"lead_hours": 6, "surface": {"t2m_c": 10.0 + 100, "wind_speed_ms": 5.0, "sp_hpa": 950.0},
                 "levels": [{"pressure_hpa": p, "temperature_c": -40.0, "wind_speed_ms": 20.0, "height_m": 5000.0 + p}
                            for p in lw.COMPARABLE_LEVELS]}]
    model_result = {"forecast": forecast}
    cmp = lw.compare_forecast_to_open_meteo(model_result, h, obs_time_iso)
    t2m_entry = next(v for v in cmp["by_lead"][0]["surface"] if v["variable"] == "t2m")
    # vs_typical_error_pct: very low but never negative -- exponential decay, not a hard clamp
    assert 0.0 <= t2m_entry["vs_typical_error_pct"] < 5.0
    assert cmp["overall_accuracy_pct"] is not None


def test_moderate_overshoot_is_not_flattened_to_zero():
    """An error a bit past 1x the typical scale should NOT show the same 0%
    as a wildly-wrong prediction -- this is exactly the bug the user caught
    (e.g. 30 hPa height: 53.4m error vs ~31m scale, only 1.7x, used to show
    0% under the old linear-clamp formula)."""
    just_over_scale = lw._skill_relative_pct(abs_error=1.7, rmse_scale=1.0)
    way_over_scale = lw._skill_relative_pct(abs_error=20.0, rmse_scale=1.0)
    assert just_over_scale > 15.0            # meaningfully nonzero, distinguishable from "wildly wrong"
    assert way_over_scale < 1.0              # still correctly shows as very low
    assert just_over_scale > way_over_scale  # and clearly distinguishes the two magnitudes


def test_value_relative_pct_matches_intuitive_expectation():
    """6.6 hPa off on an 866.6 hPa reading should read as CLOSE (~99%), not
    the near-0% the old single-metric design showed -- this is the exact
    case the user flagged as 'wrong percentage'."""
    pct = lw._value_relative_pct(abs_error=6.6, reference_value=866.6)
    assert pct > 95.0


def test_value_relative_pct_uses_kelvin_safely_across_zero_celsius():
    """0C is a normal, common temperature -- must not divide by zero or blow
    up near it. This is a generic safety check on the raw function; note that
    temperature no longer actually uses this path in the real pipeline (see
    the next test) because even the Kelvin version is misleading, not just
    unsafe -- but the function itself must still not crash if ever called
    this way."""
    pct_at_freezing = lw._value_relative_pct(abs_error=3.9, reference_value=0.0 + 273.15)
    assert 0.0 <= pct_at_freezing <= 100.0
    assert not __import__("math").isnan(pct_at_freezing)


def test_temperature_pct_of_value_is_none_not_a_fake_high_number():
    """A 5.4C miss (a genuinely bad forecast) must NOT show as ~98% via a
    Kelvin-ratio trick -- pct_of_value should be explicitly None for
    temperature so the frontend shows 'n/a' rather than a misleadingly
    reassuring number. vs_typical_error_pct must still work normally.

    It must be None, not NaN: Starlette's JSONResponse has allow_nan=False,
    so a raw NaN anywhere in the response crashes the WHOLE endpoint with a
    500 ('Out of range float values are not JSON compliant') -- this exact
    bug happened once already. None serializes as JSON null and is safe."""
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    base = now - timedelta(hours=1)
    h = fake_hourly(base)
    obs_time_iso = base.isoformat()

    forecast = [{"lead_hours": 6,
                 "surface": {"t2m_c": h["temperature_2m"][1] + 5.4, "wind_speed_ms": 5.0, "sp_hpa": 950.0},
                 "levels": [{"pressure_hpa": p, "temperature_c": -40.0, "wind_speed_ms": 20.0, "height_m": 5000.0 + p}
                            for p in lw.COMPARABLE_LEVELS]}]
    cmp = lw.compare_forecast_to_open_meteo({"forecast": forecast}, h, obs_time_iso)
    t2m = next(v for v in cmp["by_lead"][0]["surface"] if v["variable"] == "t2m")

    assert t2m["pct_of_value"] is None                   # never a fake high number for a real miss, and never NaN
    assert t2m["vs_typical_error_pct"] is not None        # the honest metric still works
    # overall_accuracy_pct must not itself become None/crash just because temperature's pct_of_value is None
    assert cmp["overall_accuracy_pct"] is not None


def test_response_is_actually_json_serializable():
    """Directly reproduces the real bug: NaN anywhere in the response crashes
    Starlette's JSONResponse with a 500, even though the Python dict itself
    looks fine. This builds a case that includes a temperature entry (which
    used to carry a raw NaN) and asserts the WHOLE response round-trips
    through json.dumps with allow_nan=False, exactly like Starlette does."""
    import json
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    base = now - timedelta(hours=1)
    h = fake_hourly(base)
    obs_time_iso = base.isoformat()

    forecast = [{"lead_hours": 6, "surface": {"t2m_c": 999.0, "wind_speed_ms": 5.0, "sp_hpa": 950.0},
                 "levels": [{"pressure_hpa": p, "temperature_c": -40.0, "wind_speed_ms": 20.0, "height_m": 5000.0 + p}
                            for p in lw.COMPARABLE_LEVELS]}]
    cmp = lw.compare_forecast_to_open_meteo({"forecast": forecast}, h, obs_time_iso)
    json.dumps(cmp, allow_nan=False)  # raises ValueError if any NaN slipped through -- the exact crash from before


def test_negative_height_triggers_a_warning():
    """Reproduces the real -23.1m @ 1000hPa case the user hit twice -- this
    must surface as an explicit warning, not pass through silently."""
    model_result = {"forecast": [{"lead_hours": 6, "surface": {"t2m_c": 20, "wind_speed_ms": 5, "sp_hpa": 950},
                                   "levels": [{"pressure_hpa": 1000, "temperature_c": 20, "wind_speed_ms": 5, "height_m": -23.1},
                                              {"pressure_hpa": 975, "temperature_c": 19, "wind_speed_ms": 5, "height_m": 200.0}]}]}
    warnings = lw._check_model_output_sanity(model_result)
    assert any("PHYSICALLY IMPOSSIBLE" in w and "1000" in w for w in warnings)


def test_physically_sane_output_triggers_no_sanity_warning():
    model_result = {"forecast": [{"lead_hours": 6, "surface": {"t2m_c": 20, "wind_speed_ms": 5, "sp_hpa": 950},
                                   "levels": [{"pressure_hpa": 1000, "temperature_c": 20, "wind_speed_ms": 5, "height_m": 80.0},
                                              {"pressure_hpa": 975, "temperature_c": 19, "wind_speed_ms": 5, "height_m": 300.0},
                                              {"pressure_hpa": 950, "temperature_c": 18, "wind_speed_ms": 5, "height_m": 520.0}]}]}
    assert lw._check_model_output_sanity(model_result) == []


def test_non_monotonic_height_triggers_a_warning():
    model_result = {"forecast": [{"lead_hours": 6, "surface": {"t2m_c": 20, "wind_speed_ms": 5, "sp_hpa": 950},
                                   "levels": [{"pressure_hpa": 1000, "temperature_c": 20, "wind_speed_ms": 5, "height_m": 300.0},
                                              {"pressure_hpa": 975, "temperature_c": 19, "wind_speed_ms": 5, "height_m": 250.0}]}]}  # height DROPPED going up
    warnings = lw._check_model_output_sanity(model_result)
    assert any("not monotonically increasing" in w for w in warnings)


def test_large_systematic_temperature_bias_triggers_a_warning_even_though_excluded_from_headline():
    """Reproduces the real 13-15C surface/low-level bias case -- this must
    show up as an explicit warning, since pct_of_value excludes temperature
    from the headline % entirely and could otherwise hide exactly this."""
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    base = now - timedelta(hours=1)
    h = fake_hourly(base)
    obs_time_iso = base.isoformat()

    forecast = [{"lead_hours": 6, "surface": {"t2m_c": h["temperature_2m"][1] + 14.0, "wind_speed_ms": 5.0, "sp_hpa": 950.0},
                 "levels": [{"pressure_hpa": p, "temperature_c": h[f"temperature_{p}hPa"][1] + 14.0,
                             "wind_speed_ms": 20.0, "height_m": 5000.0 + p} for p in lw.COMPARABLE_LEVELS]}]
    cmp = lw.compare_forecast_to_open_meteo({"forecast": forecast}, h, obs_time_iso)
    assert any("temperature" in w.lower() and "typical error" in w for w in cmp["warnings"])