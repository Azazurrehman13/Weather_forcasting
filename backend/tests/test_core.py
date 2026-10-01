"""Run: python -m pytest tests -q   (no torch or checkpoint needed -- uses synthetic stats
and a persistence stub in place of the network, which makes the data plumbing checkable:
if the pipeline is consistent, 'predict no change' must return exactly what was entered.)"""
import os, sys
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import met_core as mc


def make_stats(descending=True):
    lv = sorted(mc.KNOWN_LEVELS, reverse=descending)
    p = np.array(lv, float)
    T = 288 - 6.5 * np.clip(44330 * (1 - (p / 1013.25) ** 0.1903), 0, 11000) / 1000
    z = 9.80665 * 44330 * (1 - (p / 1013.25) ** 0.1903)
    u, v = 5 + 30 * np.exp(-((p - 250) / 150) ** 2), -1 + 0 * p
    pl_mean = np.concatenate([T, u, v, z]).astype(np.float32)
    pl_std = np.concatenate([8 + 0 * p, 10 + 0 * p, 8 + 0 * p, 300 + 0 * p]).astype(np.float32)
    rows = np.arange(mc.PAD_H)[:, None] * np.ones((1, mc.PAD_W))
    elev = 1 - rows / mc.PAD_H            # north (row 0) = high terrain, like the real domain
    sfc_mean = np.stack([300 - 30 * elev, 0.5 + 0 * elev, -1 + 0 * elev, 101000 - 45000 * elev]).astype(np.float32)
    sfc_std = np.stack([6 + 0 * elev, 1.5 + 0 * elev, 1.5 + 0 * elev, 300 + 0 * elev]).astype(np.float32)
    return mc.Stats(pl_mean, pl_std, sfc_mean, sfc_std)


def persistence(x):   # stub "model": nothing changes
    return np.repeat(x[-1][None], mc.HORIZON, axis=0)[:, :, :mc.ORIG_H, :mc.ORIG_W]


def full_request(stats, lat=34.0, lon=73.0, **over):
    row, col = mc.nearest_grid_index(lat, lon)
    lv = []
    for i, p in enumerate(stats.levels):
        n = mc.N_LEVELS
        lv.append({"pressure_hpa": p, "temperature": stats.pl_mean[i] - 273.15 + 1.5,
                   "wind_speed": 12.0, "wind_dir": 270.0, "height": stats.pl_mean[3 * n + i] / mc.G + 20})
    sfc = {"t2m": stats.sfc_mean[0, row, col] - 273.15 + 2, "sp_hpa": stats.sfc_mean[3, row, col] / 100 - 3,
           "wind_speed": 4.0, "wind_dir": 45.0}
    r = {"lat": lat, "lon": lon, "units": {"temperature": "C", "wind": "ms"}, "surface": sfc, "levels": lv}
    r.update(over)
    return r


def test_grid_index_north_is_row0():
    assert mc.nearest_grid_index(37.25, 60.75) == (0, 0)
    assert mc.nearest_grid_index(23.5, 78.0) == (55, 69)
    assert mc.nearest_grid_index(34.15, 73.22) == (12, 50)
    with pytest.raises(ValueError):
        mc.nearest_grid_index(10, 70)


@pytest.mark.parametrize("descending", [True, False])
def test_level_order_detected(descending):
    assert make_stats(descending).levels == sorted(mc.KNOWN_LEVELS, reverse=descending)


def test_wind_roundtrip_and_convention():
    u, v = mc.wind_to_uv(10, 270)           # from the west -> blowing east
    assert u == pytest.approx(10) and v == pytest.approx(0, abs=1e-9)
    u, v = mc.wind_to_uv(10, 0)             # from the north -> blowing south
    assert v == pytest.approx(-10)
    s, d = mc.uv_to_wind(*mc.wind_to_uv(7.3, 123.0))
    assert (s, d) == (pytest.approx(7.3), pytest.approx(123.0))


@pytest.mark.parametrize("descending", [True, False])
def test_persistence_returns_inputs(descending):
    st = make_stats(descending)
    fc = mc.Forecaster(st, persistence)
    req = full_request(st)
    out = fc.predict(req)
    assert len(out["forecast"]) == 12 and out["forecast"][0]["lead_hours"] == 6
    f = out["forecast"][0]
    e = {l["pressure_hpa"]: l for l in req["levels"]}
    for lvl in f["levels"]:
        i = e[lvl["pressure_hpa"]]
        assert lvl["temperature_c"] == pytest.approx(i["temperature"], abs=1e-3)
        assert lvl["wind_speed_ms"] == pytest.approx(12.0, abs=1e-3)
        assert lvl["wind_dir_deg"] == pytest.approx(270.0, abs=1e-2)
        assert lvl["height_m"] == pytest.approx(i["height"], abs=1e-2)
    s = req["surface"]
    assert f["surface"]["t2m_c"] == pytest.approx(s["t2m"], abs=1e-3)
    assert f["surface"]["sp_hpa"] == pytest.approx(s["sp_hpa"], abs=1e-3)
    assert f["surface"]["wind_speed_ms"] == pytest.approx(4.0, abs=1e-3)
    assert f["surface"]["wind_dir_deg"] == pytest.approx(45.0, abs=1e-2)
    assert out["warnings"] == []


def test_surface_uses_anomaly_at_user_pixel():
    """Same anomaly at a high-terrain and a low-terrain location -> same normalized input."""
    st = make_stats()
    seen = []
    fc = mc.Forecaster(st, lambda x: (seen.append(x[0, mc.N_PL:, 0, 0].copy()), persistence(x))[1])
    fc.predict(full_request(st, lat=36.5))
    fc.predict(full_request(st, lat=24.5))
    assert np.allclose(seen[0][0], seen[1][0], atol=1e-4) and np.allclose(seen[0][3], seen[1][3], atol=1e-4)


def test_input_window_shape_and_uniform():
    st = make_stats()
    box = {}
    fc = mc.Forecaster(st, lambda x: (box.setdefault("x", x), persistence(x))[1])
    fc.predict(full_request(st))
    x = box["x"]
    assert x.shape == (4, 124, 56, 72) and x.dtype == np.float32
    assert np.all(x == x[:1]) and np.all(x == x[:, :, :1, :1])


def test_partial_levels_extrapolated_and_climatology():
    st = make_stats()
    req = full_request(st)
    req["levels"] = [l for l in req["levels"] if l["pressure_hpa"] in (925, 850, 700, 500, 300)]
    out = mc.Forecaster(st, persistence).predict(req)
    src = {l["pressure_hpa"]: l["source"]["temperature"] for l in out["input_used"]["levels"]}
    assert src[850] == 0 and src[800] == 1 and src[30] == 2          # entered / nearby-extrap / far-away climatology
    assert any("Only 5 levels" in w for w in out["warnings"])
    # 800 hPa sits between the 850 and 700 hPa entries -> should land between their temperatures,
    # not jump straight to the flat climatology value.
    rows = {l["pressure_hpa"]: l for l in out["input_used"]["levels"]}
    t850, t700, t800 = rows[850]["temperature_c"], rows[700]["temperature_c"], rows[800]["temperature_c"]
    assert min(t850, t700) < t800 < max(t850, t700)


def test_single_level_allowed_and_warns_strongly():
    st = make_stats()
    req = full_request(st)
    req["levels"] = [l for l in req["levels"] if l["pressure_hpa"] == 850]
    out = mc.Forecaster(st, persistence).predict(req)
    rows = {l["pressure_hpa"]: l for l in out["input_used"]["levels"]}
    assert rows[850]["source"]["temperature"] == 0
    assert rows[825]["source"]["temperature"] == 1                  # close enough to be nudged
    assert rows[30]["source"]["temperature"] == 2                   # far away -> pure climatology
    # entered anomaly should pull nearby levels away from climatology, same sign as the entered anomaly
    clim850 = st.pl_mean[list(st.levels).index(850)]
    entered_anom = req["levels"][0]["temperature"] + 273.15 - clim850
    clim825 = st.pl_mean[list(st.levels).index(825)]
    nudge825 = rows[825]["temperature_k"] - clim825
    assert np.sign(nudge825) == np.sign(entered_anom) and abs(nudge825) < abs(entered_anom)
    assert any("Only 1 level" in w for w in out["warnings"])


def test_errors():
    st = make_stats()
    fc = mc.Forecaster(st, persistence)
    with pytest.raises(ValueError, match="plausible range"):          # Kelvin typed while unit = C
        r = full_request(st); r["levels"][0]["temperature"] = 288; fc.predict(r)
    with pytest.raises(ValueError, match="BOTH speed and direction"):
        r = full_request(st); r["levels"][0]["wind_dir"] = None; fc.predict(r)
    with pytest.raises(ValueError, match="not a model level"):
        r = full_request(st); r["levels"][0]["pressure_hpa"] = 111; fc.predict(r)
    with pytest.raises(ValueError, match="more than once"):
        r = full_request(st); r["levels"].append(dict(r["levels"][0])); fc.predict(r)
    with pytest.raises(ValueError, match="at least 1"):
        r = full_request(st); r["levels"] = []; fc.predict(r)
    with pytest.raises(ValueError, match="required"):
        r = full_request(st); r["surface"]["sp_hpa"] = None; fc.predict(r)


def test_units_kelvin_knots_and_uv():
    st = make_stats()
    req = full_request(st)
    req["units"] = {"temperature": "K", "wind": "kt"}
    for l in req["levels"]:
        l["temperature"] += 273.15
    req["surface"]["t2m"] += 273.15
    req["surface"]["wind_speed"] = 8.0          # knots now
    out = mc.Forecaster(st, persistence).predict(req)
    assert out["forecast"][0]["surface"]["wind_speed_ms"] == pytest.approx(8 * 0.514444, abs=1e-3)
    assert out["forecast"][0]["levels"][0]["wind_speed_ms"] == pytest.approx(12 * 0.514444, abs=1e-3)
    req2 = full_request(st)
    for l in req2["levels"]:
        l["wind_speed"] = l["wind_dir"] = None; l["u"], l["v"] = 3.0, -4.0
    out2 = mc.Forecaster(st, persistence).predict(req2)
    assert out2["forecast"][0]["levels"][0]["wind_speed_ms"] == pytest.approx(5.0, abs=1e-3)


def test_unusual_value_warns():
    st = make_stats()
    r = full_request(st); r["surface"]["sp_hpa"] = 646      # ~6 std away: unusual but not impossible
    out = mc.Forecaster(st, persistence).predict(r)
    assert any("Surface sp" in w for w in out["warnings"])


def test_impossible_surface_value_blocked():
    st = make_stats()
    r = full_request(st); r["surface"]["sp_hpa"] = 300      # physically absurd for this pixel (~-120 std)
    with pytest.raises(ValueError, match="far beyond anything the model was trained on"):
        mc.Forecaster(st, persistence).predict(r)
    try:
        mc.Forecaster(st, persistence).predict(r)
    except ValueError as e:
        assert "typical here" in str(e) and "hPa" in str(e)


def test_impossible_pressure_level_value_blocked():
    st = make_stats()
    r = full_request(st)
    r["levels"][0]["height"] = 9000       # physically plausible on its own, but absurd at this pressure level
    with pytest.raises(ValueError, match="far beyond anything the model was trained on"):
        mc.Forecaster(st, persistence).predict(r)


def test_api_end_to_end():
    import main
    st = make_stats()
    main.load_forecaster = lambda: mc.Forecaster(st, persistence, {"checkpoint_epoch": 0})
    with TestClient(main.app) as c:
        assert c.get("/api/health").json()["model_loaded"]
        sc = c.get("/api/schema").json()
        assert sc["levels_hpa"][0] == 1000 and sc["lead_hours"][:2] == [6, 12]
        ty = c.get("/api/typical", params={"lat": 34, "lon": 73}).json()
        assert len(ty["levels"]) == 30 and ty["levels"][0]["pressure_hpa"] == 1000
        assert c.get("/api/typical", params={"lat": 5, "lon": 73}).status_code == 422
        r = c.post("/api/predict", json=full_request(st))
        assert r.status_code == 200 and r.json()["forecast"][0]["lead_hours"] == 6
        bad = full_request(st); bad["levels"][0]["temperature"] = 500
        r = c.post("/api/predict", json=bad)
        assert r.status_code == 422 and "plausible range" in r.json()["detail"]
        r = c.post("/api/predict", json={"lat": 34, "lon": 73, "obs_time": "2026-09-24T06:00", **{k: v for k, v in full_request(st).items() if k in ("surface", "levels", "units")}})
        assert r.json()["forecast"][0]["valid_time"] == "2026-09-24T12:00:00"
        # typical -> predict round trip: the pre-fill values must be accepted as-is
        req = {"lat": 34, "lon": 73, "units": {"temperature": "C", "wind": "ms"},
               "surface": {"t2m": ty["surface"]["t2m_c"], "sp_hpa": ty["surface"]["sp_hpa"],
                           "wind_speed": ty["surface"]["wind_speed_ms"], "wind_dir": ty["surface"]["wind_dir_deg"]},
               "levels": [{"pressure_hpa": l["pressure_hpa"], "temperature": l["temperature_c"], "wind_speed": l["wind_speed_ms"],
                           "wind_dir": l["wind_dir_deg"], "height": l["height_m"]} for l in ty["levels"]]}
        r = c.post("/api/predict", json=req)
        assert r.status_code == 200 and r.json()["warnings"] == []