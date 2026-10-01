"""
Run this ONCE with your real checkpoint + stats before using the UI:
    python smoke_test.py
It loads the model, feeds it the training-average profile for a location and
prints what came out, so you can confirm loading, channel order and shapes are sane.
"""
import sys
import main

lat, lon = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) == 3 else (30.5, 69.5)
fc = main.load_forecaster()
print("checkpoint:", fc.model_info)
print("level order (channel index 0..29):", fc.s.levels)

t = fc.climatology(lat, lon)
req = {"lat": lat, "lon": lon, "units": {"temperature": "C", "wind": "ms"},
       "surface": {"t2m": t["surface"]["t2m_c"], "sp_hpa": t["surface"]["sp_hpa"],
                   "wind_speed": t["surface"]["wind_speed_ms"], "wind_dir": t["surface"]["wind_dir_deg"]},
       "levels": [{"pressure_hpa": l["pressure_hpa"], "temperature": l["temperature_c"], "wind_speed": l["wind_speed_ms"],
                   "wind_dir": l["wind_dir_deg"], "height": l["height_m"]} for l in t["levels"]]}
out = fc.predict(req)
print("warnings:", out["warnings"] or "none")
for k in (0, 3, 11):
    f = out["forecast"][k]
    l850 = next(l for l in f["levels"] if l["pressure_hpa"] == 850)
    print(f"+{f['lead_hours']:>2}h  t2m {f['surface']['t2m_c']:6.1f} C | sp {f['surface']['sp_hpa']:7.1f} hPa | "
          f"850hPa T {l850['temperature_c']:6.1f} C, wind {l850['wind_speed_ms']:5.1f} m/s from {l850['wind_dir_deg']:5.0f}, "
          f"height {l850['height_m']:7.0f} m")
print("\nInput was the climatological average, so +6h values should sit close to the input. "
      "Large jumps (e.g. 850 hPa height off by hundreds of metres, temperatures far from the input) "
      "point to a level-order, stats-file or checkpoint mismatch.")
