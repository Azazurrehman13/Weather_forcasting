"""
Manual-entry MET forecast API.  Run:  uvicorn main:app --port 8000
Then open http://localhost:8000  (the frontend is served from ../frontend).
Prediction is synchronous -- there is no external data fetch, so no job queue.
"""
import os
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import met_core as mc
import live_weather as lw

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Default: backend/model/ (falls back to backend/ itself). Override with env vars if yours live elsewhere.
_default_dir = os.path.join(BASE_DIR, "model") if os.path.isdir(os.path.join(BASE_DIR, "model")) else BASE_DIR
STATS_DIR = os.environ.get("MET_STATS_DIR", _default_dir)
CHECKPOINT_PATH = os.environ.get("MET_CHECKPOINT_PATH", os.path.join(STATS_DIR, "best_model_wind_weighted.pt"))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
REQUIRED_FILES = ["pl_mean.npy", "pl_std.npy", "sfc_pixel_mean.npy", "sfc_pixel_std.npy"]

state = {"fc": None}


def load_forecaster():
    missing = [p for p in [CHECKPOINT_PATH] + [os.path.join(STATS_DIR, f) for f in REQUIRED_FILES] if not os.path.isfile(p)]
    if missing:
        raise RuntimeError("Missing model file(s):\n  " + "\n  ".join(os.path.abspath(p) for p in missing) +
                           "\nPut them there, or set MET_CHECKPOINT_PATH / MET_STATS_DIR.")
    from model_utils import load_forecaster as _load   # torch imported lazily
    return _load(CHECKPOINT_PATH, STATS_DIR, device="cpu")


@asynccontextmanager
async def lifespan(app):
    print("Loading model and normalization stats...")
    state["fc"] = load_forecaster()
    print(f"Model loaded {state['fc'].model_info}. Level order: {state['fc'].s.levels[:3]}...")
    yield


app = FastAPI(title="MET Forecast (manual entry)", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Units(BaseModel):
    temperature: Literal["C", "K"] = "C"
    wind: Literal["ms", "kt", "kmh"] = "ms"


class Surface(BaseModel):
    t2m: Optional[float] = None
    sp_hpa: Optional[float] = None
    wind_speed: Optional[float] = None
    wind_dir: Optional[float] = None
    u10: Optional[float] = None
    v10: Optional[float] = None


class Level(BaseModel):
    pressure_hpa: int
    temperature: Optional[float] = None
    wind_speed: Optional[float] = None
    wind_dir: Optional[float] = None
    u: Optional[float] = None
    v: Optional[float] = None
    height: Optional[float] = None          # geopotential height, metres


class PredictRequest(BaseModel):
    lat: float
    lon: float
    obs_time: Optional[str] = None           # ISO, UTC; only used to label valid times
    units: Units = Units()
    surface: Surface
    levels: List[Level]


def fc():
    if state["fc"] is None:
        raise HTTPException(503, "Model not loaded yet.")
    return state["fc"]


@app.get("/api/health")
def health():
    return {"model_loaded": state["fc"] is not None}


@app.get("/api/schema")
def schema():
    f = fc()
    return {"levels_hpa": sorted(f.s.levels, reverse=True), "lead_hours": [(i + 1) * mc.STEP_HOURS for i in range(mc.HORIZON)],
            "domain": {"lat_min": mc.LAT_MIN, "lat_max": mc.LAT_MAX, "lon_min": mc.LON_MIN, "lon_max": mc.LON_MAX},
            "skill": mc.SKILL, "model": f.model_info}


@app.get("/api/typical")
def typical(lat: float = Query(...), lon: float = Query(...)):
    """Training-climatology profile -- used only to pre-fill the form as a starting point."""
    try:
        return fc().climatology(lat, lon)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.post("/api/predict")
def predict(req: PredictRequest):
    try:
        return fc().predict(req.model_dump())
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/live/verify")
def live_verify(lat: float = Query(...), lon: float = Query(...)):
    """Fetches real current conditions from Open-Meteo, runs our model on
    them, then compares our forecast against Open-Meteo's OWN forecast for
    the same future times. This is a model-vs-model check (see the
    'disclaimer' field in the response) -- not a comparison against
    observed ground truth, which isn't freely available for upper-air data."""
    try:
        om = lw.fetch_open_meteo(lat, lon)
        req, now_idx, hourly, missing = lw.build_predict_request_from_live(lat, lon, om)
        result = fc().predict(req)
        comparison = lw.compare_forecast_to_open_meteo(result, hourly, req["obs_time"])
        return {
            "input_used_from_open_meteo": req,
            "missing_levels_from_open_meteo": missing,
            "model_result": result,
            "comparison": comparison,
        }
    except RuntimeError as e:
        raise HTTPException(502, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))


if os.path.isdir(FRONTEND_DIR):   # mounted last so /api/* keeps priority
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")