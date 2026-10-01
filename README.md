# MET Forecast App

Local FastAPI backend + browser frontend. You manually enter a current
pressure-level profile (temperature, wind, height per level) and surface
conditions; the model predicts the next 72 hours, shown MET-style. An
optional second panel checks the model's forecast against Open-Meteo's own
forecast as a live sanity check.

**No ERA5/CDS account is needed to run this.** An earlier version fetched
live ERA5 data automatically; that was replaced with manual entry, which
better matches how a real MET sounding actually arrives (from an operator or
radiosonde, not a satellite reanalysis pipeline). `era5_fetch.py` is left in
the repo but is no longer used by anything — ignore it unless you
specifically want to resurrect live-ERA5 mode.

## Before running this — required setup

**1. Confirm your pressure levels are correct, using real evidence, not a guess.**
`backend/config/pressure_levels.json` must contain your model's exact 30
pressure levels, in the exact order the model was trained on. A wrong-but-
same-length list will NOT error — it will silently misalign channels and
produce confident-looking garbage. Two ways to get this right:

- **If you still have an original training GRIB file**: run
  `python extract_pressure_levels.py <path_to_a_pressure_level_grib_file>`
  — it reads the real `isobaricInhPa` coordinate directly from the file.
- **If you only have the trained checkpoint + saved `.npy` stats**: run
  `python verify_level_order.py` from `backend/` — it determines the true
  order from `pl_mean.npy` using a physical fact that can't be faked:
  geopotential height increases strictly monotonically with altitude. The
  script checks that monotonicity directly on your saved stats and writes
  the confirmed order to `config/pressure_levels.json` for you.

The app refuses to start if this file is empty or the wrong length — but
that check can't catch a wrong-but-plausible-length list, so use one of the
two methods above rather than typing a list from memory.

**2. Place your model files** in `backend/model/`:
- `best_model_wind_weighted.pt` (or whichever checkpoint you're using)
- `pl_mean.npy`, `pl_std.npy`, `sfc_pixel_mean.npy`, `sfc_pixel_std.npy`

If your files live elsewhere, update the paths in `model_utils.py` (or set
the environment variables it reads, if you've kept that pattern).

## Install

```
cd backend
pip install -r requirements.txt
```

No `cdsapi`, `xarray`, or `cfgrib` needed anymore — those were only for the
now-unused live-ERA5 path.

## Run the backend

```
cd backend
uvicorn main:app --reload --port 8000
```

Wait for `Model loaded {...}. Level order: [...]` before using the frontend.

## Run the frontend

Open `frontend/index.html` directly in a browser. It calls
`http://localhost:8000` — no separate frontend server needed.

## What the app actually does

**Main panel — manual profile entry.** You enter temperature, wind
speed/direction, and height for each of the 30 pressure levels, plus surface
conditions (t2m, wind, pressure) and a lat/lon. The model predicts all 12
lead times (+6h to +72h) in one pass.

Important simplification, not hidden: your entered profile is broadcast
**uniformly across the model's full spatial grid** and held **steady-state
across the input timesteps** — the model was trained on real spatial fields,
and a single point profile can't recreate that. This works reasonably for
upper levels (less sensitive to local surface effects) but can produce large
errors at low levels when the real local conditions have sharp gradients the
uniform input can't represent (see "Known issues" below — this is observed,
not theoretical).

**"Check against live weather" panel (optional).** Fetches Open-Meteo's
current conditions (free, no API key) as input, runs the model, then
compares the forecast against Open-Meteo's **own forecast** (ECMWF/GFS) for
the same future times. Read this carefully:

- This is a **model-vs-model** comparison, not a comparison against observed
  ground truth. Open-Meteo's free historical archive has no upper-air data
  for past dates, so true backtesting against observations isn't available
  without a paid data source.
- Only the **19 of your 30 pressure levels** that Open-Meteo also supports
  are compared (1000 down to 30 hPa, standard set). The other 11
  (875, 825, 775, 750, 650, 550, 450, 350, 225, 175, 125 hPa) are skipped
  entirely, never interpolated.
- Each variable shows **two percentages**:
  - **% of value** — the intuitive one: how close the prediction is to
    Open-Meteo's number, as a plain percentage. **Not shown for
    temperature** — there's no physically meaningful non-arbitrary zero for
    temperature (unlike wind speed, where 0 = no wind), so a ratio-based
    percentage would be misleading (Kelvin's huge baseline makes even a bad
    5-10°C miss falsely look like ~97-100%).
  - **vs. typical error** — stricter: how this single miss compares to the
    model's own known typical error from real evaluation. 100% only if the
    miss is much smaller than that. This is the one to trust for
    temperature.
  - The headline "Overall accuracy" percentage averages **% of value only**
    (wind/pressure/height), which means a bad temperature result can be
    invisible in that one number — see the warning system below for why
    that's covered.
- **Automatic warnings** fire above the headline number if: any level shows
  a physically impossible height (negative, or not increasing with
  altitude), or any variable group (surface temp, pressure-level temp,
  winds, height, surface pressure) averages below 20% on "vs. typical
  error" across the run. Read these before trusting the percentage.

## Known issues (observed during real testing, not hypothetical)

- **Low-level temperature bias from the uniform-broadcast simplification**:
  real testing showed errors of 13-15°C at the surface and lowest pressure
  levels (1000-800 hPa) when local conditions had a sharp near-surface
  gradient the single uniform profile couldn't represent. The error shrinks
  with altitude (down to 1-3°C above ~250 hPa). The warning system flags
  this when it happens; it is not currently fixed at the architecture level.
- **Occasional physically impossible low-level height** (e.g. negative
  height at 1000 hPa) has occurred more than once under the same
  simplification. The app now detects and warns on this automatically
  rather than silently showing it.
- **`v_component_of_wind`** has essentially no real forecasting skill beyond
  +48h (normalized R² near zero at +72h, confirmed in evaluation — not a
  pooling artifact). **Surface wind** (`u10`/`v10`) is weak throughout
  (R² 0.14-0.31). Treat these with less confidence than temperature or
  geopotential, which perform much better.
- **This does not encode the exact NUTECH MET Message format** (the 16-line
  altitude bands, `MET INTRO` codes, etc.) — that requires the actual
  message spec or app source code, which wasn't available when this was
  built. Output is the model's raw pressure-level values labeled by actual
  hPa level, not the encoded message format.

## Tests

```
cd backend
python -m pytest tests -q
```

Covers `met_core.py`'s profile logic (extrapolation, outlier blocking, unit
conversion) and `live_weather.py`'s comparison/warning logic, including
regression tests built directly from real bugs found during testing (the
JSON-serialization crash from `NaN` values, the accuracy-formula bug where
a 1.7x-typical miss and a 20x-typical miss both showed 0%, and the
misleading-temperature-percentage issue).