# Personalized cycling speed predictor

A local Streamlit application that learns segment-level cycling performance from your own
Strava recordings. Each project dependency and its Python environment lives inside `project2`.
Private recordings, caches and fitted models live in the ignored `data` directory.

## Windows quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Open PowerShell in this folder:

```powershell
uv sync --python 3.12
uv run streamlit run app.py
```

The app listens only on `127.0.0.1:8501`. `launch.ps1` runs the same command. Python 3.12 is
downloaded automatically by uv when needed. No Docker, account credentials or API key are required.

## Workflow

1. **Historical Data:** upload the complete account ZIP, or enter its absolute disk path. No manual
   extraction is needed. Inspect the import report, then process and train. Incremental imports
   skip activities already identified by activity ID or original-recording SHA-256.
2. **Route Prediction:** upload a GPX track or route exported from Komoot, enter a moving-time
   average HR, select a bike, stop scenario, departure date/time/timezone, and personal breaks.
   A GPX without usable elevation works, with an explicit terrain warning. A trusted optional CSV
   elevation profile can replace GPX elevations; use `distance_m,elevation_m` columns covering
   the whole route, including distance zero.
3. **Results:** review cruising, rolling and overall speeds, the time ledger, route overlays,
   elevation, speed and progress charts, surface composition, pacing HR, and segment details.
   Download private prediction data if desired. Personal breaks are a separate total; the progress
   graph does not assign those breaks to arbitrary locations along the route.
4. **Model Performance:** inspect chronological errors, baseline comparisons, recency candidates,
   recent-fitness comparison, uncertainty coverage, data coverage, and errors by distance/surface/
   elevation. More features do not guarantee better accuracy; the dashboard explicitly reports
   when the segment model does not beat a simple baseline.

Strava account exports can be requested from **Settings → My Account → Download your account →
Get Started → Request download**. Download the account ZIP when Strava makes it available.
See [Strava's export documentation](https://support.strava.com/en-us/articles/15401919-how-do-i-export-my-strava-data).
Komoot's route export menu provides the GPX file for a planned tour; export access depends on
your Komoot account and region entitlement. See [Komoot GPX export help](https://support.komoot.com/hc/en-us/articles/10115477099674-Export-and-import-Routes-and-Activities).

## Separate uv environment

This directory is a standalone Python project, independent of the repository's parent project.
It has its own `pyproject.toml`, committed `uv.lock`, `.python-version` and ignored `.venv`.
Always run commands **from `project2`** or use `uv run --project project2 …` from the repository root.
Verify the interpreter with:

```powershell
uv run python -c "import sys; print(sys.executable)"
uv run pytest -q
uv run ruff check .
```

The interpreter path ends in `project2\.venv\Scripts\python.exe`. Dependencies are pinned by
`uv.lock`; `uv sync --locked` reproduces them. The optional PBF reader needs `uv sync --extra pbf`.

## CLI

The command-line pipeline is the same as the web app. Example PowerShell commands:

```powershell
uv run cycling import "C:\Users\you\Downloads\export.zip"
uv run cycling process
uv run cycling train --optimize-half-life --validation-rides 16
uv run cycling predict "C:\Users\you\Downloads\planned.gpx" --hr 142 --scenario full_stop --break-minutes 15
uv run cycling status
uv run cycling config
```

Use `cycling process --refresh-osm` after changing OSM providers or fetching more tiles.
`cycling train --no-validation` provides a quick fit but no calibrated prediction interval.
`cycling predict --output data\prediction.csv` saves private segment results locally.
`cycling --data-dir <path> …` or `CYCLING_DATA_DIR` can override the default data directory.
If you choose another directory inside Git, add it to your ignore rules before storing private data.

## OSM and map matching

Configure the sidebar and save, or create ignored `config.local.json` with selected fields:

```json
{
  "osm_provider": "overpass",
  "overpass_url": "https://overpass-api.de/api/interpreter",
  "osm_max_tiles": 12,
  "osm_min_interval_s": 3.0,
  "half_life_days": 120,
  "optimize_half_life": true,
  "validation_rides": 16
}
```

Three working modes are provided:

| Mode | Configuration | Behavior |
|---|---|---|
| Offline, default | `osm_provider: "offline"` | Reuse cached road tiles; uncovered surface/control data remains unknown. No road-service requests. |
| Bounded Overpass | `osm_provider: "overpass"` | Query coarse 0.02° road tiles, cache for 180 days, at least 3 seconds between requests, up to 12 new tiles per processing batch, with up to three attempts. Both limits are configurable. |
| Local extract | `osm_provider: "local"`, `local_osm_path: "C:/maps/region.osm"` | Read OSM XML, `.osm.gz`, Overpass JSON, or optional PBF on the local machine. No public service or Docker required. |

Overpass sends **coarse geographic bounding boxes**, not activity recordings, heart rates, account
details or ZIP contents. Those bounding boxes still disclose an approximate area to the provider.
For no location disclosure, use local OSM or offline mode and leave the online basemap disabled.
Overpass defaults are intentionally bounded. Large histories need repeated batches or a regional
local extract; cached tiles are reused, and exhausted requests leave explicit unknown coverage.
Public services can return 429/504 errors; the app falls back to cached/unknown roads. An endpoint
can be replaced with your own Overpass server. See [Overpass public-instance usage](https://dev.overpass-api.de/overpass-doc/en/preface/commons.html).

The built-in matcher searches projected OSM road edges and scores distance, heading, one-way
restrictions and continuity. It preserves travel direction and the original route geometry. Surface,
smoothness, tracktype, highway, bicycle and original tags are stored for each segment. Inferred
tracktype labels remain separate from known surface tags. A proximity/heading confidence is
reported; it is a heuristic match score, not a calibrated probability. Parallel roads, trails and
complex junctions can be ambiguous. Missing surface tags are never assumed to mean asphalt.

Controls must belong to a matched road and pass forward/backward approach tests. A side-road
sign cannot be assigned solely because it is near the route. Shared-junction controls without
approach information, numeric directional tags, and low-confidence matches are flagged or
omitted instead of charged as confirmed stops. This can undercount incompletely mapped controls.
See [OSM stop direction tags](https://wiki.openstreetmap.org/wiki/Tag:highway%3Dstop).

**Optional Valhalla:** set `valhalla_url` to an existing local service such as
`http://127.0.0.1:8002`. The app calls `/trace_attributes` in bounded chunks and uses matching
OSM way IDs to confirm compatible local matches; it never adopts a rerouted geometry. This is a
conservative integration, not a full network-wide sequence matcher. An unavailable service falls
back to the working directional matcher. No Valhalla installation is required. Running a full
Valhalla service and building regional tiles is an advanced, separately managed option; this
project does not claim a bundled native Windows Valhalla server. A remote URL sends route
coordinates to that configured service. See [Valhalla map-matching API](https://valhalla.github.io/valhalla/api/map-matching/api-reference/).

Local XML/PBF extracts are loaded into memory; use a regional road extract, not a planet file.
PBF support uses pyosmium with node-location indexing. Relations, lane-specific signal programs,
turn restrictions and road-network routing are not fully interpreted. OSM attribution is included
on the map. Online basemaps load from CARTO/OSM only when enabled.

## Import and cache behavior

* Discover nested `activities.csv`, FIT/FIT.GZ, TCX and GPX, with gzip support for XML recordings.
  Match CSV filenames before IDs because a FIT upload filename can differ from the Strava activity
  ID. Prefer original FIT, then TCX, then GPX when duplicate activities are present.
* Parse duplicate CSV headers carefully: Strava's first `Distance` is display kilometres and its
  repeated raw `Distance` is metres. Explicit unit-bearing aliases are accepted. A single plain
  `Distance` uses the usual Strava display-km convention. Internal distances, speeds and times
  are metres, metres/second and seconds, with UTC timestamps.
* Default inclusion: outdoor Ride, GravelRide, MountainBikeRide. E-bike inclusion is off unless
  selected. Running, swimming, virtual/indoor rides and recordings without usable outdoor GPS
  are excluded. A recording without trustworthy cycling metadata is not guessed to be a ride.
  `allowed_sports` is configurable; changing it does not delete already imported rides.
* Prefer timestamped recordings. Check coordinate/sensor ranges, CRCs, clock continuity and GPS
  jumps. Use valid sensor-distance increments when present, otherwise geodesic distances. Track
  gaps are not connected. Long sparse intervals estimate traversed moving time and keep unknown
  pause duration separate; explicit FIT timer events recover observed pauses when available.
* SQLite indexes rides and compressed OSM tiles. Zstandard Parquet stores cleaned traces and
  adaptive segments. Joblib stores only the app's own local model; uploaded pickle/joblib models
  are never accepted. Files are written atomically. New imports mark the current model stale.
* ZIPs are streamed without extraction. Reject absolute/traversal/symlink paths, encryption,
  excessive ratios, more than 50,000 entries, entries over 128 MB, ZIPs over 2 GB, and advertised
  decompressed totals over 8 GB. Gzip reads and point counts have independent bounds. The web
  uploader is capped at 512 MB; larger permitted ZIPs can be imported by disk path or CLI.

English Strava headers and common aliases are supported; arbitrary localized CSV schemas may
need an adapter. Timestamped cycling recordings are essential; CSV summaries alone cannot train
a segment model. Malformed activities are reported individually without stopping the full import.

## Personalized models

Adaptive segments vary with gradient, surface, HR and interruption changes, with a maximum
of approximately 220 m or 35 s once enough distance has accumulated. Geometry uses distance-based
elevation smoothing and roughly 20 m heading neighborhoods to reject GPS micro-turns. Missing
elevation retains an explicit flag; gradients fall back to zero and ascent is reported unknown.

The cruising model is a regularized histogram gradient-boosted regressor on log speed. Inputs
include gradient, surface and original surface tags, smoothness, tracktype, road type, bike,
historical date, planned-distance proxy, distance completed, elevation accumulated, progress
and accumulated moving duration. Cruise labels exclude mapped/geometry interruption windows,
stops, large accelerations and unreliable recording gaps. Inference updates moving duration
iteratively using its own predicted time, never the held-out ride's recorded time. No fixed fatigue
percentage is applied. Feature support is capped beyond observed distance/duration/fitness dates;
warnings identify long-distance extrapolation. The actual final historical ride distance is used
as the best available proxy for planned ride distance; an export cannot reveal original intent.

Every ride receives an equal observation-weight budget before exponential recency weighting:
`2 ** (-age_days / half_life_days)`, default 120 days. Rare surfaces receive a small explicit support
boost. Training on four or more rides and 100 valid cruise segments enables boosted models.
Smaller usable datasets get labeled personal, gradient-binned cruising baselines. Fewer than ten
usable cruising segments results in an actionable error, not a generic guessed speed.

Missing-HR rides use the **median of actual recorded ride-average HR** and retain measured/imputed
flags. They receive full terrain-learning weight. An independent HR residual model and the pacing
profile are trained **only on timestamped measured HR**, giving imputed HR zero influence on
learning the HR-dependent relationship. No artificial curves become training truth. If fewer than
four measured-HR rides / 100 HR segments or too little HR variation exists, the app predicts the
personal terrain baseline and explains that target HR is unavailable. Summary-only HR can inform
the median and the validation ride's requested average, but is not treated as timestamped HR.

When supported, a monotone HR residual adjustment relates effort to speed, and a regularized
within-ride HR model learns gradient/duration/distance variations, including possible cardiac
drift. An iterative constrained solve finds a variable segment HR profile whose **moving-time
weighted** average matches the target. Supported HR is the measured 2nd–98th percentile range;
requests beyond it are limited and shown explicitly. This is a data-driven pacing estimate,
not a physiological prescription. Power/cadence/temperature are extracted for diagnostics but
are not required as planned-route inputs or used to silently assume future sensor values.

## Interruptions and time accounting

Nearby controls and turns are consolidated, and observation windows have non-overlapping
boundaries. This prevents charging a corner twice at the same stop/light. Historical traces
provide approach/minimum/exit speeds, braking/acceleration/stationary durations, straight/turn
behavior, direction, intersection, hour and weekday. Full/rolling/minimal stops use actual speed
and stationary evidence; sparse recordings are labeled unknown rather than assigned precise waits.

Behavioral predictions use a comparable road/turn distribution and shrink repeated local encounters
toward broader behavior with a 12-encounter prior. Traffic signals have smoothed stop probabilities
and conditional waits learned from reliable encounters. The observed scenario is descriptive;
the default full-stop scenario imposes deceleration to zero at applicable stop signs and learns
dwell only from measured full stops. Geometry-only fallback uses explicitly labeled kinematics
(0.9 m/s² braking, 0.45 m/s² acceleration, surface-dependent lateral acceleration). These are
fallback parameters, not claimed learned traffic distributions. Unobserved red-light/railway/dwell
waits remain unknown, contribute zero *known* stationary delay, and produce a lower-bound warning.

| Metric | Definition |
|---|---|
| Cruising speed | Total route distance / uninterrupted cruising time |
| Rolling speed | Distance / (cruising + exclusive braking/corner/acceleration time) |
| Overall speed | Distance / (moving + known stationary waits + selected personal breaks) |

Segment speeds are never averaged arithmetically. Waits and moving losses occupy separate ledgers.
Personal breaks default to zero and are always visible. Unclassified historical coffee/photo/mechanical
stops are not silently turned into predicted breaks. Sparse/auto-paused traces can prevent confident
separation of stops and waiting; GPS alone cannot reconstruct what was not recorded.

## Validation and confidence

Backtests train on rides **strictly before each test ride**. Test routes use geometry, elevation,
and reusable OSM annotations only: speed/HR traces, power, recorded moving duration and user
behavior statistics are excluded. Recorded ride-average measured HR is used as the test request
when available; missing HR does not become artificial evaluation truth. Imputation medians,
category encoding, feature ranges, recency normalization and intersection observations are
recomputed from each training prefix. No random segment validation is used.

The earlier half of eligible test rides selects among 60/90/120/180/365-day half-lives when enabled.
It also compares a restrained recent-fitness correction based on **earlier forecast residuals**.
That variant is selected only with at least three selection rides and an improvement over 3%.
The later half is held out for final accuracy and uncertainty calibration. Models may use those
rides only when predicting subsequent rides. At least seven rides are needed for the separate
selection/calibration workflow. More tests are configurable, but repeated fitting costs time.

Report speed MAE, moving-time MAE, elapsed-time MAE for sufficiently reliable observations,
median absolute moving-time percentage error, and historical median-speed/gradient baselines.
Elapsed-time errors are unavailable when road/control coverage, modeled waits or timestamps are insufficient;
the diagnostic lower-bound elapsed error is still retained. Errors are grouped by distance,
dominant surface, and elevation gain. Unknown controls and personal stops can still bias elapsed
accuracy, so its quality is reported separately from moving speed.

90% rolling-speed intervals use a finite-sample upper quantile of absolute speed errors from
the later chronological calibration set, and transform endpoints into moving-time bounds.
Coverage on historical rides uses **earlier residuals only**, never the test ride's own error.
Fewer than 20 calibration rides, unknown surfaces/waits, or terrain outside training support make
intervals provisional. No residuals means no fabricated interval. Time-correlated rides and changing
weather do not satisfy independent/exchangeable sampling, so nominal coverage is not guaranteed.
The displayed interval describes moving speed/time; it does not calibrate unknown stationary waits
or total elapsed-time uncertainty. Maps expose heuristic matching scores separately.

## Testing and architecture

```powershell
uv run pytest -q
uv run ruff check .
uv run python -c "import cycling.pipeline; import cycling.ui"
```

Tests build tiny synthetic ZIP/GPX/TCX/FIT fixtures in memory (including valid FIT CRCs) and mock
OSM responses. They cover formats/units, archive bounds, duplicates, HR imputation/no-HR behavior,
surface uncertainty, travel direction, measured full versus rolling stops, signal waits, recording
gaps, pause events, disconnected routes, adaptive fatigue features, cutoff separation, held-out
sensor stripping, time-weighted HR optimization, fallback prediction, non-overlapping time
accounting, caching and the Streamlit workflow. Tests do not call public OSM services.

| Module | Responsibility |
|---|---|
| `recordings.py`, `importer.py` | Secure parsing, unit normalization, metadata, incremental imports |
| `geometry.py`, `osm.py` | Trace quality, terrain, cache/providers, directional road/control annotations |
| `features.py`, `events.py` | Adaptive segments, cruising labels, turn/traffic behavior and fallback physics |
| `model.py`, `prediction.py` | Personal training, measured-HR effects, constrained pacing, exclusive time ledger |
| `validation.py`, `pipeline.py` | Earlier-only evaluation, calibration, reusable workflows |
| `storage.py`, `config.py` | Local persistence and configuration |
| `ui.py`, `app.py`, `cli.py` | Four-page UI, maps/charts and reproducible local commands |

## Limits and future improvements

* Weather/wind, drafting, traffic density, equipment changes, rider intent, sleep and nutrition are
  not independently modeled. Speed at an HR is observational, not a causal guarantee.
* Local matching is conservative rather than a global HMM. Valhalla currently confirms compatible
  annotations; ambiguous route-wide alignment, traffic lane programs and road relations are limited.
* Geometry-derived corner radius is approximate; OSM matching retains original GPX coordinates.
  A supplied DEM/elevation correction is supported, but no automatic global elevation download is bundled.
* HR drift and fatigue can be hard to separate from hills/effort. The model controls for mapped terrain,
  uses regularization, and exposes weak support; it cannot prove a physiological fatigue mechanism.
* Intersection wait distributions need repeated well-recorded mapped encounters. They are smoothed
  empirical distributions, not signal-cycle forecasts. Unknown waits are not invented. Some apparent
  full stops may still be incidental pauses beside an intersection; recordings cannot prove why.
* The basic median/gradient baseline comparison is reported; a boosted model that underperforms is
  flagged instead of advertised as an improvement. Feature-attribution/causal explanations, nested
  hyperparameter searches, probabilistic HR dynamics and calibrated elapsed intervals are not included.
* Configured ZIP/point limits bound individual inputs; parsing a large allowed archive can still take
  substantial memory/time. Regional local OSM extracts are loaded in memory. No background job queue
  or multi-user shared-data deployment is provided; run this as a single-user local app.

All actual exports, GPS files, Parquet, SQLite state, JSON reports, configuration and fitted models
must stay in ignored locations. The default `data/` and private file extensions are ignored.
Do not commit personal artifacts or credentials. No Strava API account or hardcoded key is needed.
