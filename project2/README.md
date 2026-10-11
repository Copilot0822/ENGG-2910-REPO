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

Strava account exports can be requested from **Settings > My Account > Download your account >
Get Started > Request download**. Download the account ZIP when Strava makes it available.
See [Strava's export documentation](https://support.strava.com/en-us/articles/15401919-how-do-i-export-my-strava-data).
Komoot's route export menu provides the GPX file for a planned tour; export access depends on
your Komoot account and region entitlement. See [Komoot GPX export help](https://support.komoot.com/hc/en-us/articles/10115477099674-Export-and-import-Routes-and-Activities).

## Separate uv environment

This directory is a standalone Python project, independent of the repository's parent project.
It has its own `pyproject.toml`, committed `uv.lock`, `.python-version` and ignored `.venv`.
Always run commands **from `project2`** or use `uv run --project project2 ...` from the repository root.
Verify the interpreter with:

```powershell
uv run python -c "import sys; print(sys.executable)"
uv run pytest -q
uv run ruff check .
```

The interpreter path ends in `project2\.venv\Scripts\python.exe`. Dependencies are pinned by
`uv.lock`; `uv sync --locked` reproduces them. PBF support is included in the normal sync; pyosmium has a native Windows wheel.

## CLI

The command-line pipeline is the same as the web app. Save `osm_provider: "local"` in the
sidebar or configuration before processing these extracts. Example PowerShell commands:

```powershell
uv run cycling import "C:\Users\you\Downloads\export.zip"
uv run cycling osm regions
uv run cycling osm download --regions ontario quebec new-brunswick prince-edward-island
uv run cycling osm index
uv run cycling process
uv run cycling train --optimize-half-life --validation-rides 48
uv run cycling predict "C:\Users\you\Downloads\planned.gpx" --hr 142 --scenario full_stop --break-minutes 15
uv run cycling status
uv run cycling config
```

`cycling process` detects new rides and changed relevant extracts/matching versions automatically.
Use `--refresh-osm` only to force a rebuild. `cycling osm report` recalculates coverage from caches.
`cycling compare --validation-rides 48` performs the same paired A/B/C evaluation as validated
training in local mode; it preserves all candidate models and retains Model C as the default regardless of accuracy ranking.
When ECCC enrichment is available, it compares B, C and C+Weather on the same folds.
`cycling train --no-validation` provides a quick fit but no calibrated prediction interval.
`cycling predict --output data\prediction.csv` saves private segment results locally.
`cycling --data-dir <path> ...` or `CYCLING_DATA_DIR` can override the default data directory.
If you choose another directory inside Git, add it to your ignore rules before storing private data.

## OSM and map matching

For full history, select **local** in the sidebar. Download public regional extracts once from
[Geofabrik Canada](https://download.geofabrik.de/north-america/canada.html), using the CLI above
or the Historical Data extract controls. The `osm regions` command summarizes historical distance
in coarse geographic cells locally. Choose extracts covering those cells; the downloader supports
Ontario, Quebec, New Brunswick and Prince Edward Island. Other regions can be downloaded from
Geofabrik and configured by path. No GPS recordings are sent to a matching service.

Save the settings in the sidebar or use ignored `config.local.json`:

```json
{
  "osm_provider": "local",
  "local_osm_paths": [],
  "half_life_days": 365,
  "optimize_half_life": true,
  "validation_rides": 48,
  "osm_workers": 3
}
```

Empty paths discover `data/osm/extracts/*.osm.pbf`. To use externally stored extracts, set
`local_osm_path` for one file or `local_osm_paths` to a list of absolute paths. XML, compressed XML,
Overpass JSON and PBF are accepted. PBF processing streams highway ways and control nodes using
[pyosmium](https://docs.osmcode.org/pyosmium/latest/reference/File-Processing/), storing complete
road geometry and tags in a separate SQLite/RTree index per extract. Node location indexing uses
memory while building the index; province extracts require several GB of available memory and
disk. No Docker, routing server, credentials or API keys are needed.

The local directed HMM evaluates GPS distance and heading together with bounded shortest-path
transitions between consecutive anchors, using forward/backward costs and Viterbi. Connectivity
uses OSM node IDs: geometric bridge/tunnel crossings are never joined. Bicycle prohibitions and
oneway:bicycle overrides are respected. Timestamp bounds reject impossible transitions; sparse
samples retain their original geometry. The matcher annotates the trace and **never replaces it
with a routed path**. Persistent graph and spatial-index caches avoid rebuilding province data.

Each point retains way/edge IDs, raw tags, original and matched geometry, travel direction,
represented distance, offset and confidence. Confidence is a heuristic score, not a calibrated
probability or surveyed truth. Scores below 0.50 remain explicit low-confidence matches and receive
unknown model labels; high confidence starts at 0.75. Indistinguishable parallel roads/path sections
remain uncertain. The inspector overlays original GPS, selected OSM edges and uncertain/untagged
sections, with random historical activity selection and distance-window inspection.

Surfaces come from actual surface tags. Tracktype inference is flagged separately; untagged highway
classes are never silently labeled paved. The dashboards weight coverage and composition by
represented distance rather than GPS point counts. Original unusual tags remain available even
when classification is unknown. Current extracts describe today's map, not the exact historical
network at each ride date. Downloaded source timestamps, SHA-256 and index metadata are retained.

Controls require traversal of a connected matched approach, appropriate direction tags and adequate
confidence. A nearby side-road stop or pedestrian-only signal is excluded. Shared signals on short
connected approaches are clustered; repeated traversals are separate encounters. Shared stop
nodes with unspecified approach and unsupported numeric directional tags are uncertain. Sparse
crossings can be detected between anchors, without requiring a sample precisely at the control.
Relations/turn restrictions, lanes, signal programs and every OSM tagging convention are not
fully interpreted; these are limitations of matching and control coverage, not inferred certainty.

**Offline** retains bounded cached road lookups or unknown labels without network requests.
**Overpass** is available only for small route lookups, with the existing tile budget, throttling,
retries and caching; bulk history deliberately rejects this mode. Coarse bounding boxes disclose
approximate route areas to the provider. For private matching use local extracts and leave the
online basemap disabled. Optional Valhalla confirmation accepts localhost endpoints only; the
built-in HMM is operational without it. Online map tiles are requested only when explicitly enabled.

Extract updates rebuild only their source indexes. Historical signatures include relevant source
fingerprints, matcher version and segmentation version. Each completed ride persists annotations,
segments and encounters, making interrupted processing resumable without reparsing FIT files.
Up to three local workers process different rides; `osm_workers: 1` reduces memory/CPU use.
Paired backtests also use up to three separate local processes, each with bounded numerical
threads; set `comparison_workers: 1` for serial evaluation.
GPX caches include content/elevation correction and relevant OSM revisions. Uncovered routes remain
unchanged with unknown attributes and warning messages.

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

In local mode with at least 20 rides, validated training compares three variants on the same
chronological test rides: **A** preserves the original geometry model and its cruising labels;
**B** adds surface/road attributes with geometry-only interruption accounting; **C** also excludes
mapped traffic windows from cruising labels and adds their moving and stationary effects once.
A private Phase 1 snapshot can preserve the exact original segments/model for comparison. When
absent, geometry-only labels are reconstructed from cleaned traces. No private baseline is tracked.

Earlier selection folds choose each variant's half-life (60/90/120/180/365 days) and evaluate the
restrained prefix-only fitness correction. Hyperparameter choices are frozen before final evaluation. The traffic-aware default follows the user preference and is not selected from final errors.
The final folds follow all selection folds; every prediction refits only earlier rides. Distance
stratification includes long rides before inspecting errors. With `validation_rides: 48`, the
comparison uses 12 selection and up to 32 final rides, retaining sufficient training history.
Smaller datasets use fewer folds; ordinary validation remains available below 20 rides.

Reports include paired speed MAE/RMSE, moving-time MAE/bias, measured-HR status, losses/waits,
interval coverage and errors by distance, elevation, paved/gravel composition and month. The prior
reported MAE on eight late rides is preserved for context; it is not directly comparable with a
new 32-ride sample. Use the A/B/C results on identical folds to assess improvement or regression.
Current OSM coverage/ambiguity and sparse control observations can worsen accuracy. The UI reports
regressions without claiming that additional features guarantee improvement.

Model C is the persistent default, with C+Weather preferred after weather-aware training. B stays selectable as a comparator/fallback. C uses explicit mapped-traffic waits and the full-stop scenario. A/B do not add
mapped traffic penalties to labels already containing those effects; their elapsed totals are
lower bounds. The same variant-specific feature/label construction is used for new GPX prediction.

Elapsed accuracy requires reliable timestamps, sufficient matched coverage, no unlearned modeled
waits, and at least 90% of recorded stationary time attributable to reliable traffic windows.
Unclassified personal stops and recording gaps therefore often leave elapsed MAE unavailable.
This is reported separately from rolling-speed accuracy. A diagnostic elapsed lower-bound error
is retained, but is not presented as reliable traffic-delay accuracy.

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
accounting, caching and the Streamlit workflow. Tests mock OSM and ECCC responses and do not call public weather services.

| Module | Responsibility |
|---|---|
| `recordings.py`, `importer.py` | Secure parsing, unit normalization, metadata, incremental imports |
| `geometry.py`, `osm.py`, `network.py`, `matching.py` | Trace quality, streaming regional indexes, directed HMM and control annotations |
| `coverage.py`, `comparison.py` | Distance-weighted quality reports and paired A/B/C chronological evaluation |
| `features.py`, `events.py` | Adaptive segments, cruising labels, turn/traffic behavior and fallback physics |
| `model.py`, `prediction.py` | Personal training, measured-HR effects, constrained pacing, exclusive time ledger |
| `validation.py`, `pipeline.py` | Earlier-only evaluation, calibration, reusable workflows |
| `storage.py`, `config.py` | Local persistence and configuration |
| `ui.py`, `app.py`, `cli.py` | Four-page UI, maps/charts and reproducible local commands |

## Limits and future improvements

* Station weather is modeled observationally; road shelter, drafting, signal cycles, equipment changes, rider intent, sleep and nutrition are
  not independently measured. Speed at an HR is observational, not a causal guarantee.
* Local HMM search is bounded to six candidates and local transitions. Ambiguous parallel roads,
  incomplete topology, turn restriction relations and lane-specific signal programs remain limited.
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
  substantial memory/time. PBF node-location indexing requires memory during the one-time index build. No background job queue
  or multi-user shared-data deployment is provided; run this as a single-user local app.

All actual exports, GPS files, Parquet, SQLite state, JSON reports, configuration and fitted models
must stay in ignored locations. The default `data/` and private file extensions are ignored.
Do not commit personal artifacts or credentials. No Strava API account or hardcoded key is needed.

## ECCC weather and Phase 3 traffic behavior

```powershell
uv run cycling weather enrich
uv run cycling weather report
uv run cycling train --enrich-weather --optimize-half-life --validation-rides 48
uv run cycling predict "planned-route.gpx" --hr 145 --departure "2026-10-11T09:00:00-04:00" --forecast
uv run cycling predict "planned-route.gpx" --model-variant CW --weather-neutral
```

The Historical Data page provides weather enrichment/retry, cache-only operation, station metadata,
missing values and coverage reports. UI training enriches missing weather by default; CLI training
uses `--enrich-weather` for this step. `weather --offline` uses cached records. Retraining never
switches the default away from the C family. B, C and C+Weather remain distinct saved models.
Weather-neutral C+Weather inference uses its separately fitted terrain model, rather than invented
zero wind or artificial typical weather.

The primary sources are [ECCC hourly observations](https://api.weather.gc.ca/collections/climate-hourly)
and [station metadata](https://api.weather.gc.ca/collections/climate-stations). The API only covers
a subset of archived stations, so missing stations are checked against the official monthly
[historical downloads](https://climate.weather.gc.ca/FAQ_e.html). Hourly archive CSV times are local
**standard** time, converted using station offsets without daylight saving. API UTC timestamps
remain UTC. Coarse whole-degree metadata bounds disclose a general area; observation requests use
station IDs and calendar months, not GPS traces. Monthly cache reuse avoids repeated queries.
Requests are serial, throttled to one per second, bounded to 700 per operation, retried on failure,
and paginated at 1,000 items with a page cap. Transient failures are not cached as successful empties.
ECCC attribution, meaningful User-Agent and caching follow the
[MSC usage policy](https://eccc-msc.github.io/open-data/usage-policy/readme_en/).

Segments use original recorded passage timestamps and local route bearings. Candidate selection
considers distance, available observations, elevation difference and quality flags. Up to two
suitable stations are blended, with station IDs, distance, timestamp age, interpolation status,
source records and an explicitly heuristic uncertainty estimate retained. Temporal wind interpolation
operates on vector components, so northward wraparound is correct; hourly precipitation accumulations
are not linearly interpolated. Long missing intervals remain missing. Station winds are generally
measured at 10 m and do not measure rider-height wind or shelter. ECCC describes wind in km/h and
from-direction in tens of degrees; calm is distinguished from missing/variable direction.
See the [official glossary](https://climate.weather.gc.ca/glossary_e.html).

Headwind is `W*cos(from_direction-travel_bearing)` and signed crosswind is the corresponding sine,
in radians. Positive headwind opposes travel. The boosted weather model learns nonlinear speed
relationships using headwind, signed/absolute crosswind, wind speed, temperature, humidity,
precipitation and pressure. Missing-weather sections use the separately fitted terrain model.
Measured-HR training remains distinct from median-imputed ride HR; training cutoffs are unchanged.

The initial working forecast provider uses official
[City Page hourly forecasts](https://eccc-msc.github.io/open-data/msc-data/citypage-weather/readme_citypageweather_en/),
selecting nearby cities within 50 km. Each segment records the source city, distance, issue time and
valid time. This is a regional city forecast, **not a road-scale forecast or gridded product**.
Conditions are reevaluated at predicted segment arrival times until timing stabilizes, and traffic
context also advances along the journey. Outside the hourly horizon, predictions become weather-neutral.
Precipitation probability is not treated as a rainfall amount. Unspecified personal-break locations
are not inserted into segment arrival times. Forecast and neutral scenarios are available side by side.

[HRDPS](https://eccc-msc.github.io/open-data/msc-data/nwp_hrdps/readme_hrdps_en/) and GeoMet WCS were
investigated. Automated gridded HRDPS retrieval/resampling remains an extension to the modular provider;
it is not bundled. Forecast skill has not been validated against archived issued forecasts.
Retrospective validation uses actual observed weather at recorded passage times and is labeled accordingly.

Stop analysis now requires consecutive near-zero recorded speed close to the control and short sample
intervals. Sparse sampling, unobserved gaps, incomplete approach/recovery, uncertain road matches and
long potentially personal dwells are excluded. Metrics include speed at the sign, braking onset distance,
braking and acceleration duration, recovery distance, gradient, surface, turn direction and stable
approach identity. Contextual pools consider road type, turning, surface, gradient, hour/weekend and
approach direction. Sparse intersections borrow a 12-encounter prior; full/rolling/minimal probabilities
sum to one. Predicted loss uses the mixture expectation; signal wait is stop probability times the
conditional stopped-wait distribution. These descriptive probabilities are not legal advice or exact
signal-phase forecasts. Full-stop planning still reaches zero at stop signs.

The map supports headwind/tailwind overlays and clickable traffic markers with probabilities,
minimum speed, moving loss, wait, local/broad counts, confidence and generalized/location-specific source.
Performance reports include paired errors by wind, HR, surface, distance and traffic density, plus
held-out stop Brier scores and wait errors. Individual encounter truth is used only to score predictions,
never supplied to the held-out behavior model. Weather uncertainty and probability intervals are
provisional; future forecast and elapsed-time intervals are not claimed calibrated.

Private Phase 3 reports and inspection artifacts live in `data/reports/`, including
`phase3-report.md`, `weather-coverage.json`, `traffic-behavior.json`, and `model-comparison.json`.
Phase 2 reports and models are preserved in `data/baseline_phase2/`. No new Python dependency is needed.
