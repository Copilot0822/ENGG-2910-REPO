# Protocol-independent CAN signal inference

This project generates deterministic NMEA 2000 and SAE J1939 traffic, then
tests whether a protocol-independent mathematical inference engine can recover
hidden signal definitions from only:

- raw physical CAN frames; and
- synchronized, known physical measurements.

For every candidate raw field `x` and known physical value `y`, discovery fits

```text
y = a*x + b
```

by least squares. It ranks candidates using fit quality and pairing coverage,
then reports Top-K location recovery and complete encoding recovery. Signal
names are opaque labels: discovery does not use names, units, SPNs, database
definitions, or ground truth as hints.

## Install and run

Run commands from this `project` directory. The project uses `uv`; do not use
`pip`.

```powershell
uv sync
uv run pytest

# NMEA 2000
uv run python -m simulator --protocol nmea2000 --mode coverage --min-observations-per-field 200 --seed 42
uv run python -m inference infer --protocol nmea2000 --can data/simulated/can_frames.csv --known data/simulated/known_values.csv --output data/inference --mode aligned --top-k 10
uv run python -m inference evaluate --protocol nmea2000 --inferred data/inference/inferred_signals.json --ground-truth data/simulated/ground_truth.json

# J1939: first provide your own licensed normalized JSON (see below)
uv run python -m simulator --protocol j1939 --mode coverage --min-observations-per-field 200 --seed 42 --category aligned_8_16_32 --messages-per-sample 96
uv run python -m inference infer --protocol j1939 --can data/j1939/simulated/can_frames.csv --known data/j1939/simulated/known_values.csv --output data/j1939/inference/aligned_full --mode aligned --top-k 10
uv run python -m inference evaluate --protocol j1939 --inferred data/j1939/inference/aligned_full/inferred_signals.json --ground-truth data/j1939/simulated/ground_truth.json
```

The default seeds and deterministic 0.1-second timestamps make simulator output
reproducible. The wall clock is never used to construct a dataset.

## Strict discovery and answer-key separation

Discovery reads only these clean inputs:

- `can_frames.csv`: one row per physical CAN frame; and
- `known_values.csv`: synchronized physical values with opaque unique labels.

Neither file contains decoded PGNs, SPNs, signal locations, widths, factors, or
offsets. The inference entry point imports no simulator or database loader.
Leakage tests copy only these two CSV files to a temporary directory and run
discovery there.

The evaluator runs only after inference and may read:

- `ground_truth.json`, the machine-readable answer key;
- `signal_catalog.csv`, the human-readable catalog; and
- `simulation_metadata.json`, reproducibility and coverage statistics.

An experiment orchestrator may use ground truth to define a declared benchmark
population, but it passes discovery only opaque target labels and a candidate
space configured independently of the answer key.

## J1939 database

The J1939 research uses the **paid/licensed CSS Electronics SAE J1939
DBC/database**, intentionally not included in this repository because
redistribution is not permitted. To reproduce database-dependent experiments,
users must supply their own licensed copy. Obtain it from
[CSS Electronics](https://www.csselectronics.com/products/j1939-dbc-file).
The local research version is currently `J1939_2024-06_v2.4`.

The runtime expects the project's normalized JSON conversion, not a raw DBC.
There is no automatic download. All database-dependent J1939 entry points
(simulator, observation study, and full-library experiment) use this precedence:

1. Explicit `--database` path.
2. `J1939_DATABASE_PATH` environment variable.
3. Local `vendor/j1939/J1939_2024-06_v2.4.json` under this project.

For example, from the `project` directory:

```powershell
# Use an external licensed JSON without copying it into the repository
$env:J1939_DATABASE_PATH = 'C:\LicensedData\J1939_2024-06_v2.4.json'
uv run python -m simulator --protocol j1939 --samples 100
uv run python -m inference experiment --protocol j1939 --modes aligned
uv run python -m inference full-library --resume

# An explicit path overrides the environment setting
uv run python -m simulator --protocol j1939 --database 'C:\LicensedData\local.json'
```

Alternatively place the licensed normalized JSON in
[`vendor/j1939/`](vendor/j1939/README.md). Database files and checksum sidecars
there are ignored by Git. If the selected file is absent, loading fails with
instructions to provide `--database` or `J1939_DATABASE_PATH`; it does not fall
back from an invalid explicit or environment path.

Public unit tests use small, invented definitions and run without the paid
database. The licensed-database integration test skips cleanly when no local
copy is available. Run only public tests with `uv run pytest -m 'not integration'`
or local licensed checks with `uv run pytest -m integration`.

Simulation uses the local definitions to encode frames, and evaluation uses
local ground truth after discovery. **The inference algorithm does not receive
ground-truth definitions during discovery.** The simulator, inference engine,
least-squares implementation, evaluation, experiment definitions, methodology,
and aggregate results remain public.

The loader validates the supplied file rather than hard-coding database
properties. The research version yielded these aggregate counts:

| Property | Validated count |
| --- | ---: |
| Messages / unique PGNs | 2,043 |
| Signals / SPNs | 13,774 |
| Aligned 8/16/32 signals in the complete database | 6,930 |
| Messages with payload length at most 8 bytes | 1,891 |
| Messages requiring J1939 TP | 152 |
| All-linear signals in single-frame messages | 12,165 |
| Aligned 8/16/32 targets in single-frame messages | 5,493 |

J1939 checksum values and sidecars are not stored, generated, or required by the
runtime. A normalized JSON without checksum metadata is sufficient. Version,
schema, seed, experiment options and aggregate counts document reproducibility.
The original DBC is not needed at runtime.

The simulator reads start bit, width, byte order, signedness, factor, offset,
physical limits, and raw limits from the JSON. It generates a valid raw integer
first and calculates `physical = raw * factor + offset`, ensuring exact agreement
between encoded payloads and synchronized known values. Unsafe definitions raise
an error rather than producing out-of-range data.

## J1939 identifiers and source addresses

J1939 uses 29-bit extended CAN identifiers. The identifier module constructs
and parses priority, reserved bit, data page, PDU format, PDU specific, source,
destination, and PGN.

- For PDU1 (`PF < 240`), PDU Specific is a destination address and is excluded
  from the PGN.
- For PDU2 (`PF >= 240`), PDU Specific is the group extension and is included
  in the PGN.

Database CAN IDs are definition templates. The simulator assigns messages to a
deterministic set of engine, transmission, brake, body, instrument,
aftertreatment, chassis, and miscellaneous ECU source addresses. Evaluation
defines location by PGN, start bit, and bit length; source address is not part
of correctness.

## Physical framing and J1939 Transport Protocol

Payloads of at most eight bytes produce one physical frame. Longer J1939
payloads use modular TP.CM/TP.DT framing; no physical frame exceeds eight bytes.
Broadcast messages use BAM, while destination-specific messages emit RTS, CTS,
data-transfer, and end-of-message acknowledgement frames. The inference loader
recognizes and reassembles TP traffic from the raw CAN identifiers and TP bytes,
without reading simulator definitions or ground truth.

Use `--include-transport` to include long messages:

```powershell
uv run python -m simulator --protocol j1939 --mode random --samples 100 --seed 42 --category all_linear --include-transport --output data/j1939/transport_demo
```

Unit and integration tests cover BAM and RTS/CTS fragmentation, physical frame
limits, metadata-free reassembly, and recovery of a signal from a transported
logical payload.

## Candidate-search modes

`aligned` searches byte-aligned 8-, 16-, and 32-bit signed and unsigned fields.
`arbitrary` searches configurable widths at every possible start bit.
`extended` is the explicit generic wider search used for J1939. Candidate
lengths are supplied on the command line and are never derived from the database
or ground truth.

```powershell
# Search every width from 1 through 32 bits at arbitrary bit positions
uv run python -m inference infer --protocol j1939 --can data/j1939/extended_dataset/can_frames.csv --known data/j1939/extended_dataset/known_values.csv --output data/j1939/inference/extended --mode extended --min-bit-length 1 --max-bit-length 32 --max-signals 25

# Or choose an explicit generic list
uv run python -m inference infer --protocol j1939 --can data/j1939/extended_dataset/can_frames.csv --known data/j1939/extended_dataset/known_values.csv --output data/j1939/inference/common_widths --mode extended --candidate-lengths 1 2 4 8 12 16 24 32 --max-signals 25
```

Whole-byte candidates test both little- and big-endian interpretations.
Sub-byte candidates use the payload's LSB-first DBC bit numbering and therefore
have only one extraction order.

The reproducible aligned-versus-extended benchmark selects one opaque target
for every available width from 1 through 32 and runs both searches over the same
15 signals:

```powershell
uv run python -m simulator --protocol j1939 --mode coverage --min-observations-per-field 25 --seed 42 --category all_linear --messages-per-sample 96 --output data/j1939/extended_dataset
uv run python -m inference compare-search --dataset data/j1939/extended_dataset --output data/j1939/extended_comparison --min-bit-length 1 --max-bit-length 32
```

| Mode | Target coverage | Top-1 | Top-5 / Top-10 | Full encoding | Candidates | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Aligned 8/16/32 | 20.00% | 20.00% overall / 100% in scope | 20.00% | 20.00% overall / 100% in scope | 960 | 3.52 s |
| Extended 1–32 | 100.00% | 80.00% | 93.33% | 73.33% | 50,250 | 10.64 s |

This benchmark makes the search-space trade-off visible: candidate count grew
by 52.3 times, while candidate coverage grew from 3 of 15 targets to all 15.

## Full aligned J1939 baseline

The completed single-frame coverage dataset contains 200 observations per
eligible target:

- 3,473 samples;
- 333,400 logical and physical messages;
- 1,098,600 synchronized target observations; and
- 5,493 aligned 8/16/32-bit targets.

Measured full-population inference results are:

| Metric | Result |
| --- | ---: |
| Candidates tested | 351,192 |
| Top-1 location accuracy | 99.9272% |
| Top-3 location accuracy | 99.9818% |
| Top-5 location accuracy | 100.0000% |
| Top-10 location accuracy | 100.0000% |
| Full encoding accuracy | 98.9077% |
| Mean absolute factor error | 3.256e-16 |
| Mean absolute offset error | 2.174e-7 |
| Runtime | 287.14 s |

Location accuracy is higher than full encoding accuracy because signed and
unsigned extraction can be observationally indistinguishable when generated raw
values never cross the sign bit. Full encoding deliberately requires correct
PGN, start, width, byte order, signedness, factor, and offset.

## Observation-count study

Run the standard J1939 study with:

```powershell
uv run python -m inference experiment --protocol j1939 --observation-counts 10 25 50 100 200 --seed 42 --modes aligned --max-signals 25 --messages-per-sample 96
```

It writes `data/j1939/experiments/recovery_vs_observations.csv` plus Top-1,
Top-5, and full-encoding plots. The measured 25-target aligned results were:

| Observations | Top-1 | Top-5 | Full encoding | Candidates | Runtime |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 100% | 100% | 88% | 1,600 | 8.83 s |
| 25 | 100% | 100% | 88% | 1,600 | 9.52 s |
| 50 | 100% | 100% | 88% | 1,600 | 10.75 s |
| 100 | 100% | 100% | 88% | 1,600 | 13.18 s |
| 200 | 100% | 100% | 92% | 1,600 | 19.08 s |
| 10,000 | 100% | 100% | 92% | 1,600 | 61.99 s |

The 10,000-observation run used the same deterministic 25-target selection but
generated only its 13 required PGNs on disjoint sample ranges. This avoids a
needless 50-times expansion of the complete 5,493-target baseline while keeping
the inference inputs and 1,600-candidate search unchanged. Its two remaining
full-encoding misses had correct locations, byte order, factor, and offset; only
signedness was observationally ambiguous because neither 8-bit unsigned target
entered the sign-bit region.

## NMEA 2000 versus J1939

The J1939 experiment reuses existing NMEA 2000 observation-study results and
writes `data/experiments/n2k_vs_j1939.csv` without changing the NMEA data. For
the deterministic 25-target aligned subsets, NMEA 2000 Top-1/Top-5 accuracy was
96% at every tested count, while J1939 was 100%. Full encoding was 96% for NMEA
2000 and rose from 88% to 92% for J1939. These are matched-size benchmark
subsets, not claims about equal database-wide difficulty; the full J1939 result
above covers all 5,493 baseline targets.

## Complete J1939 library experiment

The full-library command generates 200 observations for every signal in all
2,043 PGNs, including BAM and RTS/CTS payloads, then runs an aligned baseline
and a generic full search:

```powershell
uv run python -m inference full-library --output data/j1939/full_library --observations-per-signal 200 --seed 42 --batch-size 250 --candidate-batch-size 512 --resume
```

Checkpoints are written after each deterministic target batch. `--resume`
validates the saved configuration and skips completed batches. Evaluation and
plots can be regenerated without rerunning simulation or discovery:

```powershell
uv run python -m inference full-library --output data/j1939/full_library --evaluation-only
```

The reconciled population is:

| Population | Signals |
| --- | ---: |
| Aligned 8/16/32 | 6,930 |
| Additional non-aligned or other-width signals through 32 bits | 6,788 |
| Larger 40/56/64-bit signals | 56 |
| Unsupported | 0 |
| Total | 13,774 |

The generic policy is fixed independently of ground truth: every width from 1
through 32 bits plus 40, 48, 56, and 64 bits, at every possible start bit. The
dataset contains 408,600 logical messages, 547,600 physical frames, and
2,754,800 synchronized target observations.

Measured category results are:

| Category/search | Signals | Top-1 | Top-3 | Top-5 | Top-10 | Raw full | Identifiable full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A: aligned baseline | 6,930 | 99.9278% | 99.9711% | 100% | 100% | 98.2107% | 98.2828% |
| B: extended, at most 32 bits | 13,718 | 98.6587% | 99.4460% | 99.4679% | 99.5626% | 97.8787% | 98.8118% |
| C: 40/56/64-bit | 56 | 0% | 0% | 0% | 0% | 0% | 92.8571% |
| D: complete inferable library | 13,774 | 98.2576% | 99.0417% | 99.0635% | 99.1578% | 97.4808% | 98.7876% |

The 56 large fields deliberately expose identifiability rather than transport
failure. Their valid raw intervals leave most declared upper bits permanently
zero, so shorter fields at the same start reproduce the physical values exactly.
The conservative ambiguity rule classifies 52 as equivalent and leaves four as
incorrect because their best exact fit crosses a neighboring field boundary.

For the complete generic run, best-candidate component accuracy was 100% PGN,
99.6515% start bit, 98.2576% declared width, 100% byte order, and 99.9419%
signedness. Mean absolute factor and offset errors across all best candidates
were 0.07568 and 0.10943. The final classifications were 13,427
`IDENTIFIABLE`, 180 `AMBIGUOUS`, and 167 `INCORRECT`.

Ambiguity uses the complete effective raw interval, calculated from the
intersection of declared raw and physical ranges. Signedness is ambiguous only
when the whole interval lies below the sign bit. Width is ambiguous only when
upper bits remain zero across the whole interval and a shorter candidate at the
same PGN/start has the same byte order, factor, and offset. This is stricter than
calling two candidates ambiguous merely because they happened to agree in the
200 generated samples.

Transport and alignment results under the generic full search were:

| Group | Signals | Top-1 | Identifiable full |
| --- | ---: | ---: | ---: |
| Byte-aligned 8/16/32 | 6,930 | 97.4315% | 97.6623% |
| Non-aligned or other width | 6,844 | 99.0941% | 99.9269% |
| Single-frame | 12,165 | 98.8492% | 99.7863% |
| BAM | 1,575 | 93.7143% | 91.1111% |
| RTS/CTS | 34 | 97.0588% | 97.0588% |

The aligned pass extracted 143,344 candidate definitions and evaluated 967,480
regressions in 19.54 seconds. The complete search extracted 9,465,786 reusable
candidate definitions and evaluated 89,775,236 regressions in 83.07 seconds:
27.13 seconds extracting candidate matrices, 32.40 seconds fitting/ranking, and
13.60 seconds loading inputs. Peak candidate-matrix size was 2.44 MiB. Complete
simulation, both searches, evaluation, serialization, and plotting took 142.76
seconds.

Local runs write per-signal results, aggregate accuracy tables, search-space
measurements, and summary plots under the ignored `data/j1939/full_library/`.
Catalogs, ground truth, per-signal inferred encodings, and checkpoints may
reconstruct licensed definitions and must remain local. Publish only plots or
aggregate results that do not reproduce those definitions. Custom output paths
outside the ignored J1939 data directory also need protection before committing.

## Output schemas

`data/j1939/simulated/can_frames.csv` uses:

```text
sample_id,timestamp,message_id,frame_index,can_id,data_hex
```

`can_id` is the raw 29-bit identifier. `data_hex` contains at most eight bytes.
There are no decoded PGNs or signal definitions in this file.

`data/j1939/simulated/known_values.csv` uses:

```text
sample_id,timestamp,observation_id,signal_name,value,unit
```

J1939 measurement labels are opaque, for example
`signal_000001 [j1939_signal_000001]`, and the `unit` column is blank. The mapping
to SPN, real name, PGN, location, encoding and message length is retained only
in the private local ground truth and catalog.

## Generated data and public results

Licensed-derived answer keys, catalogs, inferred encodings, candidate rankings,
decoded/converted definition exports, debug dumps and checkpoints remain private.
The root `.gitignore` protects their filenames throughout the repository. Writers
also add output-local ignore rules so custom output directories protect private
artifacts. Ignore rules do not prevent deliberate force-adds; review exports
before publishing them.

Public evaluation JSON contains summary accuracy/error statistics and opaque
per-signal IDs with correctness/error metrics. Full-library result CSVs exclude
real names, units, PGNs, start bits, widths, byte order, signedness, factors,
offsets, ranges, and per-signal transport/alignment mappings. True widths and
categories are used only in memory to compute aggregate breakdowns and plots.

Raw frame CSVs and new opaque measurement CSVs can be shared if they contain no
hidden definitions. Legacy named measurement files and detailed exports remain
private under `data/j1939/`; regenerate them before sharing. Existing full-library
checkpoints use an older configuration schema: choose a new output directory
when regenerating the experiment. Database-independent NMEA 2000 and synthetic
unit tests remain available without the licensed J1939 file.

## Project layout

```text
src/simulator/                 shared NMEA 2000 simulator and CLI
src/simulator/j1939/           J1939 database, IDs, encoding, TP, datasets, experiments
src/inference/                 protocol-neutral regression, ranking, output, evaluation
src/inference/j1939_tp.py      inference-side metadata-free TP reassembly
tests/                         unit, integration, leakage, and reproducibility tests
vendor/canboat/                pinned Canboat JSON
vendor/j1939/                  public setup README; ignored licensed local JSON
data/simulated/                generated NMEA 2000 output (ignored by Git)
data/j1939/                    generated J1939 output (ignored by Git)
```

The central research question is: can a protocol-independent linear
signal-inference method reconstruct unknown signal mappings across both NMEA
2000 and SAE J1939, and how do observation count and candidate-search
complexity affect recovery accuracy?
