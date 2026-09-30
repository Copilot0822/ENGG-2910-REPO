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

# J1939 single-frame aligned baseline
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

`vendor/j1939/J1939_2024-06_v2.4.json` is the unmodified normalized conversion
supplied for this project. The loader validates the actual file rather than
hard-coding database properties. It found:

| Property | Validated count |
| --- | ---: |
| Messages / unique PGNs | 2,043 |
| Signals / SPNs | 13,774 |
| Aligned 8/16/32 signals in the complete database | 6,930 |
| Messages with payload length at most 8 bytes | 1,891 |
| Messages requiring J1939 TP | 152 |
| All-linear signals in single-frame messages | 12,165 |
| Aligned 8/16/32 targets in single-frame messages | 5,493 |

The database metadata contains the SHA-256 of the source DBC, not the converted
JSON. Therefore the project preserves that embedded source digest and pins the
JSON itself with the adjacent `.json.sha256` file. Loading fails if the JSON's
calculated SHA-256 does not match the sidecar. The pinned JSON SHA-256 is
``.
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

J1939 labels include a neutral stable identifier to disambiguate duplicate
names. The mapping to SPN, real name, PGN, location, encoding, database identity,
and message length is retained only in ground truth and the catalog.

## Project layout

```text
src/simulator/                 shared NMEA 2000 simulator and CLI
src/simulator/j1939/           J1939 database, IDs, encoding, TP, datasets, experiments
src/inference/                 protocol-neutral regression, ranking, output, evaluation
src/inference/j1939_tp.py      inference-side metadata-free TP reassembly
tests/                         unit, integration, leakage, and reproducibility tests
vendor/canboat/                pinned Canboat JSON
vendor/j1939/                  pinned J1939 JSON and checksum sidecar
data/simulated/                generated NMEA 2000 output (ignored by Git)
data/j1939/                    generated J1939 output (ignored by Git)
```

The central research question is: can a protocol-independent linear
signal-inference method reconstruct unknown signal mappings across both NMEA
2000 and SAE J1939, and how do observation count and candidate-search
complexity affect recovery accuracy?
