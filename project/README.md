# NMEA 2000/CAN dataset simulator

This project generates deterministic synthetic NMEA 2000 traffic from the pinned
Canboat machine-readable database. It writes actual physical CAN frames and a
synchronized long-format table of known physical values. It does **not** contain
the future signal-discovery or machine-learning algorithm.

## Why this exists

The intended experiment gives a later algorithm only raw CAN traffic and known
physical measurements, then asks it to reconstruct a hidden signal map: PGN/CAN
identifier, start bit, width, signedness, scale, and offset. Numeric targets use
the linear model

```text
physical_value = scale * raw_value + offset
raw_value = round((physical_value - offset) / scale)
```

The simulator generates the raw integer first and derives the physical value, so
the bus data and synchronized answer always agree without quantization drift.
Ordinary values exclude Canboat's unknown, error, and reserved sentinels.

## Dataset separation

Only these files are inference inputs:

- `can_frames.csv`: one row per physical CAN frame, with no decoded PGN or field
  metadata.
- `known_values.csv`: synchronized physical values, with no CAN identifier, PGN,
  bit location, width, signedness, scale, or offset.

These files are answer-key/evaluation data and must **never** be supplied to the
signal-discovery algorithm:

- `ground_truth.json`
- `signal_catalog.csv`

`simulation_metadata.json` records reproducibility and coverage statistics.
`unsupported_pgns.json` explains every partial or skipped Canboat definition.

## Canboat database

`vendor/canboat/canboat.json` is an unmodified copy of the supplied Canboat
database. The loader reads its version and schema version from the file and
records its SHA-256 in every run. PGNs and variants are normalized from JSON;
they are not hand-coded or scraped from a website.

Fixed-width complete definitions are supported generically. Fixed-width
incomplete/reverse-engineered definitions are marked `PARTIAL` and use random
opaque fields as distractors where safe. Ambiguous variable-length/dynamic
layouts, multi-repeat structures, non-NMEA packet types, and payloads that cannot
fit Fast Packet are explicitly `SKIPPED` rather than fabricated. One fixed-width
repeating field set is supported with a randomized count; fields within a moving
repeat are intentionally excluded as first-stage linear targets.

## Installation and commands

Run all commands from this `project` directory. The project uses `uv` only:

```powershell
uv sync
uv run python -m simulator --help
uv run python -m simulator --mode random --samples 10000 --seed 42
uv run python -m simulator --mode coverage --min-observations-per-field 200 --seed 42
uv run python -m inference infer --can data/simulated/can_frames.csv --known data/simulated/known_values.csv --output data/inference --mode aligned --top-k 10
uv run python -m inference evaluate --inferred data/inference/inferred_signals.json --ground-truth data/simulated/ground_truth.json
uv run python -m inference experiment --observation-counts 10 25 50 100 200 --seed 42
uv run pytest
```

The default output is `data/simulated/`. Override it with `--output PATH`.
`--messages-per-sample` controls the maximum random traffic density (and the
coverage scheduler's batch size). Timestamps use deterministic 0.1-second steps;
the wall clock is never read. Repeating a command with the same simulator,
database, options, and seed produces byte-identical output.

## Physical framing

Single-frame definitions produce one frame of at most eight bytes. Fast Packet
definitions use the NMEA sequence ID, frame counter, first-frame length byte,
six-byte first chunk, seven-byte continuation chunks, and `0xFF` padding. Every
generated Fast Packet is immediately reassembled and checked byte-for-byte.
CAN identifiers are full 29-bit extended identifiers with PDU1 destination and
PDU2 group-extension handling kept in a separately tested module.

## Project layout

```text
src/simulator/       loader, encoders, generator, output, and CLI
src/inference/       discovery, regression, evaluation, and experiments
tests/               unit, integration, coverage, and reproducibility tests
vendor/canboat/      pinned unmodified Canboat JSON
data/simulated/      generated output (ignored by Git)
```

## Signal inference research question

The inference engine asks how many synchronized observations are needed to
recover an unknown linear CAN signal reliably. Discovery receives only
`can_frames.csv` and `known_values.csv`; names are treated as opaque labels and
are never used to guess a PGN. It searches candidate payload locations and fits

```text
y = a*x + b
```

where `x` is a candidate raw integer and `y` is the synchronized physical value.
The least-squares problem is

```text
X = [[x_1, 1], [x_2, 1], ..., [x_n, 1]]
theta = [a, b]
theta = argmin ||X*theta - y||²
```

or equivalently, minimize

```text
sum((y_i - (a*x_i + b))²)
```

The equivalent centered one-predictor least-squares formula is used instead of
explicitly inverting `X.T @ X` or invoking a general decomposition for every
candidate. Every accepted candidate retains its inferred scale `a`, offset `b`,
R², RMSE, MAE, pair count, raw variance, target variance, normalized error, and
ranking score. Constant candidates, constant targets, and candidates with fewer
than `--min-pairs` synchronized observations are rejected. The configurable
`--min-pairing-fraction` (default 0.5) also rejects streams present for too small
a fraction of a target's samples; this prevents a fixed minimum from admitting
many chance overlaps as datasets grow while still allowing missing messages.
Ranking primarily
rewards R² while penalizing normalized error and low pairing coverage.
An intentionally tiny width penalty breaks ties in favor of the most
parsimonious field when a wider candidate merely appends constant reserved bits.

Random independent simulator fields are valuable here because unrelated CAN
fields do not move together; this reduces accidental correlations and makes a
correct linear encoding stand out from distractors.

### Candidate modes

`--mode aligned` is the Stage 1 baseline. It searches 8-, 16-, and 32-bit signed
and unsigned fields beginning on byte boundaries. Both NMEA little-endian and
big-endian alternatives are tested (8-bit fields have no distinct byte order).

`--mode arbitrary` is Stage 2. It searches every possible bit start while keeping
the same configurable widths, signedness, and byte-order alternatives. The mode
usually evaluates substantially more candidates and is therefore slower.

Candidate streams are grouped by PGN, source, and logical payload length. A
candidate need not be present at every known-value sample; only intersecting
sample IDs form `(x, y)` pairs. If multiple indistinguishable messages collide in
the same stream and sample, that ambiguous sample is excluded rather than paired
arbitrarily.

### Raw message reconstruction

The loader parses each raw 29-bit CAN identifier into priority, PGN, source, and
PDU1 destination. It groups physical rows using the inference-facing
`message_id`/`frame_index` fields, validates Fast Packet sequence IDs and frame
counters, and reconstructs the logical payload before candidate extraction.
Multi-frame groups are unambiguous. A one-frame Fast Packet is recognized only
when its header, declared length, and `0xFF` padding are internally consistent;
this deliberately conservative heuristic can theoretically mistake a matching
single-frame payload for a short Fast Packet.

### Strict answer-key separation

The discovery entry point in `inference/infer.py` loads only the two CSV inputs.
It does not import the simulator or refer to `ground_truth.json`,
`signal_catalog.csv`, or `canboat.json`. Tests run discovery successfully in a
temporary directory containing only the two inference files.

Evaluation is a separate command that runs after discovery. It may read
`ground_truth.json` and reports exact location recovery (PGN, start bit, width),
full encoding recovery (also signedness, scale, and offset within tolerances),
and top-1/top-3/top-5/top-10 location accuracy. Scale and offset absolute and
relative errors are retained for inspection.

### Observation-count experiment

The experiment command generates one deterministic coverage dataset at the
largest requested count, creates known-value subsets for each count, runs both
search modes, evaluates them afterward, and writes:

- `data/experiments/recovery_vs_observations.csv`
- `data/experiments/aligned_vs_arbitrary.csv`
- `data/experiments/top1_location_accuracy.png`
- `data/experiments/top5_location_accuracy.png`
- `data/experiments/full_encoding_accuracy.png`

The declared experiment population uses 8/16/32-bit targets, matching the
implemented Stage 1/2 scope, and includes an unaligned member first when one is
available. Ground truth is used only by this experiment-orchestration selection
and the later evaluation; candidate discovery still sees only opaque signal
labels and the two inference-facing CSVs. Seeds and configuration are recorded
in `experiment_metadata.json`; runtimes are measured rather than hard-coded.
