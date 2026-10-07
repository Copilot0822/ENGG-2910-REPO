# Local licensed J1939 database

The research uses the paid/licensed CSS Electronics SAE J1939 DBC/database.
Redistribution is not permitted, so its definitions are intentionally excluded
from this public repository. The local research version is currently
`J1939_2024-06_v2.4`.

Obtain your own licensed copy from
[CSS Electronics](https://www.csselectronics.com/products/j1939-dbc-file).
Place your normalized JSON at `J1939_2024-06_v2.4.json` in this directory, or
configure another location with `--database` or `J1939_DATABASE_PATH`. The
runtime loads the project's normalized JSON schema; it does not load the DBC
directly or download a database. Newer compatible JSON versions can be supplied
using either path setting, but may produce different aggregate research results.

Path selection is explicit CLI argument, environment variable, then this local
default. J1939 checksums and checksum sidecars are neither required nor generated.
Database JSON, DBC, and any stray checksum files here are ignored by Git,
including future versions and subdirectories. Setup documentation remains public.

The repository contains inference and simulation code, least-squares fitting,
evaluation, experiment definitions, methodology, and aggregate results. It
does not contain the licensed J1939 definitions. Discovery receives only CAN
frames and synchronized measurements; ground-truth definitions are used locally
by simulation and subsequent evaluation.

Generated catalogs, ground truth, inferred encodings, conversion/debug exports,
and inference checkpoints can expose or reconstruct licensed definitions. Keep
these artifacts local. The repository ignores their filenames throughout the
tree; output writers also create local ignore rules for custom directories.

New CAN-frame and measurement CSVs contain only raw traffic, opaque labels and
values, with blank units. Public evaluation JSON/CSV contains aggregate results
and opaque per-signal accuracy/error metrics, without names or true definitions.
Legacy datasets with real names or units remain private and must be regenerated
before sharing. Use a new output directory for old checkpoint-based runs.
