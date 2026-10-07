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

Any dataset generated from the licensed CSS Electronics J1939 database remains
private, including raw CAN frames and synchronized measurements. Opaque labels
and blank units do not make these inputs safe to publish: the inference system
can reconstruct signal encodings from the paired data. Catalogs, ground truth,
inferred encodings, candidate rankings, private metadata, decoded/converted
exports and checkpoints must also stay local. Writers install output-local
ignore rules even for custom directories and resumed runs.

Only aggregate statistics, plots, methodology, source code and deliberately
invented synthetic test datasets are intended for public distribution. Public
per-signal results use opaque IDs with correctness/error metrics and omit
definitions, names, units and original database ID mappings. The invented test
fixtures keep the project testable without a paid database. J1939 sources default
to private output; only deliberately invented definitions may explicitly use
`private_derived_output=False`. Use a new directory for legacy checkpoint runs.

Historical licensed objects were removed from reachable refs, but GitHub
server-side retained/cached objects may require GitHub Support cleanup before
the repository is made public. Keep it private until that cleanup is confirmed.
