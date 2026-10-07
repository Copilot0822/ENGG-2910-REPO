# Development summary

This project simulates CAN traffic and recovers candidate signal encodings
through least-squares fitting. Discovery uses traffic and synchronized
measurements; ground truth is used only by simulation and later evaluation.

Use uv for dependencies and run `uv run pytest --tb=short` for validation.
Public tests use small, deliberately invented J1939 definitions. The licensed
integration test skips when no local licensed database is available.

The paid CSS Electronics J1939 database is supplied locally and is not
redistributed. All datasets derived from it, including raw CAN frames and
known-value measurements, remain private. J1939 sources default to private
output; only deliberately invented definitions may explicitly opt out.
Public evaluation serializers expose aggregate metrics and opaque result IDs.
The unrelated public Canboat database remains included.

See [README.md](README.md) for current setup, methodology and privacy rules,
and [J1939_PRIVACY_AUDIT.md](J1939_PRIVACY_AUDIT.md) for the publication gate.
