"""Inference result serialization."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from inference.infer import InferenceRun


RANKING_COLUMNS = [
    "signal_name",
    "signal_id",
    "rank",
    "pgn",
    "source",
    "payload_length",
    "start_bit",
    "bit_length",
    "signed",
    "byte_order",
    "scale",
    "offset",
    "r2",
    "rmse",
    "mae",
    "observation_count",
    "raw_variance",
    "target_variance",
    "normalized_rmse",
    "pairing_fraction",
    "ranking_score",
]


def write_inference_outputs(output_directory: str | Path, run: InferenceRun) -> None:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    signals = {signal.signal_name: signal.to_dict() for signal in run.signals}
    (output / "inferred_signals.json").write_text(
        json.dumps({"metadata": run.metadata, "signals": signals}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (output / "inference_metadata.json").write_text(
        json.dumps(run.metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "candidate_rankings.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=RANKING_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for signal in run.signals:
            for rank, candidate in enumerate(signal.candidates, start=1):
                row = candidate.to_dict()
                row.update(
                    {
                        "signal_name": signal.signal_name,
                        "signal_id": signal.signal_id,
                        "rank": rank,
                    }
                )
                writer.writerow(row)

