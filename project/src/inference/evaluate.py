"""Post-inference evaluation. This is the only module that reads ground truth."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from inference.output_privacy import public_signal_results, public_summary


def _location_matches(candidate: dict[str, Any], truth: dict[str, Any]) -> bool:
    return (
        int(candidate["pgn"]) == int(truth["pgn"])
        and int(candidate["start_bit"]) == int(truth["start_bit"])
        and int(candidate["bit_length"]) == int(truth["bit_length"])
    )


def _relative_error(actual: float, expected: float, absolute_tolerance: float) -> float:
    return abs(actual - expected) / max(abs(expected), absolute_tolerance)


def evaluate_results(
    inferred_path: str | Path,
    ground_truth_path: str | Path,
    *,
    scale_relative_tolerance: float = 1e-5,
    scale_absolute_tolerance: float = 1e-8,
    offset_relative_tolerance: float = 1e-5,
    offset_absolute_tolerance: float = 1e-6,
) -> dict[str, Any]:
    inferred = json.loads(Path(inferred_path).read_text(encoding="utf-8"))
    truth_document = json.loads(Path(ground_truth_path).read_text(encoding="utf-8"))
    truths = truth_document["signals"]
    details: list[dict[str, Any]] = []
    location_hits = {1: 0, 3: 0, 5: 0, 10: 0}
    full_hits = 0
    scale_absolute_errors: list[float] = []
    scale_relative_errors: list[float] = []
    offset_absolute_errors: list[float] = []
    offset_relative_errors: list[float] = []
    located_scale_absolute_errors: list[float] = []
    located_offset_absolute_errors: list[float] = []

    for signal_name, signal in inferred.get("signals", {}).items():
        signal_id = signal.get("signal_id")
        truth = truths.get(signal_id)
        if truth is None:
            continue
        candidates = signal.get("candidates", [])
        hit_ranks = [
            rank
            for rank, candidate in enumerate(candidates, start=1)
            if _location_matches(candidate, truth)
        ]
        first_location_rank = min(hit_ranks, default=None)
        for k in location_hits:
            if first_location_rank is not None and first_location_rank <= k:
                location_hits[k] += 1

        best = candidates[0] if candidates else None
        full_match = False
        scale_abs = scale_rel = offset_abs = offset_rel = None
        if best is not None:
            expected_scale = float(truth["scale"])
            expected_offset = float(truth["offset"])
            actual_scale = float(best["scale"])
            actual_offset = float(best["offset"])
            scale_abs = abs(actual_scale - expected_scale)
            scale_rel = _relative_error(
                actual_scale, expected_scale, scale_absolute_tolerance
            )
            offset_abs = abs(actual_offset - expected_offset)
            offset_rel = _relative_error(
                actual_offset, expected_offset, offset_absolute_tolerance
            )
            scale_absolute_errors.append(scale_abs)
            scale_relative_errors.append(scale_rel)
            offset_absolute_errors.append(offset_abs)
            offset_relative_errors.append(offset_rel)
            if _location_matches(best, truth):
                located_scale_absolute_errors.append(scale_abs)
                located_offset_absolute_errors.append(offset_abs)
            full_match = (
                _location_matches(best, truth)
                and str(best.get("byte_order", "little_endian"))
                == str(truth.get("byte_order", "little_endian"))
                and bool(best["signed"]) == bool(truth["signed"])
                and math.isclose(
                    actual_scale,
                    expected_scale,
                    rel_tol=scale_relative_tolerance,
                    abs_tol=scale_absolute_tolerance,
                )
                and math.isclose(
                    actual_offset,
                    expected_offset,
                    rel_tol=offset_relative_tolerance,
                    abs_tol=offset_absolute_tolerance,
                )
            )
        if full_match:
            full_hits += 1
        details.append(
            {
                "signal_name": signal_name,
                "signal_id": signal_id,
                "location_rank": first_location_rank,
                "top1_location_correct": first_location_rank == 1,
                "full_encoding_correct": full_match,
                "scale_absolute_error": scale_abs,
                "scale_relative_error": scale_rel,
                "offset_absolute_error": offset_abs,
                "offset_relative_error": offset_rel,
            }
        )

    total = len(details)

    def accuracy(count: int) -> float:
        return count / total if total else 0.0

    def average(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    metadata = inferred.get("metadata", {})
    mode = metadata.get("inference_mode")
    bit_lengths = set(metadata.get("supported_bit_lengths", []))
    recoverable = sum(
        int(truths[item["signal_id"]]["bit_length"] in bit_lengths)
        and (mode != "aligned" or truths[item["signal_id"]]["start_bit"] % 8 == 0)
        for item in details
    )
    recoverable_details = [
        item
        for item in details
        if truths[item["signal_id"]]["bit_length"] in bit_lengths
        and (
            mode != "aligned"
            or int(truths[item["signal_id"]]["start_bit"]) % 8 == 0
        )
    ]

    def recoverable_accuracy(predicate: Any) -> float:
        if not recoverable_details:
            return 0.0
        return sum(bool(predicate(item)) for item in recoverable_details) / len(
            recoverable_details
        )

    return {
        "summary": {
            "total_signals": total,
            "recoverable_signals_in_mode": recoverable,
            "top1_location_accuracy": accuracy(location_hits[1]),
            "top3_location_accuracy": accuracy(location_hits[3]),
            "top5_location_accuracy": accuracy(location_hits[5]),
            "top10_location_accuracy": accuracy(location_hits[10]),
            "full_encoding_accuracy": accuracy(full_hits),
            "top1_location_accuracy_within_scope": recoverable_accuracy(
                lambda item: item["location_rank"] == 1
            ),
            "top3_location_accuracy_within_scope": recoverable_accuracy(
                lambda item: item["location_rank"] is not None
                and item["location_rank"] <= 3
            ),
            "top5_location_accuracy_within_scope": recoverable_accuracy(
                lambda item: item["location_rank"] is not None
                and item["location_rank"] <= 5
            ),
            "top10_location_accuracy_within_scope": recoverable_accuracy(
                lambda item: item["location_rank"] is not None
                and item["location_rank"] <= 10
            ),
            "full_encoding_accuracy_within_scope": recoverable_accuracy(
                lambda item: item["full_encoding_correct"]
            ),
            "average_scale_absolute_error": average(scale_absolute_errors),
            "average_scale_relative_error": average(scale_relative_errors),
            "average_offset_absolute_error": average(offset_absolute_errors),
            "average_offset_relative_error": average(offset_relative_errors),
            "average_scale_absolute_error_when_location_correct": average(
                located_scale_absolute_errors
            ),
            "average_offset_absolute_error_when_location_correct": average(
                located_offset_absolute_errors
            ),
            "scale_relative_tolerance": scale_relative_tolerance,
            "scale_absolute_tolerance": scale_absolute_tolerance,
            "offset_relative_tolerance": offset_relative_tolerance,
            "offset_absolute_tolerance": offset_absolute_tolerance,
        },
        "signals": details,
    }


def write_evaluation(path: str | Path, evaluation: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({
            "summary": public_summary(evaluation["summary"]),
            "signals": public_signal_results(evaluation.get("signals", [])),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
