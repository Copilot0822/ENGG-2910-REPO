"""Detailed post-inference evaluation for the J1939 full-library experiment."""

from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inference.output_privacy import public_signal_results


RESULT_COLUMNS = [
    "signal_id",
    "signal_name",
    "result_status",
    "true_bit_length",
    "true_alignment_category",
    "transport_category",
    "top1_location_correct",
    "top3_location_correct",
    "top5_location_correct",
    "top10_location_correct",
    "full_encoding_correct",
    "identifiable_full_encoding_correct",
    "identifiability_status",
    "correct_pgn",
    "correct_start_bit",
    "correct_width",
    "correct_byte_order",
    "correct_signedness",
    "scale_error",
    "offset_error",
    "candidate_count",
]


def _location(candidate: dict[str, Any], truth: dict[str, Any]) -> bool:
    return (
        int(candidate["pgn"]) == int(truth["pgn"])
        and int(candidate["start_bit"]) == int(truth["start_bit"])
        and int(candidate["bit_length"]) == int(truth["bit_length"])
    )


def _close_scale(candidate: dict[str, Any], truth: dict[str, Any]) -> bool:
    return math.isclose(
        float(candidate["scale"]),
        float(truth["scale"]),
        rel_tol=1e-5,
        abs_tol=1e-8,
    )


def _close_offset(candidate: dict[str, Any], truth: dict[str, Any]) -> bool:
    return math.isclose(
        float(candidate["offset"]),
        float(truth["offset"]),
        rel_tol=1e-5,
        abs_tol=1e-6,
    )


def _effective_raw_bounds(truth: dict[str, Any]) -> tuple[int, int]:
    factor = Decimal(str(truth["scale"]))
    offset = Decimal(str(truth["offset"]))
    physical_minimum = Decimal(str(truth["minimum"]))
    physical_maximum = Decimal(str(truth["maximum"]))
    raw_min = int(truth["raw_min"])
    raw_max = int(truth["raw_max"])
    if factor > 0:
        physical_raw_min = int(
            ((physical_minimum - offset) / factor).to_integral_value(
                rounding=ROUND_CEILING
            )
        )
        physical_raw_max = int(
            ((physical_maximum - offset) / factor).to_integral_value(
                rounding=ROUND_FLOOR
            )
        )
    else:
        physical_raw_min = int(
            ((physical_maximum - offset) / factor).to_integral_value(
                rounding=ROUND_CEILING
            )
        )
        physical_raw_max = int(
            ((physical_minimum - offset) / factor).to_integral_value(
                rounding=ROUND_FLOOR
            )
        )
    raw_min = max(raw_min, physical_raw_min)
    raw_max = min(raw_max, physical_raw_max)
    return raw_min, raw_max


def signedness_is_inherently_ambiguous(truth: dict[str, Any]) -> bool:
    """Return whether signed and unsigned decode identically on the valid range."""

    bit_length = int(truth["bit_length"])
    raw_min, raw_max = _effective_raw_bounds(truth)
    sign_boundary = 1 << (bit_length - 1)
    return raw_min >= 0 and raw_max < sign_boundary


def width_is_inherently_ambiguous(truth: dict[str, Any]) -> bool:
    """Return whether the valid range leaves declared high bits permanently zero."""

    raw_min, raw_max = _effective_raw_bounds(truth)
    if raw_min < 0:
        return False
    minimum_unsigned_width = max(1, raw_max.bit_length())
    return minimum_unsigned_width < int(truth["bit_length"])


def _full(candidate: dict[str, Any], truth: dict[str, Any]) -> bool:
    return (
        _location(candidate, truth)
        and str(candidate["byte_order"]) == str(truth["byte_order"])
        and bool(candidate["signed"]) == bool(truth["signed"])
        and _close_scale(candidate, truth)
        and _close_offset(candidate, truth)
    )


def _observationally_equivalent_encoding(
    candidate: dict[str, Any], truth: dict[str, Any]
) -> bool:
    if (
        int(candidate["pgn"]) != int(truth["pgn"])
        or int(candidate["start_bit"]) != int(truth["start_bit"])
        or str(candidate["byte_order"]) != str(truth["byte_order"])
        or not _close_scale(candidate, truth)
        or not _close_offset(candidate, truth)
    ):
        return False
    candidate_width = int(candidate["bit_length"])
    true_width = int(truth["bit_length"])
    if candidate_width > true_width:
        return False
    raw_min, raw_max = _effective_raw_bounds(truth)
    if raw_min < 0:
        return False
    usable_bits = candidate_width - int(bool(candidate["signed"]))
    return usable_bits > 0 and raw_max < (1 << usable_bits)


def _accuracy(rows: list[dict[str, Any]], key: str) -> float:
    return (
        sum(bool(row[key]) for row in rows) / len(rows)
        if rows
        else 0.0
    )


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float | None:
        values = [float(row[key]) for row in rows if row[key] is not None]
        return sum(values) / len(values) if values else None

    return {
        "signals": len(rows),
        "top1_location_accuracy": _accuracy(rows, "top1_location_correct"),
        "top3_location_accuracy": _accuracy(rows, "top3_location_correct"),
        "top5_location_accuracy": _accuracy(rows, "top5_location_correct"),
        "top10_location_accuracy": _accuracy(rows, "top10_location_correct"),
        "raw_full_encoding_accuracy": _accuracy(rows, "full_encoding_correct"),
        "identifiable_full_encoding_accuracy": _accuracy(
            rows, "identifiable_full_encoding_correct"
        ),
        "correct_pgn_accuracy": _accuracy(rows, "correct_pgn"),
        "correct_start_bit_accuracy": _accuracy(rows, "correct_start_bit"),
        "correct_width_accuracy": _accuracy(rows, "correct_width"),
        "correct_byte_order_accuracy": _accuracy(rows, "correct_byte_order"),
        "correct_signedness_accuracy": _accuracy(rows, "correct_signedness"),
        "mean_scale_absolute_error": mean("scale_error"),
        "mean_offset_absolute_error": mean("offset_error"),
        "ambiguous_signals": sum(
            row["identifiability_status"] == "AMBIGUOUS" for row in rows
        ),
        "identifiable_signals": sum(
            row["identifiability_status"] == "IDENTIFIABLE" for row in rows
        ),
        "incorrect_signals": sum(
            row["identifiability_status"] == "INCORRECT" for row in rows
        ),
        "candidate_regressions": sum(int(row["candidate_count"]) for row in rows),
    }


def evaluate_full_library(
    inferred_path: str | Path,
    ground_truth_path: str | Path,
    result_csv_path: str | Path,
    *,
    allowed_signal_ids: set[str] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    inferred = json.loads(Path(inferred_path).read_text(encoding="utf-8"))
    truths = json.loads(Path(ground_truth_path).read_text(encoding="utf-8"))[
        "signals"
    ]
    rows: list[dict[str, Any]] = []
    for signal_name, signal in inferred["signals"].items():
        signal_id = signal["signal_id"]
        if allowed_signal_ids is not None and signal_id not in allowed_signal_ids:
            continue
        truth = truths.get(signal_id)
        if truth is None:
            continue
        candidates = signal.get("candidates", [])
        location_ranks = [
            rank
            for rank, candidate in enumerate(candidates, start=1)
            if _location(candidate, truth)
        ]
        location_rank = min(location_ranks, default=None)
        best = candidates[0] if candidates else None
        raw_full = bool(best and _full(best, truth))
        ambiguity_equivalent = bool(
            best and _observationally_equivalent_encoding(best, truth)
        )
        adjusted_full = raw_full or ambiguity_equivalent
        inherently_ambiguous = signedness_is_inherently_ambiguous(
            truth
        ) or width_is_inherently_ambiguous(truth)
        if adjusted_full and inherently_ambiguous:
            status = "AMBIGUOUS"
        elif raw_full:
            status = "IDENTIFIABLE"
        else:
            status = "INCORRECT"
        aligned = (
            int(truth["start_bit"]) % 8 == 0
            and int(truth["bit_length"]) in {8, 16, 32}
        )
        row: dict[str, Any] = {
            "signal_id": signal_id,
            "signal_name": signal_name,
            "result_status": status,
            "true_bit_length": int(truth["bit_length"]),
            "true_alignment_category": (
                "aligned_8_16_32" if aligned else "non_aligned_or_other_width"
            ),
            "transport_category": truth["transport_category"],
            "top1_location_correct": location_rank == 1,
            "top3_location_correct": location_rank is not None and location_rank <= 3,
            "top5_location_correct": location_rank is not None and location_rank <= 5,
            "top10_location_correct": location_rank is not None and location_rank <= 10,
            "full_encoding_correct": raw_full,
            "identifiable_full_encoding_correct": adjusted_full,
            "identifiability_status": status,
            "correct_pgn": bool(best and int(best["pgn"]) == int(truth["pgn"])),
            "correct_start_bit": bool(
                best and int(best["start_bit"]) == int(truth["start_bit"])
            ),
            "correct_width": bool(
                best and int(best["bit_length"]) == int(truth["bit_length"])
            ),
            "correct_byte_order": bool(
                best and str(best["byte_order"]) == str(truth["byte_order"])
            ),
            "correct_signedness": bool(
                best and bool(best["signed"]) == bool(truth["signed"])
            ),
            "scale_error": (
                abs(float(best["scale"]) - float(truth["scale"]))
                if best
                else None
            ),
            "offset_error": (
                abs(float(best["offset"]) - float(truth["offset"]))
                if best
                else None
            ),
            "candidate_count": int(signal.get("candidate_count", 0)),
        }
        rows.append(row)
    rows.sort(key=lambda item: item["signal_id"])
    destination = Path(result_csv_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        public_rows = public_signal_results(rows)
        columns = [key for key in RESULT_COLUMNS if key in public_rows[0]] if public_rows else ["signal_id"]
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(public_rows)

    by_width: dict[str, dict[str, Any]] = {}
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(row["true_bit_length"])].append(row)
    for width in sorted(groups):
        by_width[str(width)] = summarize_rows(groups[width])
    by_alignment = {
        category: summarize_rows(
            [row for row in rows if row["true_alignment_category"] == category]
        )
        for category in ("aligned_8_16_32", "non_aligned_or_other_width")
    }
    by_transport = {
        category: summarize_rows(
            [row for row in rows if row["transport_category"] == category]
        )
        for category in ("single_frame", "BAM", "RTS_CTS")
    }
    return {
        "summary": summarize_rows(rows),
        "by_bit_length": by_width,
        "by_alignment": by_alignment,
        "by_transport": by_transport,
        "ambiguity_rule": (
            "The effective raw interval is derived from the intersection of the "
            "declared raw and physical ranges. Signedness is ambiguous when that "
            "entire interval lies below the sign bit. Declared width is ambiguous "
            "when its upper bits are zero throughout the interval and a shorter "
            "candidate at the same PGN/start produces the same raw values. A "
            "result is AMBIGUOUS only when PGN, start, byte order, scale, and "
            "offset are otherwise correct over the complete valid interval."
        ),
        "evaluation_runtime_seconds": time.perf_counter() - started,
        "rows": rows,
    }


def write_evaluation_json(path: str | Path, evaluation: dict[str, Any]) -> None:
    document = {key: evaluation[key] for key in (
        "summary", "by_bit_length", "by_alignment", "by_transport",
        "ambiguity_rule", "evaluation_runtime_seconds",
    ) if key in evaluation}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _plot_grouped(
    labels: list[str],
    values: list[float],
    ylabel: str,
    destination: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    axis.bar(labels, values, color="#2474A6")
    axis.set_ylim(0, 1.02)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def write_full_library_plots(
    evaluation: dict[str, Any],
    output_directory: str | Path,
    search_modes: list[dict[str, Any]],
) -> None:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    widths = sorted(evaluation["by_bit_length"], key=int)
    _plot_grouped(
        widths,
        [evaluation["by_bit_length"][width]["top1_location_accuracy"] for width in widths],
        "Top-1 location accuracy",
        output / "top1_accuracy_by_bit_length.png",
    )
    _plot_grouped(
        widths,
        [evaluation["by_bit_length"][width]["raw_full_encoding_accuracy"] for width in widths],
        "Raw full-encoding accuracy",
        output / "full_encoding_accuracy_by_bit_length.png",
    )
    alignment = list(evaluation["by_alignment"])
    _plot_grouped(
        alignment,
        [evaluation["by_alignment"][key]["top1_location_accuracy"] for key in alignment],
        "Top-1 location accuracy",
        output / "accuracy_by_alignment.png",
    )
    transport = list(evaluation["by_transport"])
    _plot_grouped(
        transport,
        [evaluation["by_transport"][key]["top1_location_accuracy"] for key in transport],
        "Top-1 location accuracy",
        output / "accuracy_by_transport.png",
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for row in search_modes:
        axis.scatter(
            float(row["candidate_regressions"]),
            float(row["runtime_seconds"]),
            s=80,
            label=str(row["mode"]),
        )
    axis.set_xlabel("Candidate regressions")
    axis.set_ylabel("Runtime (seconds)")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "candidate_search_size_vs_runtime.png", dpi=160)
    plt.close(figure)
