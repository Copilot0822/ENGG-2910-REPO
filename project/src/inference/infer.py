"""Signal-discovery pipeline using only raw CAN and known-value inputs."""

from __future__ import annotations

import heapq
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from inference.candidate import (
    ByteOrder,
    CandidateResult,
    InferenceMode,
    extract_raw,
    generate_specs,
)
from inference.loader import KnownValue, LogicalMessage, load_known_values, load_logical_messages
from inference.regression import fit_linear
from inference.scoring import score_candidate


@dataclass(frozen=True, slots=True)
class StreamKey:
    pgn: int
    source: int
    payload_length: int


@dataclass(slots=True)
class MessageStream:
    key: StreamKey
    by_sample: dict[int, LogicalMessage]


@dataclass(slots=True)
class SignalInference:
    signal_name: str
    signal_id: str | None
    unit: str
    target_observation_count: int
    candidates: list[CandidateResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        ranked = [candidate.to_dict() for candidate in self.candidates]
        return {
            "signal_name": self.signal_name,
            "signal_id": self.signal_id,
            "unit": self.unit,
            "target_observation_count": self.target_observation_count,
            "best": ranked[0] if ranked else None,
            "candidates": ranked,
        }


@dataclass(slots=True)
class InferenceRun:
    signals: list[SignalInference]
    metadata: dict[str, object]


def _build_streams(messages: list[LogicalMessage]) -> list[MessageStream]:
    grouped: dict[StreamKey, dict[int, list[LogicalMessage]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for message in messages:
        key = StreamKey(message.pgn, message.source, len(message.payload))
        grouped[key][message.sample_id].append(message)
    streams: list[MessageStream] = []
    for key, sample_groups in grouped.items():
        # If indistinguishable variants collide in one stream at one sample, no
        # answer-key-free rule can choose the correct payload. Exclude that sample
        # instead of silently creating a false pair.
        unique = {
            sample: rows[0] for sample, rows in sample_groups.items() if len(rows) == 1
        }
        if unique:
            streams.append(MessageStream(key, unique))
    return sorted(
        streams,
        key=lambda item: (item.key.pgn, item.key.source, item.key.payload_length),
    )


def _group_known(values: list[KnownValue]) -> dict[str, list[KnownValue]]:
    grouped: dict[str, list[KnownValue]] = defaultdict(list)
    for value in values:
        grouped[value.signal_name].append(value)
    return {
        name: sorted(rows, key=lambda row: (row.sample_id, row.observation_id))
        for name, rows in grouped.items()
    }


def infer_signals(
    can_path: str | Path,
    known_path: str | Path,
    *,
    mode: InferenceMode = "aligned",
    bit_lengths: tuple[int, ...] = (8, 16, 32),
    byte_orders: tuple[ByteOrder, ...] = ("little_endian", "big_endian"),
    min_pairs: int = 10,
    min_pairing_fraction: float = 0.5,
    top_k: int = 10,
    max_signals: int | None = None,
    selected_signals: set[str] | None = None,
) -> InferenceRun:
    """Infer signal locations without reading simulator or answer-key files."""

    started = time.perf_counter()
    if mode not in {"aligned", "arbitrary"}:
        raise ValueError("mode must be 'aligned' or 'arbitrary'")
    if min_pairs < 3 or top_k <= 0:
        raise ValueError("min_pairs must be at least 3 and top_k must be positive")
    if not 0 <= min_pairing_fraction <= 1:
        raise ValueError("min_pairing_fraction must be between 0 and 1")
    if not bit_lengths or any(length <= 0 for length in bit_lengths):
        raise ValueError("bit lengths must be positive")

    messages = load_logical_messages(can_path)
    known_values = load_known_values(known_path)
    streams = _build_streams(messages)
    grouped_known = _group_known(known_values)
    signal_names = sorted(grouped_known)
    if selected_signals is not None:
        signal_names = [name for name in signal_names if name in selected_signals]
    if max_signals is not None:
        signal_names = signal_names[:max_signals]

    spec_cache: dict[StreamKey, tuple[object, ...]] = {}
    candidates_tested = 0
    candidates_rejected = 0
    signal_results: list[SignalInference] = []
    heap_counter = 0

    for signal_name in signal_names:
        rows = grouped_known[signal_name]
        # There should be one observation per signal/sample. Keeping the last is
        # deterministic and avoids double-weighting malformed input.
        target_by_sample = {row.sample_id: row.value for row in rows}
        signal = SignalInference(
            signal_name=signal_name,
            signal_id=rows[0].signal_id,
            unit=rows[0].unit,
            target_observation_count=len(target_by_sample),
        )
        target_values = np.fromiter(target_by_sample.values(), dtype=np.float64)
        if len(target_by_sample) < min_pairs or float(np.var(target_values)) <= 1e-15:
            signal_results.append(signal)
            continue
        required_pairs = max(
            min_pairs, math.ceil(len(target_by_sample) * min_pairing_fraction)
        )
        best: list[tuple[tuple[float, float, int, float], int, CandidateResult]] = []

        for stream in streams:
            common_samples = sorted(target_by_sample.keys() & stream.by_sample.keys())
            if len(common_samples) < required_pairs:
                continue
            payloads = [stream.by_sample[sample].payload for sample in common_samples]
            y = np.asarray(
                [target_by_sample[sample] for sample in common_samples], dtype=np.float64
            )
            specs = spec_cache.get(stream.key)
            if specs is None:
                specs = generate_specs(
                    stream.key.pgn,
                    stream.key.source,
                    stream.key.payload_length,
                    mode,
                    bit_lengths,
                    byte_orders,
                )
                spec_cache[stream.key] = specs
            for spec in specs:
                candidates_tested += 1
                x = np.asarray(
                    [
                        extract_raw(
                            payload,
                            spec.start_bit,
                            spec.bit_length,
                            signed=spec.signed,
                            byte_order=spec.byte_order,
                        )
                        for payload in payloads
                    ],
                    dtype=np.float64,
                )
                metrics = fit_linear(x, y)
                if metrics is None:
                    candidates_rejected += 1
                    continue
                score, normalized_rmse, pairing_fraction = score_candidate(
                    metrics, len(common_samples), len(target_by_sample)
                )
                # Reserved/constant neighboring bytes can make a wider field an
                # exactly equivalent affine encoding. Prefer the smallest exact
                # representation without materially changing fit-based ranking.
                score -= spec.bit_length * 1e-8
                result = CandidateResult(
                    spec=spec,
                    scale=metrics.scale,
                    offset=metrics.offset,
                    r2=metrics.r2,
                    rmse=metrics.rmse,
                    mae=metrics.mae,
                    observation_count=len(common_samples),
                    raw_variance=metrics.raw_variance,
                    target_variance=metrics.target_variance,
                    normalized_rmse=normalized_rmse,
                    pairing_fraction=pairing_fraction,
                    ranking_score=score,
                )
                rank_key = (
                    result.ranking_score,
                    result.r2,
                    result.observation_count,
                    -result.normalized_rmse,
                )
                heap_counter += 1
                item = (rank_key, heap_counter, result)
                if len(best) < top_k:
                    heapq.heappush(best, item)
                elif rank_key > best[0][0]:
                    heapq.heapreplace(best, item)
        signal.candidates = [
            item[2] for item in sorted(best, key=lambda item: item[0], reverse=True)
        ]
        signal_results.append(signal)

    runtime = time.perf_counter() - started
    metadata: dict[str, object] = {
        "can_input": str(can_path),
        "known_values_input": str(known_path),
        "inference_mode": mode,
        "supported_bit_lengths": list(bit_lengths),
        "byte_orders": list(byte_orders),
        "minimum_pairs": min_pairs,
        "minimum_pairing_fraction": min_pairing_fraction,
        "top_k": top_k,
        "number_of_target_signals": len(signal_results),
        "number_of_logical_messages": len(messages),
        "number_of_message_streams": len(streams),
        "number_of_candidates_tested": candidates_tested,
        "number_of_candidates_rejected": candidates_rejected,
        "runtime_seconds": runtime,
        "random_seed": None,
    }
    return InferenceRun(signal_results, metadata)
