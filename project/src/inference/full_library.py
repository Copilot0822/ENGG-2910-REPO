"""Batched, cached full-library signal discovery using only clean CSV inputs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from numpy.typing import NDArray

from inference.loader import LogicalMessage, load_logical_messages
from inference.regression import fit_linear
from inference.scoring import score_candidate


ByteOrder = Literal["little_endian", "big_endian"]
SearchMode = Literal["aligned", "extended"]
_SIGNAL_ID = re.compile(r"\[(j1939_signal_\d+)\]\s*$")


@dataclass(frozen=True, slots=True)
class FullCandidateSpec:
    start_bit: int
    bit_length: int
    signed: bool
    byte_order: ByteOrder
    ordinal: int


@dataclass(slots=True)
class FullTarget:
    signal_name: str
    signal_id: str
    unit: str
    values: NDArray[np.float64]
    candidate_count: int = 0
    candidates: list[dict[str, object]] = field(default_factory=list)


@dataclass(slots=True)
class FullStream:
    pgn: int
    source: int
    payload_length: int
    samples: NDArray[np.int64]
    payloads: NDArray[np.uint8]
    targets: list[FullTarget] = field(default_factory=list)


@dataclass(slots=True)
class _TargetBuilder:
    signal_name: str
    signal_id: str
    unit: str
    values: NDArray[np.float64]


class CandidateVectorCache:
    """Reusable vectorized candidate extractor for one logical message stream.

    The 64-bit rolling window matrix is computed once. Candidate vectors are
    keyed by location, width, signedness, and byte order. The full-library path
    extracts a block once and regresses it against every target in the stream.
    """

    def __init__(
        self,
        payloads: NDArray[np.uint8],
        *,
        pgn: int = 0,
        max_vector_entries: int = 256,
    ) -> None:
        if payloads.ndim != 2 or payloads.shape[1] == 0:
            raise ValueError("payloads must be a nonempty two-dimensional array")
        self.payloads = np.ascontiguousarray(payloads, dtype=np.uint8)
        self.pgn = pgn
        self.max_vector_entries = max_vector_entries
        self._windows: NDArray[np.uint64] | None = None
        self._vectors: OrderedDict[
            tuple[int, int, int, bool, ByteOrder], NDArray[np.float64]
        ] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        self.peak_matrix_bytes = 0

    @property
    def payload_bits(self) -> int:
        return self.payloads.shape[1] * 8

    def _rolling_windows(self) -> NDArray[np.uint64]:
        if self._windows is not None:
            return self._windows
        rows, byte_count = self.payloads.shape
        words = np.zeros((rows, byte_count), dtype=np.uint64)
        for byte_offset in range(8):
            usable = byte_count - byte_offset
            if usable <= 0:
                break
            words[:, :usable] |= (
                self.payloads[:, byte_offset:].astype(np.uint64)
                << np.uint64(8 * byte_offset)
            )
        windows = np.empty((rows, byte_count * 8), dtype=np.uint64)
        for bit_offset in range(8):
            shifted = words >> np.uint64(bit_offset)
            if bit_offset:
                for byte_index in range(byte_count - 8):
                    shifted[:, byte_index] |= (
                        self.payloads[:, byte_index + 8].astype(np.uint64)
                        << np.uint64(64 - bit_offset)
                    )
            windows[:, bit_offset::8] = shifted
        self._windows = windows
        self.peak_matrix_bytes = max(self.peak_matrix_bytes, windows.nbytes)
        return windows

    @staticmethod
    def _reverse_bytes(
        raw: NDArray[np.uint64], bit_length: int
    ) -> NDArray[np.uint64]:
        result = np.zeros_like(raw)
        for byte_index in range(bit_length // 8):
            result = (result << np.uint64(8)) | (
                (raw >> np.uint64(byte_index * 8)) & np.uint64(0xFF)
            )
        return result

    def matrix(
        self,
        starts: NDArray[np.int64],
        bit_length: int,
        *,
        signed: bool,
        byte_order: ByteOrder,
    ) -> NDArray[np.float64]:
        if starts.size == 0:
            return np.empty((self.payloads.shape[0], 0), dtype=np.float64)
        if bit_length <= 0 or int(starts[-1]) + bit_length > self.payload_bits:
            raise ValueError("candidate block extends beyond the payload")
        if byte_order == "big_endian" and bit_length % 8:
            raise ValueError("big-endian candidates require whole bytes")
        windows = self._rolling_windows()[:, starts]
        mask = (
            np.uint64(0xFFFFFFFFFFFFFFFF)
            if bit_length == 64
            else np.uint64((1 << bit_length) - 1)
        )
        raw = windows & mask
        if byte_order == "big_endian":
            raw = self._reverse_bytes(raw, bit_length)
        values = raw.astype(np.float64)
        if signed:
            negative = (raw & np.uint64(1 << (bit_length - 1))) != 0
            values[negative] -= float(1 << bit_length)
        self.peak_matrix_bytes = max(self.peak_matrix_bytes, values.nbytes)
        return values

    def vector(self, spec: FullCandidateSpec) -> NDArray[np.float64]:
        key = (
            self.pgn,
            spec.start_bit,
            spec.bit_length,
            spec.signed,
            spec.byte_order,
        )
        cached = self._vectors.get(key)
        if cached is not None:
            self.cache_hits += 1
            self._vectors.move_to_end(key)
            return cached
        self.cache_misses += 1
        vector = self.matrix(
            np.asarray([spec.start_bit], dtype=np.int64),
            spec.bit_length,
            signed=spec.signed,
            byte_order=spec.byte_order,
        )[:, 0]
        self._vectors[key] = vector
        if len(self._vectors) > self.max_vector_entries:
            self._vectors.popitem(last=False)
        return vector


def candidate_count(
    payload_length: int, mode: SearchMode, bit_lengths: tuple[int, ...]
) -> int:
    payload_bits = payload_length * 8
    count = 0
    step = 8 if mode == "aligned" else 1
    for bit_length in bit_lengths:
        starts = max(0, (payload_bits - bit_length) // step + 1)
        orders = 1 if bit_length == 8 or bit_length % 8 else 2
        count += starts * orders * 2
    return count


def _candidate_blocks(
    cache: CandidateVectorCache,
    mode: SearchMode,
    bit_lengths: tuple[int, ...],
    maximum_columns: int,
) -> Iterable[tuple[NDArray[np.float64], list[FullCandidateSpec], float]]:
    payload_bits = cache.payload_bits
    step = 8 if mode == "aligned" else 1
    ordinal = 0
    for bit_length in bit_lengths:
        all_starts = np.arange(
            0, payload_bits - bit_length + 1, step, dtype=np.int64
        )
        if all_starts.size == 0:
            continue
        orders: tuple[ByteOrder, ...] = (
            ("little_endian",)
            if bit_length == 8 or bit_length % 8
            else ("little_endian", "big_endian")
        )
        variants = len(orders) * 2
        starts_per_block = max(1, maximum_columns // variants)
        for offset in range(0, all_starts.size, starts_per_block):
            extraction_started = time.perf_counter()
            starts = all_starts[offset : offset + starts_per_block]
            variant_matrices: list[NDArray[np.float64]] = []
            specs: list[FullCandidateSpec] = []
            for byte_order in orders:
                variant_matrices.extend(
                    (
                        cache.matrix(
                            starts,
                            bit_length,
                            signed=False,
                            byte_order=byte_order,
                        ),
                        cache.matrix(
                            starts,
                            bit_length,
                            signed=True,
                            byte_order=byte_order,
                        ),
                    )
                )
            for start in starts:
                for byte_order in orders:
                    specs.extend(
                        (
                            FullCandidateSpec(
                                int(start),
                                bit_length,
                                False,
                                byte_order,
                                ordinal,
                            ),
                            FullCandidateSpec(
                                int(start),
                                bit_length,
                                True,
                                byte_order,
                                ordinal + 1,
                            ),
                        )
                    )
                    ordinal += 2
            matrix = np.stack(variant_matrices, axis=2).reshape(
                cache.payloads.shape[0], -1
            )
            cache.peak_matrix_bytes = max(cache.peak_matrix_bytes, matrix.nbytes)
            yield matrix, specs, time.perf_counter() - extraction_started


def _stream_key(message: LogicalMessage) -> tuple[int, int, int]:
    return message.pgn, message.source, len(message.payload)


def load_full_library_inputs(
    can_path: str | Path,
    known_path: str | Path,
    *,
    selected_signals: set[str] | None = None,
) -> tuple[list[FullStream], dict[str, object]]:
    """Load and associate targets with streams without answer-key metadata."""

    started = time.perf_counter()
    messages = load_logical_messages(can_path, "j1939")
    grouped: dict[tuple[int, int, int], list[LogicalMessage]] = defaultdict(list)
    max_sample = max((message.sample_id for message in messages), default=-1)
    for message in messages:
        grouped[_stream_key(message)].append(message)
    keys = sorted(grouped)
    streams: list[FullStream] = []
    sample_stream = np.full(max_sample + 1, -1, dtype=np.int32)
    sample_position = np.full(max_sample + 1, -1, dtype=np.int32)
    for stream_index, key in enumerate(keys):
        rows = sorted(grouped[key], key=lambda item: item.sample_id)
        samples = np.asarray([item.sample_id for item in rows], dtype=np.int64)
        payloads = np.asarray([list(item.payload) for item in rows], dtype=np.uint8)
        if np.any(sample_stream[samples] != -1):
            raise ValueError(
                "full-library optimized input requires one logical message per sample"
            )
        sample_stream[samples] = stream_index
        sample_position[samples] = np.arange(len(rows), dtype=np.int32)
        streams.append(FullStream(*key, samples, payloads))
    del messages, grouped

    builders: list[dict[str, _TargetBuilder]] = [dict() for _ in streams]
    known_rows = 0
    with Path(known_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "signal_name", "value", "unit"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("known-values input is missing required columns")
        for row in reader:
            name = row["signal_name"]
            if selected_signals is not None and name not in selected_signals:
                continue
            sample = int(row["sample_id"])
            if not 0 <= sample < sample_stream.size:
                raise ValueError(f"known value refers to unknown sample {sample}")
            stream_index = int(sample_stream[sample])
            if stream_index < 0:
                raise ValueError(f"known value refers to missing CAN sample {sample}")
            match = _SIGNAL_ID.search(name)
            signal_id = match.group(1) if match else name
            target = builders[stream_index].get(name)
            if target is None:
                target = _TargetBuilder(
                    name,
                    signal_id,
                    row["unit"],
                    np.full(len(streams[stream_index].samples), np.nan),
                )
                builders[stream_index][name] = target
            position = int(sample_position[sample])
            if math.isfinite(float(target.values[position])):
                raise ValueError(f"duplicate known value for {name}, sample {sample}")
            target.values[position] = float(row["value"])
            known_rows += 1
    for stream, stream_builders in zip(streams, builders, strict=True):
        for target in sorted(stream_builders.values(), key=lambda item: item.signal_name):
            if not np.all(np.isfinite(target.values)):
                raise ValueError(
                    f"target {target.signal_name} is missing synchronized observations"
                )
            stream.targets.append(
                FullTarget(
                    target.signal_name,
                    target.signal_id,
                    target.unit,
                    target.values,
                )
            )
    streams = [stream for stream in streams if stream.targets]
    metadata = {
        "input_loading_seconds": time.perf_counter() - started,
        "logical_messages_loaded": int(sum(len(stream.samples) for stream in streams)),
        "message_streams": len(streams),
        "known_value_rows_loaded": known_rows,
        "target_signals_loaded": sum(len(stream.targets) for stream in streams),
    }
    return streams, metadata


def make_stream_batches(
    streams: list[FullStream], batch_size: int
) -> list[list[FullStream]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batches: list[list[FullStream]] = []
    current: list[FullStream] = []
    count = 0
    for stream in streams:
        stream_count = len(stream.targets)
        if current and count + stream_count > batch_size:
            batches.append(current)
            current = []
            count = 0
        current.append(stream)
        count += stream_count
    if current:
        batches.append(current)
    return batches


def _rank_key(candidate: dict[str, object]) -> tuple[float, float, float, int]:
    return (
        float(candidate["ranking_score"]),
        float(candidate["r2"]),
        -float(candidate["normalized_rmse"]),
        -int(candidate["ordinal"]),
    )


def _process_stream(
    stream: FullStream,
    *,
    mode: SearchMode,
    bit_lengths: tuple[int, ...],
    top_k: int,
    candidate_batch_size: int,
) -> tuple[dict[str, dict[str, object]], dict[str, float | int]]:
    cache = CandidateVectorCache(stream.payloads, pgn=stream.pgn)
    y = np.column_stack([target.values for target in stream.targets])
    y_mean = np.mean(y, axis=0)
    y_centered = y - y_mean
    y_sum_squares = np.sum(y_centered * y_centered, axis=0)
    target_variance = y_sum_squares / y.shape[0]
    target_best: list[list[dict[str, object]]] = [
        [] for _ in stream.targets
    ]
    total_specs = candidate_count(stream.payload_length, mode, bit_lengths)
    extraction_seconds = 0.0
    regression_seconds = 0.0

    for matrix, specs, block_extraction_seconds in _candidate_blocks(
        cache, mode, bit_lengths, candidate_batch_size
    ):
        extraction_seconds += block_extraction_seconds
        extraction_finished = time.perf_counter()
        x_mean = np.mean(matrix, axis=0)
        x_centered = matrix - x_mean
        denominator = np.sum(x_centered * x_centered, axis=0)
        valid = denominator > 1e-15
        covariance = x_centered.T @ y_centered
        scale = np.divide(
            covariance,
            denominator[:, None],
            out=np.zeros_like(covariance),
            where=valid[:, None],
        )
        offset = y_mean[None, :] - scale * x_mean[:, None]
        squared_error = y_sum_squares[None, :] - covariance * scale
        squared_error = np.maximum(squared_error, 0.0)
        error_ratio = np.divide(
            squared_error,
            y_sum_squares[None, :],
            out=np.full_like(squared_error, np.inf),
            where=y_sum_squares[None, :] > 1e-15,
        )
        r2 = 1.0 - error_ratio
        normalized_rmse = np.sqrt(error_ratio)
        score = r2 - 0.10 * normalized_rmse + 0.05
        score[~valid, :] = -np.inf
        ordinals = np.asarray([spec.ordinal for spec in specs], dtype=np.int64)
        for target_index, target in enumerate(stream.targets):
            finite = np.flatnonzero(np.isfinite(score[:, target_index]))
            if finite.size == 0:
                continue
            take = min(top_k, finite.size)
            ordering = np.lexsort(
                (
                    ordinals[finite],
                    normalized_rmse[finite, target_index],
                    -r2[finite, target_index],
                    -score[finite, target_index],
                )
            )
            chosen = finite[ordering[:take]]
            for candidate_index in chosen:
                spec = specs[int(candidate_index)]
                target_best[target_index].append(
                    {
                        "pgn": stream.pgn,
                        "source": stream.source,
                        "payload_length": stream.payload_length,
                        "start_bit": spec.start_bit,
                        "bit_length": spec.bit_length,
                        "signed": spec.signed,
                        "byte_order": spec.byte_order,
                        "scale": float(scale[candidate_index, target_index]),
                        "offset": float(offset[candidate_index, target_index]),
                        "r2": float(r2[candidate_index, target_index]),
                        "rmse": float(
                            math.sqrt(
                                squared_error[candidate_index, target_index]
                                / y.shape[0]
                            )
                        ),
                        "mae": 0.0,
                        "observation_count": y.shape[0],
                        "raw_variance": float(
                            denominator[candidate_index] / y.shape[0]
                        ),
                        "target_variance": float(target_variance[target_index]),
                        "normalized_rmse": float(
                            normalized_rmse[candidate_index, target_index]
                        ),
                        "pairing_fraction": 1.0,
                        "ranking_score": float(score[candidate_index, target_index]),
                        "ordinal": spec.ordinal,
                    }
                )
            target_best[target_index] = sorted(
                target_best[target_index], key=_rank_key, reverse=True
            )[:top_k]
        regression_seconds += time.perf_counter() - extraction_finished
    results: dict[str, dict[str, object]] = {}
    for target, candidates in zip(stream.targets, target_best, strict=True):
        for candidate in candidates:
            spec = FullCandidateSpec(
                int(candidate["start_bit"]),
                int(candidate["bit_length"]),
                bool(candidate["signed"]),
                str(candidate["byte_order"]),  # type: ignore[arg-type]
                int(candidate["ordinal"]),
            )
            raw = cache.vector(spec)
            metrics = fit_linear(raw, target.values)
            if metrics is None:
                continue
            score, normalized_rmse, pairing_fraction = score_candidate(
                metrics, len(raw), len(target.values)
            )
            candidate.update(
                {
                    "scale": metrics.scale,
                    "offset": metrics.offset,
                    "r2": metrics.r2,
                    "rmse": metrics.rmse,
                    "mae": metrics.mae,
                    "raw_variance": metrics.raw_variance,
                    "target_variance": metrics.target_variance,
                    "normalized_rmse": normalized_rmse,
                    "pairing_fraction": pairing_fraction,
                    "ranking_score": score,
                }
            )
        candidates.sort(key=_rank_key, reverse=True)
        for candidate in candidates:
            candidate.pop("ordinal", None)
        target.candidate_count = total_specs
        target.candidates = candidates
        results[target.signal_name] = {
            "signal_name": target.signal_name,
            "signal_id": target.signal_id,
            "unit": target.unit,
            "target_observation_count": len(target.values),
            "candidate_count": total_specs,
            "best": candidates[0] if candidates else None,
            "candidates": candidates,
        }
    return results, {
        "candidate_definitions_extracted": total_specs,
        "candidate_regressions": total_specs * len(stream.targets),
        "candidate_extraction_seconds": extraction_seconds,
        "regression_scoring_seconds": regression_seconds,
        "cache_hits": cache.cache_hits,
        "cache_misses": cache.cache_misses,
        "peak_candidate_matrix_bytes": cache.peak_matrix_bytes,
    }


def _config_digest(config: dict[str, object]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_batch_checkpoint(
    path: Path, batch_index: int, digest: str, payload: dict[str, object]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"batch_index": batch_index, "config_digest": digest, **payload}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_batch_checkpoint(path: Path, expected_digest: str) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("config_digest") != expected_digest:
        raise ValueError(f"checkpoint configuration mismatch: {path}")
    return document


def _merge_outputs(
    output: Path,
    checkpoint_paths: list[Path],
    metadata: dict[str, object],
    digest: str,
) -> None:
    signals: dict[str, object] = {}
    for path in checkpoint_paths:
        checkpoint = load_batch_checkpoint(path, digest)
        signals.update(checkpoint["signals"])
    document = {"metadata": metadata, "signals": signals}
    (output / "inferred_signals.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "inference_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    columns = [
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
    with (output / "candidate_rankings.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for name in sorted(signals):
            signal = signals[name]
            for rank, candidate in enumerate(signal["candidates"], start=1):
                writer.writerow(
                    {
                        "signal_name": name,
                        "signal_id": signal["signal_id"],
                        "rank": rank,
                        **candidate,
                    }
                )


def infer_full_library(
    can_path: str | Path,
    known_path: str | Path,
    output_directory: str | Path,
    *,
    mode: SearchMode,
    bit_lengths: tuple[int, ...],
    selected_signals: set[str] | None = None,
    batch_size: int = 250,
    candidate_batch_size: int = 512,
    top_k: int = 10,
    resume: bool = False,
) -> dict[str, object]:
    """Run resumable full-library discovery without reading any answer key."""

    started = time.perf_counter()
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    streams, load_metadata = load_full_library_inputs(
        can_path, known_path, selected_signals=selected_signals
    )
    batches = make_stream_batches(streams, batch_size)
    selection_digest = hashlib.sha256(
        "\n".join(sorted(selected_signals or ())).encode()
    ).hexdigest()
    config: dict[str, object] = {
        "algorithm_version": 2,
        "protocol": "j1939",
        "mode": mode,
        "bit_lengths": list(bit_lengths),
        "batch_size": batch_size,
        "candidate_batch_size": candidate_batch_size,
        "top_k": top_k,
        "selection_sha256": selection_digest,
        "can_input": str(can_path),
        "known_values_input": str(known_path),
        "target_count": sum(len(stream.targets) for stream in streams),
        "stream_count": len(streams),
    }
    digest = _config_digest(config)
    config_path = output / "run_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("config_digest") != digest:
            if resume:
                raise ValueError(
                    "existing full-library run has a different configuration: "
                    f"{output}"
                )
            config_path.write_text(
                json.dumps(
                    {**config, "config_digest": digest}, indent=2, sort_keys=True
                )
                + "\n",
                encoding="utf-8",
            )
    else:
        config_path.write_text(
            json.dumps({**config, "config_digest": digest}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

    totals: dict[str, float | int] = {
        "candidate_definitions_extracted": 0,
        "candidate_regressions": 0,
        "candidate_extraction_seconds": 0.0,
        "regression_scoring_seconds": 0.0,
        "cache_hits": 0,
        "cache_misses": 0,
        "peak_candidate_matrix_bytes": 0,
    }
    completed_targets = 0
    checkpoint_paths: list[Path] = []
    for batch_index, batch in enumerate(batches):
        checkpoint_path = checkpoints / f"batch_{batch_index:05d}.json"
        checkpoint_paths.append(checkpoint_path)
        batch_targets = sum(len(stream.targets) for stream in batch)
        if resume and checkpoint_path.exists():
            checkpoint = load_batch_checkpoint(checkpoint_path, digest)
            metrics = checkpoint["metrics"]
            for key in totals:
                if key == "peak_candidate_matrix_bytes":
                    totals[key] = max(int(totals[key]), int(metrics[key]))
                else:
                    totals[key] += metrics[key]
            completed_targets += batch_targets
            print(
                f"Targets complete: {completed_targets} / {config['target_count']} | "
                f"batch {batch_index + 1}/{len(batches)} resumed | "
                f"candidates evaluated: {int(totals['candidate_regressions'])} | "
                f"elapsed: {time.perf_counter() - started:.1f}s",
                flush=True,
            )
            continue
        batch_started = time.perf_counter()
        signals: dict[str, dict[str, object]] = {}
        batch_metrics = {key: 0 for key in totals}
        for stream in batch:
            stream_results, metrics = _process_stream(
                stream,
                mode=mode,
                bit_lengths=bit_lengths,
                top_k=top_k,
                candidate_batch_size=candidate_batch_size,
            )
            signals.update(stream_results)
            for key in batch_metrics:
                if key == "peak_candidate_matrix_bytes":
                    batch_metrics[key] = max(
                        int(batch_metrics[key]), int(metrics[key])
                    )
                else:
                    batch_metrics[key] += metrics[key]
        batch_metrics["batch_runtime_seconds"] = time.perf_counter() - batch_started
        write_batch_checkpoint(
            checkpoint_path,
            batch_index,
            digest,
            {"signals": signals, "metrics": batch_metrics},
        )
        for key in totals:
            if key == "peak_candidate_matrix_bytes":
                totals[key] = max(int(totals[key]), int(batch_metrics[key]))
            else:
                totals[key] += batch_metrics[key]
        completed_targets += batch_targets
        print(
            f"Targets complete: {completed_targets} / {config['target_count']} | "
            f"batch {batch_index + 1}/{len(batches)} | "
            f"candidates evaluated: {int(totals['candidate_regressions'])} | "
            f"elapsed: {time.perf_counter() - started:.1f}s",
            flush=True,
        )

    metadata: dict[str, object] = {
        **config,
        **load_metadata,
        **totals,
        "number_of_target_signals": config["target_count"],
        "number_of_candidates_tested": totals["candidate_regressions"],
        "candidate_definitions_per_target_average": (
            float(totals["candidate_regressions"]) / int(config["target_count"])
        ),
        "peak_candidate_matrix_megabytes": float(
            totals["peak_candidate_matrix_bytes"]
        )
        / (1024 * 1024),
        "runtime_seconds": time.perf_counter() - started,
        "completed_batches": len(batches),
        "resumed": resume,
    }
    _merge_outputs(output, checkpoint_paths, metadata, digest)
    return metadata
