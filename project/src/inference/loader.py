"""Load inference-facing CSVs and reconstruct logical NMEA 2000 messages."""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from inference.can_id import parse_can_id
from inference.fast_packet import (
    looks_like_single_frame_fast_packet,
    reassemble_physical_frames,
)


@dataclass(frozen=True, slots=True)
class PhysicalFrame:
    sample_id: int
    timestamp: float
    message_id: int
    frame_index: int
    can_id: int
    data: bytes


@dataclass(frozen=True, slots=True)
class LogicalMessage:
    sample_id: int
    timestamp: float
    message_id: int
    can_id: int
    pgn: int
    source: int
    destination: int | None
    priority: int
    payload: bytes
    is_fast_packet: bool


@dataclass(frozen=True, slots=True)
class KnownValue:
    sample_id: int
    timestamp: float
    observation_id: int
    signal_name: str
    signal_id: str | None
    value: float
    unit: str


_SIGNAL_ID = re.compile(r"\[(signal_\d+)\]\s*$")


def _require_columns(reader: csv.DictReader, required: set[str], path: Path) -> None:
    actual = set(reader.fieldnames or ())
    missing = required - actual
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def load_physical_frames(path: str | Path) -> list[PhysicalFrame]:
    source = Path(path)
    frames: list[PhysicalFrame] = []
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require_columns(
            reader,
            {"sample_id", "timestamp", "message_id", "frame_index", "can_id", "data_hex"},
            source,
        )
        for line_number, row in enumerate(reader, start=2):
            try:
                data = bytes.fromhex(row["data_hex"])
                if len(data) > 8:
                    raise ValueError("physical payload exceeds eight bytes")
                frames.append(
                    PhysicalFrame(
                        sample_id=int(row["sample_id"]),
                        timestamp=float(row["timestamp"]),
                        message_id=int(row["message_id"]),
                        frame_index=int(row["frame_index"]),
                        can_id=int(row["can_id"], 0),
                        data=data,
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid CAN row {line_number} in {source}: {exc}") from exc
    return frames


def reconstruct_logical_messages(frames: list[PhysicalFrame]) -> list[LogicalMessage]:
    grouped: dict[int, list[PhysicalFrame]] = defaultdict(list)
    for frame in frames:
        grouped[frame.message_id].append(frame)
    messages: list[LogicalMessage] = []
    for message_id, group in grouped.items():
        ordered = sorted(group, key=lambda frame: frame.frame_index)
        first = ordered[0]
        if [frame.frame_index for frame in ordered] != list(range(len(ordered))):
            raise ValueError(f"message {message_id} has non-contiguous frame indexes")
        if any(
            frame.sample_id != first.sample_id or frame.can_id != first.can_id
            for frame in ordered
        ):
            raise ValueError(f"message {message_id} changes sample or CAN identifier")
        if len(ordered) > 1:
            payload = reassemble_physical_frames([frame.data for frame in ordered])
            is_fast = True
        elif looks_like_single_frame_fast_packet(first.data):
            payload = first.data[2 : 2 + first.data[1]]
            is_fast = True
        else:
            payload = first.data
            is_fast = False
        parsed = parse_can_id(first.can_id)
        messages.append(
            LogicalMessage(
                sample_id=first.sample_id,
                timestamp=first.timestamp,
                message_id=message_id,
                can_id=first.can_id,
                pgn=parsed.pgn,
                source=parsed.source,
                destination=parsed.destination,
                priority=parsed.priority,
                payload=payload,
                is_fast_packet=is_fast,
            )
        )
    return sorted(messages, key=lambda item: (item.sample_id, item.message_id))


def load_logical_messages(path: str | Path) -> list[LogicalMessage]:
    return reconstruct_logical_messages(load_physical_frames(path))


def load_known_values(path: str | Path) -> list[KnownValue]:
    source = Path(path)
    values: list[KnownValue] = []
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require_columns(
            reader,
            {"sample_id", "timestamp", "observation_id", "signal_name", "value", "unit"},
            source,
        )
        for line_number, row in enumerate(reader, start=2):
            try:
                value = float(row["value"])
                if not math.isfinite(value):
                    raise ValueError("value is not finite")
                match = _SIGNAL_ID.search(row["signal_name"])
                values.append(
                    KnownValue(
                        sample_id=int(row["sample_id"]),
                        timestamp=float(row["timestamp"]),
                        observation_id=int(row["observation_id"]),
                        signal_name=row["signal_name"],
                        signal_id=match.group(1) if match else None,
                        value=value,
                        unit=row["unit"],
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid known-value row {line_number} in {source}: {exc}"
                ) from exc
    return values

