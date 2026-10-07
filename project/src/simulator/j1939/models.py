"""Normalized J1939 message and signal models."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Literal


TargetCategory = Literal["all_linear", "aligned_8_16_32"]


@dataclass(frozen=True, slots=True)
class J1939Signal:
    signal_id: str
    name: str
    spn: int
    start_bit: int
    bit_length: int
    byte_order: str
    signed: bool
    factor: Decimal
    offset: Decimal
    minimum: Decimal
    maximum: Decimal
    raw_min: int
    raw_max: int
    unit: str
    comment: str
    is_linear: bool
    aligned_8_16_32: bool


@dataclass(frozen=True, slots=True)
class J1939Message:
    definition_id: str
    dbc_id: int
    template_can_id: int
    name: str
    comment: str
    length_bytes: int
    priority: int
    reserved_bit: int
    data_page: int
    pdu_format: int
    pdu_specific: int
    pdu_type: str
    pgn: int
    template_source: int
    template_destination: int | None
    signals: tuple[J1939Signal, ...]

    @property
    def requires_transport(self) -> bool:
        return self.length_bytes > 8

    def target_signals(self, category: TargetCategory) -> tuple[J1939Signal, ...]:
        if category == "all_linear":
            return tuple(signal for signal in self.signals if signal.is_linear)
        return tuple(
            signal
            for signal in self.signals
            if signal.is_linear and signal.aligned_8_16_32
        )


@dataclass(frozen=True, slots=True)
class J1939Database:
    path: Path
    database_version: str
    schema_version: str
    messages: tuple[J1939Message, ...]
    signal_count: int
    aligned_signal_count: int

    @property
    def single_frame_messages(self) -> tuple[J1939Message, ...]:
        return tuple(message for message in self.messages if not message.requires_transport)

    @property
    def transport_messages(self) -> tuple[J1939Message, ...]:
        return tuple(message for message in self.messages if message.requires_transport)


@dataclass(frozen=True, slots=True)
class GeneratedSignalValue:
    signal: J1939Signal
    raw_value: int
    physical_value: Decimal


@dataclass(frozen=True, slots=True)
class GeneratedJ1939Message:
    definition: J1939Message
    payload: bytes
    source: int
    destination: int
    values: tuple[GeneratedSignalValue, ...]


@dataclass(frozen=True, slots=True)
class J1939PhysicalFrame:
    sample_id: int
    timestamp: Decimal
    message_id: int
    frame_index: int
    can_id: int
    data: bytes


@dataclass(frozen=True, slots=True)
class J1939Observation:
    sample_id: int
    timestamp: Decimal
    observation_id: int
    signal_name: str
    signal_id: str
    value: Decimal
    unit: str


@dataclass(slots=True)
class J1939SimulationResult:
    frames: list[J1939PhysicalFrame] = field(default_factory=list)
    observations: list[J1939Observation] = field(default_factory=list)
    logical_message_count: int = 0
    time_step_count: int = 0
    options: dict[str, object] = field(default_factory=dict)
