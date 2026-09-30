"""Typed internal models normalized from Canboat's JSON schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any


class SupportStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True, slots=True)
class FieldDefinition:
    order: int
    field_id: str
    name: str
    field_type: str
    bit_length: int | None
    bit_offset: int | None
    signed: bool
    resolution: Decimal
    offset: Decimal
    unit: str
    range_min: Decimal | None
    range_max: Decimal | None
    unknown_value: int | None
    out_of_range_value: int | None
    reserved_value: int | None
    match: int | None
    lookup_name: str | None
    indirect_lookup_name: str | None
    indirect_field_order: int | None
    bit_lookup_name: str | None
    physical_quantity: str | None
    part_of_primary_key: bool
    bit_length_variable: bool
    signal_id: str
    in_repeating_group: bool = False

    @property
    def is_linear_target(self) -> bool:
        return (
            self.field_type in {"NUMBER", "DURATION"}
            and self.bit_length is not None
            and not self.bit_length_variable
            and not self.in_repeating_group
            and self.match is None
            and self.resolution != 0
        )

    @property
    def sentinels(self) -> frozenset[int]:
        return frozenset(
            value
            for value in (
                self.unknown_value,
                self.out_of_range_value,
                self.reserved_value,
            )
            if value is not None
        )


@dataclass(frozen=True, slots=True)
class RepeatingGroup:
    start_order: int
    size: int
    count_field_order: int | None


@dataclass(frozen=True, slots=True)
class PGNDefinition:
    pgn: int
    canboat_id: str
    variant_id: str
    description: str
    priority: int
    packet_type: str
    complete: bool
    declared_length: int | None
    min_length: int | None
    fields: tuple[FieldDefinition, ...]
    repeating_groups: tuple[RepeatingGroup, ...]
    status: SupportStatus
    support_reason: str | None
    source_index: int

    @property
    def match_conditions(self) -> dict[str, int]:
        return {
            item.field_id: item.match
            for item in self.fields
            if item.match is not None
        }

    @property
    def eligible_fields(self) -> tuple[FieldDefinition, ...]:
        if self.status == SupportStatus.SKIPPED:
            return ()
        return tuple(item for item in self.fields if item.is_linear_target)


@dataclass(frozen=True, slots=True)
class CanboatDatabase:
    path: Path
    version: str
    schema_version: str
    sha256: str
    definitions: tuple[PGNDefinition, ...]
    lookups: dict[str, tuple[tuple[int, str], ...]]
    indirect_lookups: dict[str, tuple[tuple[int, int, str], ...]]
    bit_lookups: dict[str, tuple[tuple[int, str], ...]]

    @property
    def generatable_definitions(self) -> tuple[PGNDefinition, ...]:
        return tuple(
            item for item in self.definitions if item.status != SupportStatus.SKIPPED
        )

    @property
    def eligible_signal_count(self) -> int:
        return sum(len(item.eligible_fields) for item in self.definitions)


@dataclass(frozen=True, slots=True)
class GeneratedValue:
    field: FieldDefinition
    raw_value: int
    physical_value: Decimal
    repetition_index: int | None = None
    lookup_label: str | None = None


@dataclass(frozen=True, slots=True)
class GeneratedMessage:
    definition: PGNDefinition
    payload: bytes
    values: tuple[GeneratedValue, ...]
    source_address: int
    destination_address: int


@dataclass(frozen=True, slots=True)
class PhysicalFrame:
    sample_id: int
    timestamp: Decimal
    message_id: int
    frame_index: int
    can_id: int
    data: bytes


@dataclass(frozen=True, slots=True)
class KnownObservation:
    sample_id: int
    timestamp: Decimal
    observation_id: int
    signal_id: str
    signal_name: str
    value: Decimal
    unit: str


@dataclass(slots=True)
class SimulationResult:
    frames: list[PhysicalFrame] = field(default_factory=list)
    observations: list[KnownObservation] = field(default_factory=list)
    logical_message_count: int = 0
    time_step_count: int = 0
    options: dict[str, Any] = field(default_factory=dict)
