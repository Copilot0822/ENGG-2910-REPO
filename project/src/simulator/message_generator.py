"""Serialize normalized Canboat variants into logical NMEA 2000 payloads."""

from __future__ import annotations

import random
from dataclasses import dataclass

from simulator.bit_encoder import get_bits
from simulator.models import (
    CanboatDatabase,
    FieldDefinition,
    GeneratedMessage,
    GeneratedValue,
    PGNDefinition,
    SupportStatus,
)
from simulator.value_generator import ValueGenerator, physical_from_raw


@dataclass(frozen=True, slots=True)
class FieldOccurrence:
    field: FieldDefinition
    bit_offset: int
    repetition_index: int | None


def source_address_for(definition: PGNDefinition) -> int:
    text = f"{definition.canboat_id} {definition.description}".lower()
    keyword_sources = (
        (("engine",), 0x01),
        (("battery", "electrical", "charger", "inverter"), 0x15),
        (("gnss", "gps", "position", "navigation", "route"), 0x23),
        (("heading", "attitude", "compass"), 0x31),
        (("tank", "fluid", "fuel"), 0x45),
        (("temperature", "environment", "wind", "weather"), 0x52),
        (("autopilot", "rudder", "steering"), 0x61),
        (("display", "alert", "alarm"), 0x70),
    )
    for keywords, address in keyword_sources:
        if any(keyword in text for keyword in keywords):
            return address
    return (0x80, 0x81, 0x82, 0x83)[definition.pgn % 4]


def _repeat_count(definition: PGNDefinition, rng: random.Random) -> int:
    if not definition.repeating_groups:
        return 1
    group = definition.repeating_groups[0]
    grouped = [
        field
        for field in definition.fields
        if group.start_order <= field.order < group.start_order + group.size
    ]
    base = min(field.bit_offset or 0 for field in grouped)
    end = max((field.bit_offset or 0) + (field.bit_length or 0) for field in grouped)
    group_width = end - base
    maximum = min(3, max(1, (223 * 8 - base) // max(1, group_width)))
    return rng.randint(1, maximum)


def field_occurrences(
    definition: PGNDefinition, repeat_count: int
) -> tuple[FieldOccurrence, ...]:
    if not definition.repeating_groups:
        return tuple(
            FieldOccurrence(field, int(field.bit_offset), None)
            for field in definition.fields
            if field.bit_offset is not None
        )
    group = definition.repeating_groups[0]
    grouped = [
        field
        for field in definition.fields
        if group.start_order <= field.order < group.start_order + group.size
    ]
    group_base = min(int(field.bit_offset) for field in grouped if field.bit_offset is not None)
    group_end = max(
        int(field.bit_offset) + int(field.bit_length)
        for field in grouped
        if field.bit_offset is not None and field.bit_length is not None
    )
    group_width = group_end - group_base
    occurrences: list[FieldOccurrence] = []
    for field in definition.fields:
        assert field.bit_offset is not None
        if field.order < group.start_order:
            occurrences.append(FieldOccurrence(field, field.bit_offset, None))
        elif field.order < group.start_order + group.size:
            relative = field.bit_offset - group_base
            for repetition in range(repeat_count):
                occurrences.append(
                    FieldOccurrence(
                        field,
                        group_base + repetition * group_width + relative,
                        repetition,
                    )
                )
        else:
            shifted = field.bit_offset + (repeat_count - 1) * group_width
            occurrences.append(FieldOccurrence(field, shifted, None))
    return tuple(occurrences)


class MessageGenerator:
    def __init__(self, database: CanboatDatabase, rng: random.Random) -> None:
        self.database = database
        self.rng = rng
        self.values = ValueGenerator(database, rng)

    def generate(self, definition: PGNDefinition) -> GeneratedMessage:
        if definition.status == SupportStatus.SKIPPED:
            raise ValueError(f"cannot generate skipped variant {definition.variant_id}")
        repeat_count = _repeat_count(definition, self.rng)
        occurrences = field_occurrences(definition, repeat_count)
        maximum_bit = max(
            occurrence.bit_offset + int(occurrence.field.bit_length)
            for occurrence in occurrences
            if occurrence.field.bit_length is not None
        )
        minimum_bytes = definition.min_length or 0
        if not definition.repeating_groups:
            minimum_bytes = max(minimum_bytes, definition.declared_length or 0)
        payload_length = max(minimum_bytes, (maximum_bit + 7) // 8)
        if definition.packet_type == "Single" and payload_length > 8:
            raise ValueError("single-frame payload exceeds 8 bytes")
        if definition.packet_type == "Fast" and payload_length > 223:
            raise ValueError("Fast Packet payload exceeds 223 bytes")
        payload = bytearray([0xFF] * payload_length)
        count_order = (
            definition.repeating_groups[0].count_field_order
            if definition.repeating_groups
            else None
        )
        forced_by_order: dict[int, int] = {}
        for field in definition.fields:
            if field.field_type != "INDIRECT_LOOKUP":
                continue
            pairs = self.database.indirect_lookups.get(
                field.indirect_lookup_name or "", ()
            )
            if pairs and field.indirect_field_order is not None:
                selector, value, _label = self.rng.choice(pairs)
                forced_by_order[field.indirect_field_order] = selector
                forced_by_order[field.order] = value
        generated: list[GeneratedValue] = []
        for occurrence in occurrences:
            field = occurrence.field
            forced = forced_by_order.get(field.order)
            if field.order == count_order:
                forced = repeat_count
            encoded = self.values.encode(
                payload, field, occurrence.bit_offset, forced_raw=forced
            )
            if encoded is not None and field.is_linear_target:
                generated.append(
                    GeneratedValue(
                        field=field,
                        raw_value=encoded.raw_value,
                        physical_value=encoded.physical_value,
                        repetition_index=occurrence.repetition_index,
                        lookup_label=encoded.lookup_label,
                    )
                )
        for field in definition.fields:
            if field.match is not None:
                occurrence = next(item for item in occurrences if item.field is field)
                actual = get_bits(
                    payload, occurrence.bit_offset, int(field.bit_length), signed=False
                )
                expected = field.match & ((1 << int(field.bit_length)) - 1)
                if actual != expected:
                    raise AssertionError(
                        f"variant match field {field.field_id} encoded as {actual}, expected {expected}"
                    )
        message = GeneratedMessage(
            definition=definition,
            payload=bytes(payload),
            values=tuple(generated),
            source_address=source_address_for(definition),
            destination_address=0x23,
        )
        validate_round_trip(message)
        return message


def decode_linear_fields(message: GeneratedMessage) -> dict[str, object]:
    result: dict[str, object] = {}
    for value in message.values:
        field = value.field
        assert field.bit_offset is not None and field.bit_length is not None
        raw = get_bits(
            message.payload, field.bit_offset, field.bit_length, signed=field.signed
        )
        result[field.signal_id] = physical_from_raw(field, raw)
    return result


def validate_round_trip(message: GeneratedMessage) -> None:
    decoded = decode_linear_fields(message)
    for value in message.values:
        if value.repetition_index is not None:
            continue
        actual = decoded[value.field.signal_id]
        if actual != value.physical_value:
            raise AssertionError(
                f"round trip mismatch for {value.field.signal_id}: "
                f"{actual} != {value.physical_value}"
            )
