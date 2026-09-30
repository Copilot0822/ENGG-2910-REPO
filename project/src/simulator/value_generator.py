"""Generate valid encoded values without selecting NMEA sentinel states."""

from __future__ import annotations

import random
import struct
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR

from simulator.bit_encoder import set_bits
from simulator.models import CanboatDatabase, FieldDefinition


@dataclass(frozen=True, slots=True)
class EncodedValue:
    raw_value: int
    physical_value: Decimal
    lookup_label: str | None = None


def raw_encoding_limits(field: FieldDefinition) -> tuple[int, int]:
    if field.bit_length is None:
        raise ValueError(f"field {field.field_id} has no fixed bit length")
    if field.signed:
        return -(1 << (field.bit_length - 1)), (1 << (field.bit_length - 1)) - 1
    return 0, (1 << field.bit_length) - 1


def valid_raw_limits(field: FieldDefinition) -> tuple[int, int]:
    lower, upper = raw_encoding_limits(field)
    if field.range_min is not None:
        from_physical = int(
            ((field.range_min - field.offset) / field.resolution).to_integral_value(
                rounding=ROUND_CEILING
            )
        )
        lower = max(lower, from_physical)
    if field.range_max is not None:
        from_physical = int(
            ((field.range_max - field.offset) / field.resolution).to_integral_value(
                rounding=ROUND_FLOOR
            )
        )
        upper = min(upper, from_physical)
    if lower > upper:
        raise ValueError(f"field {field.field_id} has an empty valid raw range")
    return lower, upper


def choose_valid_raw(field: FieldDefinition, rng: random.Random) -> int:
    lower, upper = valid_raw_limits(field)
    forbidden = field.sentinels
    allowed_count = upper - lower + 1 - sum(lower <= value <= upper for value in forbidden)
    if allowed_count <= 0:
        raise ValueError(f"field {field.field_id} has no non-sentinel raw values")
    for _ in range(32):
        candidate = rng.randint(lower, upper)
        if candidate not in forbidden:
            return candidate
    # Tiny ranges can be sentinel-heavy. A deterministic scan guarantees progress.
    start = rng.randint(lower, upper)
    for delta in range(upper - lower + 1):
        candidate = lower + ((start - lower + delta) % (upper - lower + 1))
        if candidate not in forbidden:
            return candidate
    raise AssertionError("allowed-count calculation disagrees with range scan")


def physical_from_raw(field: FieldDefinition, raw_value: int) -> Decimal:
    return field.resolution * raw_value + field.offset


class ValueGenerator:
    def __init__(self, database: CanboatDatabase, rng: random.Random) -> None:
        self.database = database
        self.rng = rng

    def encode(
        self,
        buffer: bytearray,
        field: FieldDefinition,
        bit_offset: int,
        *,
        forced_raw: int | None = None,
    ) -> EncodedValue | None:
        if field.bit_length is None:
            raise ValueError(f"cannot encode variable field {field.field_id}")
        width = field.bit_length
        field_type = field.field_type

        if field_type == "RESERVED":
            set_bits(buffer, bit_offset, width, (1 << width) - 1)
            return None
        if field_type == "SPARE":
            set_bits(buffer, bit_offset, width, 0)
            return None
        if field_type == "STRING_FIX":
            self._encode_fixed_string(buffer, bit_offset, width)
            return None
        if field_type == "FLOAT":
            if width != 32:
                raise ValueError("only Canboat's 32-bit FLOAT representation is supported")
            packed = struct.pack("<f", self.rng.uniform(-1000.0, 1000.0))
            set_bits(buffer, bit_offset, 32, int.from_bytes(packed, "little"))
            return None
        if field_type == "DECIMAL":
            if width % 8:
                raise ValueError("DECIMAL fields must be byte aligned in length")
            encoded = bytes(self.rng.randint(0, 99) for _ in range(width // 8))
            set_bits(buffer, bit_offset, width, int.from_bytes(encoded, "little"))
            return None
        if field_type == "BINARY":
            set_bits(buffer, bit_offset, width, self.rng.getrandbits(width))
            return None
        if field_type == "BITLOOKUP":
            choices = self.database.bit_lookups.get(field.bit_lookup_name or "", ())
            raw = 0
            for bit, _label in choices:
                if bit < width and self.rng.choice((False, True)):
                    raw |= 1 << bit
            set_bits(buffer, bit_offset, width, raw)
            return EncodedValue(raw, physical_from_raw(field, raw))
        if field_type in {"LOOKUP", "INDIRECT_LOOKUP"}:
            raw, label = self._lookup_value(field, forced_raw)
            set_bits(buffer, bit_offset, width, raw)
            return EncodedValue(raw, physical_from_raw(field, raw), label)

        if forced_raw is not None or field.match is not None:
            raw = int(forced_raw if forced_raw is not None else field.match)
        elif field_type == "MMSI":
            raw = self.rng.randint(2_000_000, min(999_999_999, (1 << width) - 1))
        elif field_type == "PGN":
            raw = self.rng.randint(0, min(0x3FFFF, (1 << width) - 1))
        elif field_type == "ADDRESS":
            raw = self.rng.randint(0, min(0xFF, (1 << width) - 1))
        elif field_type == "ISO_NAME":
            raw = self.rng.getrandbits(width)
        else:
            raw = choose_valid_raw(field, self.rng)
        set_bits(buffer, bit_offset, width, raw)
        return EncodedValue(raw, physical_from_raw(field, raw))

    def _lookup_value(
        self, field: FieldDefinition, forced_raw: int | None
    ) -> tuple[int, str | None]:
        table = self.database.lookups.get(field.lookup_name or "", ())
        labels = {value: label for value, label in table}
        if forced_raw is not None or field.match is not None:
            raw = int(forced_raw if forced_raw is not None else field.match)
            return raw, labels.get(raw)
        lower, upper = valid_raw_limits(field)
        candidates = [
            (value, label)
            for value, label in table
            if lower <= value <= upper and value not in field.sentinels
        ]
        if candidates:
            return self.rng.choice(candidates)
        raw = choose_valid_raw(field, self.rng)
        return raw, labels.get(raw)

    def _encode_fixed_string(
        self, buffer: bytearray, bit_offset: int, bit_length: int
    ) -> None:
        if bit_offset % 8 or bit_length % 8:
            raise ValueError("fixed strings must be byte aligned")
        byte_count = bit_length // 8
        value = f"SIM{self.rng.randint(0, 999999):06d}".encode("ascii")[:byte_count]
        value = value + bytes([0xFF]) * (byte_count - len(value))
        set_bits(buffer, bit_offset, bit_length, int.from_bytes(value, "little"))

