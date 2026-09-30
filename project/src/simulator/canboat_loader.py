"""Load and normalize the pinned Canboat JSON database."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from simulator.models import (
    CanboatDatabase,
    FieldDefinition,
    PGNDefinition,
    RepeatingGroup,
    SupportStatus,
)


def _decimal(value: object, default: str = "0") -> Decimal:
    if value is None:
        return Decimal(default)
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _optional_decimal(value: object) -> Decimal | None:
    return None if value is None else _decimal(value)


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _load_lookup_table(
    rows: list[dict[str, Any]], value_key: str, list_key: str
) -> dict[str, tuple[tuple[int, str], ...]]:
    result: dict[str, tuple[tuple[int, str], ...]] = {}
    for row in rows:
        values = tuple(
            (int(item[value_key]), str(item["Name"]))
            for item in row.get(list_key, [])
            if value_key in item and "Name" in item
        )
        result[str(row["Name"])] = values
    return result


def _load_indirect_lookup_table(
    rows: list[dict[str, Any]],
) -> dict[str, tuple[tuple[int, int, str], ...]]:
    result: dict[str, tuple[tuple[int, int, str], ...]] = {}
    for row in rows:
        result[str(row["Name"])] = tuple(
            (int(item["Value1"]), int(item["Value2"]), str(item["Name"]))
            for item in row.get("EnumValues", [])
            if all(key in item for key in ("Value1", "Value2", "Name"))
        )
    return result


def _repeating_groups(raw: dict[str, Any]) -> tuple[RepeatingGroup, ...]:
    groups: list[RepeatingGroup] = []
    for number in (1, 2):
        size = raw.get(f"RepeatingFieldSet{number}Size")
        start = raw.get(f"RepeatingFieldSet{number}StartField")
        if size is None or start is None:
            continue
        groups.append(
            RepeatingGroup(
                start_order=int(start),
                size=int(size),
                count_field_order=_optional_int(
                    raw.get(f"RepeatingFieldSet{number}CountField")
                ),
            )
        )
    return tuple(groups)


def _field_in_repeat(order: int, groups: tuple[RepeatingGroup, ...]) -> bool:
    return any(
        group.start_order <= order < group.start_order + group.size
        for group in groups
    )


def _classify(
    raw: dict[str, Any],
    fields: list[dict[str, Any]],
    repeats: tuple[RepeatingGroup, ...],
) -> tuple[SupportStatus, str | None]:
    packet_type = str(raw.get("Type", ""))
    if packet_type not in {"Single", "Fast"}:
        return SupportStatus.SKIPPED, f"unsupported packet type {packet_type or 'missing'}"
    if not fields:
        return SupportStatus.SKIPPED, "definition has no fields"
    if len(repeats) > 1:
        return SupportStatus.SKIPPED, "multiple repeating field sets are not yet supported"
    for item in fields:
        if item.get("BitLengthVariable") or item.get("BitLength") is None:
            return (
                SupportStatus.SKIPPED,
                f"dynamic or missing bit length for field {item.get('Id', item.get('Order'))}",
            )
        if item.get("BitOffset") is None:
            return (
                SupportStatus.SKIPPED,
                f"dynamic or missing bit offset for field {item.get('Id', item.get('Order'))}",
            )
        if item.get("FieldType") in {
            "VARIABLE",
            "DYNAMIC_FIELD_VALUE",
            "DYNAMIC_FIELD_KEY",
            "DYNAMIC_FIELD_LENGTH",
        }:
            return (
                SupportStatus.SKIPPED,
                f"unsupported dynamic field type {item.get('FieldType')}",
            )
    maximum_bit = max(int(item["BitOffset"]) + int(item["BitLength"]) for item in fields)
    declared = raw.get("Length")
    effective_length = max((maximum_bit + 7) // 8, int(declared or 0))
    if packet_type == "Single" and effective_length > 8:
        return SupportStatus.SKIPPED, "single-frame definition exceeds 8 bytes"
    if packet_type == "Fast" and not repeats and effective_length > 223:
        return SupportStatus.SKIPPED, "logical payload exceeds Fast Packet maximum of 223 bytes"
    if repeats:
        group = repeats[0]
        orders = {int(item["Order"]) for item in fields}
        expected = set(range(group.start_order, group.start_order + group.size))
        if not expected.issubset(orders):
            return SupportStatus.SKIPPED, "repeating field set is incomplete"
    if not bool(raw.get("Complete", False)):
        return SupportStatus.PARTIAL, "Canboat marks this definition incomplete"
    return SupportStatus.SUPPORTED, None


def load_canboat(path: str | Path) -> CanboatDatabase:
    """Read *path* and turn every Canboat PGN entry into a normalized variant."""

    source = Path(path).resolve()
    raw_bytes = source.read_bytes()
    document = json.loads(raw_bytes, parse_float=Decimal)
    lookups = _load_lookup_table(
        document.get("LookupEnumerations", []), "Value", "EnumValues"
    )
    indirect_lookups = _load_indirect_lookup_table(
        document.get("LookupIndirectEnumerations", [])
    )
    bit_lookups = _load_lookup_table(
        document.get("LookupBitEnumerations", []), "Bit", "EnumBitValues"
    )

    definitions: list[PGNDefinition] = []
    signal_counter = 0
    for index, raw in enumerate(document.get("PGNs", []), start=1):
        raw_fields = list(raw.get("Fields", []))
        repeats = _repeating_groups(raw)
        status, reason = _classify(raw, raw_fields, repeats)
        fields: list[FieldDefinition] = []
        for raw_field in raw_fields:
            signal_counter += 1
            field_type = str(raw_field.get("FieldType", "UNKNOWN"))
            signed_default = field_type in {"FLOAT"}
            order = int(raw_field.get("Order", len(fields) + 1))
            fields.append(
                FieldDefinition(
                    order=order,
                    field_id=str(raw_field.get("Id", f"field{order}")),
                    name=str(raw_field.get("Name", raw_field.get("Id", f"Field {order}"))),
                    field_type=field_type,
                    bit_length=_optional_int(raw_field.get("BitLength")),
                    bit_offset=_optional_int(raw_field.get("BitOffset")),
                    signed=bool(raw_field.get("Signed", signed_default)),
                    resolution=_decimal(raw_field.get("Resolution"), "1"),
                    offset=_decimal(raw_field.get("Offset"), "0"),
                    unit=str(raw_field.get("Unit", "")),
                    range_min=_optional_decimal(raw_field.get("RangeMin")),
                    range_max=_optional_decimal(raw_field.get("RangeMax")),
                    unknown_value=_optional_int(raw_field.get("UnknownValue")),
                    out_of_range_value=_optional_int(raw_field.get("OutOfRangeValue")),
                    reserved_value=_optional_int(raw_field.get("ReservedValue")),
                    match=_optional_int(raw_field.get("Match")),
                    lookup_name=raw_field.get("LookupEnumeration"),
                    indirect_lookup_name=raw_field.get("LookupIndirectEnumeration"),
                    indirect_field_order=_optional_int(
                        raw_field.get("LookupIndirectEnumerationFieldOrder")
                    ),
                    bit_lookup_name=raw_field.get("LookupBitEnumeration"),
                    physical_quantity=raw_field.get("PhysicalQuantity"),
                    part_of_primary_key=bool(raw_field.get("PartOfPrimaryKey", False)),
                    bit_length_variable=bool(raw_field.get("BitLengthVariable", False)),
                    signal_id=f"signal_{signal_counter:06d}",
                    in_repeating_group=_field_in_repeat(order, repeats),
                )
            )
        definitions.append(
            PGNDefinition(
                pgn=int(raw["PGN"]),
                canboat_id=str(raw.get("Id", f"pgn{raw['PGN']}")),
                variant_id=f"variant_{index:06d}",
                description=str(raw.get("Description", raw.get("Id", raw["PGN"]))),
                priority=int(raw.get("Priority", 6)),
                packet_type=str(raw.get("Type", "")),
                complete=bool(raw.get("Complete", False)),
                declared_length=_optional_int(raw.get("Length")),
                min_length=_optional_int(raw.get("MinLength")),
                fields=tuple(sorted(fields, key=lambda item: item.order)),
                repeating_groups=repeats,
                status=status,
                support_reason=reason,
                source_index=index,
            )
        )

    return CanboatDatabase(
        path=source,
        version=str(document.get("Version", "unknown")),
        schema_version=str(document.get("SchemaVersion", "unknown")),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        definitions=tuple(definitions),
        lookups=lookups,
        indirect_lookups=indirect_lookups,
        bit_lookups=bit_lookups,
    )
