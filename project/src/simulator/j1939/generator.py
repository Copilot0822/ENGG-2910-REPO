"""Deterministic J1939 message and dataset generation."""

from __future__ import annotations

import random
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR

from simulator.j1939.encoding import decode_signal, encode_signal
from simulator.j1939.identifier import build_j1939_identifier
from simulator.j1939.models import (
    GeneratedJ1939Message,
    GeneratedSignalValue,
    J1939Database,
    J1939Message,
    J1939Observation,
    J1939PhysicalFrame,
    J1939Signal,
    J1939SimulationResult,
    TargetCategory,
)
from simulator.j1939.transport import TransportFrame, fragment_transport, reassemble_transport


SOURCE_ADDRESSES = {
    "engine": 0x00,
    "transmission": 0x03,
    "brake": 0x0B,
    "body": 0x21,
    "instrument": 0x17,
    "aftertreatment": 0x3D,
    "chassis": 0x31,
}


def source_address_for(message: J1939Message) -> int:
    text = f"{message.name} {message.comment}".lower()
    for keyword, address in SOURCE_ADDRESSES.items():
        if keyword in text:
            return address
    return (0x80, 0x81, 0x82, 0x83)[message.pgn % 4]


def destination_address_for(message: J1939Message, source: int) -> int:
    if message.pdu_type != "PDU1":
        return 0xFF
    destination = message.template_destination
    if destination is None or destination >= 0xFE or destination == source:
        return 0x17 if source != 0x17 else 0x21
    return destination


def valid_raw_limits(signal: J1939Signal) -> tuple[int, int]:
    if signal.signed:
        encoding_min = -(1 << (signal.bit_length - 1))
        encoding_max = (1 << (signal.bit_length - 1)) - 1
    else:
        encoding_min = 0
        encoding_max = (1 << signal.bit_length) - 1
    lower = max(encoding_min, signal.raw_min)
    upper = min(encoding_max, signal.raw_max)
    if signal.factor > 0:
        physical_lower = int(
            ((signal.minimum - signal.offset) / signal.factor).to_integral_value(
                rounding=ROUND_CEILING
            )
        )
        physical_upper = int(
            ((signal.maximum - signal.offset) / signal.factor).to_integral_value(
                rounding=ROUND_FLOOR
            )
        )
    else:
        physical_lower = int(
            ((signal.maximum - signal.offset) / signal.factor).to_integral_value(
                rounding=ROUND_CEILING
            )
        )
        physical_upper = int(
            ((signal.minimum - signal.offset) / signal.factor).to_integral_value(
                rounding=ROUND_FLOOR
            )
        )
    lower = max(lower, physical_lower)
    upper = min(upper, physical_upper)
    if lower > upper:
        raise ValueError(f"signal {signal.name} has no valid raw range")
    return lower, upper


class J1939MessageGenerator:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def generate(
        self, message: J1939Message, category: TargetCategory
    ) -> GeneratedJ1939Message:
        payload = bytearray(
            self.rng.getrandbits(8) for _ in range(message.length_bytes)
        )
        target_ids = {signal.signal_id for signal in message.target_signals(category)}
        values: list[GeneratedSignalValue] = []
        for signal in message.signals:
            lower, upper = valid_raw_limits(signal)
            raw = self.rng.randint(lower, upper)
            encode_signal(payload, signal, raw)
            decoded = decode_signal(payload, signal)
            if decoded != raw:
                raise AssertionError(
                    f"J1939 round trip failed for {message.name}/{signal.name}"
                )
            physical = signal.factor * raw + signal.offset
            if not signal.minimum <= physical <= signal.maximum:
                raise AssertionError(f"generated value for {signal.name} is out of range")
            if signal.signal_id in target_ids:
                values.append(GeneratedSignalValue(signal, raw, physical))
        source = source_address_for(message)
        return GeneratedJ1939Message(
            definition=message,
            payload=bytes(payload),
            source=source,
            destination=destination_address_for(message, source),
            values=tuple(values),
        )


class J1939DatasetGenerator:
    def __init__(self, database: J1939Database, seed: int) -> None:
        self.database = database
        self.seed = seed
        self.rng = random.Random(seed)
        self.messages = J1939MessageGenerator(self.rng)
        self.message_id = 0
        self.observation_id = 0

    def _pool(
        self, category: TargetCategory, include_transport: bool
    ) -> list[J1939Message]:
        return [
            message
            for message in self.database.messages
            if (include_transport or not message.requires_transport)
            and message.target_signals(category)
        ]

    def random_mode(
        self,
        samples: int,
        *,
        category: TargetCategory,
        include_transport: bool,
        messages_per_sample: int = 12,
    ) -> J1939SimulationResult:
        if samples <= 0 or messages_per_sample <= 0:
            raise ValueError("samples and messages_per_sample must be positive")
        pool = self._pool(category, include_transport)
        result = J1939SimulationResult(
            options={
                "protocol": "j1939",
                "mode": "random",
                "samples": samples,
                "category": category,
                "include_transport": include_transport,
                "messages_per_sample": messages_per_sample,
                "seed": self.seed,
            }
        )
        for sample_id in range(samples):
            count = self.rng.randint(1, min(messages_per_sample, len(pool)))
            self._emit_sample(result, sample_id, self.rng.sample(pool, count), category)
        result.time_step_count = samples
        return result

    def coverage_mode(
        self,
        minimum: int,
        *,
        category: TargetCategory,
        include_transport: bool,
        messages_per_sample: int = 48,
    ) -> J1939SimulationResult:
        if minimum <= 0 or messages_per_sample <= 0:
            raise ValueError("coverage minimum and messages_per_sample must be positive")
        pool = self._pool(category, include_transport)
        counts = {
            signal.signal_id: 0
            for message in pool
            for signal in message.target_signals(category)
        }
        result = J1939SimulationResult(
            options={
                "protocol": "j1939",
                "mode": "coverage",
                "min_observations_per_field": minimum,
                "category": category,
                "include_transport": include_transport,
                "messages_per_sample": messages_per_sample,
                "seed": self.seed,
            }
        )
        sample_id = 0
        while min(counts.values()) < minimum:
            candidates = [
                message
                for message in pool
                if any(
                    counts[signal.signal_id] < minimum
                    for signal in message.target_signals(category)
                )
            ]
            self.rng.shuffle(candidates)
            candidates.sort(
                key=lambda message: min(
                    counts[signal.signal_id]
                    for signal in message.target_signals(category)
                )
            )
            selected = candidates[:messages_per_sample]
            before = len(result.observations)
            self._emit_sample(result, sample_id, selected, category)
            for observation in result.observations[before:]:
                counts[observation.signal_id] += 1
            sample_id += 1
        result.time_step_count = sample_id
        return result

    def targeted_coverage_mode(
        self,
        minimum: int,
        definitions: list[J1939Message],
        *,
        category: TargetCategory,
        include_transport: bool,
    ) -> J1939SimulationResult:
        """Cover selected messages on disjoint sample ranges.

        Disjoint ranges preserve the normal inference experiment's pairing
        behavior: a signal is synchronized with its own message stream without
        making every selected PGN appear spuriously synchronized with it.  This
        is substantially smaller than generating the entire database for very
        high observation counts.
        """

        if minimum <= 0 or not definitions:
            raise ValueError("minimum and definitions must be nonempty")
        if not include_transport and any(
            definition.requires_transport for definition in definitions
        ):
            raise ValueError("selected definitions require J1939 TP")
        result = J1939SimulationResult(
            options={
                "protocol": "j1939",
                "mode": "targeted_coverage",
                "min_observations_per_field": minimum,
                "category": category,
                "include_transport": include_transport,
                "messages_per_sample": 1,
                "selected_message_count": len(definitions),
                "seed": self.seed,
            }
        )
        sample_id = 0
        for definition in definitions:
            for _ in range(minimum):
                self._emit_sample(result, sample_id, [definition], category)
                sample_id += 1
        result.time_step_count = sample_id
        return result

    def _emit_sample(
        self,
        result: J1939SimulationResult,
        sample_id: int,
        definitions: list[J1939Message],
        category: TargetCategory,
    ) -> None:
        timestamp = Decimal(sample_id) / Decimal(10)
        for definition in definitions:
            generated = self.messages.generate(definition, category)
            self._emit_message(result, sample_id, timestamp, generated)

    def _emit_message(
        self,
        result: J1939SimulationResult,
        sample_id: int,
        timestamp: Decimal,
        message: GeneratedJ1939Message,
    ) -> None:
        definition = message.definition
        if definition.requires_transport:
            destination = (
                message.destination if definition.pdu_type == "PDU1" else 0xFF
            )
            physical = fragment_transport(
                message.payload, definition.pgn, message.source, destination
            )
            reassembled = reassemble_transport(physical)
            if reassembled.payload != message.payload or reassembled.target_pgn != definition.pgn:
                raise AssertionError("J1939 TP round trip failed")
        else:
            physical = [
                TransportFrame(
                    build_j1939_identifier(
                        definition.priority,
                        definition.pgn,
                        message.source,
                        message.destination,
                    ),
                    message.payload,
                )
            ]
        for frame_index, frame in enumerate(physical):
            if len(frame.data) > 8:
                raise AssertionError("physical J1939 frame exceeds eight bytes")
            result.frames.append(
                J1939PhysicalFrame(
                    sample_id,
                    timestamp,
                    self.message_id,
                    frame_index,
                    frame.can_id,
                    frame.data,
                )
            )
        for value in message.values:
            signal = value.signal
            result.observations.append(
                J1939Observation(
                    sample_id,
                    timestamp,
                    self.observation_id,
                    f"{signal.name} [{signal.signal_id}]",
                    signal.signal_id,
                    value.physical_value,
                    signal.unit,
                )
            )
            self.observation_id += 1
        self.message_id += 1
        result.logical_message_count += 1
