"""Deterministic traffic scheduling and physical CAN frame production."""

from __future__ import annotations

import random
from decimal import Decimal

from simulator.fast_packet import fragment_fast_packet, reassemble_fast_packet
from simulator.message_generator import MessageGenerator
from simulator.models import (
    CanboatDatabase,
    GeneratedMessage,
    KnownObservation,
    PhysicalFrame,
    PGNDefinition,
    SimulationResult,
)
from simulator.n2k_identifier import build_can_id


class DatasetGenerator:
    def __init__(self, database: CanboatDatabase, seed: int) -> None:
        self.database = database
        self.seed = seed
        self.rng = random.Random(seed)
        self.messages = MessageGenerator(database, self.rng)
        self.message_id = 0
        self.observation_id = 0
        self.fast_sequence = 0

    def random_mode(
        self, samples: int, messages_per_sample: int = 6
    ) -> SimulationResult:
        if samples <= 0 or messages_per_sample <= 0:
            raise ValueError("samples and messages-per-sample must be positive")
        pool = list(self.database.generatable_definitions)
        if not pool:
            raise ValueError("Canboat database has no generatable definitions")
        result = SimulationResult(
            options={
                "mode": "random",
                "samples": samples,
                "messages_per_sample": messages_per_sample,
                "seed": self.seed,
            }
        )
        for sample_id in range(samples):
            count = self.rng.randint(1, min(messages_per_sample, len(pool)))
            selected = self.rng.sample(pool, count)
            self._emit_sample(result, sample_id, selected)
        result.time_step_count = samples
        return result

    def coverage_mode(
        self, min_observations_per_field: int, messages_per_sample: int = 12
    ) -> SimulationResult:
        if min_observations_per_field <= 0 or messages_per_sample <= 0:
            raise ValueError("coverage minimum and messages-per-sample must be positive")
        pool = [
            definition
            for definition in self.database.generatable_definitions
            if definition.eligible_fields
        ]
        if not pool:
            raise ValueError("Canboat database has no eligible linear targets")
        counts = {
            field.signal_id: 0
            for definition in pool
            for field in definition.eligible_fields
        }
        result = SimulationResult(
            options={
                "mode": "coverage",
                "min_observations_per_field": min_observations_per_field,
                "messages_per_sample": messages_per_sample,
                "seed": self.seed,
            }
        )
        sample_id = 0
        while min(counts.values()) < min_observations_per_field:
            candidates = [
                definition
                for definition in pool
                if any(
                    counts[field.signal_id] < min_observations_per_field
                    for field in definition.eligible_fields
                )
            ]
            self.rng.shuffle(candidates)
            candidates.sort(
                key=lambda definition: min(
                    counts[field.signal_id] for field in definition.eligible_fields
                )
            )
            selected = candidates[:messages_per_sample]
            before = len(result.observations)
            self._emit_sample(result, sample_id, selected)
            for observation in result.observations[before:]:
                counts[observation.signal_id] += 1
            sample_id += 1
        result.time_step_count = sample_id
        return result

    def _emit_sample(
        self,
        result: SimulationResult,
        sample_id: int,
        definitions: list[PGNDefinition],
    ) -> None:
        timestamp = Decimal(sample_id) / Decimal(10)
        for definition in definitions:
            message = self.messages.generate(definition)
            self._emit_message(result, sample_id, timestamp, message)

    def _emit_message(
        self,
        result: SimulationResult,
        sample_id: int,
        timestamp: Decimal,
        message: GeneratedMessage,
    ) -> None:
        definition = message.definition
        can_id = build_can_id(
            definition.priority,
            definition.pgn,
            message.source_address,
            message.destination_address,
        )
        if definition.packet_type == "Fast":
            physical_payloads = fragment_fast_packet(message.payload, self.fast_sequence)
            if reassemble_fast_packet(physical_payloads) != message.payload:
                raise AssertionError("Fast Packet did not reassemble to the logical payload")
            self.fast_sequence = (self.fast_sequence + 1) % 8
        else:
            physical_payloads = [message.payload]
        for frame_index, payload in enumerate(physical_payloads):
            if len(payload) > 8:
                raise AssertionError("physical CAN payload exceeded 8 bytes")
            result.frames.append(
                PhysicalFrame(
                    sample_id=sample_id,
                    timestamp=timestamp,
                    message_id=self.message_id,
                    frame_index=frame_index,
                    can_id=can_id,
                    data=payload,
                )
            )
        for value in message.values:
            if value.repetition_index is not None:
                continue
            field = value.field
            result.observations.append(
                KnownObservation(
                    sample_id=sample_id,
                    timestamp=timestamp,
                    observation_id=self.observation_id,
                    signal_id=field.signal_id,
                    # Definition descriptions are deliberately excluded here: a
                    # small number of Canboat descriptions literally contain the
                    # PGN number. The opaque ID keeps duplicate field names unique
                    # without revealing the CAN mapping.
                    signal_name=f"{field.name} [{field.signal_id}]",
                    value=value.physical_value,
                    unit=field.unit,
                )
            )
            self.observation_id += 1
        self.message_id += 1
        result.logical_message_count += 1
