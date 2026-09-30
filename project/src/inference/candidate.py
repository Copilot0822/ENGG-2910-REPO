"""Candidate signal definitions and extraction primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


InferenceMode = Literal["aligned", "arbitrary"]
ByteOrder = Literal["little_endian", "big_endian"]


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    pgn: int
    source: int
    payload_length: int
    start_bit: int
    bit_length: int
    signed: bool
    byte_order: ByteOrder


@dataclass(frozen=True, slots=True)
class CandidateResult:
    spec: CandidateSpec
    scale: float
    offset: float
    r2: float
    rmse: float
    mae: float
    observation_count: int
    raw_variance: float
    target_variance: float
    normalized_rmse: float
    pairing_fraction: float
    ranking_score: float

    def to_dict(self) -> dict[str, object]:
        result = asdict(self.spec)
        result.update(
            {
                "scale": self.scale,
                "offset": self.offset,
                "r2": self.r2,
                "rmse": self.rmse,
                "mae": self.mae,
                "observation_count": self.observation_count,
                "raw_variance": self.raw_variance,
                "target_variance": self.target_variance,
                "normalized_rmse": self.normalized_rmse,
                "pairing_fraction": self.pairing_fraction,
                "ranking_score": self.ranking_score,
            }
        )
        return result


def candidate_starts(payload_bits: int, bit_length: int, mode: InferenceMode) -> range:
    stop = payload_bits - bit_length + 1
    if stop <= 0:
        return range(0)
    return range(0, stop, 8 if mode == "aligned" else 1)


def extract_raw(
    payload: bytes,
    start_bit: int,
    bit_length: int,
    *,
    signed: bool,
    byte_order: ByteOrder,
) -> int:
    if start_bit < 0 or bit_length <= 0 or start_bit + bit_length > len(payload) * 8:
        raise ValueError("candidate field does not fit in payload")
    raw = 0
    for bit in range(bit_length):
        absolute = start_bit + bit
        if payload[absolute // 8] & (1 << (absolute % 8)):
            raw |= 1 << bit
    if byte_order == "big_endian":
        if bit_length % 8:
            raise ValueError("big-endian candidates require a whole number of bytes")
        raw = int.from_bytes(
            raw.to_bytes(bit_length // 8, "little"), "big", signed=False
        )
    elif byte_order != "little_endian":
        raise ValueError(f"unsupported byte order {byte_order}")
    if signed and raw & (1 << (bit_length - 1)):
        raw -= 1 << bit_length
    return raw


def generate_specs(
    pgn: int,
    source: int,
    payload_length: int,
    mode: InferenceMode,
    bit_lengths: tuple[int, ...],
    byte_orders: tuple[ByteOrder, ...],
) -> tuple[CandidateSpec, ...]:
    specs: list[CandidateSpec] = []
    for bit_length in bit_lengths:
        for start_bit in candidate_starts(payload_length * 8, bit_length, mode):
            orders: tuple[ByteOrder, ...] = (
                ("little_endian",) if bit_length == 8 else byte_orders
            )
            for byte_order in orders:
                for signed in (False, True):
                    specs.append(
                        CandidateSpec(
                            pgn=pgn,
                            source=source,
                            payload_length=payload_length,
                            start_bit=start_bit,
                            bit_length=bit_length,
                            signed=signed,
                            byte_order=byte_order,
                        )
                    )
    return tuple(specs)

