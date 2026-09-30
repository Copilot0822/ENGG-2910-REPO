import pytest

from simulator.n2k_identifier import build_can_id, parse_can_id


def test_pdu1_identifier_uses_destination_in_ps_byte():
    can_id = build_can_id(6, 0xEA00, 0x15, 0x23)
    parsed = parse_can_id(can_id)
    assert can_id <= 0x1FFFFFFF
    assert (parsed.priority, parsed.pgn, parsed.source) == (6, 0xEA00, 0x15)
    assert parsed.is_pdu1 and parsed.destination == 0x23


def test_pdu2_identifier_preserves_group_extension():
    can_id = build_can_id(2, 127488, 0x01, 0x44)
    parsed = parse_can_id(can_id)
    assert parsed.pgn == 127488
    assert not parsed.is_pdu1 and parsed.destination is None


def test_invalid_pdu1_group_extension_is_rejected():
    with pytest.raises(ValueError):
        build_can_id(6, 0xEA01, 1)

