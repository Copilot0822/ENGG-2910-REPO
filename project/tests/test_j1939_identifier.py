from simulator.j1939.identifier import build_j1939_identifier, parse_j1939_identifier


def test_j1939_pdu1_destination_is_not_part_of_pgn():
    first = build_j1939_identifier(6, 0xEA00, 0x22, 0x17)
    second = build_j1939_identifier(6, 0xEA00, 0x99, 0x44)
    parsed_first = parse_j1939_identifier(first)
    parsed_second = parse_j1939_identifier(second)
    assert parsed_first.pgn == parsed_second.pgn == 0xEA00
    assert parsed_first.destination == 0x17
    assert parsed_second.destination == 0x44


def test_j1939_pdu2_group_extension_and_source_substitution():
    first = build_j1939_identifier(3, 0xFEEE, 0x01)
    second = build_j1939_identifier(3, 0xFEEE, 0x82)
    assert parse_j1939_identifier(first).pgn == 0xFEEE
    assert parse_j1939_identifier(second).pgn == 0xFEEE
    assert parse_j1939_identifier(first).source != parse_j1939_identifier(second).source
    assert parse_j1939_identifier(first).destination is None

