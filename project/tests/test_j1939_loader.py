import shutil

import pytest

from conftest import J1939_DATABASE_PATH
from simulator.j1939.loader import load_j1939_database


def test_j1939_database_counts_hash_and_spn_metadata(j1939_database):
    assert len(j1939_database.messages) == 2043
    assert len({message.pgn for message in j1939_database.messages}) == 2043
    assert j1939_database.signal_count == 13774
    assert j1939_database.aligned_signal_count == 6930
    assert len(j1939_database.single_frame_messages) == 1891
    assert len(j1939_database.transport_messages) == 152
    assert j1939_database.json_sha256 == ""
    assert len(j1939_database.source_dbc_sha256) == 64
    assert all(signal.spn >= 0 for message in j1939_database.messages for signal in message.signals)


def test_j1939_json_checksum_mismatch_is_rejected(tmp_path):
    copied = tmp_path / J1939_DATABASE_PATH.name
    shutil.copyfile(J1939_DATABASE_PATH, copied)
    copied.with_suffix(copied.suffix + ".sha256").write_text(
        "0" * 64 + "  " + copied.name + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_j1939_database(copied)

