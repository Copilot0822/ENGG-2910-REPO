import json
from hashlib import sha256

from conftest import DATABASE_PATH
from simulator.models import SupportStatus


def test_database_loads_and_reports_versions_from_file(database):
    source = json.loads(DATABASE_PATH.read_text(encoding="utf-8"))
    assert database.version == source["Version"] == "8.3.0"
    assert database.schema_version == source["SchemaVersion"]
    assert len(database.definitions) == len(source["PGNs"])
    assert len({item.variant_id for item in database.definitions}) == len(
        database.definitions
    )
    # Git may check out this JSON with LF or CRLF on different systems.
    assert database.sha256 == sha256(DATABASE_PATH.read_bytes()).hexdigest()


def test_loader_classifies_every_definition(database):
    counts = {status: 0 for status in SupportStatus}
    for definition in database.definitions:
        counts[definition.status] += 1
        if definition.status == SupportStatus.SKIPPED:
            assert definition.support_reason
    assert sum(counts.values()) == len(database.definitions)
    assert counts[SupportStatus.SUPPORTED] > 0
    assert counts[SupportStatus.PARTIAL] > 0
    assert counts[SupportStatus.SKIPPED] > 0
    assert database.eligible_signal_count > 0
