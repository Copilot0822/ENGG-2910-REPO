from pathlib import Path

import pytest

from simulator.canboat_loader import load_canboat
from simulator.j1939.loader import load_j1939_database, resolve_j1939_database_path
from synthetic_j1939 import write_synthetic_database


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "vendor" / "canboat" / "canboat.json"


@pytest.fixture(scope="session")
def database():
    return load_canboat(DATABASE_PATH)


@pytest.fixture(scope="session")
def synthetic_j1939_path(tmp_path_factory):
    return write_synthetic_database(tmp_path_factory.mktemp("j1939") / "synthetic.json")


@pytest.fixture(scope="session")
def j1939_database(synthetic_j1939_path):
    return load_j1939_database(synthetic_j1939_path, private_derived_output=False)


@pytest.fixture(scope="session")
def licensed_j1939_database():
    path = resolve_j1939_database_path()
    if not path.is_file():
        pytest.skip(
            "Licensed J1939 database unavailable; provide a local copy with "
            "J1939_DATABASE_PATH to run this integration test."
        )
    return load_j1939_database(path)
