from pathlib import Path

import pytest

from simulator.canboat_loader import load_canboat
from simulator.j1939.loader import load_j1939_database


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "vendor" / "canboat" / "canboat.json"
J1939_DATABASE_PATH = PROJECT_ROOT / "vendor" / "j1939" / "J1939_2024-06_v2.4.json"


@pytest.fixture(scope="session")
def database():
    return load_canboat(DATABASE_PATH)


@pytest.fixture(scope="session")
def j1939_database():
    return load_j1939_database(J1939_DATABASE_PATH)
