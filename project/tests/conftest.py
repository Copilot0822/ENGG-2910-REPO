from pathlib import Path

import pytest

from simulator.canboat_loader import load_canboat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "vendor" / "canboat" / "canboat.json"


@pytest.fixture(scope="session")
def database():
    return load_canboat(DATABASE_PATH)

