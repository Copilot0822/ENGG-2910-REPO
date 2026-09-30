from simulator.dataset_generator import DatasetGenerator
from simulator.output import write_outputs


OUTPUT_FILES = (
    "can_frames.csv",
    "known_values.csv",
    "ground_truth.json",
    "simulation_metadata.json",
    "unsupported_pgns.json",
    "signal_catalog.csv",
)


def _generate(database, destination, seed):
    result = DatasetGenerator(database, seed).random_mode(6, 5)
    write_outputs(destination, database, result)


def test_same_seed_is_byte_identical(database, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _generate(database, first, 99)
    _generate(database, second, 99)
    assert all((first / name).read_bytes() == (second / name).read_bytes() for name in OUTPUT_FILES)


def test_different_seed_changes_raw_dataset(database, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _generate(database, first, 1)
    _generate(database, second, 2)
    assert (first / "can_frames.csv").read_bytes() != (second / "can_frames.csv").read_bytes()
