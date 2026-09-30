from inference.experiments import run_experiment


def _stable(row):
    return {key: value for key, value in row.items() if key != "runtime_seconds"}


def test_experiment_metrics_are_reproducible_with_fixed_seed(database, tmp_path):
    first = run_experiment(
        database_path=database.path,
        output_directory=tmp_path / "first",
        observation_counts=[3],
        seed=123,
        modes=("arbitrary",),
        max_signals=2,
        min_pairs=3,
        top_k=3,
        messages_per_sample=48,
    )
    second = run_experiment(
        database_path=database.path,
        output_directory=tmp_path / "second",
        observation_counts=[3],
        seed=123,
        modes=("arbitrary",),
        max_signals=2,
        min_pairs=3,
        top_k=3,
        messages_per_sample=48,
    )
    assert [_stable(row) for row in first] == [_stable(row) for row in second]
    for name in (
        "recovery_vs_observations.csv",
        "aligned_vs_arbitrary.csv",
        "top1_location_accuracy.png",
        "top5_location_accuracy.png",
        "full_encoding_accuracy.png",
    ):
        assert (tmp_path / "first" / name).stat().st_size > 0
