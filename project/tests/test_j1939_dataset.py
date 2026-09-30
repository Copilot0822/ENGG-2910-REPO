from simulator.j1939.generator import J1939DatasetGenerator


def test_j1939_frames_and_fixed_seed_reproducibility(j1939_database):
    first = J1939DatasetGenerator(j1939_database, 42).random_mode(
        8, category="all_linear", include_transport=True, messages_per_sample=20
    )
    second = J1939DatasetGenerator(j1939_database, 42).random_mode(
        8, category="all_linear", include_transport=True, messages_per_sample=20
    )
    third = J1939DatasetGenerator(j1939_database, 43).random_mode(
        8, category="all_linear", include_transport=True, messages_per_sample=20
    )
    assert first.frames == second.frames
    assert first.observations == second.observations
    assert first.frames != third.frames
    assert all(len(frame.data) <= 8 for frame in first.frames)


def test_j1939_coverage_reaches_aligned_single_frame_targets(j1939_database):
    result = J1939DatasetGenerator(j1939_database, 7).coverage_mode(
        1,
        category="aligned_8_16_32",
        include_transport=False,
        messages_per_sample=96,
    )
    observed = {item.signal_id for item in result.observations}
    expected = {
        signal.signal_id
        for message in j1939_database.single_frame_messages
        for signal in message.target_signals("aligned_8_16_32")
    }
    assert observed == expected
    assert len(expected) == 5493


def test_all_linear_definitions_generate_with_transport(j1939_database):
    result = J1939DatasetGenerator(j1939_database, 19).coverage_mode(
        1,
        category="all_linear",
        include_transport=True,
        messages_per_sample=256,
    )
    observed = {item.signal_id for item in result.observations}
    expected = {
        signal.signal_id
        for message in j1939_database.messages
        for signal in message.target_signals("all_linear")
    }
    assert observed == expected
    assert len(expected) == 13774
    assert result.logical_message_count == 2043
    assert len(result.frames) > result.logical_message_count
    assert all(len(frame.data) <= 8 for frame in result.frames)
