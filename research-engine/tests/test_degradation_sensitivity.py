from ablate_degradation_sensitivity import paired_monotonicity_summary


def test_paired_monotonicity_keeps_two_statement_pairs_separate():
    records = []
    for pair_id, values in (("pair-match", [0.1, 0.2, 0.3, 0.4]), ("pair-mismatch", [0.8, 0.7, 0.6, 0.5])):
        for sigma, value in zip((0.0, 1.5, 3.0, 6.0), values):
            records.append({
                "pair_id": pair_id,
                "source_recording_id": "same-recording",
                "sigma": sigma,
                "discarded_mass": value,
                "sinkhorn_divergence": value,
            })
    report = paired_monotonicity_summary(records)
    assert report["paired_image_statement_pair_count"] == 2
    assert report["recordings_with_complete_pairs"] == 1
    assert report["pair_level"]["discarded_non_decreasing"] == 0.5
    assert report["pair_level"]["discarded_non_increasing"] == 0.5
    assert report["recording_level_all_pairs"]["discarded_non_decreasing"] == 0.0
