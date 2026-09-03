from __future__ import annotations

from faultline.generation import build_ood_diagnostic_pair, validate_diagnostic_pair


def test_ood_profile_is_outside_training_ranges_and_remains_epistemically_valid() -> None:
    pairs = [build_ood_diagnostic_pair(seed) for seed in range(4)]

    assert all(13 <= pair.graph.node_count <= 20 for pair in pairs)
    assert all(dict(pair.parameters)["family"] == "larger_linear_chain_ood" for pair in pairs)
    assert all(
        dict(pair.parameters)["rate"] not in {1.0, 1.5, 2.0, 2.5, 3.0}
        for pair in pairs
    )
    for pair in pairs:
        validation = validate_diagnostic_pair(pair)
        assert validation.valid
        assert validation.passive_recovery_probability == 0.5
        assert validation.active_recovery_probability == 1.0
        assert validation.epistemic_pressure > 0.0
