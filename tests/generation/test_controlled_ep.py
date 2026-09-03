from __future__ import annotations

import pytest

from faultline.generation import (
    CueCondition,
    CueTask,
    build_generated_diagnostic_pair,
    build_matched_ep_block,
    evaluate_matched_block,
)


def test_matched_block_holds_fault_cue_and_factory_marginals_fixed() -> None:
    pair = build_generated_diagnostic_pair(17)
    block = build_matched_ep_block(pair)
    value = evaluate_matched_block(block)

    assert all(task.pair is pair for task in block.ambiguous_tasks)
    assert block.revealed_task.pair is pair
    assert value.fault_marginals_equal
    assert value.cue_marginals_equal
    assert not value.cue_fault_joint_equal


def test_only_ambiguous_condition_has_positive_active_value() -> None:
    block = build_matched_ep_block(build_generated_diagnostic_pair(29))
    value = evaluate_matched_block(block)

    assert value.ambiguous.passive_recovery_probability == 0.5
    assert value.ambiguous.active_recovery_probability == 1.0
    assert value.ambiguous.epistemic_pressure > 0.0
    assert value.revealed.passive_recovery_probability == 1.0
    assert value.revealed.active_recovery_probability == 1.0
    assert value.revealed.epistemic_pressure == pytest.approx(0.0, abs=1e-12)
    assert value.revealed.passive_expected_return > value.ambiguous.passive_expected_return


def test_matched_control_holds_across_procedural_seeds() -> None:
    values = [
        evaluate_matched_block(build_matched_ep_block(build_generated_diagnostic_pair(seed)))
        for seed in range(8)
    ]

    assert all(value.fault_marginals_equal for value in values)
    assert all(value.cue_marginals_equal for value in values)
    assert all(not value.cue_fault_joint_equal for value in values)
    assert all(value.ambiguous.epistemic_pressure > 0.0 for value in values)
    assert all(
        value.revealed.epistemic_pressure == pytest.approx(0.0, abs=1e-12)
        for value in values
    )


def test_cue_condition_invariants_are_enforced() -> None:
    pair = build_generated_diagnostic_pair(3)
    with pytest.raises(ValueError, match="share one cue"):
        CueTask("bad-ambiguous", pair, CueCondition.AMBIGUOUS, (0, 1))
    with pytest.raises(ValueError, match="distinguish every world"):
        CueTask("bad-revealed", pair, CueCondition.REVEALED, (1, 1))
