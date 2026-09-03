from __future__ import annotations

from faultline.env import ClearBlockage, Replace
from faultline.generation import (
    build_manual_diagnostic_pair,
    create_world_env,
    diagnostic_evidence,
    evaluate_repair,
    full_passive_snapshot,
    run_contingent_active_policy,
)


def test_worlds_have_exactly_equal_complete_passive_snapshot() -> None:
    pair = build_manual_diagnostic_pair(seed=42)
    left = full_passive_snapshot(create_world_env(pair, pair.worlds[0]))
    right = full_passive_snapshot(create_world_env(pair, pair.worlds[1]))

    assert left == right
    assert pair.worlds[0].correct_repair != pair.worlds[1].correct_repair


def test_each_world_has_a_different_unique_successful_repair() -> None:
    pair = build_manual_diagnostic_pair(seed=42)
    candidates = (ClearBlockage("delivery"), Replace("processor"))

    for world in pair.worlds:
        outcomes = [evaluate_repair(pair, world, repair) for repair in candidates]
        successful = [outcome for outcome in outcomes if outcome.recovered]
        assert len(successful) == 1
        assert successful[0].repair == world.correct_repair
        failed = next(outcome for outcome in outcomes if not outcome.recovered)
        assert successful[0].total_return > failed.total_return
        assert successful[0].false_repairs == 0
        assert failed.false_repairs == 1


def test_common_isolation_intervention_produces_distinct_public_evidence() -> None:
    pair = build_manual_diagnostic_pair(seed=42)

    evidence = [dict(diagnostic_evidence(pair, world)) for world in pair.worlds]

    assert evidence[0] != evidence[1]
    blocked_evidence = next(item for item in evidence if item["output_buffer"] == 2.0)
    failed_evidence = next(item for item in evidence if item["output_buffer"] == 0.0)
    assert blocked_evidence["input_buffer"] == 0.0
    assert failed_evidence["input_buffer"] == 2.0


def test_evidence_contingent_policy_recovers_both_worlds_without_false_repairs() -> None:
    pair = build_manual_diagnostic_pair(seed=42)

    active = [run_contingent_active_policy(pair, world) for world in pair.worlds]

    assert all(result.recovered for result in active)
    assert [result.selected_repair for result in active] == [
        world.correct_repair for world in pair.worlds
    ]
    assert all(result.metrics.false_repair_count == 0 for result in active)


def test_active_policy_has_positive_value_gap_over_best_shared_repair() -> None:
    pair = build_manual_diagnostic_pair(seed=42)
    candidates = (ClearBlockage("delivery"), Replace("processor"))
    passive_values = [
        sum(evaluate_repair(pair, world, repair).total_return for world in pair.worlds)
        / len(pair.worlds)
        for repair in candidates
    ]
    active_value = sum(
        run_contingent_active_policy(pair, world).total_return for world in pair.worlds
    ) / len(pair.worlds)

    assert active_value > max(passive_values)
    assert sum(
        evaluate_repair(pair, world, candidates[0]).recovered for world in pair.worlds
    ) == 1
