from __future__ import annotations

import pytest

from faultline.env import Advance, ClearBlockage, Inspect, Replace
from faultline.generation import build_manual_diagnostic_pair, create_world_env
from faultline.oracle import (
    DecisionKind,
    action_partitions,
    analyze_actions,
    bayes_update,
    best_shared_decision_mass,
    diagnostic_pair_problem,
    entropy_bits,
    solve_active,
    solve_passive,
)
from faultline.oracle.passive import root_branches


def test_exact_bayesian_belief_update_and_passive_ceiling() -> None:
    assert bayes_update((0.5, 0.5), (1.0, 0.0)) == (1.0, 0.0)
    assert bayes_update((0.25, 0.75), (0.5, 0.5)) == (0.25, 0.75)
    assert entropy_bits((0.5, 0.5)) == 1.0
    assert entropy_bits((1.0, 0.0)) == 0.0
    assert best_shared_decision_mass(
        (0.5, 0.5),
        (frozenset({0}), frozenset({1})),
    ) == 0.5


def test_passive_oracle_matches_exhaustive_shared_repair_values() -> None:
    pair = build_manual_diagnostic_pair(42)
    problem = diagnostic_pair_problem(pair)

    passive = solve_passive(problem)

    assert passive.expected_return == pytest.approx(1.25)
    assert passive.recovery_probability == 0.5
    assert passive.plan.name == "clear_delivery"
    assert passive.world_recovered == (True, False)


def test_advance_then_inspect_performs_exact_belief_separation() -> None:
    pair = build_manual_diagnostic_pair(42)
    problem = diagnostic_pair_problem(pair)
    root = root_branches(problem)

    after_advance = action_partitions(problem, root, Advance(1))
    assert len(after_advance) == 1
    assert after_advance[0].posterior == (0.5, 0.5)

    after_inspect = action_partitions(problem, after_advance[0].branches, Inspect("processor"))
    assert len(after_inspect) == 2
    assert {partition.posterior for partition in after_inspect} == {(1.0, 0.0), (0.0, 1.0)}
    assert sum(partition.probability for partition in after_inspect) == 1.0


def test_active_solver_finds_optimal_two_action_diagnostic_policy() -> None:
    pair = build_manual_diagnostic_pair(42)
    problem = diagnostic_pair_problem(pair)
    passive = solve_passive(problem)

    depth_zero = solve_active(problem, 0)
    depth_one = solve_active(problem, 1)
    depth_two = solve_active(problem, 2)
    depth_three = solve_active(problem, 3)

    assert depth_zero.expected_return == passive.expected_return
    assert depth_one.expected_return == passive.expected_return
    assert depth_two.kind is DecisionKind.DIAGNOSE
    assert depth_two.diagnostic_action == Advance(1)
    assert len(depth_two.outcomes) == 1
    assert depth_two.outcomes[0].decision.diagnostic_action == Inspect("processor")
    assert depth_two.recovery_probability == 1.0
    assert depth_two.expected_return == pytest.approx(8.58)
    assert depth_three.expected_return == depth_two.expected_return
    assert depth_two.expected_return - passive.expected_return == pytest.approx(7.33)


def test_solver_value_matches_direct_enumeration_of_discovered_policy() -> None:
    pair = build_manual_diagnostic_pair(42)
    returns: list[float] = []
    for world in pair.worlds:
        env = create_world_env(pair, world, with_reward=True)
        env.act(Advance(1))
        evidence = env.act(Inspect("processor"))
        repair = (
            ClearBlockage("delivery")
            if float(evidence.observation["output_buffer"]) > 0.0
            else Replace("processor")
        )
        env.act(repair)
        env.act(Advance(pair.reward.max_ticks))
        assert env.reward_tracker is not None
        assert env.reward_tracker.recovered
        returns.append(env.reward_tracker.total_reward)

    oracle = solve_active(diagnostic_pair_problem(pair), 2)
    assert sum(returns) / len(returns) == pytest.approx(oracle.expected_return)


def test_single_root_actions_have_no_information_before_dynamics() -> None:
    problem = diagnostic_pair_problem(build_manual_diagnostic_pair(42))

    analyses = analyze_actions(problem)

    assert all(analysis.information_gain_bits == 0.0 for analysis in analyses)
    assert all(analysis.outcome_count == 1 for analysis in analyses)
    assert max(analysis.expected_return for analysis in analyses) <= solve_passive(
        problem
    ).expected_return
