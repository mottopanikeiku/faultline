from __future__ import annotations

import numpy as np
import pytest

from faultline.faults import BlockedEdge
from faultline.generation import CueCondition, build_generated_diagnostic_pair
from faultline.training.rl_env import DiagnosticAction, DiagnosticEpisode


def test_ambiguous_worlds_have_identical_initial_policy_observation() -> None:
    pair = build_generated_diagnostic_pair(12)
    left = DiagnosticEpisode(pair, CueCondition.AMBIGUOUS, world_index=0, cue=1)
    right = DiagnosticEpisode(pair, CueCondition.AMBIGUOUS, world_index=1, cue=1)

    left_observation = left.observe()
    right_observation = right.observe()

    np.testing.assert_array_equal(left_observation.nodes, right_observation.nodes)
    np.testing.assert_array_equal(left_observation.adjacency, right_observation.adjacency)
    np.testing.assert_array_equal(left_observation.node_mask, right_observation.node_mask)
    np.testing.assert_array_equal(
        left_observation.global_features,
        right_observation.global_features,
    )
    assert not left_observation.nodes[:, 9].any()
    np.testing.assert_array_equal(
        left_observation.action_mask,
        np.asarray([True, False, True, True]),
    )


def test_advance_inspect_repair_recovers_each_latent_world() -> None:
    pair = build_generated_diagnostic_pair(31)
    for world_index, world in enumerate(pair.worlds):
        episode = DiagnosticEpisode(
            pair,
            CueCondition.AMBIGUOUS,
            world_index=world_index,
            cue=0,
        )
        _, first_reward, done, _ = episode.step(DiagnosticAction.ADVANCE)
        assert not done
        observation, inspect_reward, done, _ = episode.step(DiagnosticAction.INSPECT)
        assert not done
        target = pair.graph.node_index[pair.evidence_node]
        action = (
            DiagnosticAction.CLEAR_BLOCKAGE
            if observation.nodes[target, 11] > 0.0
            else DiagnosticAction.REPLACE_PROCESSOR
        )
        _, repair_reward, done, summary = episode.step(action)

        assert done
        assert summary is not None
        assert summary.recovered
        assert summary.correct_repair
        assert summary.informative_inspection
        assert summary.advance_count == 1
        assert summary.inspect_count == 1
        assert summary.false_repair_count == 0
        assert first_reward + inspect_reward + repair_reward == pytest.approx(
            summary.total_reward
        )
        assert isinstance(world.fault, BlockedEdge) is (
            action is DiagnosticAction.CLEAR_BLOCKAGE
        )


def test_immediate_wrong_repair_commits_and_cannot_be_retried() -> None:
    pair = build_generated_diagnostic_pair(4)
    blocked_index = next(
        index for index, world in enumerate(pair.worlds) if isinstance(world.fault, BlockedEdge)
    )
    episode = DiagnosticEpisode(
        pair,
        CueCondition.AMBIGUOUS,
        world_index=blocked_index,
        cue=0,
    )

    _, _, done, summary = episode.step(DiagnosticAction.REPLACE_PROCESSOR)

    assert done
    assert summary is not None
    assert not summary.recovered
    assert not summary.correct_repair
    assert summary.false_repair_count == 1
    with pytest.raises(RuntimeError, match="already terminated"):
        episode.step(DiagnosticAction.CLEAR_BLOCKAGE)


def test_revealed_condition_balances_fault_marginals_but_provides_correct_cue() -> None:
    pair = build_generated_diagnostic_pair(55)
    rng = np.random.default_rng(9)
    episodes = [
        DiagnosticEpisode.sample(pair, CueCondition.REVEALED, rng) for _ in range(200)
    ]

    assert {episode.cue for episode in episodes} == {0, 1}
    for episode in episodes:
        blocked = isinstance(pair.worlds[episode.world_index].fault, BlockedEdge)
        assert episode.cue == (0 if blocked else 1)


def test_decision_step_limit_forces_operational_timeout() -> None:
    episode = DiagnosticEpisode(
        build_generated_diagnostic_pair(8),
        CueCondition.AMBIGUOUS,
        world_index=0,
        cue=0,
        max_decision_steps=2,
    )

    after_advance, _, done, _ = episode.step(DiagnosticAction.ADVANCE)
    assert not done
    np.testing.assert_array_equal(
        after_advance.action_mask,
        np.asarray([False, True, True, True]),
    )
    after_inspect, _, done, summary = episode.step(DiagnosticAction.INSPECT)
    np.testing.assert_array_equal(
        after_inspect.action_mask,
        np.asarray([False, False, False, False]),
    )

    assert done
    assert summary is not None
    assert not summary.recovered
    assert summary.selected_repair is None
