from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from faultline.agents.recurrent import GraphRecurrentPolicy
from faultline.env import Inspect
from faultline.evaluation.counterfactual import (
    evaluate_evidence_interventions,
    remove_diagnostic_observation,
    stale_diagnostic_observation,
    swap_diagnostic_observation,
)
from faultline.evaluation.counterfactual_study import load_counterfactual_protocol
from faultline.generation import (
    CueCondition,
    build_generated_diagnostic_pair,
    create_world_env,
)
from faultline.training.checkpoint import load_policy_checkpoint
from faultline.training.rl_env import (
    DiagnosticAction,
    DiagnosticEpisode,
    PolicyObservation,
)


def evidence_observations() -> tuple[PolicyObservation, PolicyObservation]:
    pair = build_generated_diagnostic_pair(88)
    observations = []
    for world_index in (0, 1):
        episode = DiagnosticEpisode(
            pair,
            CueCondition.AMBIGUOUS,
            world_index,
            cue=0,
        )
        episode.step(DiagnosticAction.ADVANCE)
        observation, _, _, _ = episode.step(DiagnosticAction.INSPECT)
        observations.append(observation)
    return observations[0], observations[1]


def test_swap_changes_only_revealed_telemetry_columns() -> None:
    left, right = evidence_observations()
    swapped = swap_diagnostic_observation(left, right)

    np.testing.assert_array_equal(swapped.nodes[:, :9], left.nodes[:, :9])
    np.testing.assert_array_equal(swapped.nodes[:, 9:12], right.nodes[:, 9:12])
    np.testing.assert_array_equal(swapped.adjacency, left.adjacency)
    np.testing.assert_array_equal(swapped.node_mask, left.node_mask)
    np.testing.assert_array_equal(swapped.global_features, left.global_features)
    np.testing.assert_array_equal(swapped.action_mask, left.action_mask)
    removed = remove_diagnostic_observation(left)
    assert not removed.nodes[:, 9:12].any()


def test_stale_evidence_matches_a_pre_dynamics_inspection_in_telemetry_units() -> None:
    pair = build_generated_diagnostic_pair(88)
    left, _ = evidence_observations()
    target = pair.graph.node_index[pair.evidence_node]
    assert pair.graph.input_capacities.max() > pair.graph.rates.max()
    before_dynamics = create_world_env(pair, pair.worlds[0]).act(Inspect(pair.evidence_node))

    stale = stale_diagnostic_observation(left, pair)

    np.testing.assert_array_equal(stale.nodes[:, :9], left.nodes[:, :9])
    assert stale.nodes[:, 9].sum() == stale.nodes[target, 9] == 1.0
    assert stale.nodes[target, 10] == np.float32(
        float(before_dynamics.observation["input_buffer"]) / pair.graph.rates.max()
    )
    assert not stale.nodes[:, 11].any()



def test_saved_diagnostic_policy_switches_to_swapped_world_repair() -> None:
    repo = Path(__file__).parents[2]
    policy, _ = load_policy_checkpoint(
        repo / "artifacts/runs/ep-pilot-30k-masked-seed00-20260903/policy.pt"
    )

    result = evaluate_evidence_interventions(
        policy,
        [1_000_000, 1_000_001],
        randomizations_per_world=4,
        random_seed=4,
    )

    assert result["diagnostic_eligibility_rate"] == 1.0
    assert result["normal_correct_repair_rate"] == 1.0
    assert result["swap_repair_switch_rate"] == 1.0
    assert result["swap_donor_repair_rate"] == 1.0
    assert result["causal_evidence_use_rate"] == 1.0


def test_immediate_repair_policy_is_not_misclassified_as_evidence_user() -> None:
    policy = GraphRecurrentPolicy(hidden_size=16, message_layers=1)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
        policy.actor.bias[DiagnosticAction.CLEAR_BLOCKAGE] = 1.0

    result = evaluate_evidence_interventions(policy, [1_000_000])

    assert result["diagnostic_eligibility_rate"] == 0.0
    assert result["causal_evidence_use_rate"] is None
    assert result["swap_repair_switch_rate"] is None


def test_counterfactual_protocol_keeps_test_split_sealed() -> None:
    repo = Path(__file__).parents[2]
    protocol = load_counterfactual_protocol(
        repo / "configs/evaluation/counterfactual-v1.toml"
    )

    assert protocol.evaluation_split == "validation"
    assert protocol.seeds == list(range(1_000_000, 1_000_032))
    assert protocol.primary_metric == "overall_causal_evidence_use_rate"
