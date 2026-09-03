from __future__ import annotations

from pathlib import Path

import torch

from faultline.agents.recurrent import GraphRecurrentPolicy
from faultline.evaluation.policy import POLICY_EVALUATION_VERSION, evaluate_policy
from faultline.training.checkpoint import load_policy_checkpoint
from faultline.training.rl_env import DiagnosticAction


def test_paired_evaluation_exposes_passive_fixed_repair_ceiling() -> None:
    policy = GraphRecurrentPolicy(hidden_size=16, message_layers=1)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
        policy.actor.bias[DiagnosticAction.CLEAR_BLOCKAGE] = 1.0

    evaluation = evaluate_policy(policy, [1_000_000, 1_000_001])

    assert evaluation["evaluation_version"] == POLICY_EVALUATION_VERSION
    assert evaluation["base_pair_count"] == 2
    assert evaluation["ambiguous"]["episode_count"] == 8
    assert evaluation["revealed"]["episode_count"] == 4
    assert evaluation["ambiguous"]["recovery_rate"] == 0.5
    assert evaluation["ambiguous"]["correct_repair_rate"] == 0.5
    assert evaluation["ambiguous"]["experiment_then_correct_repair_rate"] == 0.0
    assert evaluation["revealed"]["recovery_rate"] == 0.5
    assert len(evaluation["rows"]) == 12


def test_probe_action_ablation_reduces_diagnostic_policy_to_passive_ceiling() -> None:
    repo = Path(__file__).parents[2]
    policy, _ = load_policy_checkpoint(
        repo / "artifacts/runs/ep-pilot-30k-masked-seed00-20260903/policy.pt"
    )

    evaluation = evaluate_policy(
        policy,
        [1_000_000, 1_000_001],
        disable_diagnostics=True,
    )

    assert evaluation["diagnostics_disabled"]
    assert evaluation["ambiguous"]["recovery_rate"] == 0.5
    assert evaluation["ambiguous"]["experiment_then_correct_repair_rate"] == 0.0
    assert evaluation["ambiguous"]["mean_advance_count"] == 0.0
    assert evaluation["ambiguous"]["mean_inspect_count"] == 0.0
