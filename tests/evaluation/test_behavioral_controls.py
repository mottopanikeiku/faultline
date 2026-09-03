from __future__ import annotations

from pathlib import Path

from faultline.evaluation.behavioral_controls import evaluate_behavioral_controls
from faultline.evaluation.behavioral_study import load_behavioral_protocol
from faultline.training.checkpoint import load_policy_checkpoint


def test_probe_ablation_cost_invariance_and_ood_execution() -> None:
    repo = Path(__file__).parents[2]
    policy, _ = load_policy_checkpoint(
        repo / "artifacts/runs/ep-pilot-30k-masked-seed00-20260903/policy.pt"
    )

    result = evaluate_behavioral_controls(
        policy,
        [1_000_000, 1_000_001],
        [0, 1],
    )

    assert not result["costs_observed_by_policy"]
    assert result["baseline"]["ambiguous"]["recovery_rate"] == 1.0
    assert result["probe_action_ablation"]["ambiguous"]["recovery_rate"] == 0.5
    assert result["probe_action_ablation"]["ambiguous_recovery_drop"] == 0.5
    assert {
        item["trace_change_rate"] for item in result["probe_cost_sensitivity"]
    } == {0.0}
    assert {
        item["trace_change_rate"] for item in result["repair_cost_sensitivity"]
    } == {0.0}
    assert result["probe_cost_sensitivity"][0]["ambiguous"]["mean_return"] != result[
        "probe_cost_sensitivity"
    ][-1]["ambiguous"]["mean_return"]
    assert result["ood"]["generator_version"] == "diagnostic-chain-ood-v1"
    assert result["ood"]["ambiguous"]["recovery_rate"] == 1.0


def test_behavioral_protocol_uses_development_ranges_only() -> None:
    repo = Path(__file__).parents[2]
    protocol = load_behavioral_protocol(
        repo / "configs/evaluation/behavioral-controls-v1.toml"
    )

    assert protocol.iid_split == "validation"
    assert protocol.iid_seeds == list(range(1_000_000, 1_000_032))
    assert protocol.ood_seeds == tuple(range(16))
    assert protocol.probe_cost_multipliers == (0.25, 1.0, 4.0)
