"""Probe ablation, cost sensitivity, and development OOD policy controls."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import Any

from faultline.agents.recurrent import GraphRecurrentPolicy
from faultline.artifacts import canonical_sha256
from faultline.env import RewardConfig
from faultline.evaluation.policy import evaluate_policy
from faultline.generation import DiagnosticPair, build_ood_diagnostic_pair

BEHAVIORAL_CONTROLS_VERSION = "behavioral-controls-v1"


def _trace_records(evaluation: dict[str, Any]) -> list[dict[str, object]]:
    return [
        {
            "pair_seed": row["pair_seed"],
            "condition": row["condition"],
            "world_index": row["world_index"],
            "cue": row["cue"],
            "actions": row["actions"],
        }
        for row in evaluation["rows"]
    ]


def _trace_change_rate(
    baseline: dict[str, Any],
    intervention: dict[str, Any],
) -> float:
    baseline_rows = _trace_records(baseline)
    intervention_rows = _trace_records(intervention)
    if len(baseline_rows) != len(intervention_rows):
        raise ValueError("behavioral evaluations are not aligned")
    changed = sum(
        left["actions"] != right["actions"]
        for left, right in zip(baseline_rows, intervention_rows, strict=True)
    )
    return changed / len(baseline_rows)


def _scale_probe_cost(pair: DiagnosticPair, multiplier: float) -> DiagnosticPair:
    reward = pair.reward
    return replace(
        pair,
        reward=replace(
            reward,
            time_cost=reward.time_cost * multiplier,
            passive_cost=reward.passive_cost * multiplier,
            diagnostic_cost=reward.diagnostic_cost * multiplier,
        ),
    )


def _scale_repair_cost(pair: DiagnosticPair, multiplier: float) -> DiagnosticPair:
    reward: RewardConfig = pair.reward
    return replace(
        pair,
        reward=replace(
            reward,
            repair_cost=reward.repair_cost * multiplier,
            false_repair_cost=reward.false_repair_cost * multiplier,
        ),
    )


def evaluate_behavioral_controls(
    policy: GraphRecurrentPolicy,
    iid_seeds: list[int],
    ood_seeds: list[int],
    *,
    probe_cost_multipliers: tuple[float, ...] = (0.25, 1.0, 4.0),
    repair_cost_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0),
) -> dict[str, Any]:
    """Evaluate controls without changing latent state or retraining the policy."""
    if not iid_seeds or not ood_seeds:
        raise ValueError("behavioral controls require IID and OOD seeds")
    if any(multiplier <= 0.0 for multiplier in (*probe_cost_multipliers, *repair_cost_multipliers)):
        raise ValueError("cost multipliers must be positive")

    baseline = evaluate_policy(policy, iid_seeds)
    ablated = evaluate_policy(policy, iid_seeds, disable_diagnostics=True)
    probe_costs: list[dict[str, Any]] = []
    for multiplier in probe_cost_multipliers:
        evaluation = evaluate_policy(
            policy,
            iid_seeds,
            pair_transform=partial(_scale_probe_cost, multiplier=multiplier),
        )
        probe_costs.append(
            {
                "multiplier": multiplier,
                "trace_change_rate": _trace_change_rate(baseline, evaluation),
                "trace_sha256": canonical_sha256(_trace_records(evaluation)),
                "ambiguous": evaluation["ambiguous"],
                "revealed": evaluation["revealed"],
            }
        )

    repair_costs: list[dict[str, Any]] = []
    for multiplier in repair_cost_multipliers:
        evaluation = evaluate_policy(
            policy,
            iid_seeds,
            pair_transform=partial(_scale_repair_cost, multiplier=multiplier),
        )
        repair_costs.append(
            {
                "multiplier": multiplier,
                "trace_change_rate": _trace_change_rate(baseline, evaluation),
                "trace_sha256": canonical_sha256(_trace_records(evaluation)),
                "ambiguous": evaluation["ambiguous"],
                "revealed": evaluation["revealed"],
            }
        )

    ood = evaluate_policy(
        policy,
        ood_seeds,
        pair_builder=build_ood_diagnostic_pair,
    )
    return {
        "evaluation_version": BEHAVIORAL_CONTROLS_VERSION,
        "iid_base_pair_count": len(iid_seeds),
        "ood_base_pair_count": len(ood_seeds),
        "baseline": {
            "trace_sha256": canonical_sha256(_trace_records(baseline)),
            "ambiguous": baseline["ambiguous"],
            "revealed": baseline["revealed"],
        },
        "probe_action_ablation": {
            "trace_sha256": canonical_sha256(_trace_records(ablated)),
            "ambiguous": ablated["ambiguous"],
            "revealed": ablated["revealed"],
            "ambiguous_recovery_drop": (
                float(baseline["ambiguous"]["recovery_rate"])
                - float(ablated["ambiguous"]["recovery_rate"])
            ),
        },
        "probe_cost_sensitivity": probe_costs,
        "repair_cost_sensitivity": repair_costs,
        "ood": {
            "generator_version": "diagnostic-chain-ood-v1",
            "seeds": ood_seeds,
            "trace_sha256": canonical_sha256(_trace_records(ood)),
            "ambiguous": ood["ambiguous"],
            "revealed": ood["revealed"],
        },
        "costs_observed_by_policy": False,
    }
