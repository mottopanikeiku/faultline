"""Exact audit of cue-controlled high- and zero-EP task blocks."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean, median
from typing import Any

import numpy as np

from faultline.generation import (
    build_generated_diagnostic_pair,
    build_matched_ep_block,
    evaluate_matched_block,
)

CONTROL_AUDIT_VERSION = "matched-ep-control-v1"


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(median(values)),
        "mean": float(fmean(values)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
        "std_population": float(array.std(ddof=0)),
    }


def _standardized_mean_difference(left: list[float], right: list[float]) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    difference = float(left_array.mean() - right_array.mean())
    pooled_standard_deviation = float(
        np.sqrt(0.5 * (left_array.var(ddof=0) + right_array.var(ddof=0)))
    )
    if pooled_standard_deviation == 0.0:
        return 0.0 if difference == 0.0 else float("inf")
    return difference / pooled_standard_deviation


def analyze_matched_ep_controls(seeds: Sequence[int]) -> dict[str, Any]:
    """Compare ambiguous and revealed cue dependence on identical base factories."""
    if not seeds:
        raise ValueError("matched control audit requires at least one seed")
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        pair = build_generated_diagnostic_pair(seed)
        block = build_matched_ep_block(pair)
        value = evaluate_matched_block(block)
        rows.append(
            {
                "block_id": block.block_id,
                "seed": seed,
                **dict(pair.parameters),
                "ambiguous_passive_expected_return": (
                    value.ambiguous.passive_expected_return
                ),
                "ambiguous_active_expected_return": value.ambiguous.active_expected_return,
                "ambiguous_ep": value.ambiguous.epistemic_pressure,
                "ambiguous_passive_recovery": (
                    value.ambiguous.passive_recovery_probability
                ),
                "ambiguous_active_recovery": value.ambiguous.active_recovery_probability,
                "revealed_passive_expected_return": value.revealed.passive_expected_return,
                "revealed_active_expected_return": value.revealed.active_expected_return,
                "revealed_ep": value.revealed.epistemic_pressure,
                "revealed_passive_recovery": value.revealed.passive_recovery_probability,
                "revealed_active_recovery": value.revealed.active_recovery_probability,
                "fault_marginals_equal": value.fault_marginals_equal,
                "cue_marginals_equal": value.cue_marginals_equal,
                "cue_fault_joint_equal": value.cue_fault_joint_equal,
            }
        )

    nuisance_features = (
        "node_count",
        "fault_position",
        "rate",
        "transport_rate",
        "capacity_multiple",
        "preload",
        "time_cost",
        "diagnostic_cost",
    )
    standardized_mean_differences = {
        feature: _standardized_mean_difference(
            [float(row[feature]) for row in rows],
            [float(row[feature]) for row in rows],
        )
        for feature in nuisance_features
    }
    ambiguous_ep = [float(row["ambiguous_ep"]) for row in rows]
    revealed_ep = [float(row["revealed_ep"]) for row in rows]
    return {
        "analysis_version": CONTROL_AUDIT_VERSION,
        "count": len(rows),
        "seeds": list(seeds),
        "ambiguous_ep": _distribution(ambiguous_ep),
        "revealed_ep": _distribution(revealed_ep),
        "ambiguous_passive_recovery_mean": fmean(
            float(row["ambiguous_passive_recovery"]) for row in rows
        ),
        "ambiguous_active_recovery_mean": fmean(
            float(row["ambiguous_active_recovery"]) for row in rows
        ),
        "revealed_passive_recovery_mean": fmean(
            float(row["revealed_passive_recovery"]) for row in rows
        ),
        "revealed_active_recovery_mean": fmean(
            float(row["revealed_active_recovery"]) for row in rows
        ),
        "all_fault_marginals_equal": all(
            bool(row["fault_marginals_equal"]) for row in rows
        ),
        "all_cue_marginals_equal": all(bool(row["cue_marginals_equal"]) for row in rows),
        "all_cue_fault_joints_differ": all(
            not bool(row["cue_fault_joint_equal"]) for row in rows
        ),
        "nuisance_standardized_mean_differences": standardized_mean_differences,
        "max_abs_nuisance_standardized_mean_difference": max(
            map(abs, standardized_mean_differences.values())
        ),
        "rows": rows,
    }
