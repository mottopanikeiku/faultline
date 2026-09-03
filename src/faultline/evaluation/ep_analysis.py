"""Empirical epistemic-pressure and intervention-value analysis."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean, median
from typing import Any

import numpy as np
from numpy.typing import NDArray

from faultline.env import Advance, Inspect
from faultline.generation import build_generated_diagnostic_pair, validate_diagnostic_pair
from faultline.oracle import (
    action_partitions,
    analyze_action_at_belief,
    diagnostic_pair_problem,
)
from faultline.oracle.passive import root_branches

EP_ANALYSIS_VERSION = "ep-distribution-v1"
FloatArray = NDArray[np.float64]


def _ranks(values: FloatArray) -> FloatArray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _pearson(left: FloatArray, right: FloatArray) -> float | None:
    if left.size != right.size or left.size < 2:
        raise ValueError("correlation inputs must be aligned with at least two values")
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(
        np.sqrt(
            np.dot(left_centered, left_centered)
            * np.dot(right_centered, right_centered)
        )
    )
    if denominator == 0.0:
        return None
    return float(np.dot(left_centered, right_centered) / denominator)


def _distribution(values: Sequence[float]) -> dict[str, float]:
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


def analyze_ep_distribution(seeds: Sequence[int]) -> dict[str, Any]:
    """Regenerate and exactly analyze a fixed ordered set of pair seeds."""
    if not seeds:
        raise ValueError("EP analysis requires at least one seed")
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        pair = build_generated_diagnostic_pair(seed)
        validation = validate_diagnostic_pair(pair)
        if not validation.valid:
            raise RuntimeError(f"pair {pair.pair_id} failed current validation")
        problem = diagnostic_pair_problem(pair)
        advance_action = next(
            action for action in problem.diagnostic_actions if isinstance(action, Advance)
        )
        inspect_action = next(
            action for action in problem.diagnostic_actions if isinstance(action, Inspect)
        )
        roots = root_branches(problem)
        immediate_advance = analyze_action_at_belief(problem, roots, advance_action)
        after_advance = action_partitions(problem, roots, advance_action)
        if len(after_advance) != 1:
            raise AssertionError("generated pair leaks its latent world through advance status")
        diagnostic_inspect = analyze_action_at_belief(
            problem,
            after_advance[0].branches,
            inspect_action,
        )
        parameters = dict(pair.parameters)
        rows.append(
            {
                "pair_id": pair.pair_id,
                "seed": seed,
                **parameters,
                "passive_expected_return": validation.passive_expected_return,
                "active_expected_return": validation.active_expected_return,
                "passive_difficulty": -validation.passive_expected_return,
                "epistemic_pressure": validation.epistemic_pressure,
                "repair_margin": validation.repair_margin,
                "immediate_advance_decision_value": immediate_advance.decision_value,
                "immediate_advance_information_gain_bits": (
                    immediate_advance.information_gain_bits
                ),
                "post_advance_inspect_decision_value": diagnostic_inspect.decision_value,
                "post_advance_inspect_information_gain_bits": (
                    diagnostic_inspect.information_gain_bits
                ),
            }
        )

    pressures = np.asarray([row["epistemic_pressure"] for row in rows], dtype=np.float64)
    correlation_features = (
        "node_count",
        "fault_position",
        "rate",
        "transport_rate",
        "capacity_multiple",
        "preload",
        "time_cost",
        "diagnostic_cost",
        "passive_difficulty",
        "repair_margin",
    )
    correlations: dict[str, dict[str, float | None]] = {}
    pressure_ranks = _ranks(pressures)
    for feature in correlation_features:
        values = np.asarray([row[feature] for row in rows], dtype=np.float64)
        correlations[feature] = {
            "pearson": _pearson(values, pressures),
            "spearman": _pearson(_ranks(values), pressure_ranks),
        }

    return {
        "analysis_version": EP_ANALYSIS_VERSION,
        "count": len(rows),
        "seeds": list(seeds),
        "epistemic_pressure": _distribution(
            [float(row["epistemic_pressure"]) for row in rows]
        ),
        "passive_expected_return": _distribution(
            [float(row["passive_expected_return"]) for row in rows]
        ),
        "active_expected_return": _distribution(
            [float(row["active_expected_return"]) for row in rows]
        ),
        "immediate_advance_decision_value": _distribution(
            [float(row["immediate_advance_decision_value"]) for row in rows]
        ),
        "post_advance_inspect_decision_value": _distribution(
            [float(row["post_advance_inspect_decision_value"]) for row in rows]
        ),
        "immediate_advance_information_gain_bits": _distribution(
            [float(row["immediate_advance_information_gain_bits"]) for row in rows]
        ),
        "post_advance_inspect_information_gain_bits": _distribution(
            [float(row["post_advance_inspect_information_gain_bits"]) for row in rows]
        ),
        "correlations_with_ep": correlations,
        "rows": rows,
    }
