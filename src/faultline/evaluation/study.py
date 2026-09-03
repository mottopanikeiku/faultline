"""Frozen multi-arm small-policy kill-test analysis."""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from faultline.evaluation.statistics import bootstrap_mean, paired_bootstrap_difference

KILL_TEST_ANALYSIS_VERSION = "small-kill-analysis-v1"


@dataclass(frozen=True, slots=True)
class KillTestProtocol:
    protocol_id: str
    status: str
    training_config: str
    training_decision_steps: int
    arms: tuple[str, ...]
    training_seeds: tuple[int, ...]
    run_id_template: str
    evaluation_split: str
    evaluation_offset: int
    evaluation_base_pair_count: int
    primary_metric: str
    primary_comparisons: tuple[str, ...]
    minimum_relevant_effect: float
    confidence_level: float
    bootstrap_resamples: int
    bootstrap_seed: int
    saturation_threshold: float
    secondary_metrics: tuple[str, ...]


def load_kill_test_protocol(path: Path) -> KillTestProtocol:
    with path.open("rb") as source:
        raw: dict[str, Any] = tomllib.load(source)
    interpretation = raw["interpretation"]
    secondary = raw["secondary_metrics"]
    protocol = KillTestProtocol(
        protocol_id=str(raw["protocol_id"]),
        status=str(raw["status"]),
        training_config=str(raw["training_config"]),
        training_decision_steps=int(raw["training_decision_steps"]),
        arms=tuple(str(value) for value in raw["arms"]),
        training_seeds=tuple(int(value) for value in raw["training_seeds"]),
        run_id_template=str(raw["run_id_template"]),
        evaluation_split=str(raw["evaluation_split"]),
        evaluation_offset=int(raw["evaluation_offset"]),
        evaluation_base_pair_count=int(raw["evaluation_base_pair_count"]),
        primary_metric=str(raw["primary_metric"]),
        primary_comparisons=tuple(str(value) for value in raw["primary_comparisons"]),
        minimum_relevant_effect=float(raw["minimum_relevant_effect"]),
        confidence_level=float(raw["confidence_level"]),
        bootstrap_resamples=int(raw["bootstrap_resamples"]),
        bootstrap_seed=int(raw["bootstrap_seed"]),
        saturation_threshold=float(interpretation["saturation_threshold"]),
        secondary_metrics=tuple(str(value) for value in secondary["metrics"]),
    )
    if len(protocol.arms) < 2 or len(protocol.training_seeds) < 2:
        raise ValueError("kill test requires multiple arms and training seeds")
    if len(set(protocol.arms)) != len(protocol.arms):
        raise ValueError("kill-test arms must be unique")
    if len(set(protocol.training_seeds)) != len(protocol.training_seeds):
        raise ValueError("kill-test training seeds must be unique")
    return protocol


def _metric(result: dict[str, Any], path: str) -> float:
    value: Any = result["evaluation"]
    for part in path.split("."):
        value = value[part]
    return float(value)


def _run_id(protocol: KillTestProtocol, arm: str, seed: int) -> str:
    return protocol.run_id_template.format(arm=arm, seed=seed)


def analyze_kill_test(repo: Path, protocol: KillTestProtocol) -> dict[str, Any]:
    """Load every preregistered run; missing or mismatched runs fail the analysis."""
    results: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in protocol.arms}
    source_runs: list[dict[str, Any]] = []
    for arm in protocol.arms:
        for seed in protocol.training_seeds:
            run_id = _run_id(protocol, arm, seed)
            result_path = repo / "artifacts" / "results" / f"{run_id}.json"
            manifest_path = repo / "artifacts" / "manifests" / f"{run_id}.json"
            if not result_path.is_file() or not manifest_path.is_file():
                raise FileNotFoundError(f"missing preregistered run {run_id}")
            result: dict[str, Any] = json.loads(result_path.read_text(encoding="utf-8"))
            manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
            if result["training_seed"] != seed or result["curriculum"] != arm:
                raise ValueError(f"run identity mismatch for {run_id}")
            if result["resolved_config"]["ppo"]["total_decision_steps"] != (
                protocol.training_decision_steps
            ):
                raise ValueError(f"training budget mismatch for {run_id}")
            evaluation_config = result["resolved_config"]["evaluation"]
            expected_evaluation = {
                "split": protocol.evaluation_split,
                "offset": protocol.evaluation_offset,
                "base_pair_count": protocol.evaluation_base_pair_count,
            }
            if evaluation_config != expected_evaluation:
                raise ValueError(f"evaluation protocol mismatch for {run_id}")
            if manifest["git_dirty"]:
                raise ValueError(f"dirty source run {run_id}")
            results[arm][seed] = result
            source_runs.append(
                {
                    "run_id": run_id,
                    "arm": arm,
                    "seed": seed,
                    "git_commit": manifest["git_commit"],
                    "result_sha256": manifest["metrics"]["result_sha256"],
                    "checkpoint_sha256": manifest["metrics"]["checkpoint_sha256"],
                    "actual_decision_steps": result["decision_steps"],
                }
            )

    arm_summaries: dict[str, Any] = {}
    primary_by_arm: dict[str, list[float]] = {}
    for arm_index, arm in enumerate(protocol.arms):
        ordered = [results[arm][seed] for seed in protocol.training_seeds]
        primary = [_metric(result, protocol.primary_metric) for result in ordered]
        primary_by_arm[arm] = primary
        arm_summaries[arm] = {
            "primary": asdict(
                bootstrap_mean(
                    primary,
                    confidence_level=protocol.confidence_level,
                    resamples=protocol.bootstrap_resamples,
                    seed=protocol.bootstrap_seed + arm_index,
                )
            ),
            "individual_seeds": [
                {"seed": seed, "value": value}
                for seed, value in zip(protocol.training_seeds, primary, strict=True)
            ],
            "secondary_means": {
                metric: fmean(_metric(result, metric) for result in ordered)
                for metric in protocol.secondary_metrics
            },
        }

    comparisons: dict[str, Any] = {}
    for comparison_index, comparison in enumerate(protocol.primary_comparisons):
        left, separator, right = comparison.partition("-")
        if not separator or left not in primary_by_arm or right not in primary_by_arm:
            raise ValueError(f"invalid primary comparison {comparison!r}")
        comparisons[comparison] = asdict(
            paired_bootstrap_difference(
                primary_by_arm[left],
                primary_by_arm[right],
                confidence_level=protocol.confidence_level,
                resamples=protocol.bootstrap_resamples,
                seed=protocol.bootstrap_seed + 100 + comparison_index,
            )
        )

    epistemic_comparisons = [
        value
        for name, value in comparisons.items()
        if name.startswith("epistemic-")
    ]
    supports_effect = bool(epistemic_comparisons) and all(
        value["estimate"] >= protocol.minimum_relevant_effect and value["lower"] > 0.0
        for value in epistemic_comparisons
    )
    saturated = all(
        summary["primary"]["estimate"] >= protocol.saturation_threshold
        for summary in arm_summaries.values()
    )
    if supports_effect:
        decision = "supports_curriculum_specific_effect"
    elif saturated:
        decision = "benchmark_saturated_no_curriculum_separation"
    else:
        decision = "no_preregistered_curriculum_effect"

    return {
        "analysis_version": KILL_TEST_ANALYSIS_VERSION,
        "protocol": asdict(protocol),
        "primary_metric": protocol.primary_metric,
        "arms": arm_summaries,
        "paired_comparisons": comparisons,
        "minimum_relevant_effect": protocol.minimum_relevant_effect,
        "supports_curriculum_specific_effect": supports_effect,
        "all_arms_saturated": saturated,
        "decision": decision,
        "source_runs": source_runs,
    }
