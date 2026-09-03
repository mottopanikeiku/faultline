"""Counterfactual evidence-use study across frozen small-policy checkpoints."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from faultline.evaluation.counterfactual import (
    COUNTERFACTUAL_VERSION,
    evaluate_evidence_interventions,
)
from faultline.evaluation.statistics import bootstrap_mean, paired_bootstrap_difference
from faultline.evaluation.study import KillTestProtocol, load_kill_test_protocol
from faultline.generation import get_split
from faultline.training.checkpoint import load_policy_checkpoint

COUNTERFACTUAL_STUDY_VERSION = "counterfactual-study-v1"


@dataclass(frozen=True, slots=True)
class CounterfactualProtocol:
    protocol_id: str
    source_study_protocol: str
    evaluation_split: str
    evaluation_offset: int
    base_pair_count: int
    randomizations_per_world: int
    random_seed: int
    interventions: tuple[str, ...]
    primary_metric: str

    @property
    def seeds(self) -> list[int]:
        split = get_split(self.evaluation_split)
        start = split.seed_start + self.evaluation_offset
        return list(range(start, start + self.base_pair_count))


def load_counterfactual_protocol(path: Path) -> CounterfactualProtocol:
    with path.open("rb") as source:
        raw: dict[str, Any] = tomllib.load(source)
    protocol = CounterfactualProtocol(
        protocol_id=str(raw["protocol_id"]),
        source_study_protocol=str(raw["source_study_protocol"]),
        evaluation_split=str(raw["evaluation_split"]),
        evaluation_offset=int(raw["evaluation_offset"]),
        base_pair_count=int(raw["base_pair_count"]),
        randomizations_per_world=int(raw["randomizations_per_world"]),
        random_seed=int(raw["random_seed"]),
        interventions=tuple(str(value) for value in raw["interventions"]),
        primary_metric=str(raw["primary_metric"]),
    )
    if protocol.evaluation_split == "test":
        raise ValueError("counterfactual development study cannot use sealed test split")
    if protocol.base_pair_count <= 0 or protocol.randomizations_per_world <= 0:
        raise ValueError("counterfactual counts must be positive")
    return protocol


def _run_id(study: KillTestProtocol, arm: str, seed: int) -> str:
    return study.run_id_template.format(arm=arm, seed=seed)


def analyze_counterfactual_study(
    repo: Path,
    protocol: CounterfactualProtocol,
) -> dict[str, Any]:
    source_path = repo / protocol.source_study_protocol
    source_study = load_kill_test_protocol(source_path)
    evaluation_seeds = protocol.seeds
    run_results: list[dict[str, Any]] = []
    metric_names = (
        "diagnostic_eligibility_rate",
        "normal_correct_repair_rate",
        "swap_repair_switch_rate",
        "swap_donor_repair_rate",
        "causal_evidence_use_rate",
        "overall_causal_evidence_use_rate",
        "removal_change_rate",
        "stale_change_rate",
        "uninformative_change_rate",
        "randomized_correct_rate",
    )

    for arm in source_study.arms:
        for seed in source_study.training_seeds:
            run_id = _run_id(source_study, arm, seed)
            manifest = json.loads(
                (repo / "artifacts/manifests" / f"{run_id}.json").read_text(
                    encoding="utf-8"
                )
            )
            checkpoint = repo / manifest["metrics"]["checkpoint_path"]
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if digest != manifest["metrics"]["checkpoint_sha256"]:
                raise ValueError(f"checkpoint hash mismatch for {run_id}")
            policy, _ = load_policy_checkpoint(checkpoint)
            evaluation = evaluate_evidence_interventions(
                policy,
                evaluation_seeds,
                randomizations_per_world=protocol.randomizations_per_world,
                random_seed=protocol.random_seed,
            )
            conditional = evaluation["causal_evidence_use_rate"]
            evaluation["overall_causal_evidence_use_rate"] = evaluation[
                "diagnostic_eligibility_rate"
            ] * (float(conditional) if conditional is not None else 0.0)
            run_results.append(
                {
                    "run_id": run_id,
                    "arm": arm,
                    "training_seed": seed,
                    "checkpoint_sha256": digest,
                    "evaluation": evaluation,
                }
            )

    arms: dict[str, Any] = {}
    primary_by_arm: dict[str, list[float]] = {}
    for arm_index, arm in enumerate(source_study.arms):
        runs = [run for run in run_results if run["arm"] == arm]
        per_metric: dict[str, Any] = {}
        for metric_index, metric in enumerate(metric_names):
            values = [
                float(run["evaluation"][metric])
                for run in runs
                if run["evaluation"][metric] is not None
            ]
            per_metric[metric] = {
                "mean": fmean(values) if values else None,
                "available_training_seeds": len(values),
            }
            if values:
                per_metric[metric]["bootstrap"] = asdict(
                    bootstrap_mean(
                        values,
                        confidence_level=source_study.confidence_level,
                        resamples=source_study.bootstrap_resamples,
                        seed=(
                            source_study.bootstrap_seed
                            + 1_000
                            + arm_index * len(metric_names)
                            + metric_index
                        ),
                    )
                )
        primary = [
            float(run["evaluation"][protocol.primary_metric]) for run in runs
        ]
        primary_by_arm[arm] = primary
        arms[arm] = {
            "metrics": per_metric,
            "individual_training_seeds": [
                {
                    "seed": run["training_seed"],
                    "value": run["evaluation"][protocol.primary_metric],
                    "eligible": run["evaluation"]["diagnostic_eligibility_rate"],
                    "conditional_causal_use": run["evaluation"][
                        "causal_evidence_use_rate"
                    ],
                }
                for run in runs
            ],
        }

    comparisons = {
        f"epistemic-{baseline}": asdict(
            paired_bootstrap_difference(
                primary_by_arm["epistemic"],
                primary_by_arm[baseline],
                confidence_level=source_study.confidence_level,
                resamples=source_study.bootstrap_resamples,
                seed=source_study.bootstrap_seed + 2_000 + index,
            )
        )
        for index, baseline in enumerate(("random", "difficulty"))
    }
    return {
        "analysis_version": COUNTERFACTUAL_STUDY_VERSION,
        "evaluator_version": COUNTERFACTUAL_VERSION,
        "protocol": asdict(protocol),
        "source_study_protocol": asdict(source_study),
        "evaluation_seeds": evaluation_seeds,
        "arms": arms,
        "paired_primary_comparisons": comparisons,
        "runs": run_results,
    }
