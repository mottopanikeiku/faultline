"""Aggregate probe-ablation, cost, and OOD controls over frozen checkpoints."""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from faultline.evaluation.behavioral_controls import (
    BEHAVIORAL_CONTROLS_VERSION,
    evaluate_behavioral_controls,
)
from faultline.evaluation.statistics import bootstrap_mean
from faultline.evaluation.study import load_kill_test_protocol
from faultline.generation import get_split
from faultline.training.checkpoint import load_policy_checkpoint

BEHAVIORAL_STUDY_VERSION = "behavioral-control-study-v1"


@dataclass(frozen=True, slots=True)
class BehavioralProtocol:
    protocol_id: str
    source_study_protocol: str
    iid_split: str
    iid_offset: int
    iid_base_pair_count: int
    ood_generator_version: str
    ood_seeds: tuple[int, ...]
    probe_cost_multipliers: tuple[float, ...]
    repair_cost_multipliers: tuple[float, ...]

    @property
    def iid_seeds(self) -> list[int]:
        split = get_split(self.iid_split)
        start = split.seed_start + self.iid_offset
        return list(range(start, start + self.iid_base_pair_count))


def load_behavioral_protocol(path: Path) -> BehavioralProtocol:
    with path.open("rb") as source:
        raw: dict[str, Any] = tomllib.load(source)
    protocol = BehavioralProtocol(
        protocol_id=str(raw["protocol_id"]),
        source_study_protocol=str(raw["source_study_protocol"]),
        iid_split=str(raw["iid_split"]),
        iid_offset=int(raw["iid_offset"]),
        iid_base_pair_count=int(raw["iid_base_pair_count"]),
        ood_generator_version=str(raw["ood_generator_version"]),
        ood_seeds=tuple(int(value) for value in raw["ood_seeds"]),
        probe_cost_multipliers=tuple(float(value) for value in raw["probe_cost_multipliers"]),
        repair_cost_multipliers=tuple(float(value) for value in raw["repair_cost_multipliers"]),
    )
    if protocol.iid_split == "test":
        raise ValueError("behavioral development controls cannot use sealed test split")
    if not protocol.ood_seeds or protocol.iid_base_pair_count <= 0:
        raise ValueError("behavioral protocol requires IID and OOD tasks")
    return protocol


def analyze_behavioral_study(repo: Path, protocol: BehavioralProtocol) -> dict[str, Any]:
    source_study = load_kill_test_protocol(repo / protocol.source_study_protocol)
    runs: list[dict[str, Any]] = []
    for arm in source_study.arms:
        for training_seed in source_study.training_seeds:
            run_id = source_study.run_id_template.format(arm=arm, seed=training_seed)
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
            evaluation = evaluate_behavioral_controls(
                policy,
                protocol.iid_seeds,
                list(protocol.ood_seeds),
                probe_cost_multipliers=protocol.probe_cost_multipliers,
                repair_cost_multipliers=protocol.repair_cost_multipliers,
            )
            runs.append(
                {
                    "run_id": run_id,
                    "arm": arm,
                    "training_seed": training_seed,
                    "checkpoint_sha256": digest,
                    "evaluation": evaluation,
                }
            )

    metric_extractors: dict[str, Callable[[dict[str, Any]], Any]] = {
        "baseline_ambiguous_recovery": lambda run: run["evaluation"]["baseline"][
            "ambiguous"
        ]["recovery_rate"],
        "ablated_ambiguous_recovery": lambda run: run["evaluation"][
            "probe_action_ablation"
        ]["ambiguous"]["recovery_rate"],
        "ablation_recovery_drop": lambda run: run["evaluation"][
            "probe_action_ablation"
        ]["ambiguous_recovery_drop"],
        "ood_ambiguous_recovery": lambda run: run["evaluation"]["ood"]["ambiguous"][
            "recovery_rate"
        ],
        "ood_diagnostic_success": lambda run: run["evaluation"]["ood"]["ambiguous"][
            "experiment_then_correct_repair_rate"
        ],
        "max_probe_cost_trace_change": lambda run: max(
            item["trace_change_rate"]
            for item in run["evaluation"]["probe_cost_sensitivity"]
        ),
        "max_repair_cost_trace_change": lambda run: max(
            item["trace_change_rate"]
            for item in run["evaluation"]["repair_cost_sensitivity"]
        ),
    }
    arms: dict[str, Any] = {}
    for arm_index, arm in enumerate(source_study.arms):
        arm_runs = [run for run in runs if run["arm"] == arm]
        metrics: dict[str, Any] = {}
        for metric_index, (name, extractor) in enumerate(metric_extractors.items()):
            values = [float(extractor(run)) for run in arm_runs]
            metrics[name] = {
                "mean": fmean(values),
                "bootstrap": asdict(
                    bootstrap_mean(
                        values,
                        confidence_level=source_study.confidence_level,
                        resamples=source_study.bootstrap_resamples,
                        seed=(
                            source_study.bootstrap_seed
                            + 3_000
                            + arm_index * len(metric_extractors)
                            + metric_index
                        ),
                    )
                ),
            }
        arms[arm] = {
            "metrics": metrics,
            "individual_training_seeds": [
                {
                    "seed": run["training_seed"],
                    **{
                        name: float(extractor(run))
                        for name, extractor in metric_extractors.items()
                    },
                }
                for run in arm_runs
            ],
        }

    return {
        "analysis_version": BEHAVIORAL_STUDY_VERSION,
        "evaluator_version": BEHAVIORAL_CONTROLS_VERSION,
        "protocol": asdict(protocol),
        "source_study_protocol": asdict(source_study),
        "arms": arms,
        "runs": runs,
    }
