from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from faultline.evaluation.study import analyze_kill_test, load_kill_test_protocol


def write_fake_run(repo: Path, arm: str, seed: int, value: float) -> None:
    run_id = f"small-kill-v1-{arm}-seed-{seed}"
    result = {
        "training_seed": seed,
        "curriculum": arm,
        "decision_steps": 30000,
        "resolved_config": {
            "ppo": {"total_decision_steps": 30000},
            "evaluation": {
                "split": "validation",
                "offset": 0,
                "base_pair_count": 128,
            },
        },
        "evaluation": {
            "ambiguous": {
                "experiment_then_correct_repair_rate": value,
                "recovery_rate": value,
                "mean_return": value,
                "false_repair_rate": 1.0 - value,
            },
            "revealed": {
                "informative_inspection_rate": value,
                "mean_return": value,
            },
        },
    }
    manifest = {
        "git_dirty": False,
        "git_commit": "a" * 40,
        "metrics": {"result_sha256": "b" * 64, "checkpoint_sha256": "c" * 64},
    }
    result_path = repo / "artifacts/results" / f"{run_id}.json"
    manifest_path = repo / "artifacts/manifests" / f"{run_id}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_frozen_protocol_excludes_exploratory_seed_zero() -> None:
    repo = Path(__file__).parents[2]
    protocol = load_kill_test_protocol(repo / "configs/evaluation/small-kill-v1.toml")

    assert protocol.training_seeds == tuple(range(100, 108))
    assert protocol.primary_metric == "ambiguous.experiment_then_correct_repair_rate"
    assert protocol.evaluation_split == "validation"


def test_study_analysis_uses_paired_seed_units_and_detects_saturation(tmp_path: Path) -> None:
    source_repo = Path(__file__).parents[2]
    protocol = load_kill_test_protocol(
        source_repo / "configs/evaluation/small-kill-v1.toml"
    )
    protocol = replace(
        protocol,
        training_seeds=(100, 101),
        bootstrap_resamples=100,
    )
    for arm in protocol.arms:
        for seed in protocol.training_seeds:
            write_fake_run(tmp_path, arm, seed, 1.0)

    analysis = analyze_kill_test(tmp_path, protocol)

    assert analysis["all_arms_saturated"]
    assert not analysis["supports_curriculum_specific_effect"]
    assert analysis["decision"] == "benchmark_saturated_no_curriculum_separation"
    assert analysis["paired_comparisons"]["epistemic-random"]["estimate"] == 0.0


def test_study_analysis_refuses_missing_preregistered_run(tmp_path: Path) -> None:
    repo = Path(__file__).parents[2]
    protocol = load_kill_test_protocol(repo / "configs/evaluation/small-kill-v1.toml")

    with pytest.raises(FileNotFoundError, match="missing preregistered run"):
        analyze_kill_test(tmp_path, protocol)
