from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from faultline.artifacts.manifest import canonical_sha256

SPEC = importlib.util.spec_from_file_location(
    "seed_comparison", Path(__file__).parents[2] / "tools/analyze_seed_comparison.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_superiority_requires_both_baselines_and_five_point_mean() -> None:
    comparison = {"estimate": 0.08, "lower": 0.01, "upper": 0.15}
    assert MODULE.classify({"a": comparison, "b": comparison}, 0.05) == (
        "supports_advantage_over_both"
    )
    weak = {"estimate": 0.04, "lower": 0.01, "upper": 0.08}
    assert MODULE.classify({"a": comparison, "b": weak}, 0.05) == (
        "inconclusive_about_target_advantage"
    )
    touches_zero = {"estimate": 0.08, "lower": 0.0, "upper": 0.15}
    assert MODULE.classify({"a": comparison, "b": touches_zero}, 0.05) == (
        "inconclusive_about_target_advantage"
    )


def test_target_advantage_is_ruled_out_not_equivalence() -> None:
    comparison = {"estimate": -0.01, "lower": -0.06, "upper": 0.04}
    assert MODULE.classify({"a": comparison}, 0.05) == (
        "rules_out_target_advantage_over_both"
    )
    comparison["upper"] = 0.05
    assert MODULE.classify({"a": comparison}, 0.05) == (
        "inconclusive_about_target_advantage"
    )


def valid_run() -> tuple[dict, dict]:
    config = {"ppo": {"total_decision_steps": 30000}}
    result = {
        "checkpoint": {"sha256": "c" * 64}, "decision_steps": 30010,
        "resolved_config": config,
        "evaluation": {
            "rows": [{"condition": "ambiguous", "advance_count": 1,
                      "informative_inspection": True, "correct_repair": True}],
            "ambiguous": {"experiment_then_correct_repair_rate": 1.0},
        },
    }
    manifest = {"config": config, "config_sha256": canonical_sha256(config),
                "metrics": {"checkpoint_sha256": "c" * 64,
                            "result_sha256": canonical_sha256(result)}}
    return result, manifest


def test_run_checks_hashes_and_recomputes_primary_from_traces() -> None:
    result, manifest = valid_run()
    MODULE.verify_run(result, manifest)
    result["evaluation"]["rows"][0]["correct_repair"] = False
    with pytest.raises(ValueError, match="result digest mismatch"):
        MODULE.verify_run(result, manifest)
    manifest["metrics"]["result_sha256"] = canonical_sha256(result)
    with pytest.raises(ValueError, match="primary score disagrees"):
        MODULE.verify_run(result, manifest)


def test_partial_training_is_not_a_valid_run() -> None:
    result, manifest = valid_run()
    result["decision_steps"] = 10000
    manifest["metrics"]["result_sha256"] = canonical_sha256(result)
    with pytest.raises(ValueError, match="training stopped"):
        MODULE.verify_run(result, manifest)
