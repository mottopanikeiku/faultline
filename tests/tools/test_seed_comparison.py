from __future__ import annotations

import importlib.util
import io
import json
import tarfile
import tomllib
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree

import pytest

from faultline.artifacts.manifest import canonical_sha256
from faultline.evaluation.study import load_kill_test_protocol

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


@pytest.mark.parametrize(
    ("lower", "upper", "direction", "equivalent"),
    [
        (-0.04, 0.04, "unresolved", True),
        (0.01, 0.04, "ambiguous_only_better", True),
        (-0.04, -0.01, "ambiguous_only_worse", True),
        (0.01, 0.10, "ambiguous_only_better", False),
        (-0.10, -0.01, "ambiguous_only_worse", False),
        (-0.05, 0.05, "unresolved", False),
    ],
)
def test_direction_and_equivalence_are_distinct(lower, upper, direction, equivalent) -> None:
    result = MODULE.contrast_decision({"lower": lower, "upper": upper}, 0.05)
    assert result["direction"] == direction
    assert result["no_difference_larger_than_five_points"] == equivalent
    assert result["rules_out_five_point_advantage"] == (upper < 0.05)


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


def test_figure_shows_every_seed_and_keeps_wide_intervals_on_canvas() -> None:
    analysis = {
        "arms": {
            arm: {"primary": {"estimate": 0.5},
                  "individual_seeds": [{"seed": seed, "value": 0.5} for seed in range(32)]}
            for arm in ("random", "difficulty", "epistemic")
        },
        "paired_comparisons": {
            "epistemic-random": {"estimate": 0.0, "lower": -0.8, "upper": 0.8},
            "epistemic-difficulty": {"estimate": 0.0, "lower": -0.1, "upper": 0.1},
        },
    }
    root = ElementTree.fromstring(MODULE.render_svg(analysis))
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    circles = root.findall(".//svg:circle", namespace)
    assert len(circles) == 3 * 32 + 2
    assert all(0 <= float(circle.attrib["cx"]) <= 900 for circle in circles)
    assert len(root.findall(".//svg:title", namespace)) == 3 * 32 + 1


def test_archive_analysis_preserves_six_episodes_per_base_pair(monkeypatch, tmp_path) -> None:
    repo = Path(__file__).parents[2]
    protocol = replace(
        load_kill_test_protocol(repo / "configs/evaluation/seed-comparison.toml"),
        training_seeds=(200, 201), bootstrap_resamples=100,
    )
    monkeypatch.setattr(MODULE, "load_kill_test_protocol", lambda path: protocol)
    config_path = tmp_path / protocol.training_config
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes((repo / protocol.training_config).read_bytes())
    config = tomllib.loads(config_path.read_text())
    directory = tmp_path / "artifacts/results"
    directory.mkdir(parents=True)
    with tarfile.open(directory / "seed-comparison-runs.tar.gz", "w:gz") as archive:
        for arm in protocol.arms:
            for seed in protocol.training_seeds:
                run_id = protocol.run_id_template.format(arm=arm, seed=seed)
                result, manifest = valid_run()
                result.update(training_seed=seed, curriculum=arm)
                result["resolved_config"] = {
                    **config,
                    "curriculum": {**config["curriculum"], "kind": arm, "training_seed": seed},
                }
                ambiguous_row = result["evaluation"]["rows"][0]
                revealed_row = {**ambiguous_row, "condition": "revealed"}
                result["evaluation"]["rows"] = (
                    [ambiguous_row] * 4 + [revealed_row] * 2
                ) * protocol.evaluation_base_pair_count
                result["evaluation"]["ambiguous"].update(
                    recovery_rate=1.0, mean_return=1.0, false_repair_rate=0.0,
                )
                result["evaluation"]["revealed"] = {
                    "informative_inspection_rate": 1.0, "mean_return": 1.0,
                }
                manifest.update(
                    git_dirty=False, git_commit="a" * 40, config=result["resolved_config"],
                    config_sha256=canonical_sha256(result["resolved_config"]),
                )
                manifest["metrics"]["result_sha256"] = canonical_sha256(result)
                for kind, value in (("results", result), ("manifests", manifest)):
                    payload = json.dumps(value).encode()
                    member = tarfile.TarInfo(f"artifacts/{kind}/{run_id}.json")
                    member.size = len(payload)
                    archive.addfile(member, io.BytesIO(payload))
    analysis = MODULE.analyze(tmp_path)
    assert analysis["target_decision"] == "rules_out_target_advantage_over_both"
    assert analysis["arms"]["epistemic"]["primary"]["unit_count"] == 2
