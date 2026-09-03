from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from faultline.artifacts import canonical_sha256


def test_behavioral_control_study_records_ablation_cost_and_ood_results() -> None:
    repo = Path(__file__).parents[2]
    run_id = "behavioral-controls-v1-analysis"
    analysis = json.loads(
        (repo / "artifacts/results" / f"{run_id}.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (repo / "artifacts/manifests" / f"{run_id}.json").read_text(encoding="utf-8")
    )
    plot = repo / manifest["metrics"]["plot_path"]

    assert manifest["git_dirty"] is False
    assert canonical_sha256(analysis) == manifest["metrics"]["result_sha256"]
    assert hashlib.sha256(plot.read_bytes()).hexdigest() == manifest["metrics"]["plot_sha256"]
    assert len(analysis["runs"]) == 24

    arms = analysis["arms"]
    for arm in ("random", "difficulty", "epistemic"):
        assert arms[arm]["metrics"]["ablated_ambiguous_recovery"]["mean"] == 0.5
        assert arms[arm]["metrics"]["max_probe_cost_trace_change"]["mean"] == 0.0
        assert arms[arm]["metrics"]["max_repair_cost_trace_change"]["mean"] == 0.0

    assert arms["random"]["metrics"]["ablation_recovery_drop"]["mean"] == pytest.approx(
        0.40234375
    )
    assert arms["difficulty"]["metrics"]["ood_ambiguous_recovery"][
        "mean"
    ] == pytest.approx(0.951171875)
    assert arms["epistemic"]["metrics"]["ood_ambiguous_recovery"][
        "mean"
    ] == pytest.approx(0.98046875)

    first = analysis["runs"][0]["evaluation"]
    probe_returns = [item["ambiguous"]["mean_return"] for item in first["probe_cost_sensitivity"]]
    assert len(set(probe_returns)) > 1
    assert {item["trace_change_rate"] for item in first["probe_cost_sensitivity"]} == {0.0}
