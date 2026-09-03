from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from faultline.artifacts import canonical_sha256


def test_counterfactual_study_is_integral_and_reports_eligibility_separately() -> None:
    repo = Path(__file__).parents[2]
    run_id = "counterfactual-v1-analysis"
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

    random_metrics = analysis["arms"]["random"]["metrics"]
    difficulty_metrics = analysis["arms"]["difficulty"]["metrics"]
    epistemic_metrics = analysis["arms"]["epistemic"]["metrics"]
    assert random_metrics["diagnostic_eligibility_rate"]["mean"] == 0.8125
    assert random_metrics["causal_evidence_use_rate"]["available_training_seeds"] == 7
    assert random_metrics["causal_evidence_use_rate"]["mean"] == pytest.approx(
        0.9910714285714286
    )
    assert difficulty_metrics["overall_causal_evidence_use_rate"]["mean"] == pytest.approx(
        0.8984375
    )
    assert epistemic_metrics["diagnostic_eligibility_rate"]["mean"] == 1.0
    assert epistemic_metrics["causal_evidence_use_rate"]["mean"] == pytest.approx(
        0.90234375
    )
    assert epistemic_metrics["randomized_correct_rate"]["mean"] == pytest.approx(
        0.49664306640625
    )
    assert analysis["paired_primary_comparisons"]["epistemic-difficulty"][
        "estimate"
    ] == pytest.approx(0.00390625)
