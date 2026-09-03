from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import fmean

import pytest

from faultline.artifacts import canonical_sha256


def test_gate2_analysis_and_plots_match_task_level_rows() -> None:
    repo = Path(__file__).parents[2]
    run_id = "gate2-ep-analysis-v0.3-20260903"
    result_path = repo / "artifacts/results" / f"{run_id}.json"
    ep_plot_path = repo / "artifacts/results" / f"{run_id}-ep.svg"
    intervention_plot_path = repo / "artifacts/results" / f"{run_id}-interventions.svg"
    manifest = json.loads(
        (repo / "artifacts/manifests" / f"{run_id}.json").read_text(encoding="utf-8")
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    metrics = manifest["metrics"]

    assert manifest["git_dirty"] is False
    assert result["count"] == len(result["rows"]) == 100
    assert canonical_sha256(result) == metrics["result_sha256"]
    assert hashlib.sha256(ep_plot_path.read_bytes()).hexdigest() == metrics["ep_plot_sha256"]
    assert (
        hashlib.sha256(intervention_plot_path.read_bytes()).hexdigest()
        == metrics["intervention_plot_sha256"]
    )

    pressures = [row["epistemic_pressure"] for row in result["rows"]]
    assert min(pressures) == pytest.approx(metrics["epistemic_pressure"]["min"])
    assert fmean(pressures) == pytest.approx(metrics["epistemic_pressure"]["mean"])
    assert max(pressures) == pytest.approx(metrics["epistemic_pressure"]["max"])
    assert {
        row["immediate_advance_information_gain_bits"] for row in result["rows"]
    } == {0.0}
    assert {
        row["post_advance_inspect_information_gain_bits"] for row in result["rows"]
    } == {1.0}
