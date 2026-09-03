from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from faultline.artifacts import canonical_sha256


def _load(repo: Path, run_id: str) -> tuple[dict[str, object], dict[str, object]]:
    result = json.loads(
        (repo / "artifacts/results" / f"{run_id}.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (repo / "artifacts/manifests" / f"{run_id}.json").read_text(encoding="utf-8")
    )
    return result, manifest


def test_normalized_ep_result_is_integral_and_records_saturation() -> None:
    repo = Path(__file__).parents[2]
    run_id = "gate2b-normalized-ep-v0.3-20260903"
    result, manifest = _load(repo, run_id)
    metrics = manifest["metrics"]

    assert canonical_sha256(result) == metrics["result_sha256"]
    assert result["normalized_ep"]["mean"] == pytest.approx(0.9732872391830213)
    assert result["normalized_ep"]["q25"] > 0.98
    assert abs(result["correlations_with_normalized_ep"]["repair_margin"]["pearson"]) < 0.05
    plot = repo / metrics["normalized_plot_path"]
    assert hashlib.sha256(plot.read_bytes()).hexdigest() == metrics["normalized_plot_sha256"]


def test_matched_control_changes_only_cue_fault_joint() -> None:
    repo = Path(__file__).parents[2]
    run_id = "gate2c-matched-ep-control-v0.3-20260903"
    result, manifest = _load(repo, run_id)
    metrics = manifest["metrics"]

    assert manifest["git_dirty"] is False
    assert canonical_sha256(result) == metrics["result_sha256"]
    assert result["count"] == 100
    assert result["all_fault_marginals_equal"]
    assert result["all_cue_marginals_equal"]
    assert result["all_cue_fault_joints_differ"]
    assert result["max_abs_nuisance_standardized_mean_difference"] == 0.0
    assert result["ambiguous_ep"]["mean"] == pytest.approx(8.861625)
    assert result["revealed_ep"]["max"] == pytest.approx(0.0, abs=1e-12)
    assert result["ambiguous_passive_recovery_mean"] == 0.5
    assert result["revealed_passive_recovery_mean"] == 1.0
    plot = repo / metrics["plot_path"]
    assert hashlib.sha256(plot.read_bytes()).hexdigest() == metrics["plot_sha256"]
