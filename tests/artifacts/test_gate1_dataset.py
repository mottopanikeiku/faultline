from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean

import pytest

from faultline.artifacts import canonical_sha256


def test_gate1_metrics_recompute_from_immutable_task_records() -> None:
    repo = Path(__file__).parents[2]
    run_id = "gate1-diagnostic-pairs-v0.2-20260903"
    result = json.loads(
        (repo / "artifacts/results" / f"{run_id}.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (repo / "artifacts/manifests" / f"{run_id}.json").read_text(encoding="utf-8")
    )

    assert result["count"] == len(result["tasks"]) == 100
    assert [task["seed"] for task in result["tasks"]] == list(range(100))
    assert canonical_sha256(result) == manifest["metrics"]["result_sha256"]

    validations = [task["validation"] for task in result["tasks"]]
    pressures = [validation["epistemic_pressure"] for validation in validations]
    metrics = manifest["metrics"]
    assert sum(validation["valid"] for validation in validations) == metrics["validated_count"]
    assert min(pressures) == pytest.approx(metrics["ep_min"])
    assert fmean(pressures) == pytest.approx(metrics["ep_mean"])
    assert max(pressures) == pytest.approx(metrics["ep_max"])
    assert fmean(
        validation["passive_recovery_probability"] for validation in validations
    ) == pytest.approx(metrics["passive_recovery_mean"])
    assert fmean(
        validation["active_recovery_probability"] for validation in validations
    ) == pytest.approx(metrics["active_recovery_mean"])
