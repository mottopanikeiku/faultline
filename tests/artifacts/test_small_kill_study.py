from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from faultline.artifacts import canonical_sha256


def test_frozen_small_kill_result_links_all_preregistered_runs() -> None:
    repo = Path(__file__).parents[2]
    run_id = "small-kill-v1-analysis"
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
    assert len(analysis["source_runs"]) == 24
    assert analysis["decision"] == "no_preregistered_curriculum_effect"
    assert not analysis["supports_curriculum_specific_effect"]

    assert analysis["arms"]["random"]["primary"]["estimate"] == pytest.approx(
        0.807373046875
    )
    assert analysis["arms"]["difficulty"]["primary"]["estimate"] == pytest.approx(
        0.90234375
    )
    assert analysis["arms"]["epistemic"]["primary"]["estimate"] == pytest.approx(
        0.951171875
    )
    assert analysis["paired_comparisons"]["epistemic-random"] == {
        "estimate": pytest.approx(0.143798828125),
        "confidence_level": 0.95,
        "lower": pytest.approx(-0.083740234375),
        "upper": pytest.approx(0.43017578125),
        "resamples": 10000,
        "unit_count": 8,
    }

    for source in analysis["source_runs"]:
        source_result = json.loads(
            (repo / "artifacts/results" / f"{source['run_id']}.json").read_text(
                encoding="utf-8"
            )
        )
        checkpoint = repo / "artifacts/runs" / source["run_id"] / "policy.pt"
        assert canonical_sha256(source_result) == source["result_sha256"]
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == source[
            "checkpoint_sha256"
        ]
