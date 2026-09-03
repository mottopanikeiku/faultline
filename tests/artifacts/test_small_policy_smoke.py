from __future__ import annotations

import hashlib
import json
from pathlib import Path

from faultline.artifacts import canonical_sha256
from faultline.evaluation.policy import evaluate_policy
from faultline.training.checkpoint import load_policy_checkpoint


def test_small_policy_smoke_is_reconstructable_and_remains_negative() -> None:
    repo = Path(__file__).parents[2]
    run_id = "small-ep-smoke-seed00-20260903"
    result = json.loads(
        (repo / "artifacts/results" / f"{run_id}.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (repo / "artifacts/manifests" / f"{run_id}.json").read_text(encoding="utf-8")
    )
    checkpoint = repo / manifest["metrics"]["checkpoint_path"]

    assert manifest["git_dirty"] is False
    assert canonical_sha256(result) == manifest["metrics"]["result_sha256"]
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == manifest["metrics"][
        "checkpoint_sha256"
    ]
    assert result["evaluation"]["ambiguous"]["recovery_rate"] == 0.5
    assert (
        result["evaluation"]["ambiguous"]["experiment_then_correct_repair_rate"]
        == 0.0
    )

    policy, payload = load_policy_checkpoint(checkpoint)
    assert payload["policy_version"] == "graph-gru-v1"
    replay = evaluate_policy(policy, [1_000_000, 1_000_001])
    assert replay["ambiguous"]["recovery_rate"] == 0.5
    assert replay["ambiguous"]["experiment_then_correct_repair_rate"] == 0.0
