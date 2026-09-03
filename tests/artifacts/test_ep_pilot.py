from __future__ import annotations

import hashlib
import json
from pathlib import Path

from faultline.artifacts import canonical_sha256
from faultline.evaluation.policy import evaluate_policy
from faultline.training.checkpoint import load_policy_checkpoint


def test_30k_ep_pilot_replays_diagnostic_success_and_overprobing() -> None:
    repo = Path(__file__).parents[2]
    run_id = "ep-pilot-30k-masked-seed00-20260903"
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
    assert result["evaluation"]["ambiguous"]["experiment_then_correct_repair_rate"] == 1.0
    assert result["evaluation"]["revealed"]["informative_inspection_rate"] == 1.0

    policy, _ = load_policy_checkpoint(checkpoint)
    replay = evaluate_policy(policy, [1_000_000, 1_000_001])
    assert replay["ambiguous"]["recovery_rate"] == 1.0
    assert replay["ambiguous"]["experiment_then_correct_repair_rate"] == 1.0
    assert replay["revealed"]["informative_inspection_rate"] == 1.0
