from __future__ import annotations

import json
from pathlib import Path

import pytest

from faultline.artifacts import (
    build_manifest,
    canonical_sha256,
    repository_root,
    write_manifest,
)


def test_config_hash_is_canonical() -> None:
    assert canonical_sha256({"a": 1, "b": [2, 3]}) == canonical_sha256(
        {"b": [2, 3], "a": 1}
    )


def test_manifest_records_git_state_and_cannot_be_overwritten(tmp_path: Path) -> None:
    repo = repository_root(Path(__file__))
    manifest = build_manifest(
        repo=repo,
        run_id="test-run",
        experiment="unit-test",
        config={"seed": 7},
        metrics={"value": 1.5},
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:00:01+00:00",
        metric_version="test-v1",
        seed=7,
    )
    destination = tmp_path / "manifest.json"

    write_manifest(destination, manifest)

    restored = json.loads(destination.read_text(encoding="utf-8"))
    assert restored["schema_version"] == 1
    assert len(restored["git_commit"]) == 40
    assert isinstance(restored["git_dirty"], bool)
    assert restored["config_sha256"] == canonical_sha256({"seed": 7})
    assert restored["hardware"]["processor"]
    with pytest.raises(FileExistsError):
        write_manifest(destination, manifest)
