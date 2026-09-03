"""Immutable machine-readable experiment manifests."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def repository_root(start: Path) -> Path:
    """Resolve the containing Git worktree root."""
    search_directory = start if start.is_dir() else start.parent
    output = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=search_directory,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return Path(output)


def git_state(repo: Path) -> tuple[str, bool]:
    """Return exact HEAD and whether tracked or untracked changes exist."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, bool(status.strip())


def hardware_metadata() -> dict[str, object]:
    uname = platform.uname()
    return {
        "system": uname.system,
        "release": uname.release,
        "machine": uname.machine,
        "processor": uname.processor,
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def build_manifest(
    *,
    repo: Path,
    run_id: str,
    experiment: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    started_at: str,
    completed_at: str,
    metric_version: str,
    split: str = "not_applicable",
    status: str = "completed",
    seed: int | None = None,
) -> dict[str, Any]:
    """Construct a schema-v1 manifest from observed run data and repository state."""
    commit, dirty = git_state(repo)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "git_commit": commit,
        "git_dirty": dirty,
        "started_at": started_at,
        "completed_at": completed_at,
        "experiment": experiment,
        "config": config,
        "config_sha256": canonical_sha256(config),
        "split": split,
        "seed": seed,
        "generator_version": None,
        "policy": None,
        "curriculum": None,
        "reward": None,
        "hardware": hardware_metadata(),
        "status": status,
        "metric_version": metric_version,
        "metrics": metrics,
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Create a manifest atomically enough to reject accidental result mutation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as destination:
        destination.write(payload)
