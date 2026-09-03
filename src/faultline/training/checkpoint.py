"""Portable, versioned recurrent-policy checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from faultline.agents.recurrent import GraphRecurrentPolicy

CHECKPOINT_VERSION = 1
POLICY_VERSION = "graph-gru-v1"


def save_policy_checkpoint(
    path: Path,
    policy: GraphRecurrentPolicy,
    *,
    ppo_config: dict[str, Any],
    curriculum_config: dict[str, Any],
    training_metrics: dict[str, Any],
) -> str:
    """Create a non-overwriting CPU-portable policy artifact and return its SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "policy_version": POLICY_VERSION,
        "architecture": {
            "hidden_size": policy.hidden_size,
            "message_layers": len(policy.self_layers),
            "parameter_count": policy.parameter_count,
        },
        "state_dict": {
            name: tensor.detach().cpu() for name, tensor in policy.state_dict().items()
        },
        "ppo_config": ppo_config,
        "curriculum_config": curriculum_config,
        "training_metrics": training_metrics,
    }
    with path.open("xb") as destination:
        torch.save(payload, destination)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_policy_checkpoint(
    path: Path,
    *,
    device: str = "cpu",
) -> tuple[GraphRecurrentPolicy, dict[str, Any]]:
    """Load a known schema and reject incompatible checkpoint versions."""
    payload: dict[str, Any] = torch.load(path, map_location=device, weights_only=True)
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported policy checkpoint version")
    if payload.get("policy_version") != POLICY_VERSION:
        raise ValueError("unsupported policy architecture version")
    architecture = payload["architecture"]
    policy = GraphRecurrentPolicy(
        hidden_size=int(architecture["hidden_size"]),
        message_layers=int(architecture["message_layers"]),
    ).to(device)
    policy.load_state_dict(payload["state_dict"], strict=True)
    policy.eval()
    return policy, payload
