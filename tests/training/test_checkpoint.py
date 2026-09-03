from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from faultline.agents.recurrent import GraphRecurrentPolicy
from faultline.training.checkpoint import load_policy_checkpoint, save_policy_checkpoint


def test_checkpoint_round_trip_is_portable_and_immutable(tmp_path: Path) -> None:
    torch.manual_seed(41)
    policy = GraphRecurrentPolicy(hidden_size=24, message_layers=2)
    path = tmp_path / "policy.pt"

    digest = save_policy_checkpoint(
        path,
        policy,
        ppo_config={"learning_rate": 0.001},
        curriculum_config={"kind": "random"},
        training_metrics={"decision_steps": 100},
    )
    restored, payload = load_policy_checkpoint(path)

    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert payload["architecture"]["parameter_count"] == policy.parameter_count
    for original, loaded in zip(policy.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(original, loaded)
    with pytest.raises(FileExistsError):
        save_policy_checkpoint(
            path,
            policy,
            ppo_config={},
            curriculum_config={},
            training_metrics={},
        )
