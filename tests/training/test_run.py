from __future__ import annotations

from pathlib import Path

import pytest

from faultline.training.curriculum import CurriculumKind
from faultline.training.run import EvaluationConfig, load_training_config


def test_training_config_resolves_explicit_budget_and_curriculum() -> None:
    repo = Path(__file__).parents[2]
    resolved = load_training_config(
        repo / "configs/training/small-cpu-smoke.toml",
        curriculum_kind=CurriculumKind.EPISTEMIC,
        training_seed=9,
        max_steps=123,
    )

    assert resolved.ppo.total_decision_steps == 123
    assert resolved.curriculum.kind is CurriculumKind.EPISTEMIC
    assert resolved.curriculum.training_seed == 9
    assert resolved.evaluation.seeds[0] == 1_000_000
    assert resolved.to_dict()["curriculum"]["kind"] == "epistemic"


def test_training_runner_cannot_open_sealed_test_split() -> None:
    with pytest.raises(ValueError, match="sealed test split"):
        EvaluationConfig(split="test", offset=0, base_pair_count=1)
