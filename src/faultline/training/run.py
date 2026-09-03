"""Resolved training configuration and held-out validation runner."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

from faultline.evaluation.policy import evaluate_policy
from faultline.generation import get_split
from faultline.training.curriculum import CurriculumConfig, CurriculumKind
from faultline.training.ppo import PPOConfig, TrainingResult, train_ppo


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    split: str
    offset: int
    base_pair_count: int

    def __post_init__(self) -> None:
        split = get_split(self.split)
        if self.split == "test":
            raise ValueError("training runner must not access the sealed test split")
        if self.offset < 0 or self.base_pair_count <= 0:
            raise ValueError("evaluation offset and count must be valid")
        if self.offset + self.base_pair_count > split.count:
            raise ValueError("evaluation range exceeds split")

    @property
    def seeds(self) -> tuple[int, ...]:
        split = get_split(self.split)
        start = split.seed_start + self.offset
        return tuple(range(start, start + self.base_pair_count))


@dataclass(frozen=True, slots=True)
class ResolvedTrainingConfig:
    ppo: PPOConfig
    curriculum: CurriculumConfig
    evaluation: EvaluationConfig

    def to_dict(self) -> dict[str, Any]:
        curriculum = asdict(self.curriculum)
        curriculum["kind"] = self.curriculum.kind.value
        return {
            "ppo": self.ppo.to_dict(),
            "curriculum": curriculum,
            "evaluation": asdict(self.evaluation),
        }


@dataclass(frozen=True, slots=True)
class TrainingExperiment:
    training: TrainingResult
    evaluation: dict[str, Any]
    elapsed_seconds: float


def load_training_config(
    path: Path,
    *,
    curriculum_kind: CurriculumKind,
    training_seed: int,
    max_steps: int | None = None,
    device: str | None = None,
) -> ResolvedTrainingConfig:
    """Load a strict TOML config and apply explicit command-line cost controls."""
    with path.open("rb") as source:
        raw: dict[str, Any] = tomllib.load(source)
    expected_sections = {"ppo", "curriculum", "evaluation"}
    if set(raw) != expected_sections:
        raise ValueError(f"training config sections must be exactly {sorted(expected_sections)}")
    ppo_values: dict[str, Any] = dict(raw["ppo"])
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("maximum steps must be positive")
        ppo_values["total_decision_steps"] = min(
            int(ppo_values["total_decision_steps"]),
            max_steps,
        )
    if device is not None:
        ppo_values["device"] = device
    ppo = PPOConfig(**ppo_values)
    curriculum_values: dict[str, Any] = dict(raw["curriculum"])
    curriculum = CurriculumConfig(
        kind=curriculum_kind,
        training_seed=training_seed,
        **curriculum_values,
    )
    evaluation = EvaluationConfig(**dict(raw["evaluation"]))
    return ResolvedTrainingConfig(ppo=ppo, curriculum=curriculum, evaluation=evaluation)


def run_training(config: ResolvedTrainingConfig) -> TrainingExperiment:
    started = perf_counter()
    training = train_ppo(config.ppo, config.curriculum)
    evaluation = evaluate_policy(
        training.policy,
        config.evaluation.seeds,
        device=config.ppo.device,
    )
    elapsed = perf_counter() - started
    return TrainingExperiment(
        training=training,
        evaluation=evaluation,
        elapsed_seconds=elapsed,
    )


def with_training_seed(
    config: ResolvedTrainingConfig,
    seed: int,
) -> ResolvedTrainingConfig:
    """Create another arm/seed config without mutating a resolved protocol."""
    return replace(config, curriculum=replace(config.curriculum, training_seed=seed))
