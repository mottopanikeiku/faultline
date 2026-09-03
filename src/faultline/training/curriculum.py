"""Matched random, generic-difficulty, and epistemic curriculum samplers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import exp

import numpy as np

from faultline.faults import BlockedEdge
from faultline.generation import CueCondition, build_generated_diagnostic_pair, get_split
from faultline.training.rl_env import DiagnosticEpisode, EpisodeSummary


class CurriculumKind(StrEnum):
    RANDOM = "random"
    DIFFICULTY = "difficulty"
    EPISTEMIC = "epistemic"


@dataclass(frozen=True, slots=True)
class CurriculumConfig:
    kind: CurriculumKind
    training_seed: int
    pool_offset: int = 1_000
    pool_size: int = 512
    max_decision_steps: int = 4
    difficulty_ema: float = 0.05
    difficulty_temperature: float = 0.25
    difficulty_exploration: float = 0.1

    def __post_init__(self) -> None:
        split = get_split("train")
        if self.training_seed < 0:
            raise ValueError("training seed must be non-negative")
        if self.pool_offset < 0 or self.pool_size <= 0:
            raise ValueError("curriculum pool offset and size must be valid")
        if self.pool_offset + self.pool_size > split.count:
            raise ValueError("curriculum pool exceeds training split")
        if self.max_decision_steps <= 0:
            raise ValueError("decision-step limit must be positive")
        if not 0.0 < self.difficulty_ema <= 1.0:
            raise ValueError("difficulty EMA must be in (0, 1]")
        if self.difficulty_temperature <= 0.0:
            raise ValueError("difficulty temperature must be positive")
        if not 0.0 <= self.difficulty_exploration <= 1.0:
            raise ValueError("difficulty exploration must be in [0, 1]")


class CurriculumSampler:
    """Counter-based sampling keeps base factory/world schedules matched across arms."""

    def __init__(self, config: CurriculumConfig) -> None:
        self.config = config
        split = get_split("train")
        self._pairs = tuple(
            build_generated_diagnostic_pair(
                split.seed_start + config.pool_offset + index
            )
            for index in range(config.pool_size)
        )
        self.episode_index = 0
        self.failure_ema = {
            CueCondition.AMBIGUOUS: 0.5,
            CueCondition.REVEALED: 0.5,
        }
        self.condition_counts = {
            CueCondition.AMBIGUOUS: 0,
            CueCondition.REVEALED: 0,
        }

    def _rng(self, stream: int) -> np.random.Generator:
        sequence = np.random.SeedSequence(
            [self.config.training_seed, self.episode_index, stream]
        )
        return np.random.default_rng(sequence)

    def ambiguous_probability(self) -> float:
        kind = self.config.kind
        if kind is CurriculumKind.EPISTEMIC:
            return 1.0
        if kind is CurriculumKind.RANDOM:
            return 0.5
        ambiguous_score = self.failure_ema[CueCondition.AMBIGUOUS]
        revealed_score = self.failure_ema[CueCondition.REVEALED]
        maximum = max(ambiguous_score, revealed_score)
        ambiguous_weight = exp(
            (ambiguous_score - maximum) / self.config.difficulty_temperature
        )
        revealed_weight = exp(
            (revealed_score - maximum) / self.config.difficulty_temperature
        )
        adaptive = ambiguous_weight / (ambiguous_weight + revealed_weight)
        exploration = self.config.difficulty_exploration
        return (1.0 - exploration) * adaptive + exploration * 0.5

    def sample_episode(self) -> DiagnosticEpisode:
        base_index = int(self._rng(0).integers(0, self.config.pool_size))
        pair = self._pairs[base_index]
        condition = (
            CueCondition.AMBIGUOUS
            if float(self._rng(1).random()) < self.ambiguous_probability()
            else CueCondition.REVEALED
        )
        world_index = int(self._rng(2).integers(0, 2))
        if condition is CueCondition.AMBIGUOUS:
            cue = int(self._rng(3).integers(0, 2))
        else:
            cue = 0 if isinstance(pair.worlds[world_index].fault, BlockedEdge) else 1
        episode = DiagnosticEpisode(
            pair,
            condition,
            world_index,
            cue,
            max_decision_steps=self.config.max_decision_steps,
        )
        self.condition_counts[condition] += 1
        self.episode_index += 1
        return episode

    def update(self, summary: EpisodeSummary) -> None:
        """Difficulty uses observed failure only; EP receives no student-relative shortcut."""
        if self.config.kind is not CurriculumKind.DIFFICULTY:
            return
        target = 0.0 if summary.recovered else 1.0
        condition = summary.condition
        rate = self.config.difficulty_ema
        self.failure_ema[condition] += rate * (target - self.failure_ema[condition])

    def state_dict(self) -> dict[str, object]:
        return {
            "kind": self.config.kind.value,
            "episode_index": self.episode_index,
            "ambiguous_probability": self.ambiguous_probability(),
            "failure_ema": {
                condition.value: value for condition, value in self.failure_ema.items()
            },
            "condition_counts": {
                condition.value: value for condition, value in self.condition_counts.items()
            },
        }
