"""Finite-world diagnostic planning model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from faultline.env import Action, FactoryEnv


@dataclass(frozen=True, slots=True)
class TerminalPlan:
    """A commitment sequence ending in operational evaluation."""

    name: str
    actions: tuple[Action, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.actions:
            raise ValueError("terminal plans require a name and at least one action")


@dataclass(frozen=True, slots=True)
class DiagnosticProblem:
    """A bounded latent-world problem suitable for exact enumeration."""

    world_labels: tuple[str, ...]
    prior: tuple[float, ...]
    initial_envs: tuple[FactoryEnv, ...]
    diagnostic_actions: tuple[Action, ...]
    terminal_plans: tuple[TerminalPlan, ...]

    def __post_init__(self) -> None:
        world_count = len(self.world_labels)
        if world_count < 2:
            raise ValueError("diagnostic problem requires at least two worlds")
        if len(set(self.world_labels)) != world_count:
            raise ValueError("world labels must be unique")
        if len(self.prior) != world_count or len(self.initial_envs) != world_count:
            raise ValueError("world labels, prior, and environments must align")
        if any(not np.isfinite(probability) or probability < 0.0 for probability in self.prior):
            raise ValueError("prior probabilities must be finite and non-negative")
        if not np.isclose(sum(self.prior), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("prior probabilities must sum to one")
        if not self.diagnostic_actions or not self.terminal_plans:
            raise ValueError("diagnostic actions and terminal plans must be non-empty")
        if any(env.reward_tracker is None for env in self.initial_envs):
            raise ValueError("oracle environments require operational reward tracking")


@dataclass(slots=True)
class WorldBranch:
    """One mutable simulator branch with posterior probability."""

    world_index: int
    probability: float
    env: FactoryEnv

    def clone(self) -> WorldBranch:
        return WorldBranch(self.world_index, self.probability, self.env.clone())
