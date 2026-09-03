"""Operational reward and sustained-recovery accounting."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from faultline.env.actions import ActionKind


class TerminationReason(StrEnum):
    RECOVERED = "recovered"
    TICK_LIMIT = "tick_limit"
    ACTION_LIMIT = "action_limit"


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Reward coefficients; none directly reward information or belief change."""

    target_throughput: float
    throughput_value: float = 1.0
    time_cost: float = 0.05
    passive_cost: float = 0.02
    diagnostic_cost: float = 0.25
    repair_cost: float = 2.0
    false_repair_cost: float = 3.0
    invalid_action_cost: float = 0.5
    recovery_fraction: float = 0.9
    recovery_ticks: int = 3
    recovery_bonus: float = 5.0
    max_ticks: int = 50
    max_actions: int = 50

    def __post_init__(self) -> None:
        nonnegative = (
            self.target_throughput,
            self.throughput_value,
            self.time_cost,
            self.passive_cost,
            self.diagnostic_cost,
            self.repair_cost,
            self.false_repair_cost,
            self.invalid_action_cost,
            self.recovery_bonus,
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in nonnegative):
            raise ValueError("reward values must be finite and non-negative")
        if self.target_throughput <= 0.0:
            raise ValueError("target throughput must be positive")
        if not 0.0 < self.recovery_fraction <= 1.0:
            raise ValueError("recovery fraction must be in (0, 1]")
        if self.recovery_ticks <= 0 or self.max_ticks <= 0 or self.max_actions <= 0:
            raise ValueError("recovery, tick, and action limits must be positive")


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    total_reward: float
    production_reward: float
    time_cost: float
    action_cost: float
    false_repair_cost: float
    recovery_bonus: float
    action_count: int
    passive_observation_count: int
    diagnostic_count: int
    repair_count: int
    false_repair_count: int
    recovery_streak: int
    recovered: bool
    terminated: bool
    termination_reason: TerminationReason | None


@dataclass(slots=True)
class RewardTracker:
    """Mutable per-episode accounting kept separate from public plant state."""

    config: RewardConfig
    total_reward: float = 0.0
    production_reward: float = 0.0
    accumulated_time_cost: float = 0.0
    action_cost: float = 0.0
    accumulated_false_repair_cost: float = 0.0
    recovery_bonus_paid: float = 0.0
    action_count: int = 0
    passive_observation_count: int = 0
    diagnostic_count: int = 0
    repair_count: int = 0
    false_repair_count: int = 0
    recovery_streak: int = 0
    recovered: bool = False
    terminated: bool = False
    termination_reason: TerminationReason | None = None

    def record_action(
        self,
        kind: ActionKind,
        *,
        accepted: bool,
        repair_was_needed: bool = False,
    ) -> float:
        """Charge an action without exposing privileged necessity in the observation."""
        self.action_count += 1
        false_repair_cost = 0.0
        if not accepted:
            cost = self.config.invalid_action_cost
        elif kind in (ActionKind.INSPECT, ActionKind.MEASURE_FLOW):
            self.passive_observation_count += 1
            cost = self.config.passive_cost
        elif kind in (ActionKind.ISOLATE, ActionKind.TOGGLE):
            self.diagnostic_count += 1
            cost = self.config.diagnostic_cost
        elif kind in (ActionKind.REPLACE, ActionKind.CLEAR_BLOCKAGE):
            self.repair_count += 1
            cost = self.config.repair_cost
            if not repair_was_needed:
                self.false_repair_count += 1
                false_repair_cost = self.config.false_repair_cost
                self.accumulated_false_repair_cost += false_repair_cost
        else:
            cost = 0.0

        reward = -cost - false_repair_cost
        self.action_cost += cost
        self.total_reward += reward
        if not self.terminated and self.action_count >= self.config.max_actions:
            self.terminated = True
            self.termination_reason = TerminationReason.ACTION_LIMIT
        return reward

    def record_tick(self, delivered: float, tick: int) -> float:
        """Score production and update sustained-recovery termination."""
        production_reward = self.config.throughput_value * delivered
        reward = production_reward - self.config.time_cost
        self.production_reward += production_reward
        self.accumulated_time_cost += self.config.time_cost

        threshold = self.config.target_throughput * self.config.recovery_fraction
        if delivered >= threshold:
            self.recovery_streak += 1
        else:
            self.recovery_streak = 0

        if self.recovery_streak >= self.config.recovery_ticks:
            self.recovered = True
            self.terminated = True
            self.termination_reason = TerminationReason.RECOVERED
            reward += self.config.recovery_bonus
            self.recovery_bonus_paid = self.config.recovery_bonus
        elif tick >= self.config.max_ticks:
            self.terminated = True
            self.termination_reason = TerminationReason.TICK_LIMIT

        self.total_reward += reward
        return reward

    def snapshot(self) -> EpisodeMetrics:
        return EpisodeMetrics(
            total_reward=self.total_reward,
            production_reward=self.production_reward,
            time_cost=self.accumulated_time_cost,
            action_cost=self.action_cost,
            false_repair_cost=self.accumulated_false_repair_cost,
            recovery_bonus=self.recovery_bonus_paid,
            action_count=self.action_count,
            passive_observation_count=self.passive_observation_count,
            diagnostic_count=self.diagnostic_count,
            repair_count=self.repair_count,
            false_repair_count=self.false_repair_count,
            recovery_streak=self.recovery_streak,
            recovered=self.recovered,
            terminated=self.terminated,
            termination_reason=self.termination_reason,
        )
