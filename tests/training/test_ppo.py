from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import torch

from faultline.agents.recurrent import GraphRecurrentPolicy
from faultline.training.curriculum import (
    CurriculumConfig,
    CurriculumKind,
    CurriculumSampler,
)
from faultline.training.ppo import (
    EpisodeTrajectory,
    PPOConfig,
    _finish_trajectory,
    collect_rollout,
    ppo_update,
    train_ppo,
)


def ppo_config(*, total_steps: int = 96) -> PPOConfig:
    return PPOConfig(
        total_decision_steps=total_steps,
        rollout_min_steps=48,
        parallel_envs=4,
        hidden_size=32,
        message_layers=1,
        learning_rate=1e-3,
        update_epochs=2,
        minibatch_episodes=8,
        device="cpu",
        torch_threads=1,
    )


def curriculum_config() -> CurriculumConfig:
    return CurriculumConfig(
        kind=CurriculumKind.RANDOM,
        training_seed=5,
        pool_offset=3_000,
        pool_size=8,
    )


def test_rollout_contains_complete_recurrent_episodes() -> None:
    config = ppo_config()
    sampler = CurriculumSampler(curriculum_config())
    policy = GraphRecurrentPolicy(hidden_size=32, message_layers=1)

    rollout = collect_rollout(policy, sampler, config, minimum_steps=32)

    assert rollout.decision_steps >= 32
    assert rollout.simulator_ticks > 0
    assert len(rollout.trajectories) == len(rollout.summaries)
    for trajectory in rollout.trajectories:
        assert trajectory.summary is not None
        assert len(trajectory.actions) == len(trajectory.rewards)
        assert len(trajectory.actions) == len(trajectory.advantages)
        assert len(trajectory.actions) == len(trajectory.returns)
        assert 1 <= len(trajectory.actions) <= 4


def test_gae_matches_hand_computed_advantages_with_terminal_bootstrap_zero() -> None:
    trajectory = EpisodeTrajectory(rewards=[1.0, 0.0, 2.0], old_values=[0.5, 1.0, 1.5])
    _finish_trajectory(trajectory, replace(ppo_config(), gamma=0.5, gae_lambda=0.5))

    assert trajectory.advantages == [0.96875, -0.125, 0.5]
    assert trajectory.returns == [1.46875, 0.875, 2.0]


def test_gae_with_unit_lambda_returns_discounted_episode_returns() -> None:
    trajectory = EpisodeTrajectory(rewards=[1.0, 0.0, 2.0], old_values=[7.0, -3.0, 0.25])
    _finish_trajectory(trajectory, replace(ppo_config(), gamma=0.5, gae_lambda=1.0))

    assert trajectory.returns == [1.5, 1.0, 2.0]


def test_ppo_update_is_finite_and_changes_policy_parameters() -> None:
    torch.manual_seed(7)
    config = ppo_config()
    sampler = CurriculumSampler(curriculum_config())
    policy = GraphRecurrentPolicy(hidden_size=32, message_layers=1)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    rollout = collect_rollout(policy, sampler, config, minimum_steps=32)
    before = [parameter.detach().clone() for parameter in policy.parameters()]

    metrics = ppo_update(policy, optimizer, rollout, config, np.random.default_rng(7))

    assert all(math.isfinite(value) for value in metrics.values())
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, policy.parameters(), strict=True)
    )


def test_cpu_training_smoke_runs_to_budget() -> None:
    result = train_ppo(ppo_config(total_steps=80), curriculum_config())

    assert result.decision_steps >= 80
    assert result.simulator_ticks > result.decision_steps
    assert result.episode_count > 0
    assert result.history
    assert result.history[-1]["decision_steps"] == result.decision_steps
    assert result.curriculum_state["kind"] == "random"
