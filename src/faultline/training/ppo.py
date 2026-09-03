"""Compact recurrent PPO for the diagnostic-pair environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import fmean
from typing import Any

import numpy as np
import torch
from torch import Tensor

from faultline.agents.recurrent import GraphRecurrentPolicy
from faultline.training.curriculum import CurriculumConfig, CurriculumSampler
from faultline.training.rl_env import EpisodeSummary, PolicyObservation


@dataclass(frozen=True, slots=True)
class PPOConfig:
    total_decision_steps: int = 20_000
    rollout_min_steps: int = 1_024
    parallel_envs: int = 32
    hidden_size: int = 256
    message_layers: int = 3
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_gradient_norm: float = 0.5
    update_epochs: int = 4
    minibatch_episodes: int = 64
    target_kl: float = 0.03
    device: str = "cpu"
    torch_threads: int = 4

    def __post_init__(self) -> None:
        positive_integers = (
            self.total_decision_steps,
            self.rollout_min_steps,
            self.parallel_envs,
            self.hidden_size,
            self.message_layers,
            self.update_epochs,
            self.minibatch_episodes,
            self.torch_threads,
        )
        if any(value <= 0 for value in positive_integers):
            raise ValueError("PPO integer settings must be positive")
        if self.learning_rate <= 0.0 or self.max_gradient_norm <= 0.0:
            raise ValueError("learning rate and gradient norm must be positive")
        if not 0.0 <= self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("discount and GAE factors must be in [0, 1]")
        if self.clip_ratio <= 0.0 or self.value_coefficient < 0.0:
            raise ValueError("clip ratio must be positive and value coefficient non-negative")
        if self.entropy_coefficient < 0.0 or self.target_kl <= 0.0:
            raise ValueError("entropy coefficient must be non-negative and target KL positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EpisodeTrajectory:
    observations: list[PolicyObservation] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    old_log_probabilities: list[float] = field(default_factory=list)
    old_values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    advantages: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    summary: EpisodeSummary | None = None


@dataclass(frozen=True, slots=True)
class Rollout:
    trajectories: tuple[EpisodeTrajectory, ...]
    summaries: tuple[EpisodeSummary, ...]
    decision_steps: int
    simulator_ticks: int


@dataclass(frozen=True, slots=True)
class TrainingResult:
    policy: GraphRecurrentPolicy
    history: tuple[dict[str, float | int], ...]
    curriculum_state: dict[str, object]
    decision_steps: int
    simulator_ticks: int
    episode_count: int


def _observation_tensors(
    observations: list[PolicyObservation],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.as_tensor(np.stack([item.nodes for item in observations]), device=device),
        torch.as_tensor(np.stack([item.adjacency for item in observations]), device=device),
        torch.as_tensor(np.stack([item.node_mask for item in observations]), device=device),
        torch.as_tensor(
            np.stack([item.global_features for item in observations]),
            device=device,
        ),
        torch.as_tensor(np.stack([item.action_mask for item in observations]), device=device),
    )


def _finish_trajectory(trajectory: EpisodeTrajectory, config: PPOConfig) -> None:
    advantages = [0.0] * len(trajectory.rewards)
    gae = 0.0
    next_value = 0.0
    for index in range(len(trajectory.rewards) - 1, -1, -1):
        delta = trajectory.rewards[index] + config.gamma * next_value - trajectory.old_values[index]
        gae = delta + config.gamma * config.gae_lambda * gae
        advantages[index] = gae
        next_value = trajectory.old_values[index]
    trajectory.advantages = advantages
    trajectory.returns = [
        advantage + value
        for advantage, value in zip(advantages, trajectory.old_values, strict=True)
    ]


def collect_rollout(
    policy: GraphRecurrentPolicy,
    sampler: CurriculumSampler,
    config: PPOConfig,
    *,
    minimum_steps: int | None = None,
) -> Rollout:
    """Collect complete episodes concurrently, retiring workers after the step target."""
    target_steps = minimum_steps or config.rollout_min_steps
    device = torch.device(config.device)
    slot_count = min(config.parallel_envs, target_steps)
    episodes = [sampler.sample_episode() for _ in range(slot_count)]
    trajectories = [EpisodeTrajectory() for _ in range(slot_count)]
    hidden = [policy.initial_hidden(1, device).squeeze(0) for _ in range(slot_count)]
    active_slots = list(range(slot_count))
    completed: list[EpisodeTrajectory] = []
    summaries: list[EpisodeSummary] = []
    decision_steps = 0
    simulator_ticks = 0
    retiring = False

    policy.eval()
    while active_slots:
        observations = [episodes[slot].observe() for slot in active_slots]
        tensors = _observation_tensors(observations, device)
        hidden_batch = torch.stack([hidden[slot] for slot in active_slots])
        actions, log_probabilities, values, next_hidden = policy.act(
            *tensors[:4],
            hidden_batch,
            tensors[4],
        )
        next_active: list[int] = []
        for batch_index, slot in enumerate(active_slots):
            trajectory = trajectories[slot]
            trajectory.observations.append(observations[batch_index])
            action = int(actions[batch_index].item())
            trajectory.actions.append(action)
            trajectory.old_log_probabilities.append(float(log_probabilities[batch_index].item()))
            trajectory.old_values.append(float(values[batch_index].item()))
            _, reward, terminated, summary = episodes[slot].step(action)
            trajectory.rewards.append(reward)
            hidden[slot] = next_hidden[batch_index].detach()
            if not terminated:
                next_active.append(slot)
                continue

            if summary is None:
                raise AssertionError("terminated RL episode did not emit a summary")
            trajectory.summary = summary
            _finish_trajectory(trajectory, config)
            completed.append(trajectory)
            summaries.append(summary)
            sampler.update(summary)
            decision_steps += len(trajectory.actions)
            simulator_ticks += summary.simulator_ticks
            if decision_steps >= target_steps:
                retiring = True
            if not retiring:
                episodes[slot] = sampler.sample_episode()
                trajectories[slot] = EpisodeTrajectory()
                hidden[slot] = policy.initial_hidden(1, device).squeeze(0)
                next_active.append(slot)
        active_slots = next_active

    return Rollout(
        trajectories=tuple(completed),
        summaries=tuple(summaries),
        decision_steps=decision_steps,
        simulator_ticks=simulator_ticks,
    )


def _padded_tensors(
    trajectories: list[EpisodeTrajectory],
    device: torch.device,
    advantage_mean: float,
    advantage_std: float,
) -> dict[str, Tensor]:
    batch_size = len(trajectories)
    max_length = max(len(trajectory.actions) for trajectory in trajectories)
    sample = trajectories[0].observations[0]
    nodes = np.zeros((batch_size, max_length, *sample.nodes.shape), dtype=np.float32)
    adjacency = np.zeros((batch_size, max_length, *sample.adjacency.shape), dtype=np.float32)
    node_mask = np.zeros((batch_size, max_length, *sample.node_mask.shape), dtype=np.bool_)
    global_features = np.zeros(
        (batch_size, max_length, *sample.global_features.shape),
        dtype=np.float32,
    )
    action_mask = np.zeros(
        (batch_size, max_length, *sample.action_mask.shape),
        dtype=np.bool_,
    )
    actions = np.zeros((batch_size, max_length), dtype=np.int64)
    old_log_probabilities = np.zeros((batch_size, max_length), dtype=np.float32)
    old_values = np.zeros((batch_size, max_length), dtype=np.float32)
    advantages = np.zeros((batch_size, max_length), dtype=np.float32)
    returns = np.zeros((batch_size, max_length), dtype=np.float32)
    valid = np.zeros((batch_size, max_length), dtype=np.bool_)

    for row, trajectory in enumerate(trajectories):
        length = len(trajectory.actions)
        for column, observation in enumerate(trajectory.observations):
            nodes[row, column] = observation.nodes
            adjacency[row, column] = observation.adjacency
            node_mask[row, column] = observation.node_mask
            global_features[row, column] = observation.global_features
            action_mask[row, column] = observation.action_mask
        actions[row, :length] = trajectory.actions
        old_log_probabilities[row, :length] = trajectory.old_log_probabilities
        old_values[row, :length] = trajectory.old_values
        advantages[row, :length] = (
            np.asarray(trajectory.advantages, dtype=np.float32) - advantage_mean
        ) / advantage_std
        returns[row, :length] = trajectory.returns
        valid[row, :length] = True

    return {
        "nodes": torch.as_tensor(nodes, device=device),
        "adjacency": torch.as_tensor(adjacency, device=device),
        "node_mask": torch.as_tensor(node_mask, device=device),
        "global_features": torch.as_tensor(global_features, device=device),
        "action_mask": torch.as_tensor(action_mask, device=device),
        "actions": torch.as_tensor(actions, device=device),
        "old_log_probabilities": torch.as_tensor(old_log_probabilities, device=device),
        "old_values": torch.as_tensor(old_values, device=device),
        "advantages": torch.as_tensor(advantages, device=device),
        "returns": torch.as_tensor(returns, device=device),
        "valid": torch.as_tensor(valid, device=device),
    }


def _evaluate_padded(
    policy: GraphRecurrentPolicy,
    batch: dict[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    batch_size, sequence_length = batch["actions"].shape
    hidden = policy.initial_hidden(batch_size, batch["nodes"].device)
    log_probabilities: list[Tensor] = []
    entropies: list[Tensor] = []
    values: list[Tensor] = []
    for step in range(sequence_length):
        output = policy(
            batch["nodes"][:, step],
            batch["adjacency"][:, step],
            batch["node_mask"][:, step],
            batch["global_features"][:, step],
            hidden,
            batch["action_mask"][:, step],
        )
        hidden = output.hidden
        log_probability = torch.log_softmax(output.logits, dim=-1)
        probability = log_probability.exp()
        selected_log_probability = log_probability.gather(
            -1,
            batch["actions"][:, step].unsqueeze(-1),
        ).squeeze(-1)
        log_probabilities.append(selected_log_probability)
        entropies.append(-(probability * log_probability).sum(dim=-1))
        values.append(output.value)
    valid = batch["valid"]
    return (
        torch.stack(log_probabilities, dim=1)[valid],
        torch.stack(entropies, dim=1)[valid],
        torch.stack(values, dim=1)[valid],
    )


def ppo_update(
    policy: GraphRecurrentPolicy,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    config: PPOConfig,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Update on padded complete episodes with backpropagation through recurrent history."""
    policy.train()
    device = torch.device(config.device)
    all_advantages = np.asarray(
        [advantage for trajectory in rollout.trajectories for advantage in trajectory.advantages],
        dtype=np.float64,
    )
    advantage_mean = float(all_advantages.mean())
    advantage_std = float(all_advantages.std() + 1e-8)
    indices = np.arange(len(rollout.trajectories))
    losses: list[float] = []
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    divergences: list[float] = []
    clip_fractions: list[float] = []

    for _ in range(config.update_epochs):
        rng.shuffle(indices)
        stop_early = False
        for start in range(0, len(indices), config.minibatch_episodes):
            selection = indices[start : start + config.minibatch_episodes]
            trajectories = [rollout.trajectories[int(index)] for index in selection]
            batch = _padded_tensors(
                trajectories,
                device,
                advantage_mean,
                advantage_std,
            )
            new_log_probabilities, entropy, new_values = _evaluate_padded(policy, batch)
            valid = batch["valid"]
            old_log_probabilities = batch["old_log_probabilities"][valid]
            old_values = batch["old_values"][valid]
            advantages = batch["advantages"][valid]
            returns = batch["returns"][valid]
            log_ratio = new_log_probabilities - old_log_probabilities
            ratio = log_ratio.exp()
            unclipped = ratio * advantages
            clipped = ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_clipped = old_values + (new_values - old_values).clamp(
                -config.clip_ratio,
                config.clip_ratio,
            )
            value_loss = 0.5 * torch.maximum(
                (new_values - returns).square(),
                (value_clipped - returns).square(),
            ).mean()
            entropy_mean = entropy.mean()
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy_mean
            )
            optimizer.zero_grad(set_to_none=True)
            torch.autograd.backward(loss)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_gradient_norm)
            optimizer.step()

            with torch.no_grad():
                approximate_kl = float(((ratio - 1.0) - log_ratio).mean().item())
                clip_fraction = float(
                    ((ratio - 1.0).abs() > config.clip_ratio).float().mean().item()
                )
            losses.append(float(loss.item()))
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy_mean.item()))
            divergences.append(approximate_kl)
            clip_fractions.append(clip_fraction)
            if approximate_kl > config.target_kl:
                stop_early = True
                break
        if stop_early:
            break

    return {
        "loss": fmean(losses),
        "policy_loss": fmean(policy_losses),
        "value_loss": fmean(value_losses),
        "entropy": fmean(entropies),
        "approximate_kl": fmean(divergences),
        "clip_fraction": fmean(clip_fractions),
    }


def train_ppo(
    ppo_config: PPOConfig,
    curriculum_config: CurriculumConfig,
) -> TrainingResult:
    """Train one policy under a fixed decision-step budget."""
    torch.set_num_threads(ppo_config.torch_threads)
    torch.manual_seed(curriculum_config.training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(curriculum_config.training_seed)
    device = torch.device(ppo_config.device)
    policy = GraphRecurrentPolicy(
        hidden_size=ppo_config.hidden_size,
        message_layers=ppo_config.message_layers,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=ppo_config.learning_rate, eps=1e-5)
    sampler = CurriculumSampler(curriculum_config)
    update_rng = np.random.default_rng(curriculum_config.training_seed + 17_000_003)
    history: list[dict[str, float | int]] = []
    decision_steps = 0
    simulator_ticks = 0
    episode_count = 0

    while decision_steps < ppo_config.total_decision_steps:
        remaining = ppo_config.total_decision_steps - decision_steps
        rollout = collect_rollout(
            policy,
            sampler,
            ppo_config,
            minimum_steps=min(ppo_config.rollout_min_steps, remaining),
        )
        update_metrics = ppo_update(policy, optimizer, rollout, ppo_config, update_rng)
        decision_steps += rollout.decision_steps
        simulator_ticks += rollout.simulator_ticks
        episode_count += len(rollout.summaries)
        history.append(
            {
                "update": len(history),
                "decision_steps": decision_steps,
                "simulator_ticks": simulator_ticks,
                "episodes": episode_count,
                "mean_return": fmean(summary.total_reward for summary in rollout.summaries),
                "recovery_rate": fmean(summary.recovered for summary in rollout.summaries),
                "correct_repair_rate": fmean(
                    summary.correct_repair for summary in rollout.summaries
                ),
                "informative_inspection_rate": fmean(
                    summary.informative_inspection for summary in rollout.summaries
                ),
                **update_metrics,
            }
        )

    return TrainingResult(
        policy=policy,
        history=tuple(history),
        curriculum_state=sampler.state_dict(),
        decision_steps=decision_steps,
        simulator_ticks=simulator_ticks,
        episode_count=episode_count,
    )
