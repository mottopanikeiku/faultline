"""Paired deterministic evaluation for recurrent diagnostic policies."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from statistics import fmean
from typing import Any

import torch

from faultline.agents.recurrent import GraphRecurrentPolicy
from faultline.generation import (
    CueCondition,
    DiagnosticPair,
    build_generated_diagnostic_pair,
)
from faultline.training.rl_env import DiagnosticEpisode, EpisodeSummary, PolicyObservation

POLICY_EVALUATION_VERSION = "small-policy-eval-v2"


def _policy_step(
    policy: GraphRecurrentPolicy,
    observation: PolicyObservation,
    hidden: torch.Tensor,
    device: torch.device,
) -> tuple[int, torch.Tensor]:
    tensors = (
        torch.as_tensor(observation.nodes, device=device).unsqueeze(0),
        torch.as_tensor(observation.adjacency, device=device).unsqueeze(0),
        torch.as_tensor(observation.node_mask, device=device).unsqueeze(0),
        torch.as_tensor(observation.global_features, device=device).unsqueeze(0),
        torch.as_tensor(observation.action_mask, device=device).unsqueeze(0),
    )
    action, _, _, next_hidden = policy.act(
        *tensors[:4],
        hidden,
        tensors[4],
        deterministic=True,
    )
    return int(action.item()), next_hidden


def _disable_diagnostics(observation: PolicyObservation) -> PolicyObservation:
    action_mask = observation.action_mask.copy()
    action_mask[:2] = False
    return PolicyObservation(
        nodes=observation.nodes,
        adjacency=observation.adjacency,
        node_mask=observation.node_mask,
        global_features=observation.global_features,
        action_mask=action_mask,
    )


def _run_episode(
    policy: GraphRecurrentPolicy,
    episode: DiagnosticEpisode,
    device: torch.device,
    disable_diagnostics: bool,
) -> tuple[EpisodeSummary, list[int]]:
    hidden = policy.initial_hidden(1, device)
    actions: list[int] = []
    summary: EpisodeSummary | None = None
    while summary is None:
        observation = episode.observe()
        if disable_diagnostics:
            observation = _disable_diagnostics(observation)
        action, hidden = _policy_step(policy, observation, hidden, device)
        actions.append(action)
        _, _, _, summary = episode.step(action)
    return summary, actions


def _aggregate(summaries: list[EpisodeSummary]) -> dict[str, float | int]:
    return {
        "episode_count": len(summaries),
        "mean_return": fmean(summary.total_reward for summary in summaries),
        "recovery_rate": fmean(summary.recovered for summary in summaries),
        "correct_repair_rate": fmean(summary.correct_repair for summary in summaries),
        "false_repair_rate": fmean(
            summary.false_repair_count > 0 for summary in summaries
        ),
        "mean_decision_steps": fmean(summary.decision_steps for summary in summaries),
        "mean_advance_count": fmean(summary.advance_count for summary in summaries),
        "mean_inspect_count": fmean(summary.inspect_count for summary in summaries),
        "informative_inspection_rate": fmean(
            summary.informative_inspection for summary in summaries
        ),
        "experiment_then_correct_repair_rate": fmean(
            summary.advance_count > 0
            and summary.informative_inspection
            and summary.correct_repair
            for summary in summaries
        ),
    }


def evaluate_policy(
    policy: GraphRecurrentPolicy,
    seeds: Sequence[int],
    *,
    device: str = "cpu",
    pair_builder: Callable[[int], DiagnosticPair] = build_generated_diagnostic_pair,
    pair_transform: Callable[[DiagnosticPair], DiagnosticPair] | None = None,
    disable_diagnostics: bool = False,
) -> dict[str, Any]:
    """Evaluate every world and balanced cue on held-out base-pair seeds."""
    if not seeds:
        raise ValueError("policy evaluation requires at least one seed")
    torch_device = torch.device(device)
    policy = policy.to(torch_device)
    policy.eval()
    condition_summaries: dict[CueCondition, list[EpisodeSummary]] = {
        CueCondition.AMBIGUOUS: [],
        CueCondition.REVEALED: [],
    }
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        pair = pair_builder(seed)
        if pair_transform is not None:
            pair = pair_transform(pair)
        for condition in (CueCondition.AMBIGUOUS, CueCondition.REVEALED):
            for world_index in (0, 1):
                cues = (0, 1) if condition is CueCondition.AMBIGUOUS else (
                    0
                    if pair.worlds[world_index].fault.kind.value == "blocked_edge"
                    else 1,
                )
                for cue in cues:
                    episode = DiagnosticEpisode(pair, condition, world_index, cue)
                    summary, actions = _run_episode(
                        policy,
                        episode,
                        torch_device,
                        disable_diagnostics,
                    )
                    condition_summaries[condition].append(summary)
                    rows.append(
                        {
                            "pair_seed": seed,
                            "pair_id": pair.pair_id,
                            "condition": condition.value,
                            "world_index": world_index,
                            "cue": cue,
                            "actions": actions,
                            "total_reward": summary.total_reward,
                            "recovered": summary.recovered,
                            "correct_repair": summary.correct_repair,
                            "advance_count": summary.advance_count,
                            "inspect_count": summary.inspect_count,
                            "informative_inspection": summary.informative_inspection,
                            "false_repair_count": summary.false_repair_count,
                        }
                    )
    return {
        "evaluation_version": POLICY_EVALUATION_VERSION,
        "base_pair_count": len(seeds),
        "diagnostics_disabled": disable_diagnostics,
        "pair_builder": pair_builder.__name__,
        "seeds": list(seeds),
        "ambiguous": _aggregate(condition_summaries[CueCondition.AMBIGUOUS]),
        "revealed": _aggregate(condition_summaries[CueCondition.REVEALED]),
        "rows": rows,
    }
