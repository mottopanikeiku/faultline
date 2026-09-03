"""Causal interventions on policy-visible diagnostic evidence."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any

import numpy as np
import torch
from torch import Tensor

from faultline.agents.recurrent import GraphRecurrentPolicy
from faultline.faults import BlockedEdge
from faultline.generation import CueCondition, DiagnosticPair, build_generated_diagnostic_pair
from faultline.training.rl_env import (
    DiagnosticAction,
    DiagnosticEpisode,
    PolicyObservation,
)

COUNTERFACTUAL_VERSION = "evidence-swap-v1"
_TELEMETRY_COLUMNS = slice(9, 12)


@dataclass(frozen=True, slots=True)
class PreparedEvidence:
    observation: PolicyObservation
    hidden: Tensor
    prefix_actions: tuple[DiagnosticAction, ...]


def _policy_action(
    policy: GraphRecurrentPolicy,
    observation: PolicyObservation,
    hidden: Tensor,
    device: torch.device,
) -> tuple[DiagnosticAction, Tensor]:
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
    return DiagnosticAction(int(action.item())), next_hidden


def _prepare_evidence(
    policy: GraphRecurrentPolicy,
    episode: DiagnosticEpisode,
    device: torch.device,
) -> tuple[PreparedEvidence | None, tuple[DiagnosticAction, ...]]:
    hidden = policy.initial_hidden(1, device)
    prefix: list[DiagnosticAction] = []
    action, hidden = _policy_action(policy, episode.observe(), hidden, device)
    prefix.append(action)
    if action is not DiagnosticAction.ADVANCE:
        return None, tuple(prefix)
    episode.step(action)

    action, hidden = _policy_action(policy, episode.observe(), hidden, device)
    prefix.append(action)
    if action is not DiagnosticAction.INSPECT:
        return None, tuple(prefix)
    observation, _, terminated, _ = episode.step(action)
    if terminated:
        raise AssertionError("diagnostic preparation terminated before repair decision")
    return PreparedEvidence(observation, hidden, tuple(prefix)), tuple(prefix)


def swap_diagnostic_observation(
    receiver: PolicyObservation,
    donor: PolicyObservation,
) -> PolicyObservation:
    """Swap only revealed telemetry columns; topology, cue, history, and masks remain fixed."""
    if receiver.nodes.shape != donor.nodes.shape:
        raise ValueError("counterfactual observations must have aligned node shapes")
    nodes = receiver.nodes.copy()
    nodes[:, _TELEMETRY_COLUMNS] = donor.nodes[:, _TELEMETRY_COLUMNS]
    return PolicyObservation(
        nodes=nodes,
        adjacency=receiver.adjacency,
        node_mask=receiver.node_mask,
        global_features=receiver.global_features,
        action_mask=receiver.action_mask,
    )


def remove_diagnostic_observation(observation: PolicyObservation) -> PolicyObservation:
    nodes = observation.nodes.copy()
    nodes[:, _TELEMETRY_COLUMNS] = 0.0
    return PolicyObservation(
        nodes=nodes,
        adjacency=observation.adjacency,
        node_mask=observation.node_mask,
        global_features=observation.global_features,
        action_mask=observation.action_mask,
    )


def stale_diagnostic_observation(observation: PolicyObservation) -> PolicyObservation:
    """Replace telemetry by the public pre-dynamics target input and zero output."""
    nodes = observation.nodes.copy()
    target = int(np.argmax(nodes[:, 8]))
    nodes[:, _TELEMETRY_COLUMNS] = 0.0
    nodes[target, 9] = 1.0
    nodes[target, 10] = nodes[target, 7]
    return PolicyObservation(
        nodes=nodes,
        adjacency=observation.adjacency,
        node_mask=observation.node_mask,
        global_features=observation.global_features,
        action_mask=observation.action_mask,
    )


def uninformative_diagnostic_observation(
    left: PolicyObservation,
    right: PolicyObservation,
) -> PolicyObservation:
    """Feed the midpoint of the two valid results while retaining the receiver history."""
    nodes = left.nodes.copy()
    nodes[:, _TELEMETRY_COLUMNS] = 0.5 * (
        left.nodes[:, _TELEMETRY_COLUMNS] + right.nodes[:, _TELEMETRY_COLUMNS]
    )
    return PolicyObservation(
        nodes=nodes,
        adjacency=left.adjacency,
        node_mask=left.node_mask,
        global_features=left.global_features,
        action_mask=left.action_mask,
    )


def _correct_action(pair: DiagnosticPair, world_index: int) -> DiagnosticAction:
    return (
        DiagnosticAction.CLEAR_BLOCKAGE
        if isinstance(pair.worlds[world_index].fault, BlockedEdge)
        else DiagnosticAction.REPLACE_PROCESSOR
    )


def evaluate_evidence_interventions(
    policy: GraphRecurrentPolicy,
    seeds: list[int],
    *,
    device: str = "cpu",
    randomizations_per_world: int = 16,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Measure repair sensitivity when diagnostic evidence alone is swapped or degraded."""
    if not seeds or randomizations_per_world <= 0:
        raise ValueError("counterfactual evaluation requires seeds and randomization draws")
    torch_device = torch.device(device)
    policy = policy.to(torch_device)
    policy.eval()
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        pair = build_generated_diagnostic_pair(seed)
        for cue in (0, 1):
            prepared: list[PreparedEvidence | None] = []
            prefixes: list[tuple[DiagnosticAction, ...]] = []
            for world_index in (0, 1):
                episode = DiagnosticEpisode(
                    pair,
                    CueCondition.AMBIGUOUS,
                    world_index,
                    cue,
                )
                evidence, prefix = _prepare_evidence(policy, episode, torch_device)
                prepared.append(evidence)
                prefixes.append(prefix)
            pair_eligible = prepared[0] is not None and prepared[1] is not None
            for world_index in (0, 1):
                correct = _correct_action(pair, world_index)
                donor_correct = _correct_action(pair, 1 - world_index)
                if not pair_eligible:
                    rows.append(
                        {
                            "pair_seed": seed,
                            "cue": cue,
                            "world_index": world_index,
                            "eligible": False,
                            "prefix_actions": [
                                action.name.lower()
                                for action in prefixes[world_index]
                            ],
                            "correct_action": correct.name.lower(),
                        }
                    )
                    continue

                evidence = prepared[world_index]
                donor = prepared[1 - world_index]
                if evidence is None or donor is None:
                    raise AssertionError("eligible counterfactual pair lost prepared evidence")
                normal, _ = _policy_action(
                    policy,
                    evidence.observation,
                    evidence.hidden,
                    torch_device,
                )
                swapped, _ = _policy_action(
                    policy,
                    swap_diagnostic_observation(evidence.observation, donor.observation),
                    evidence.hidden,
                    torch_device,
                )
                removed, _ = _policy_action(
                    policy,
                    remove_diagnostic_observation(evidence.observation),
                    evidence.hidden,
                    torch_device,
                )
                stale, _ = _policy_action(
                    policy,
                    stale_diagnostic_observation(evidence.observation),
                    evidence.hidden,
                    torch_device,
                )
                uninformative, _ = _policy_action(
                    policy,
                    uninformative_diagnostic_observation(
                        evidence.observation,
                        donor.observation,
                    ),
                    evidence.hidden,
                    torch_device,
                )
                randomized_actions: list[DiagnosticAction] = []
                for _ in range(randomizations_per_world):
                    random_source = prepared[int(rng.integers(0, 2))]
                    if random_source is None:
                        raise AssertionError("random evidence source missing")
                    randomized, _ = _policy_action(
                        policy,
                        swap_diagnostic_observation(
                            evidence.observation,
                            random_source.observation,
                        ),
                        evidence.hidden,
                        torch_device,
                    )
                    randomized_actions.append(randomized)
                rows.append(
                    {
                        "pair_seed": seed,
                        "cue": cue,
                        "world_index": world_index,
                        "eligible": True,
                        "prefix_actions": [
                            action.name.lower() for action in evidence.prefix_actions
                        ],
                        "correct_action": correct.name.lower(),
                        "donor_correct_action": donor_correct.name.lower(),
                        "normal_action": normal.name.lower(),
                        "swapped_action": swapped.name.lower(),
                        "removed_action": removed.name.lower(),
                        "stale_action": stale.name.lower(),
                        "uninformative_action": uninformative.name.lower(),
                        "normal_correct": normal is correct,
                        "swapped_to_donor": swapped is donor_correct,
                        "swapped_changed_action": swapped is not normal,
                        "removed_changed_action": removed is not normal,
                        "stale_changed_action": stale is not normal,
                        "uninformative_changed_action": uninformative is not normal,
                        "randomized_correct_rate": fmean(
                            action is correct for action in randomized_actions
                        ),
                    }
                )

    eligible = [row for row in rows if row["eligible"]]
    total = len(rows)

    def eligible_mean(field: str) -> float | None:
        return fmean(float(row[field]) for row in eligible) if eligible else None

    return {
        "evaluation_version": COUNTERFACTUAL_VERSION,
        "base_pair_count": len(seeds),
        "seeds": seeds,
        "opportunity_count": total,
        "eligible_count": len(eligible),
        "diagnostic_eligibility_rate": len(eligible) / total,
        "normal_correct_repair_rate": eligible_mean("normal_correct"),
        "swap_repair_switch_rate": eligible_mean("swapped_changed_action"),
        "swap_donor_repair_rate": eligible_mean("swapped_to_donor"),
        "causal_evidence_use_rate": (
            fmean(
                bool(row["normal_correct"])
                and bool(row["swapped_to_donor"])
                and bool(row["swapped_changed_action"])
                for row in eligible
            )
            if eligible
            else None
        ),
        "removal_change_rate": eligible_mean("removed_changed_action"),
        "stale_change_rate": eligible_mean("stale_changed_action"),
        "uninformative_change_rate": eligible_mean("uninformative_changed_action"),
        "randomized_correct_rate": eligible_mean("randomized_correct_rate"),
        "rows": rows,
    }
