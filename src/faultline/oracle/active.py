"""Exact short-horizon diagnostic policy enumeration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import fsum
from typing import TypeAlias

from faultline.env import Action, ActionResult
from faultline.oracle.model import DiagnosticProblem, TerminalPlan, WorldBranch
from faultline.oracle.passive import best_terminal_plan, root_branches

ObservationValue: TypeAlias = bool | int | float | str
ObservationKey: TypeAlias = tuple[tuple[str, ObservationValue], ...]
_VALUE_TOLERANCE = 1e-12


class DecisionKind(StrEnum):
    COMMIT = "commit"
    DIAGNOSE = "diagnose"


@dataclass(slots=True)
class ActionPartition:
    observation: ObservationKey
    probability: float
    posterior: tuple[float, ...]
    branches: tuple[WorldBranch, ...]


@dataclass(frozen=True, slots=True)
class OutcomePolicy:
    observation: ObservationKey
    probability: float
    posterior: tuple[float, ...]
    decision: OracleDecision


@dataclass(frozen=True, slots=True)
class OracleDecision:
    expected_return: float
    recovery_probability: float
    kind: DecisionKind
    terminal_plan: TerminalPlan | None = None
    diagnostic_action: Action | None = None
    outcomes: tuple[OutcomePolicy, ...] = ()


def observation_key(result: ActionResult) -> ObservationKey:
    """Canonicalize everything the policy receives from an action."""
    values: dict[str, ObservationValue] = dict(result.observation)
    values["_accepted"] = result.accepted
    values["_error"] = result.error.value if result.error is not None else ""
    values["_reward"] = result.reward
    values["_terminated"] = result.terminated
    return tuple(sorted(values.items()))


def action_partitions(
    problem: DiagnosticProblem,
    branches: tuple[WorldBranch, ...],
    action: Action,
) -> tuple[ActionPartition, ...]:
    """Execute one shared action and partition posterior branches by public outcome."""
    grouped: dict[ObservationKey, list[WorldBranch]] = {}
    for branch in branches:
        next_branch = branch.clone()
        result = next_branch.env.act(action)
        key = observation_key(result)
        grouped.setdefault(key, []).append(next_branch)

    partitions: list[ActionPartition] = []
    for key, members in grouped.items():
        probability = fsum(member.probability for member in members)
        normalized = tuple(
            WorldBranch(member.world_index, member.probability / probability, member.env)
            for member in members
        )
        posterior = [0.0] * len(problem.world_labels)
        for member in normalized:
            posterior[member.world_index] = member.probability
        partitions.append(
            ActionPartition(
                observation=key,
                probability=probability,
                posterior=tuple(posterior),
                branches=normalized,
            )
        )
    return tuple(partitions)


def _solve(
    problem: DiagnosticProblem,
    branches: tuple[WorldBranch, ...],
    remaining_diagnostic_actions: int,
) -> OracleDecision:
    terminal = best_terminal_plan(problem, branches)
    best = OracleDecision(
        expected_return=terminal.expected_return,
        recovery_probability=terminal.recovery_probability,
        kind=DecisionKind.COMMIT,
        terminal_plan=terminal.plan,
    )
    if remaining_diagnostic_actions == 0:
        return best

    for action in problem.diagnostic_actions:
        outcomes: list[OutcomePolicy] = []
        expected_return = 0.0
        recovery_probability = 0.0
        for partition in action_partitions(problem, branches, action):
            child = _solve(
                problem,
                partition.branches,
                remaining_diagnostic_actions - 1,
            )
            expected_return += partition.probability * child.expected_return
            recovery_probability += partition.probability * child.recovery_probability
            outcomes.append(
                OutcomePolicy(
                    observation=partition.observation,
                    probability=partition.probability,
                    posterior=partition.posterior,
                    decision=child,
                )
            )
        if expected_return > best.expected_return + _VALUE_TOLERANCE:
            best = OracleDecision(
                expected_return=expected_return,
                recovery_probability=recovery_probability,
                kind=DecisionKind.DIAGNOSE,
                diagnostic_action=action,
                outcomes=tuple(outcomes),
            )
    return best


def solve_active_from_branches(
    problem: DiagnosticProblem,
    branches: tuple[WorldBranch, ...],
    max_diagnostic_actions: int = 3,
) -> OracleDecision:
    """Solve from an externally conditioned, normalized exact belief."""
    if not branches:
        raise ValueError("active solver requires at least one belief branch")
    if not 0 <= max_diagnostic_actions <= 6:
        raise ValueError("diagnostic horizon must be between zero and six")
    if abs(fsum(branch.probability for branch in branches) - 1.0) > _VALUE_TOLERANCE:
        raise ValueError("belief branch probabilities must sum to one")
    return _solve(problem, branches, max_diagnostic_actions)


def solve_active(
    problem: DiagnosticProblem,
    max_diagnostic_actions: int = 3,
) -> OracleDecision:
    """Enumerate the optimal contingent diagnostic policy to the requested depth."""
    return solve_active_from_branches(
        problem,
        root_branches(problem),
        max_diagnostic_actions,
    )
