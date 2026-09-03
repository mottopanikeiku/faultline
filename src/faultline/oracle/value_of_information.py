"""Decision value and information gain for bounded diagnostic actions."""

from __future__ import annotations

from dataclasses import dataclass

from faultline.env import Action
from faultline.oracle.active import action_partitions
from faultline.oracle.belief import entropy_bits
from faultline.oracle.model import DiagnosticProblem, WorldBranch
from faultline.oracle.passive import best_terminal_plan, root_branches


@dataclass(frozen=True, slots=True)
class ActionValue:
    action: Action
    expected_return: float
    recovery_probability: float
    decision_value: float
    information_gain_bits: float
    outcome_count: int


def analyze_action_at_belief(
    problem: DiagnosticProblem,
    branches: tuple[WorldBranch, ...],
    action: Action,
) -> ActionValue:
    """Evaluate one action and contingent commitment from an arbitrary exact belief."""
    if not branches:
        raise ValueError("cannot analyze an action at an empty belief")
    prior = [0.0] * len(problem.world_labels)
    for branch in branches:
        prior[branch.world_index] = branch.probability
    passive = best_terminal_plan(problem, branches)
    expected_return = 0.0
    recovery_probability = 0.0
    expected_posterior_entropy = 0.0
    partitions = action_partitions(problem, branches, action)
    for partition in partitions:
        terminal = best_terminal_plan(problem, partition.branches)
        expected_return += partition.probability * terminal.expected_return
        recovery_probability += partition.probability * terminal.recovery_probability
        expected_posterior_entropy += partition.probability * entropy_bits(partition.posterior)
    information_gain = entropy_bits(tuple(prior)) - expected_posterior_entropy
    return ActionValue(
        action=action,
        expected_return=expected_return,
        recovery_probability=recovery_probability,
        decision_value=expected_return - passive.expected_return,
        information_gain_bits=information_gain,
        outcome_count=len(partitions),
    )


def analyze_action(problem: DiagnosticProblem, action: Action) -> ActionValue:
    """Evaluate one diagnostic action from the problem's root belief."""
    return analyze_action_at_belief(problem, root_branches(problem), action)


def analyze_actions(problem: DiagnosticProblem) -> tuple[ActionValue, ...]:
    return tuple(analyze_action(problem, action) for action in problem.diagnostic_actions)
