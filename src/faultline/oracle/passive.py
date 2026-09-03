"""Exact terminal-decision evaluation for finite latent worlds."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum

from faultline.env import FactoryEnv
from faultline.oracle.model import DiagnosticProblem, TerminalPlan, WorldBranch


@dataclass(frozen=True, slots=True)
class TerminalValue:
    plan: TerminalPlan
    expected_return: float
    recovery_probability: float
    world_returns: tuple[float, ...]
    world_recovered: tuple[bool, ...]


def root_branches(problem: DiagnosticProblem) -> tuple[WorldBranch, ...]:
    return tuple(
        WorldBranch(index, probability, env.clone())
        for index, (probability, env) in enumerate(
            zip(problem.prior, problem.initial_envs, strict=True)
        )
        if probability > 0.0
    )


def execute_terminal_plan(env: FactoryEnv, plan: TerminalPlan) -> tuple[float, bool]:
    """Execute a terminal plan on a cloned environment and return final total reward."""
    branch = env.clone()
    for action in plan.actions:
        if branch.reward_tracker is not None and branch.reward_tracker.terminated:
            break
        branch.act(action)
    if branch.reward_tracker is None:
        raise AssertionError("oracle terminal branch has no reward tracker")
    metrics = branch.reward_tracker.snapshot()
    return metrics.total_reward, metrics.recovered


def evaluate_terminal_plan(
    branches: tuple[WorldBranch, ...],
    plan: TerminalPlan,
) -> TerminalValue:
    if not branches:
        raise ValueError("cannot evaluate an empty belief")
    returns: list[float] = []
    recovered: list[bool] = []
    for branch in branches:
        value, success = execute_terminal_plan(branch.env, plan)
        returns.append(value)
        recovered.append(success)
    expected_return = fsum(
        branch.probability * value
        for branch, value in zip(branches, returns, strict=True)
    )
    recovery_probability = fsum(
        branch.probability
        for branch, success in zip(branches, recovered, strict=True)
        if success
    )
    return TerminalValue(
        plan=plan,
        expected_return=expected_return,
        recovery_probability=recovery_probability,
        world_returns=tuple(returns),
        world_recovered=tuple(recovered),
    )


def best_terminal_plan(
    problem: DiagnosticProblem,
    branches: tuple[WorldBranch, ...],
) -> TerminalValue:
    """Choose the highest-return shared commitment with stable tie breaking."""
    evaluations = tuple(
        evaluate_terminal_plan(branches, plan) for plan in problem.terminal_plans
    )
    return max(evaluations, key=lambda evaluation: evaluation.expected_return)


def solve_passive(problem: DiagnosticProblem) -> TerminalValue:
    """Evaluate a policy that must commit from the shared passive history."""
    return best_terminal_plan(problem, root_branches(problem))
