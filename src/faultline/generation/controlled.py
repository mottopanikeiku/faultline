"""Matched task blocks that vary cue-fault dependence, not marginal difficulty features."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import fsum

from faultline.faults import BlockedEdge
from faultline.generation.diagnostic_pairs import DiagnosticPair
from faultline.oracle import diagnostic_pair_problem, solve_active_from_branches
from faultline.oracle.model import WorldBranch
from faultline.oracle.passive import best_terminal_plan, root_branches


class CueCondition(StrEnum):
    AMBIGUOUS = "ambiguous"
    REVEALED = "revealed"


@dataclass(frozen=True, slots=True)
class CueTask:
    """A base pair plus one public binary cue assignment over its latent worlds."""

    task_id: str
    pair: DiagnosticPair
    condition: CueCondition
    cues: tuple[int, int]

    def __post_init__(self) -> None:
        if len(self.cues) != len(self.pair.worlds) or any(cue not in (0, 1) for cue in self.cues):
            raise ValueError("cue assignments must provide one binary value per world")
        if self.condition is CueCondition.AMBIGUOUS and len(set(self.cues)) != 1:
            raise ValueError("ambiguous cue task must share one cue across worlds")
        if self.condition is CueCondition.REVEALED and len(set(self.cues)) != len(self.cues):
            raise ValueError("revealed cue task must distinguish every world")


@dataclass(frozen=True, slots=True)
class MatchedEpBlock:
    block_id: str
    ambiguous_tasks: tuple[CueTask, CueTask]
    revealed_task: CueTask


@dataclass(frozen=True, slots=True)
class CueTaskValue:
    passive_expected_return: float
    active_expected_return: float
    epistemic_pressure: float
    passive_recovery_probability: float
    active_recovery_probability: float


@dataclass(frozen=True, slots=True)
class MatchedBlockValue:
    block_id: str
    ambiguous: CueTaskValue
    revealed: CueTaskValue
    fault_marginals_equal: bool
    cue_marginals_equal: bool
    cue_fault_joint_equal: bool


def build_matched_ep_block(pair: DiagnosticPair) -> MatchedEpBlock:
    """Pair two ambiguous cue values against a perfectly predictive balanced cue."""
    revealed_cues = (
        0 if isinstance(pair.worlds[0].fault, BlockedEdge) else 1,
        0 if isinstance(pair.worlds[1].fault, BlockedEdge) else 1,
    )
    ambiguous_zero = CueTask(
        task_id=f"{pair.pair_id}-ambiguous-cue0",
        pair=pair,
        condition=CueCondition.AMBIGUOUS,
        cues=(0, 0),
    )
    ambiguous_one = CueTask(
        task_id=f"{pair.pair_id}-ambiguous-cue1",
        pair=pair,
        condition=CueCondition.AMBIGUOUS,
        cues=(1, 1),
    )
    revealed = CueTask(
        task_id=f"{pair.pair_id}-revealed",
        pair=pair,
        condition=CueCondition.REVEALED,
        cues=revealed_cues,
    )
    return MatchedEpBlock(
        block_id=f"matched-{pair.pair_id}",
        ambiguous_tasks=(ambiguous_zero, ambiguous_one),
        revealed_task=revealed,
    )


def _conditioned_branches(
    branches: tuple[WorldBranch, ...],
    cues: tuple[int, int],
) -> tuple[tuple[float, tuple[WorldBranch, ...]], ...]:
    groups: dict[int, list[WorldBranch]] = {}
    for branch in branches:
        groups.setdefault(cues[branch.world_index], []).append(branch)
    conditioned: list[tuple[float, tuple[WorldBranch, ...]]] = []
    for members in groups.values():
        mass = fsum(member.probability for member in members)
        conditioned.append(
            (
                mass,
                tuple(
                    WorldBranch(
                        member.world_index,
                        member.probability / mass,
                        member.env,
                    )
                    for member in members
                ),
            )
        )
    return tuple(conditioned)


def evaluate_cue_task(task: CueTask, *, diagnostic_depth: int = 2) -> CueTaskValue:
    """Condition on the free public cue, then solve passive and active decisions exactly."""
    problem = diagnostic_pair_problem(task.pair)
    passive_return = 0.0
    active_return = 0.0
    passive_recovery = 0.0
    active_recovery = 0.0
    for mass, branches in _conditioned_branches(root_branches(problem), task.cues):
        passive = best_terminal_plan(problem, branches)
        active = solve_active_from_branches(problem, branches, diagnostic_depth)
        passive_return += mass * passive.expected_return
        active_return += mass * active.expected_return
        passive_recovery += mass * passive.recovery_probability
        active_recovery += mass * active.recovery_probability
    return CueTaskValue(
        passive_expected_return=passive_return,
        active_expected_return=active_return,
        epistemic_pressure=active_return - passive_return,
        passive_recovery_probability=passive_recovery,
        active_recovery_probability=active_recovery,
    )


def _marginals(
    tasks: tuple[CueTask, ...],
) -> tuple[dict[str, float], dict[int, float], dict[tuple[int, str], float]]:
    fault_marginal: dict[str, float] = {}
    cue_marginal: dict[int, float] = {}
    joint: dict[tuple[int, str], float] = {}
    task_probability = 1.0 / len(tasks)
    for task in tasks:
        for world_index, (world, world_probability) in enumerate(
            zip(task.pair.worlds, (0.5, 0.5), strict=True)
        ):
            probability = task_probability * world_probability
            kind = world.fault.kind.value
            fault_marginal[kind] = fault_marginal.get(kind, 0.0) + probability
            cue = task.cues[world_index]
            cue_marginal[cue] = cue_marginal.get(cue, 0.0) + probability
            joint[(cue, kind)] = joint.get((cue, kind), 0.0) + probability
    return fault_marginal, cue_marginal, joint


def evaluate_matched_block(block: MatchedEpBlock) -> MatchedBlockValue:
    ambiguous_values = tuple(evaluate_cue_task(task) for task in block.ambiguous_tasks)
    if ambiguous_values[0] != ambiguous_values[1]:
        raise AssertionError("ambiguous cue labels changed oracle value")
    revealed_value = evaluate_cue_task(block.revealed_task)
    ambiguous_marginals = _marginals(block.ambiguous_tasks)
    revealed_marginals = _marginals((block.revealed_task,))
    return MatchedBlockValue(
        block_id=block.block_id,
        ambiguous=ambiguous_values[0],
        revealed=revealed_value,
        fault_marginals_equal=ambiguous_marginals[0] == revealed_marginals[0],
        cue_marginals_equal=ambiguous_marginals[1] == revealed_marginals[1],
        cue_fault_joint_equal=ambiguous_marginals[2] == revealed_marginals[2],
    )
