"""Oracle problem adapter for generated two-world diagnostic pairs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from faultline.env import Advance, ClearBlockage, Inspect, Isolate, MeasureFlow
from faultline.oracle.model import DiagnosticProblem, TerminalPlan

if TYPE_CHECKING:
    from faultline.generation.diagnostic_pairs import DiagnosticPair, RepairAction


def _repair_name(repair: RepairAction) -> str:
    if isinstance(repair, ClearBlockage):
        return f"clear_{repair.edge}"
    return f"replace_{repair.node}"


def diagnostic_pair_problem(pair: DiagnosticPair) -> DiagnosticProblem:
    """Create the exact bounded action model shared by manual and generated pairs."""
    # Imported here because faultline.generation imports this package at module load.
    from faultline.generation.diagnostic_pairs import create_world_env

    horizon = pair.reward.max_ticks
    repairs = tuple(dict.fromkeys(world.correct_repair for world in pair.worlds))
    blocked_edge = next(
        repair.edge for repair in repairs if isinstance(repair, ClearBlockage)
    )
    terminal_plans: list[TerminalPlan] = [
        TerminalPlan(_repair_name(repair), (repair, Advance(horizon))) for repair in repairs
    ]
    terminal_plans.append(TerminalPlan("wait", (Advance(horizon),)))
    terminal_plans.extend(
        TerminalPlan(
            f"restore_{pair.intervention_edge}_then_{_repair_name(repair)}",
            (
                Isolate(pair.intervention_edge, isolated=False),
                repair,
                Advance(horizon),
            ),
        )
        for repair in repairs
    )
    return DiagnosticProblem(
        world_labels=tuple(world.label for world in pair.worlds),
        prior=(0.5, 0.5),
        initial_envs=tuple(
            create_world_env(pair, world, with_reward=True) for world in pair.worlds
        ),
        diagnostic_actions=(
            Advance(1),
            Inspect(pair.evidence_node),
            MeasureFlow(blocked_edge),
            Isolate(pair.intervention_edge),
        ),
        terminal_plans=tuple(terminal_plans),
    )
