"""Oracle problem adapter for the first manual diagnostic pair."""

from __future__ import annotations

from faultline.env import Advance, ClearBlockage, Inspect, Isolate, MeasureFlow, Replace
from faultline.generation import DiagnosticPair, create_world_env
from faultline.oracle.model import DiagnosticProblem, TerminalPlan


def manual_pair_problem(pair: DiagnosticPair) -> DiagnosticProblem:
    """Create the exact bounded action model used by the milestone solver."""
    horizon = pair.reward.max_ticks
    return DiagnosticProblem(
        world_labels=tuple(world.label for world in pair.worlds),
        prior=(0.5, 0.5),
        initial_envs=tuple(
            create_world_env(pair, world, with_reward=True) for world in pair.worlds
        ),
        diagnostic_actions=(
            Advance(1),
            Inspect(pair.evidence_node),
            MeasureFlow("delivery"),
            Isolate(pair.intervention_edge),
        ),
        terminal_plans=(
            TerminalPlan(
                "clear_delivery",
                (ClearBlockage("delivery"), Advance(horizon)),
            ),
            TerminalPlan(
                "replace_processor",
                (Replace("processor"), Advance(horizon)),
            ),
            TerminalPlan("wait", (Advance(horizon),)),
            TerminalPlan(
                "restore_feed_then_clear_delivery",
                (
                    Isolate(pair.intervention_edge, isolated=False),
                    ClearBlockage("delivery"),
                    Advance(horizon),
                ),
            ),
            TerminalPlan(
                "restore_feed_then_replace_processor",
                (
                    Isolate(pair.intervention_edge, isolated=False),
                    Replace("processor"),
                    Advance(horizon),
                ),
            ),
        ),
    )
