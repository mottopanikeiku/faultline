"""Validated diagnostic-pair primitives and the first hand-constructed pair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from faultline.env import (
    ActionResult,
    Advance,
    ClearBlockage,
    Edge,
    EpisodeMetrics,
    FactoryEnv,
    FactoryGraph,
    Inspect,
    Isolate,
    Node,
    NodeType,
    Replace,
    RewardConfig,
)
from faultline.env.observation import inspect_node, measure_edge, observe_status
from faultline.faults import BlockedEdge, FailedProcessor, LatentFault

RepairAction: TypeAlias = Replace | ClearBlockage
CanonicalObservation: TypeAlias = tuple[tuple[str, bool | int | float | str], ...]


@dataclass(frozen=True, slots=True)
class PassiveSnapshot:
    """All telemetry available without changing plant dynamics."""

    status: CanonicalObservation
    nodes: tuple[CanonicalObservation, ...]
    edges: tuple[CanonicalObservation, ...]


@dataclass(frozen=True, slots=True)
class DiagnosticWorld:
    label: str
    fault: LatentFault
    correct_repair: RepairAction


@dataclass(frozen=True, slots=True)
class DiagnosticPair:
    seed: int
    graph: FactoryGraph
    worlds: tuple[DiagnosticWorld, DiagnosticWorld]
    reward: RewardConfig
    intervention_edge: str
    evidence_node: str


@dataclass(frozen=True, slots=True)
class RepairEvaluation:
    repair: RepairAction
    recovered: bool
    total_return: float
    final_throughput: float
    false_repairs: int


@dataclass(frozen=True, slots=True)
class ActiveEvaluation:
    selected_repair: RepairAction
    evidence: CanonicalObservation
    recovered: bool
    total_return: float
    metrics: EpisodeMetrics


def _canonical(observation: dict[str, bool | int | float | str]) -> CanonicalObservation:
    return tuple(sorted(observation.items()))


def full_passive_snapshot(env: FactoryEnv) -> PassiveSnapshot:
    """Collect every non-dynamical public measurement in stable topology order."""
    return PassiveSnapshot(
        status=_canonical(observe_status(env.state)),
        nodes=tuple(
            _canonical(inspect_node(env.graph, env.state, index))
            for index in range(env.graph.node_count)
        ),
        edges=tuple(
            _canonical(measure_edge(env.graph, env.state, index))
            for index in range(env.graph.edge_count)
        ),
    )


def build_manual_diagnostic_pair(seed: int = 42) -> DiagnosticPair:
    """Build two initially identical worlds separated by an upstream isolation experiment."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    graph = FactoryGraph.compile(
        nodes=[
            Node("source", NodeType.SOURCE, 2.0, output_capacity=4.0),
            Node(
                "processor",
                NodeType.PROCESSOR,
                2.0,
                input_capacity=4.0,
                output_capacity=4.0,
                initial_input=2.0,
            ),
            Node("sink", NodeType.SINK, 2.0, input_capacity=4.0),
        ],
        edges=[
            Edge("feed", "source", "processor", 2.0),
            Edge("delivery", "processor", "sink", 2.0),
        ],
    )
    blocked = DiagnosticWorld(
        label="A",
        fault=BlockedEdge("delivery"),
        correct_repair=ClearBlockage("delivery"),
    )
    failed = DiagnosticWorld(
        label="B",
        fault=FailedProcessor("processor"),
        correct_repair=Replace("processor"),
    )
    worlds = (blocked, failed) if seed % 2 == 0 else (failed, blocked)
    worlds = (
        DiagnosticWorld("A", worlds[0].fault, worlds[0].correct_repair),
        DiagnosticWorld("B", worlds[1].fault, worlds[1].correct_repair),
    )
    return DiagnosticPair(
        seed=seed,
        graph=graph,
        worlds=worlds,
        reward=RewardConfig(
            target_throughput=2.0,
            throughput_value=1.0,
            time_cost=0.1,
            passive_cost=0.02,
            diagnostic_cost=0.25,
            repair_cost=2.0,
            false_repair_cost=3.0,
            recovery_fraction=1.0,
            recovery_ticks=3,
            recovery_bonus=5.0,
            max_ticks=12,
            max_actions=12,
        ),
        intervention_edge="feed",
        evidence_node="processor",
    )


def create_world_env(
    pair: DiagnosticPair,
    world: DiagnosticWorld,
    *,
    with_reward: bool = False,
) -> FactoryEnv:
    return FactoryEnv.create(
        pair.graph,
        world.fault,
        reward_config=pair.reward if with_reward else None,
        check_invariants=True,
    )


def evaluate_repair(
    pair: DiagnosticPair,
    world: DiagnosticWorld,
    repair: RepairAction,
) -> RepairEvaluation:
    """Commit to one repair, then score operational recovery to termination."""
    env = create_world_env(pair, world, with_reward=True)
    env.act(repair)
    env.act(Advance(pair.reward.max_ticks))
    if env.reward_tracker is None:
        raise AssertionError("reward tracker missing from repair evaluation")
    metrics = env.reward_tracker.snapshot()
    return RepairEvaluation(
        repair=repair,
        recovered=metrics.recovered,
        total_return=metrics.total_reward,
        final_throughput=env.state.last_delivered,
        false_repairs=metrics.false_repair_count,
    )


def diagnostic_evidence(pair: DiagnosticPair, world: DiagnosticWorld) -> CanonicalObservation:
    """Run the common intervention and return its requested public evidence."""
    env = create_world_env(pair, world)
    env.act(Isolate(pair.intervention_edge))
    env.act(Advance(1))
    result = env.act(Inspect(pair.evidence_node))
    return _canonical(result.observation)


def run_contingent_active_policy(
    pair: DiagnosticPair,
    world: DiagnosticWorld,
) -> ActiveEvaluation:
    """Use the intervention response—not latent state—to select one of the two repairs."""
    env = create_world_env(pair, world, with_reward=True)
    env.act(Isolate(pair.intervention_edge))
    env.act(Advance(1))
    evidence_result: ActionResult = env.act(Inspect(pair.evidence_node))
    output_buffer = float(evidence_result.observation["output_buffer"])
    repair: RepairAction = (
        ClearBlockage("delivery") if output_buffer > 0.0 else Replace("processor")
    )
    env.act(Isolate(pair.intervention_edge, isolated=False))
    env.act(repair)
    env.act(Advance(pair.reward.max_ticks))
    if env.reward_tracker is None:
        raise AssertionError("reward tracker missing from active evaluation")
    metrics = env.reward_tracker.snapshot()
    return ActiveEvaluation(
        selected_repair=repair,
        evidence=_canonical(evidence_result.observation),
        recovered=metrics.recovered,
        total_return=metrics.total_reward,
        metrics=metrics,
    )
