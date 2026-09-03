"""Procedural construction of exactly confusable two-fault factory tasks."""

from __future__ import annotations

import numpy as np

from faultline.env import (
    ClearBlockage,
    Edge,
    FactoryGraph,
    Node,
    NodeType,
    Replace,
    RewardConfig,
)
from faultline.faults import BlockedEdge, FailedProcessor
from faultline.generation.diagnostic_pairs import DiagnosticPair, DiagnosticWorld

GENERATOR_VERSION = "diagnostic-chain-v1"
_MAX_SEED = 2**63 - 1


def build_generated_diagnostic_pair(seed: int) -> DiagnosticPair:
    """Generate a chain where blockage and processor failure share all initial telemetry."""
    if not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"seed must be between zero and {_MAX_SEED}")
    rng = np.random.default_rng(seed)
    node_count = int(rng.integers(4, 13))
    processor_positions = tuple(
        position for position in range(1, node_count - 1) if position % 2 == 1
    )
    fault_position = int(rng.choice(processor_positions))
    rate = float(rng.choice((1.0, 1.5, 2.0, 2.5, 3.0)))
    transport_fraction = float(rng.choice((0.75, 1.0)))
    transport_rate = rate * transport_fraction
    capacity_multiple = float(rng.choice((2.0, 2.5, 3.0)))
    preload_fraction = float(rng.choice((0.5, 0.75, 1.0)))
    preload = rate * preload_fraction
    time_cost = float(rng.choice((0.05, 0.1, 0.15)))
    diagnostic_cost = float(rng.choice((0.1, 0.25, 0.4)))

    nodes = [
        Node(
            "source",
            NodeType.SOURCE,
            rate,
            output_capacity=capacity_multiple * rate,
        )
    ]
    for position in range(1, node_count - 1):
        kind = NodeType.PROCESSOR if position % 2 else NodeType.BUFFER
        prefix = "processor" if kind is NodeType.PROCESSOR else "buffer"
        nodes.append(
            Node(
                f"{prefix}_{position:03d}",
                kind,
                rate,
                input_capacity=capacity_multiple * rate,
                output_capacity=capacity_multiple * rate,
                initial_input=preload if position == fault_position else 0.0,
            )
        )
    nodes.append(
        Node(
            "sink",
            NodeType.SINK,
            rate,
            input_capacity=capacity_multiple * rate,
        )
    )
    edges = [
        Edge(
            f"transport_{position:03d}",
            nodes[position].name,
            nodes[position + 1].name,
            transport_rate,
        )
        for position in range(node_count - 1)
    ]
    graph = FactoryGraph.compile(nodes, edges)

    evidence_node = nodes[fault_position].name
    blocked_edge = edges[fault_position].name
    intervention_edge = edges[fault_position - 1].name
    blocked_world = DiagnosticWorld(
        "A",
        BlockedEdge(blocked_edge),
        ClearBlockage(blocked_edge),
    )
    failed_world = DiagnosticWorld(
        "B",
        FailedProcessor(evidence_node),
        Replace(evidence_node),
    )
    ordered = (
        (blocked_world, failed_world)
        if int(rng.integers(0, 2)) == 0
        else (failed_world, blocked_world)
    )
    worlds = (
        DiagnosticWorld("A", ordered[0].fault, ordered[0].correct_repair),
        DiagnosticWorld("B", ordered[1].fault, ordered[1].correct_repair),
    )
    reward = RewardConfig(
        target_throughput=transport_rate,
        throughput_value=1.0,
        time_cost=time_cost,
        passive_cost=0.02,
        diagnostic_cost=diagnostic_cost,
        repair_cost=2.0,
        false_repair_cost=3.0,
        recovery_fraction=1.0,
        recovery_ticks=3,
        recovery_bonus=5.0 + 0.25 * node_count,
        max_ticks=2 * node_count + 6,
        max_actions=12,
    )
    return DiagnosticPair(
        pair_id=f"chain-v1-{seed:016x}",
        generator_version=GENERATOR_VERSION,
        parameters=(
            ("family", "linear_chain"),
            ("node_count", node_count),
            ("fault_position", fault_position),
            ("rate", rate),
            ("transport_rate", transport_rate),
            ("capacity_multiple", capacity_multiple),
            ("preload", preload),
            ("time_cost", time_cost),
            ("diagnostic_cost", diagnostic_cost),
        ),
        seed=seed,
        graph=graph,
        worlds=worlds,
        reward=reward,
        intervention_edge=intervention_edge,
        evidence_node=evidence_node,
    )
