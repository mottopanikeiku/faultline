"""Development-only chain profile outside v1 training parameter ranges."""

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

OOD_GENERATOR_VERSION = "diagnostic-chain-ood-v1"


def build_ood_diagnostic_pair(seed: int) -> DiagnosticPair:
    """Generate larger chains and numeric values absent from diagnostic-chain-v1."""
    if seed < 0:
        raise ValueError("OOD seed must be non-negative")
    rng = np.random.default_rng(seed)
    node_count = int(rng.integers(13, 21))
    processor_positions = tuple(
        position for position in range(1, node_count - 1) if position % 2 == 1
    )
    fault_position = int(rng.choice(processor_positions))
    rate = float(rng.choice((0.6, 3.5, 4.0)))
    transport_fraction = float(rng.choice((0.55, 0.65, 0.9)))
    transport_rate = rate * transport_fraction
    capacity_multiple = float(rng.choice((3.5, 4.0)))
    preload_fraction = float(rng.choice((1.1, 1.25)))
    preload = rate * preload_fraction
    time_cost = float(rng.choice((0.075, 0.125, 0.2)))
    diagnostic_cost = float(rng.choice((0.15, 0.3, 0.5)))

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
    blocked = DiagnosticWorld(
        "A",
        BlockedEdge(blocked_edge),
        ClearBlockage(blocked_edge),
    )
    failed = DiagnosticWorld(
        "B",
        FailedProcessor(evidence_node),
        Replace(evidence_node),
    )
    ordered = (blocked, failed) if int(rng.integers(0, 2)) == 0 else (failed, blocked)
    worlds = (
        DiagnosticWorld("A", ordered[0].fault, ordered[0].correct_repair),
        DiagnosticWorld("B", ordered[1].fault, ordered[1].correct_repair),
    )
    return DiagnosticPair(
        pair_id=f"ood-chain-v1-{seed:016x}",
        generator_version=OOD_GENERATOR_VERSION,
        parameters=(
            ("family", "larger_linear_chain_ood"),
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
        reward=RewardConfig(
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
        ),
        intervention_edge=intervention_edge,
        evidence_node=evidence_node,
    )
