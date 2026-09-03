"""Deterministic factory layouts used by tests, demos, and benchmarks."""

from __future__ import annotations

from faultline.env.graph import Edge, FactoryGraph, Node, NodeType


def chain_factory(node_count: int = 8, *, rate: float = 2.0) -> FactoryGraph:
    """Build a source-to-sink chain with alternating processors and buffers."""
    if node_count < 3:
        raise ValueError("chain factory requires at least three nodes")
    if rate <= 0.0:
        raise ValueError("chain factory rate must be positive")

    nodes = [Node("source", NodeType.SOURCE, rate, output_capacity=2.0 * rate)]
    for position in range(1, node_count - 1):
        kind = NodeType.PROCESSOR if position % 2 else NodeType.BUFFER
        prefix = "processor" if kind is NodeType.PROCESSOR else "buffer"
        nodes.append(
            Node(
                f"{prefix}_{position:03d}",
                kind,
                rate,
                input_capacity=2.0 * rate,
                output_capacity=2.0 * rate,
            )
        )
    nodes.append(Node("sink", NodeType.SINK, rate, input_capacity=2.0 * rate))

    edges = [
        Edge(
            name=f"transport_{position:03d}",
            source=nodes[position].name,
            target=nodes[position + 1].name,
            capacity=rate,
        )
        for position in range(node_count - 1)
    ]
    return FactoryGraph.compile(nodes, edges)
