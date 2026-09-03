"""Agent-visible observations with an explicit privileged-state boundary."""

from __future__ import annotations

from typing import TypeAlias

from faultline.env.graph import FactoryGraph, NodeType
from faultline.env.state import FactoryState

PublicScalar: TypeAlias = bool | int | float | str
PublicObservation: TypeAlias = dict[str, PublicScalar]


def observe_status(state: FactoryState) -> PublicObservation:
    """Return the always-visible operational summary."""
    return {
        "event": "status",
        "tick": state.tick,
        "throughput": state.last_delivered,
        "delivered_total": state.delivered_total,
    }


def inspect_node(
    graph: FactoryGraph,
    state: FactoryState,
    node_index: int,
) -> PublicObservation:
    """Reveal documented node telemetry, never latent fault flags."""
    node_type = NodeType(int(graph.node_types[node_index]))
    return {
        "event": "inspect",
        "node": graph.node_names[node_index],
        "node_type": node_type.name.lower(),
        "enabled": bool(state.node_enabled[node_index]),
        "input_buffer": float(state.inputs[node_index]),
        "output_buffer": float(state.outputs[node_index]),
        "nominal_rate": float(graph.rates[node_index]),
    }


def measure_edge(
    graph: FactoryGraph,
    state: FactoryState,
    edge_index: int,
) -> PublicObservation:
    """Reveal transport telemetry without blockage truth."""
    return {
        "event": "measure_flow",
        "edge": graph.edge_names[edge_index],
        "source": graph.node_names[int(graph.edge_sources[edge_index])],
        "target": graph.node_names[int(graph.edge_targets[edge_index])],
        "enabled": bool(state.edge_enabled[edge_index]),
        "flow": float(state.last_edge_flow[edge_index]),
        "nominal_capacity": float(graph.edge_capacities[edge_index]),
    }
