"""Dependency-free text rendering for deterministic simulator episodes."""

from __future__ import annotations

from faultline.env.environment import Interaction
from faultline.env.graph import FactoryGraph, NodeType
from faultline.env.state import FactoryState


def render_factory(
    graph: FactoryGraph,
    state: FactoryState,
    *,
    debug: bool = False,
) -> str:
    """Render topology and material state; latent markers require explicit debug mode."""
    lines = [
        f"FACTORY tick={state.tick} throughput={state.last_delivered:.2f} "
        f"delivered={state.delivered_total:.2f}",
        "NODES",
    ]
    for index, name in enumerate(graph.node_names):
        node_type = NodeType(int(graph.node_types[index])).name.lower()
        status = "on" if state.node_enabled[index] else "off"
        latent = ""
        if debug:
            markers: list[str] = []
            if state.node_failed[index]:
                markers.append("FAILED")
            if state.node_backpressured[index]:
                markers.append("BACKPRESSURE")
            if markers:
                latent = " [" + ",".join(markers) + "]"
        lines.append(
            f"  {name:<18} {node_type:<9} {status:<3} "
            f"in={state.inputs[index]:6.2f}/{graph.input_capacities[index]:.2f} "
            f"out={state.outputs[index]:6.2f}/{graph.output_capacities[index]:.2f}"
            f"{latent}"
        )

    lines.append("EDGES")
    for index, name in enumerate(graph.edge_names):
        source = graph.node_names[int(graph.edge_sources[index])]
        target = graph.node_names[int(graph.edge_targets[index])]
        status = "on" if state.edge_enabled[index] else "isolated"
        latent = " [BLOCKED]" if debug and state.edge_blocked[index] else ""
        lines.append(
            f"  {source} --{name} flow={state.last_edge_flow[index]:.2f}/"
            f"{graph.edge_capacities[index]:.2f} {status}--> {target}{latent}"
        )
    return "\n".join(lines)


def render_timeline(history: list[Interaction]) -> str:
    """Render only public interaction results and rewards."""
    lines = ["TIMELINE"]
    for index, interaction in enumerate(history, start=1):
        observation = ", ".join(
            f"{key}={value}" for key, value in sorted(interaction.result.observation.items())
        )
        lines.append(
            f"  {index:02d} {interaction.result.kind.value:<16} "
            f"reward={interaction.result.reward:7.2f} {observation}"
        )
    return "\n".join(lines)
