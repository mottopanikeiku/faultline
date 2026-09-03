"""Deterministic discrete-tick material-flow dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from faultline.env.graph import FactoryGraph
from faultline.env.state import FactoryState

_CONSERVATION_ATOL = 1e-10


@dataclass(frozen=True, slots=True)
class Transition:
    """Operational totals produced by one or more simulation ticks."""

    ticks: int
    injected: float
    transformed: float
    delivered: float


def validate_state(graph: FactoryGraph, state: FactoryState) -> None:
    """Reject state arrays that cannot belong to ``graph``."""
    node_shape = (graph.node_count,)
    edge_shape = (graph.edge_count,)
    if state.inputs.shape != node_shape or state.outputs.shape != node_shape:
        raise ValueError("state node-array shape does not match graph")
    if (
        state.node_enabled.shape != node_shape
        or state.node_failed.shape != node_shape
        or state.node_backpressured.shape != node_shape
    ):
        raise ValueError("state node-control shape does not match graph")
    if (
        state.edge_enabled.shape != edge_shape
        or state.edge_blocked.shape != edge_shape
        or state.last_edge_flow.shape != edge_shape
    ):
        raise ValueError("state edge-control shape does not match graph")
    if np.any(state.inputs < 0.0) or np.any(state.outputs < 0.0):
        raise ValueError("state contains negative material")
    if np.any(state.inputs > graph.input_capacities + _CONSERVATION_ATOL):
        raise ValueError("state input exceeds node capacity")
    if np.any(state.outputs > graph.output_capacities + _CONSERVATION_ATOL):
        raise ValueError("state output exceeds node capacity")


def step_tick(
    graph: FactoryGraph,
    state: FactoryState,
    *,
    check_invariants: bool = False,
) -> Transition:
    """Mutate ``state`` by one deterministic tick.

    Node transforms use start-of-phase inventory. Edge transfers then run in declared edge order,
    which is part of the immutable environment definition. Sink consumption runs last.
    """
    if check_invariants:
        validate_state(graph, state)

    source_indices = graph.source_indices
    source_space = graph.output_capacities[source_indices] - state.outputs[source_indices]
    injected_by_source = np.minimum(graph.rates[source_indices], source_space)
    injected_by_source *= state.node_enabled[source_indices]
    state.outputs[source_indices] += injected_by_source
    injected = float(injected_by_source.sum(dtype=np.float64))

    transform_indices = graph.transform_indices
    transform_space = graph.output_capacities[transform_indices] - state.outputs[transform_indices]
    transformed_by_node = np.minimum(
        np.minimum(state.inputs[transform_indices], graph.rates[transform_indices]),
        transform_space,
    )
    transformed_by_node *= (
        state.node_enabled[transform_indices] & ~state.node_failed[transform_indices]
    )
    state.inputs[transform_indices] -= transformed_by_node
    state.outputs[transform_indices] += transformed_by_node
    transformed = float(transformed_by_node.sum(dtype=np.float64))

    state.last_edge_flow.fill(0.0)
    for edge_index in range(graph.edge_count):
        if not state.edge_enabled[edge_index] or state.edge_blocked[edge_index]:
            continue
        source = int(graph.edge_sources[edge_index])
        target = int(graph.edge_targets[edge_index])
        if state.node_backpressured[target]:
            continue
        target_space = graph.input_capacities[target] - state.inputs[target]
        flow = min(
            state.outputs[source],
            graph.edge_capacities[edge_index],
            target_space,
        )
        if flow <= 0.0:
            continue
        state.outputs[source] -= flow
        state.inputs[target] += flow
        state.last_edge_flow[edge_index] = flow

    sink_indices = graph.sink_indices
    delivered_by_sink = np.minimum(state.inputs[sink_indices], graph.rates[sink_indices])
    delivered_by_sink *= (
        state.node_enabled[sink_indices] & ~state.node_backpressured[sink_indices]
    )
    state.inputs[sink_indices] -= delivered_by_sink
    delivered = float(delivered_by_sink.sum(dtype=np.float64))

    state.tick += 1
    state.injected_total += injected
    state.delivered_total += delivered

    if check_invariants:
        validate_state(graph, state)
        if abs(state.conservation_residual) > _CONSERVATION_ATOL:
            message = (
                f"material conservation residual {state.conservation_residual:.3e} "
                f"at tick {state.tick}"
            )
            raise AssertionError(message)

    return Transition(ticks=1, injected=injected, transformed=transformed, delivered=delivered)


def advance(
    graph: FactoryGraph,
    state: FactoryState,
    ticks: int,
    *,
    check_invariants: bool = False,
) -> Transition:
    """Advance multiple ticks without allocating a trajectory."""
    if ticks < 0:
        raise ValueError("ticks must be non-negative")
    injected = 0.0
    transformed = 0.0
    delivered = 0.0
    for _ in range(ticks):
        transition = step_tick(graph, state, check_invariants=check_invariants)
        injected += transition.injected
        transformed += transition.transformed
        delivered += transition.delivered
    return Transition(
        ticks=ticks,
        injected=injected,
        transformed=transformed,
        delivered=delivered,
    )
