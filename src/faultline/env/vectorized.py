"""NumPy-batched material-flow stepping for many copies of one topology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from faultline.env.graph import FactoryGraph

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]

_BATCH_ATOL = 1e-10


@dataclass(slots=True)
class VectorizedState:
    """Structure-of-arrays state for a homogeneous batch of factory graphs."""

    inputs: FloatArray
    outputs: FloatArray
    node_enabled: BoolArray
    edge_enabled: BoolArray
    node_failed: BoolArray
    node_backpressured: BoolArray
    edge_blocked: BoolArray
    last_edge_flow: FloatArray
    last_delivered: FloatArray
    tick: IntArray
    injected_total: FloatArray
    delivered_total: FloatArray
    initial_material: FloatArray

    @classmethod
    def healthy(cls, graph: FactoryGraph, batch_size: int) -> VectorizedState:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        node_shape = (batch_size, graph.node_count)
        edge_shape = (batch_size, graph.edge_count)
        initial_inputs = np.broadcast_to(graph.initial_inputs, node_shape).copy()
        initial_outputs = np.broadcast_to(graph.initial_outputs, node_shape).copy()
        initial_material = initial_inputs.sum(axis=1) + initial_outputs.sum(axis=1)
        return cls(
            inputs=initial_inputs,
            outputs=initial_outputs,
            node_enabled=np.ones(node_shape, dtype=np.bool_),
            edge_enabled=np.ones(edge_shape, dtype=np.bool_),
            node_failed=np.zeros(node_shape, dtype=np.bool_),
            node_backpressured=np.zeros(node_shape, dtype=np.bool_),
            edge_blocked=np.zeros(edge_shape, dtype=np.bool_),
            last_edge_flow=np.zeros(edge_shape, dtype=np.float64),
            last_delivered=np.zeros(batch_size, dtype=np.float64),
            tick=np.zeros(batch_size, dtype=np.int64),
            injected_total=np.zeros(batch_size, dtype=np.float64),
            delivered_total=np.zeros(batch_size, dtype=np.float64),
            initial_material=initial_material,
        )

    @property
    def batch_size(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def material_held(self) -> FloatArray:
        return cast(
            FloatArray,
            self.inputs.sum(axis=1, dtype=np.float64)
            + self.outputs.sum(axis=1, dtype=np.float64),
        )

    @property
    def conservation_residual(self) -> FloatArray:
        return (
            self.initial_material
            + self.injected_total
            - self.delivered_total
            - self.material_held
        )


@dataclass(frozen=True, slots=True)
class BatchTransition:
    """Per-environment totals from one batched tick."""

    injected: FloatArray
    transformed: FloatArray
    delivered: FloatArray


def validate_vectorized_state(graph: FactoryGraph, state: VectorizedState) -> None:
    batch_size = state.batch_size
    node_shape = (batch_size, graph.node_count)
    edge_shape = (batch_size, graph.edge_count)
    for node_array in (
        state.inputs,
        state.outputs,
        state.node_enabled,
        state.node_failed,
        state.node_backpressured,
    ):
        if node_array.shape != node_shape:
            raise ValueError("batched node-array shape does not match graph")
    for edge_array in (state.edge_enabled, state.edge_blocked, state.last_edge_flow):
        if edge_array.shape != edge_shape:
            raise ValueError("batched edge-array shape does not match graph")
    for scalar_array in (
        state.last_delivered,
        state.tick,
        state.injected_total,
        state.delivered_total,
        state.initial_material,
    ):
        if scalar_array.shape != (batch_size,):
            raise ValueError("batched scalar-array shape does not match batch")
    if np.any(state.inputs < 0.0) or np.any(state.outputs < 0.0):
        raise ValueError("batched state contains negative material")
    if np.any(state.inputs > graph.input_capacities[None, :] + _BATCH_ATOL):
        raise ValueError("batched input exceeds node capacity")
    if np.any(state.outputs > graph.output_capacities[None, :] + _BATCH_ATOL):
        raise ValueError("batched output exceeds node capacity")


def step_batch(
    graph: FactoryGraph,
    state: VectorizedState,
    *,
    active: BoolArray | None = None,
    check_invariants: bool = False,
) -> BatchTransition:
    """Advance all active environments one tick without a Python loop over environments."""
    if check_invariants:
        validate_vectorized_state(graph, state)
    if active is None:
        active = np.ones(state.batch_size, dtype=np.bool_)
    elif active.shape != (state.batch_size,):
        raise ValueError("active mask shape does not match batch")

    source_indices = graph.source_indices
    source_space = (
        graph.output_capacities[source_indices][None, :] - state.outputs[:, source_indices]
    )
    injected_by_source = np.minimum(graph.rates[source_indices][None, :], source_space)
    injected_by_source *= state.node_enabled[:, source_indices] & active[:, None]
    state.outputs[:, source_indices] += injected_by_source
    injected = injected_by_source.sum(axis=1)

    transform_indices = graph.transform_indices
    transform_space = (
        graph.output_capacities[transform_indices][None, :]
        - state.outputs[:, transform_indices]
    )
    transformed_by_node = np.minimum(
        np.minimum(state.inputs[:, transform_indices], graph.rates[transform_indices][None, :]),
        transform_space,
    )
    transformed_by_node *= (
        state.node_enabled[:, transform_indices]
        & ~state.node_failed[:, transform_indices]
        & active[:, None]
    )
    state.inputs[:, transform_indices] -= transformed_by_node
    state.outputs[:, transform_indices] += transformed_by_node
    transformed = transformed_by_node.sum(axis=1)

    state.last_edge_flow[active, :] = 0.0
    for edge_index in range(graph.edge_count):
        source = int(graph.edge_sources[edge_index])
        target = int(graph.edge_targets[edge_index])
        can_flow = (
            active
            & state.edge_enabled[:, edge_index]
            & ~state.edge_blocked[:, edge_index]
            & ~state.node_backpressured[:, target]
        )
        target_space = graph.input_capacities[target] - state.inputs[:, target]
        flow = np.minimum(
            np.minimum(state.outputs[:, source], graph.edge_capacities[edge_index]),
            target_space,
        )
        flow *= can_flow
        state.outputs[:, source] -= flow
        state.inputs[:, target] += flow
        state.last_edge_flow[:, edge_index] = np.where(
            active,
            flow,
            state.last_edge_flow[:, edge_index],
        )

    sink_indices = graph.sink_indices
    delivered_by_sink = np.minimum(
        state.inputs[:, sink_indices],
        graph.rates[sink_indices][None, :],
    )
    delivered_by_sink *= (
        state.node_enabled[:, sink_indices]
        & ~state.node_backpressured[:, sink_indices]
        & active[:, None]
    )
    state.inputs[:, sink_indices] -= delivered_by_sink
    delivered = delivered_by_sink.sum(axis=1)

    state.tick += active.astype(np.int64)
    state.injected_total += injected
    state.delivered_total += delivered
    state.last_delivered = np.where(active, delivered, state.last_delivered)

    if check_invariants:
        validate_vectorized_state(graph, state)
        residual = np.abs(state.conservation_residual)
        if np.any(residual > _BATCH_ATOL):
            failing = int(np.flatnonzero(residual > _BATCH_ATOL)[0])
            raise AssertionError(
                f"material conservation residual {residual[failing]:.3e} in batch row {failing}"
            )

    return BatchTransition(injected=injected, transformed=transformed, delivered=delivered)
