from __future__ import annotations

import numpy as np

from faultline.env import Edge, FactoryGraph, FactoryState, Node, NodeType, step_tick
from faultline.env.vectorized import VectorizedState, step_batch


def branched_factory() -> FactoryGraph:
    return FactoryGraph.compile(
        nodes=[
            Node("source", NodeType.SOURCE, 3.0, output_capacity=6.0),
            Node("processor", NodeType.PROCESSOR, 2.0, 5.0, 5.0),
            Node("buffer", NodeType.BUFFER, 1.5, 4.0, 4.0),
            Node("sink", NodeType.SINK, 2.5, input_capacity=5.0),
        ],
        edges=[
            Edge("feed", "source", "processor", 2.5),
            Edge("processed", "processor", "buffer", 1.75),
            Edge("delivery", "buffer", "sink", 2.0),
        ],
    )


def test_batched_kernel_is_exactly_equivalent_to_scalar_kernel() -> None:
    graph = branched_factory()
    batch = VectorizedState.healthy(graph, 4)
    scalars = [FactoryState.healthy(graph) for _ in range(4)]

    batch.edge_blocked[1, 1] = True
    scalars[1].edge_blocked[1] = True
    batch.node_failed[2, 1] = True
    scalars[2].node_failed[1] = True
    batch.node_backpressured[3, 3] = True
    scalars[3].node_backpressured[3] = True

    for tick in range(30):
        if tick == 8:
            batch.edge_blocked[1, 1] = False
            scalars[1].edge_blocked[1] = False
        if tick == 11:
            batch.node_failed[2, 1] = False
            scalars[2].node_failed[1] = False
        if tick == 14:
            batch.node_backpressured[3, 3] = False
            scalars[3].node_backpressured[3] = False
        batched_transition = step_batch(graph, batch, check_invariants=True)
        scalar_transitions = [step_tick(graph, state, check_invariants=True) for state in scalars]
        np.testing.assert_array_equal(
            batched_transition.delivered,
            np.asarray([transition.delivered for transition in scalar_transitions]),
        )

    for row, scalar in enumerate(scalars):
        np.testing.assert_array_equal(batch.inputs[row], scalar.inputs)
        np.testing.assert_array_equal(batch.outputs[row], scalar.outputs)
        np.testing.assert_array_equal(batch.last_edge_flow[row], scalar.last_edge_flow)
        assert batch.tick[row] == scalar.tick
        assert batch.injected_total[row] == scalar.injected_total
        assert batch.delivered_total[row] == scalar.delivered_total
        assert batch.last_delivered[row] == scalar.last_delivered
    np.testing.assert_allclose(batch.conservation_residual, 0.0, rtol=0.0, atol=1e-12)


def test_inactive_batch_rows_remain_bit_identical() -> None:
    graph = branched_factory()
    state = VectorizedState.healthy(graph, 3)
    step_batch(graph, state)
    frozen = (
        state.inputs[1].copy(),
        state.outputs[1].copy(),
        state.last_edge_flow[1].copy(),
        state.tick[1],
        state.injected_total[1],
        state.delivered_total[1],
    )

    step_batch(graph, state, active=np.asarray([True, False, True], dtype=np.bool_))

    np.testing.assert_array_equal(state.inputs[1], frozen[0])
    np.testing.assert_array_equal(state.outputs[1], frozen[1])
    np.testing.assert_array_equal(state.last_edge_flow[1], frozen[2])
    assert state.tick[1] == frozen[3]
    assert state.injected_total[1] == frozen[4]
    assert state.delivered_total[1] == frozen[5]


def test_batched_execution_is_deterministic() -> None:
    graph = branched_factory()
    left = VectorizedState.healthy(graph, 64)
    right = VectorizedState.healthy(graph, 64)
    left.edge_blocked[::3, 1] = True
    right.edge_blocked[::3, 1] = True

    for _ in range(20):
        step_batch(graph, left)
        step_batch(graph, right)

    for left_array, right_array in (
        (left.inputs, right.inputs),
        (left.outputs, right.outputs),
        (left.last_edge_flow, right.last_edge_flow),
        (left.delivered_total, right.delivered_total),
    ):
        np.testing.assert_array_equal(left_array, right_array)
