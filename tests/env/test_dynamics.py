from __future__ import annotations

import numpy as np
import pytest

from faultline.env import Edge, FactoryGraph, FactoryState, Node, NodeType, advance, step_tick


def linear_factory(*, initial_processor_input: float = 0.0) -> FactoryGraph:
    return FactoryGraph.compile(
        nodes=[
            Node("source", NodeType.SOURCE, rate=2.0, output_capacity=4.0),
            Node(
                "processor",
                NodeType.PROCESSOR,
                rate=2.0,
                input_capacity=4.0,
                output_capacity=4.0,
                initial_input=initial_processor_input,
            ),
            Node("buffer", NodeType.BUFFER, rate=2.0, input_capacity=4.0, output_capacity=4.0),
            Node("sink", NodeType.SINK, rate=2.0, input_capacity=4.0),
        ],
        edges=[
            Edge("source_to_processor", "source", "processor", capacity=2.0),
            Edge("processor_to_buffer", "processor", "buffer", capacity=2.0),
            Edge("buffer_to_sink", "buffer", "sink", capacity=2.0),
        ],
    )


def state_signature(state: FactoryState) -> tuple[object, ...]:
    return (
        state.inputs.tobytes(),
        state.outputs.tobytes(),
        state.node_enabled.tobytes(),
        state.edge_enabled.tobytes(),
        state.node_failed.tobytes(),
        state.node_backpressured.tobytes(),
        state.edge_blocked.tobytes(),
        state.last_edge_flow.tobytes(),
        state.last_delivered,
        state.tick,
        state.injected_total,
        state.delivered_total,
    )


def test_linear_factory_propagates_with_one_stage_per_tick() -> None:
    graph = linear_factory()
    state = FactoryState.healthy(graph)

    first = step_tick(graph, state, check_invariants=True)
    assert first.injected == 2.0
    assert first.delivered == 0.0
    assert state.inputs[graph.node_index["processor"]] == 2.0

    second = step_tick(graph, state, check_invariants=True)
    assert second.transformed == 2.0
    assert second.delivered == 0.0
    assert state.inputs[graph.node_index["buffer"]] == 2.0

    third = step_tick(graph, state, check_invariants=True)
    assert third.delivered == 2.0
    assert state.conservation_residual == pytest.approx(0.0, abs=1e-12)


def test_material_is_conserved_over_sustained_flow() -> None:
    graph = linear_factory(initial_processor_input=1.25)
    state = FactoryState.healthy(graph)

    transition = advance(graph, state, 100, check_invariants=True)

    assert transition.delivered > 0.0
    assert state.initial_material == 1.25
    assert state.conservation_residual == pytest.approx(0.0, abs=1e-12)
    assert state.initial_material + state.injected_total == pytest.approx(
        state.delivered_total + state.material_held,
        abs=1e-12,
    )


def test_same_graph_and_actions_produce_bit_identical_state() -> None:
    graph = linear_factory()
    left = FactoryState.healthy(graph)
    right = FactoryState.healthy(graph)

    for tick in range(25):
        if tick == 7:
            left.edge_enabled[1] = False
            right.edge_enabled[1] = False
        if tick == 11:
            left.edge_enabled[1] = True
            right.edge_enabled[1] = True
        step_tick(graph, left, check_invariants=True)
        step_tick(graph, right, check_invariants=True)

    assert state_signature(left) == state_signature(right)
    np.testing.assert_array_equal(left.inputs, right.inputs)
    np.testing.assert_array_equal(left.outputs, right.outputs)


def test_state_clone_is_independent() -> None:
    graph = linear_factory()
    original = FactoryState.healthy(graph)
    branch = original.clone()

    step_tick(graph, branch)

    assert original.tick == 0
    assert branch.tick == 1
    assert not np.shares_memory(original.outputs, branch.outputs)


def test_cycles_are_rejected() -> None:
    nodes = [
        Node("source", NodeType.SOURCE, rate=1.0, output_capacity=1.0),
        Node("buffer", NodeType.BUFFER, rate=1.0, input_capacity=1.0, output_capacity=1.0),
        Node("sink", NodeType.SINK, rate=1.0, input_capacity=1.0),
    ]
    edges = [
        Edge("source_buffer", "source", "buffer", 1.0),
        Edge("buffer_source", "buffer", "source", 1.0),
        Edge("buffer_sink", "buffer", "sink", 1.0),
    ]

    with pytest.raises(ValueError, match="acyclic"):
        FactoryGraph.compile(nodes, edges)
