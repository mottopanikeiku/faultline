from __future__ import annotations

import pytest

from faultline.env import Edge, FactoryGraph, FactoryState, Node, NodeType, advance
from faultline.faults import BlockedEdge, DownstreamBackpressure, FailedProcessor, inject_fault


def linear_factory() -> FactoryGraph:
    return FactoryGraph.compile(
        nodes=[
            Node("source", NodeType.SOURCE, rate=2.0, output_capacity=4.0),
            Node("processor", NodeType.PROCESSOR, 2.0, 4.0, 4.0),
            Node("buffer", NodeType.BUFFER, 2.0, 4.0, 4.0),
            Node("sink", NodeType.SINK, 2.0, input_capacity=4.0),
        ],
        edges=[
            Edge("feed", "source", "processor", 2.0),
            Edge("output", "processor", "buffer", 2.0),
            Edge("delivery", "buffer", "sink", 2.0),
        ],
    )


def test_blocked_edge_stops_and_clear_restores_material_flow() -> None:
    graph = linear_factory()
    state = FactoryState.healthy(graph)
    inject_fault(graph, state, BlockedEdge("output"))

    blocked = advance(graph, state, 10, check_invariants=True)
    assert blocked.delivered == 0.0
    assert state.edge_blocked[graph.edge_index["output"]]
    assert state.outputs[graph.node_index["processor"]] == 4.0

    state.edge_blocked[graph.edge_index["output"]] = False
    restored = advance(graph, state, 5, check_invariants=True)
    assert restored.delivered > 0.0
    assert state.conservation_residual == pytest.approx(0.0, abs=1e-12)


def test_failed_processor_propagates_starvation_and_replacement_restores_flow() -> None:
    graph = linear_factory()
    state = FactoryState.healthy(graph)
    inject_fault(graph, state, FailedProcessor("processor"))

    failed = advance(graph, state, 10, check_invariants=True)
    assert failed.transformed == 0.0
    assert failed.delivered == 0.0
    assert state.inputs[graph.node_index["processor"]] == 4.0

    state.node_failed[graph.node_index["processor"]] = False
    restored = advance(graph, state, 5, check_invariants=True)
    assert restored.delivered > 0.0


def test_sink_backpressure_fills_upstream_and_release_restores_flow() -> None:
    graph = linear_factory()
    state = FactoryState.healthy(graph)
    inject_fault(graph, state, DownstreamBackpressure("sink"))

    blocked = advance(graph, state, 10, check_invariants=True)
    assert blocked.delivered == 0.0
    assert state.outputs[graph.node_index["buffer"]] == 4.0
    assert state.inputs[graph.node_index["sink"]] == 0.0

    state.node_backpressured[graph.node_index["sink"]] = False
    restored = advance(graph, state, 5, check_invariants=True)
    assert restored.delivered > 0.0


def test_fault_targets_are_type_checked_without_mutating_state() -> None:
    graph = linear_factory()
    state = FactoryState.healthy(graph)

    with pytest.raises(ValueError, match="not a processor"):
        inject_fault(graph, state, FailedProcessor("buffer"))
    with pytest.raises(ValueError, match="not a buffer or sink"):
        inject_fault(graph, state, DownstreamBackpressure("processor"))
    with pytest.raises(ValueError, match="unknown blocked edge"):
        inject_fault(graph, state, BlockedEdge("missing"))

    assert not state.node_failed.any()
    assert not state.node_backpressured.any()
    assert not state.edge_blocked.any()
