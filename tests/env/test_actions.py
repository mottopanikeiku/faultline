from __future__ import annotations

from collections.abc import Mapping

from faultline.env import (
    ActionError,
    Advance,
    ClearBlockage,
    Edge,
    FactoryEnv,
    FactoryGraph,
    Inspect,
    Isolate,
    MeasureFlow,
    Node,
    NodeType,
    Replace,
    Toggle,
)
from faultline.faults import BlockedEdge, FailedProcessor


def linear_factory() -> FactoryGraph:
    return FactoryGraph.compile(
        nodes=[
            Node("source", NodeType.SOURCE, 2.0, output_capacity=4.0),
            Node("processor", NodeType.PROCESSOR, 2.0, 4.0, 4.0),
            Node("sink", NodeType.SINK, 2.0, input_capacity=4.0),
        ],
        edges=[
            Edge("feed", "source", "processor", 2.0),
            Edge("delivery", "processor", "sink", 2.0),
        ],
    )


def assert_no_privileged_keys(value: object) -> None:
    forbidden = ("fault", "failed", "blocked", "backpressure", "operational")
    if isinstance(value, Mapping):
        for key, child in value.items():
            assert not any(fragment in str(key).lower() for fragment in forbidden)
            assert_no_privileged_keys(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            assert_no_privileged_keys(child)


def test_latent_faults_do_not_leak_before_their_public_effects() -> None:
    graph = linear_factory()
    blocked = FactoryEnv.create(graph, BlockedEdge("delivery"), check_invariants=True)
    failed = FactoryEnv.create(graph, FailedProcessor("processor"), check_invariants=True)

    assert blocked.observe() == failed.observe()
    assert blocked.act(Inspect("processor")) == failed.act(Inspect("processor"))
    assert_no_privileged_keys(blocked.observe())
    assert_no_privileged_keys(blocked.history[0].result.observation)


def test_passive_actions_reveal_only_requested_operational_telemetry() -> None:
    graph = linear_factory()
    env = FactoryEnv.create(graph, check_invariants=True)
    env.act(Advance(3))

    inspected = env.act(Inspect("processor"))
    measured = env.act(MeasureFlow("delivery"))

    assert inspected.accepted
    assert inspected.observation["node_type"] == "processor"
    assert "input_buffer" in inspected.observation
    assert measured.accepted
    assert measured.observation["flow"] == 2.0
    assert_no_privileged_keys(inspected.observation)
    assert_no_privileged_keys(measured.observation)


def test_control_actions_change_dynamics_and_are_reversible() -> None:
    graph = linear_factory()
    env = FactoryEnv.create(graph, check_invariants=True)

    isolated = env.act(Isolate("feed"))
    env.act(Advance(3))
    assert isolated.observation["isolated"] is True
    assert env.state.delivered_total == 0.0

    env.act(Isolate("feed", isolated=False))
    env.act(Toggle("processor"))
    env.act(Advance(3))
    assert env.state.delivered_total == 0.0

    toggled_on = env.act(Toggle("processor"))
    env.act(Advance(3))
    assert toggled_on.observation["enabled"] is True
    assert env.state.delivered_total > 0.0


def test_repairs_have_identical_public_result_whether_or_not_needed() -> None:
    graph = linear_factory()
    failed = FactoryEnv.create(graph, FailedProcessor("processor"))
    healthy = FactoryEnv.create(graph)

    failed_result = failed.act(Replace("processor"))
    healthy_result = healthy.act(Replace("processor"))
    assert failed_result == healthy_result
    assert not failed.state.node_failed[graph.node_index["processor"]]

    blocked = FactoryEnv.create(graph, BlockedEdge("delivery"))
    healthy = FactoryEnv.create(graph)
    assert blocked.act(ClearBlockage("delivery")) == healthy.act(ClearBlockage("delivery"))
    assert not blocked.state.edge_blocked[graph.edge_index["delivery"]]


def test_unknown_component_errors_are_uniform_and_public() -> None:
    env = FactoryEnv.create(linear_factory())

    node_result = env.act(Inspect("missing"))
    edge_result = env.act(MeasureFlow("missing"))

    assert not node_result.accepted
    assert node_result.error is ActionError.UNKNOWN_COMPONENT
    assert node_result.observation["error"] == "unknown_component"
    assert edge_result.error is ActionError.UNKNOWN_COMPONENT
