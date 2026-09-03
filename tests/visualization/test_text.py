from __future__ import annotations

from faultline.env import Advance, FactoryEnv
from faultline.faults import BlockedEdge, FailedProcessor, inject_fault
from faultline.generation import chain_factory
from faultline.visualization import render_factory, render_timeline


def test_public_render_hides_latent_fault_markers() -> None:
    graph = chain_factory(5)
    env = FactoryEnv.create(graph, BlockedEdge("transport_001"))
    inject_fault(graph, env.state, FailedProcessor("processor_001"))
    env.act(Advance(2))

    public = render_factory(graph, env.state)
    debug = render_factory(graph, env.state, debug=True)

    assert "[BLOCKED]" not in public
    assert "[FAILED]" not in public
    assert "[BLOCKED]" in debug
    assert "[FAILED]" in debug
    assert "tick=2" in public


def test_timeline_contains_public_results_and_reward_only() -> None:
    graph = chain_factory(4)
    env = FactoryEnv.create(graph)
    env.act(Advance(2))

    timeline = render_timeline(env.history)

    assert "TIMELINE" in timeline
    assert "advance" in timeline
    assert "ticks_advanced=2" in timeline
    assert "fault" not in timeline.lower()
