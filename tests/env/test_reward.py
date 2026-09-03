from __future__ import annotations

import pytest

from faultline.env import (
    Advance,
    Edge,
    FactoryEnv,
    FactoryGraph,
    Inspect,
    Isolate,
    Node,
    NodeType,
    Replace,
    RewardConfig,
    RewardTracker,
    TerminationReason,
)
from faultline.faults import FailedProcessor


def compact_factory() -> FactoryGraph:
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


def reward_config(**overrides: float | int) -> RewardConfig:
    values: dict[str, float | int] = {
        "target_throughput": 2.0,
        "throughput_value": 1.0,
        "time_cost": 0.1,
        "passive_cost": 0.05,
        "diagnostic_cost": 0.25,
        "repair_cost": 2.0,
        "false_repair_cost": 3.0,
        "recovery_fraction": 1.0,
        "recovery_ticks": 3,
        "recovery_bonus": 5.0,
        "max_ticks": 20,
        "max_actions": 20,
    }
    values.update(overrides)
    return RewardConfig(**values)  # type: ignore[arg-type]


def test_recovery_requires_sustained_target_throughput() -> None:
    tracker = RewardTracker(reward_config())

    for tick, throughput in enumerate((2.0, 2.0, 0.0, 2.0, 2.0), start=1):
        tracker.record_tick(throughput, tick)
        assert not tracker.recovered

    final_reward = tracker.record_tick(2.0, 6)
    assert tracker.recovered
    assert tracker.recovery_streak == 3
    assert tracker.termination_reason is TerminationReason.RECOVERED
    assert final_reward == pytest.approx(6.9)


def test_advance_stops_exactly_when_sustained_recovery_is_verified() -> None:
    env = FactoryEnv.create(
        compact_factory(),
        reward_config=reward_config(),
        check_invariants=True,
    )

    result = env.act(Advance(20))
    metrics = env.reward_tracker.snapshot() if env.reward_tracker is not None else None

    assert result.terminated
    assert result.observation["ticks_advanced"] == 4
    assert result.reward == pytest.approx(10.6)
    assert metrics is not None
    assert metrics.recovered
    assert metrics.production_reward == 6.0
    assert metrics.time_cost == pytest.approx(0.4)
    assert metrics.recovery_bonus == 5.0
    assert metrics.total_reward == pytest.approx(
        metrics.production_reward
        - metrics.time_cost
        - metrics.action_cost
        - metrics.false_repair_cost
        + metrics.recovery_bonus
    )
    with pytest.raises(RuntimeError, match="episode_terminated"):
        env.act(Inspect("source"))


def test_needed_and_false_repairs_receive_operational_costs() -> None:
    graph = compact_factory()
    failed = FactoryEnv.create(
        graph,
        FailedProcessor("processor"),
        reward_config=reward_config(),
    )
    healthy = FactoryEnv.create(graph, reward_config=reward_config())

    needed = failed.act(Replace("processor"))
    unnecessary = healthy.act(Replace("processor"))

    assert needed.reward == -2.0
    assert unnecessary.reward == -5.0
    assert failed.reward_tracker is not None
    assert failed.reward_tracker.false_repair_count == 0
    assert healthy.reward_tracker is not None
    assert healthy.reward_tracker.false_repair_count == 1
    metrics = healthy.reward_tracker.snapshot()
    assert metrics.action_cost == 2.0
    assert metrics.false_repair_cost == 3.0


def test_passive_and_diagnostic_actions_pay_distinct_costs() -> None:
    env = FactoryEnv.create(compact_factory(), reward_config=reward_config())

    inspected = env.act(Inspect("processor"))
    isolated = env.act(Isolate("feed"))

    assert inspected.reward == -0.05
    assert isolated.reward == -0.25
    assert env.reward_tracker is not None
    assert env.reward_tracker.passive_observation_count == 1
    assert env.reward_tracker.diagnostic_count == 1


def test_tick_limit_terminates_unrecovered_failure() -> None:
    env = FactoryEnv.create(
        compact_factory(),
        FailedProcessor("processor"),
        reward_config=reward_config(max_ticks=5),
    )

    result = env.act(Advance(20))

    assert result.terminated
    assert result.observation["ticks_advanced"] == 5
    assert env.reward_tracker is not None
    assert env.reward_tracker.termination_reason is TerminationReason.TICK_LIMIT
    assert not env.reward_tracker.recovered
