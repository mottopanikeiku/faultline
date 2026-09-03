"""Small recurrent-RL interface over the exact factory diagnostic pairs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from faultline.env import Advance, ClearBlockage, Inspect, Replace
from faultline.faults import BlockedEdge
from faultline.generation import CueCondition, DiagnosticPair, create_world_env

MAX_NODES = 12
NODE_FEATURE_DIM = 12
GLOBAL_FEATURE_DIM = 10
ACTION_COUNT = 4

Float32Array = NDArray[np.float32]
BoolArray = NDArray[np.bool_]


class DiagnosticAction(IntEnum):
    ADVANCE = 0
    INSPECT = 1
    CLEAR_BLOCKAGE = 2
    REPLACE_PROCESSOR = 3


@dataclass(frozen=True, slots=True)
class PolicyObservation:
    nodes: Float32Array
    adjacency: Float32Array
    node_mask: BoolArray
    global_features: Float32Array
    action_mask: BoolArray


@dataclass(frozen=True, slots=True)
class EpisodeSummary:
    pair_id: str
    condition: CueCondition
    world_index: int
    cue: int
    total_reward: float
    recovered: bool
    correct_repair: bool
    selected_repair: DiagnosticAction | None
    decision_steps: int
    simulator_ticks: int
    advance_count: int
    inspect_count: int
    informative_inspection: bool
    false_repair_count: int


class DiagnosticEpisode:
    """A repair-commitment episode using only operational simulator reward."""

    def __init__(
        self,
        pair: DiagnosticPair,
        condition: CueCondition,
        world_index: int,
        cue: int,
        *,
        max_decision_steps: int = 4,
    ) -> None:
        if pair.graph.node_count > MAX_NODES:
            raise ValueError(f"policy supports at most {MAX_NODES} nodes")
        if world_index not in (0, 1) or cue not in (0, 1):
            raise ValueError("world index and cue must be binary")
        if max_decision_steps <= 0:
            raise ValueError("decision-step limit must be positive")
        self.pair = pair
        self.condition = condition
        self.world_index = world_index
        self.cue = cue
        self.max_decision_steps = max_decision_steps
        self.env = create_world_env(pair, pair.worlds[world_index], with_reward=True)
        self.decision_steps = 0
        self.advance_count = 0
        self.inspect_count = 0
        self.informative_inspection = False
        self.selected_repair: DiagnosticAction | None = None
        self.terminated = False
        self._last_action: int | None = None
        self._telemetry_mask = np.zeros(MAX_NODES, dtype=np.bool_)
        self._telemetry_inputs = np.zeros(MAX_NODES, dtype=np.float32)
        self._telemetry_outputs = np.zeros(MAX_NODES, dtype=np.float32)
        self._node_features = self._build_static_node_features()
        self._adjacency = self._build_adjacency()
        self._node_mask = np.zeros(MAX_NODES, dtype=np.bool_)
        self._node_mask[: pair.graph.node_count] = True

    @classmethod
    def sample(
        cls,
        pair: DiagnosticPair,
        condition: CueCondition,
        rng: np.random.Generator,
        *,
        max_decision_steps: int = 4,
    ) -> DiagnosticEpisode:
        world_index = int(rng.integers(0, 2))
        if condition is CueCondition.AMBIGUOUS:
            cue = int(rng.integers(0, 2))
        else:
            cue = 0 if isinstance(pair.worlds[world_index].fault, BlockedEdge) else 1
        return cls(
            pair,
            condition,
            world_index,
            cue,
            max_decision_steps=max_decision_steps,
        )

    def _build_static_node_features(self) -> Float32Array:
        graph = self.pair.graph
        features = np.zeros((MAX_NODES, NODE_FEATURE_DIM), dtype=np.float32)
        scale = max(float(graph.rates.max()), float(graph.input_capacities.max()), 1.0)
        for node_index in range(graph.node_count):
            node_type = int(graph.node_types[node_index])
            features[node_index, node_type] = 1.0
            features[node_index, 4] = float(graph.rates[node_index]) / scale
            features[node_index, 5] = float(graph.input_capacities[node_index]) / scale
            features[node_index, 6] = float(graph.output_capacities[node_index]) / scale
            features[node_index, 7] = float(graph.initial_inputs[node_index]) / scale
            features[node_index, 8] = float(
                graph.node_names[node_index] == self.pair.evidence_node
            )
        return features

    def _build_adjacency(self) -> Float32Array:
        graph = self.pair.graph
        adjacency = np.zeros((MAX_NODES, MAX_NODES), dtype=np.float32)
        for edge_index in range(graph.edge_count):
            source = int(graph.edge_sources[edge_index])
            target = int(graph.edge_targets[edge_index])
            adjacency[source, target] = 1.0
            adjacency[target, source] = 1.0
        return adjacency

    def observe(self) -> PolicyObservation:
        nodes = self._node_features.copy()
        nodes[:, 9] = self._telemetry_mask
        nodes[:, 10] = self._telemetry_inputs
        nodes[:, 11] = self._telemetry_outputs
        global_features = np.zeros(GLOBAL_FEATURE_DIM, dtype=np.float32)
        global_features[self.cue] = 1.0
        if self._last_action is None:
            global_features[2] = 1.0
        else:
            global_features[3 + self._last_action] = 1.0
        global_features[7] = float(
            self.env.state.last_delivered / self.pair.reward.target_throughput
        )
        global_features[8] = float(self.env.state.tick / self.pair.reward.max_ticks)
        global_features[9] = float(self.decision_steps / self.max_decision_steps)
        return PolicyObservation(
            nodes=nodes,
            adjacency=self._adjacency,
            node_mask=self._node_mask,
            global_features=global_features,
            action_mask=np.ones(ACTION_COUNT, dtype=np.bool_),
        )

    def step(
        self,
        action: DiagnosticAction | int,
    ) -> tuple[PolicyObservation, float, bool, EpisodeSummary | None]:
        if self.terminated:
            raise RuntimeError("episode already terminated")
        action = DiagnosticAction(action)
        self._last_action = int(action)
        self.decision_steps += 1
        reward = 0.0

        if action is DiagnosticAction.ADVANCE:
            result = self.env.act(Advance(1))
            reward += result.reward
            self.advance_count += 1
            self.terminated = result.terminated
        elif action is DiagnosticAction.INSPECT:
            result = self.env.act(Inspect(self.pair.evidence_node))
            reward += result.reward
            self.inspect_count += 1
            node_index = self.pair.graph.node_index[self.pair.evidence_node]
            scale = max(float(self.pair.graph.rates.max()), 1.0)
            self._telemetry_mask[node_index] = True
            self._telemetry_inputs[node_index] = (
                float(result.observation["input_buffer"]) / scale
            )
            self._telemetry_outputs[node_index] = (
                float(result.observation["output_buffer"]) / scale
            )
            self.informative_inspection |= self.env.state.tick > 0
        else:
            reward += self._commit_repair(action)
            self.terminated = True

        if not self.terminated and self.decision_steps >= self.max_decision_steps:
            reward += self._force_timeout()
            self.terminated = True

        summary = self.summary() if self.terminated else None
        return self.observe(), reward, self.terminated, summary

    def _commit_repair(self, action: DiagnosticAction) -> float:
        blocked_repair = next(
            world.correct_repair
            for world in self.pair.worlds
            if isinstance(world.correct_repair, ClearBlockage)
        )
        failed_repair = next(
            world.correct_repair
            for world in self.pair.worlds
            if isinstance(world.correct_repair, Replace)
        )
        repair = (
            blocked_repair
            if action is DiagnosticAction.CLEAR_BLOCKAGE
            else failed_repair
        )
        self.selected_repair = action
        result = self.env.act(repair)
        reward = result.reward
        if self.env.reward_tracker is not None and not self.env.reward_tracker.terminated:
            reward += self.env.act(Advance(self.pair.reward.max_ticks)).reward
        return reward

    def _force_timeout(self) -> float:
        if self.env.reward_tracker is None or self.env.reward_tracker.terminated:
            return 0.0
        return self.env.act(Advance(self.pair.reward.max_ticks)).reward

    def summary(self) -> EpisodeSummary:
        if self.env.reward_tracker is None:
            raise AssertionError("RL episode has no reward tracker")
        metrics = self.env.reward_tracker.snapshot()
        correct_action = (
            DiagnosticAction.CLEAR_BLOCKAGE
            if isinstance(self.pair.worlds[self.world_index].fault, BlockedEdge)
            else DiagnosticAction.REPLACE_PROCESSOR
        )
        return EpisodeSummary(
            pair_id=self.pair.pair_id,
            condition=self.condition,
            world_index=self.world_index,
            cue=self.cue,
            total_reward=metrics.total_reward,
            recovered=metrics.recovered,
            correct_repair=self.selected_repair is correct_action,
            selected_repair=self.selected_repair,
            decision_steps=self.decision_steps,
            simulator_ticks=self.env.state.tick,
            advance_count=self.advance_count,
            inspect_count=self.inspect_count,
            informative_inspection=self.informative_inspection,
            false_repair_count=metrics.false_repair_count,
        )
