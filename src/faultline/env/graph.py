"""Immutable factory topology compiled to dense arrays."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ByteArray = NDArray[np.uint8]


class NodeType(IntEnum):
    """Node behavior implemented by the deterministic material-flow kernel."""

    SOURCE = 0
    PROCESSOR = 1
    BUFFER = 2
    SINK = 3


@dataclass(frozen=True, slots=True)
class Node:
    """Readable node declaration; compiled out of the simulation hot loop."""

    name: str
    kind: NodeType
    rate: float
    input_capacity: float = 0.0
    output_capacity: float = 0.0
    initial_input: float = 0.0
    initial_output: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("node name must not be empty")
        values = (
            self.rate,
            self.input_capacity,
            self.output_capacity,
            self.initial_input,
            self.initial_output,
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError(f"node {self.name!r} has a negative or non-finite numeric field")
        if self.initial_input > self.input_capacity:
            raise ValueError(f"node {self.name!r} initial input exceeds capacity")
        if self.initial_output > self.output_capacity:
            raise ValueError(f"node {self.name!r} initial output exceeds capacity")
        if self.kind is NodeType.SOURCE and self.output_capacity <= 0.0:
            raise ValueError("source nodes require positive output capacity")
        if self.kind in (NodeType.PROCESSOR, NodeType.BUFFER) and (
            self.input_capacity <= 0.0 or self.output_capacity <= 0.0
        ):
            raise ValueError("processor and buffer nodes require input and output capacity")
        if self.kind is NodeType.SINK and self.input_capacity <= 0.0:
            raise ValueError("sink nodes require positive input capacity")


@dataclass(frozen=True, slots=True)
class Edge:
    """Readable directed transport declaration."""

    name: str
    source: str
    target: str
    capacity: float

    def __post_init__(self) -> None:
        if not self.name or not self.source or not self.target:
            raise ValueError("edge name and endpoints must not be empty")
        if not np.isfinite(self.capacity) or self.capacity <= 0.0:
            raise ValueError(f"edge {self.name!r} requires finite positive capacity")


def _readonly_float(values: list[float]) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    result.setflags(write=False)
    return result


def _readonly_int(values: list[int]) -> IntArray:
    result = np.asarray(values, dtype=np.int64)
    result.setflags(write=False)
    return result


def _readonly_byte(values: list[int]) -> ByteArray:
    result = np.asarray(values, dtype=np.uint8)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class FactoryGraph:
    """Validated directed acyclic factory graph in simulation-ready form."""

    node_names: tuple[str, ...]
    edge_names: tuple[str, ...]
    node_index: Mapping[str, int]
    edge_index: Mapping[str, int]
    node_types: ByteArray
    rates: FloatArray
    input_capacities: FloatArray
    output_capacities: FloatArray
    initial_inputs: FloatArray
    initial_outputs: FloatArray
    edge_sources: IntArray
    edge_targets: IntArray
    edge_capacities: FloatArray
    source_indices: IntArray
    transform_indices: IntArray
    sink_indices: IntArray

    @classmethod
    def compile(cls, nodes: list[Node], edges: list[Edge]) -> FactoryGraph:
        """Validate readable declarations and compile immutable dense arrays."""
        if not nodes:
            raise ValueError("factory requires at least one node")
        node_names = tuple(node.name for node in nodes)
        if len(set(node_names)) != len(node_names):
            raise ValueError("node names must be unique")
        edge_names = tuple(edge.name for edge in edges)
        if len(set(edge_names)) != len(edge_names):
            raise ValueError("edge names must be unique")

        node_lookup = {name: index for index, name in enumerate(node_names)}
        edge_sources: list[int] = []
        edge_targets: list[int] = []
        endpoint_pairs: set[tuple[int, int]] = set()
        for edge in edges:
            try:
                source = node_lookup[edge.source]
                target = node_lookup[edge.target]
            except KeyError as error:
                raise ValueError(
                    f"edge {edge.name!r} references unknown node {error.args[0]!r}"
                ) from error
            if source == target:
                raise ValueError(f"edge {edge.name!r} is a self-loop")
            if (source, target) in endpoint_pairs:
                raise ValueError("parallel transport edges are not supported")
            endpoint_pairs.add((source, target))
            edge_sources.append(source)
            edge_targets.append(target)

        cls._require_acyclic(len(nodes), edge_sources, edge_targets)
        kinds = [int(node.kind) for node in nodes]
        if not any(kind == NodeType.SOURCE for kind in kinds):
            raise ValueError("factory requires at least one source")
        if not any(kind == NodeType.SINK for kind in kinds):
            raise ValueError("factory requires at least one sink")

        return cls(
            node_names=node_names,
            edge_names=edge_names,
            node_index=MappingProxyType(node_lookup),
            edge_index=MappingProxyType({name: index for index, name in enumerate(edge_names)}),
            node_types=_readonly_byte(kinds),
            rates=_readonly_float([node.rate for node in nodes]),
            input_capacities=_readonly_float([node.input_capacity for node in nodes]),
            output_capacities=_readonly_float([node.output_capacity for node in nodes]),
            initial_inputs=_readonly_float([node.initial_input for node in nodes]),
            initial_outputs=_readonly_float([node.initial_output for node in nodes]),
            edge_sources=_readonly_int(edge_sources),
            edge_targets=_readonly_int(edge_targets),
            edge_capacities=_readonly_float([edge.capacity for edge in edges]),
            source_indices=_readonly_int(
                [index for index, kind in enumerate(kinds) if kind == NodeType.SOURCE]
            ),
            transform_indices=_readonly_int(
                [
                    index
                    for index, kind in enumerate(kinds)
                    if kind in (NodeType.PROCESSOR, NodeType.BUFFER)
                ]
            ),
            sink_indices=_readonly_int(
                [index for index, kind in enumerate(kinds) if kind == NodeType.SINK]
            ),
        )

    @staticmethod
    def _require_acyclic(node_count: int, sources: list[int], targets: list[int]) -> None:
        indegree = [0] * node_count
        outgoing: list[list[int]] = [[] for _ in range(node_count)]
        for source, target in zip(sources, targets, strict=True):
            indegree[target] += 1
            outgoing[source].append(target)
        ready = [index for index, degree in enumerate(indegree) if degree == 0]
        visited = 0
        while ready:
            node = ready.pop()
            visited += 1
            for target in outgoing[node]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
        if visited != node_count:
            raise ValueError("factory material-flow graph must be acyclic")

    @property
    def node_count(self) -> int:
        return len(self.node_names)

    @property
    def edge_count(self) -> int:
        return len(self.edge_names)
