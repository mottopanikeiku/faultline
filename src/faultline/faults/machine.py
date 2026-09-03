"""Machine and downstream-consumer fault semantics."""

from __future__ import annotations

from dataclasses import dataclass

from faultline.env.graph import FactoryGraph, NodeType
from faultline.env.state import FactoryState
from faultline.faults.base import FaultKind


@dataclass(frozen=True, slots=True)
class FailedProcessor:
    """A processor cannot transform input until it is replaced."""

    component: str

    @property
    def kind(self) -> FaultKind:
        return FaultKind.FAILED_PROCESSOR

    def inject(self, graph: FactoryGraph, state: FactoryState) -> None:
        try:
            node_index = graph.node_index[self.component]
        except KeyError as error:
            raise ValueError(f"unknown failed processor {self.component!r}") from error
        if graph.node_types[node_index] != NodeType.PROCESSOR:
            raise ValueError(f"failed-processor target {self.component!r} is not a processor")
        state.node_failed[node_index] = True


@dataclass(frozen=True, slots=True)
class DownstreamBackpressure:
    """A buffer or sink refuses inbound material; a sink also stops consuming."""

    component: str

    @property
    def kind(self) -> FaultKind:
        return FaultKind.DOWNSTREAM_BACKPRESSURE

    def inject(self, graph: FactoryGraph, state: FactoryState) -> None:
        try:
            node_index = graph.node_index[self.component]
        except KeyError as error:
            raise ValueError(f"unknown backpressured node {self.component!r}") from error
        if graph.node_types[node_index] not in (NodeType.BUFFER, NodeType.SINK):
            raise ValueError(
                f"downstream-backpressure target {self.component!r} is not a buffer or sink"
            )
        state.node_backpressured[node_index] = True
