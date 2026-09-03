"""Transport fault semantics."""

from __future__ import annotations

from dataclasses import dataclass

from faultline.env.graph import FactoryGraph
from faultline.env.state import FactoryState
from faultline.faults.base import FaultKind


@dataclass(frozen=True, slots=True)
class BlockedEdge:
    """A transport edge has zero effective capacity until explicitly cleared."""

    component: str

    @property
    def kind(self) -> FaultKind:
        return FaultKind.BLOCKED_EDGE

    def inject(self, graph: FactoryGraph, state: FactoryState) -> None:
        try:
            edge_index = graph.edge_index[self.component]
        except KeyError as error:
            raise ValueError(f"unknown blocked edge {self.component!r}") from error
        state.edge_blocked[edge_index] = True
