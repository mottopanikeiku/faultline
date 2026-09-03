"""Latent fault interface shared by concrete fault families."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from faultline.env.graph import FactoryGraph
from faultline.env.state import FactoryState


class FaultKind(StrEnum):
    BLOCKED_EDGE = "blocked_edge"
    FAILED_PROCESSOR = "failed_processor"
    DOWNSTREAM_BACKPRESSURE = "downstream_backpressure"


class LatentFault(Protocol):
    """Fault descriptors are scenario metadata and never policy observations."""

    @property
    def kind(self) -> FaultKind: ...

    @property
    def component(self) -> str: ...

    def inject(self, graph: FactoryGraph, state: FactoryState) -> None: ...


def inject_fault(graph: FactoryGraph, state: FactoryState, fault: LatentFault) -> None:
    """Apply a validated latent fault to mutable privileged state."""
    fault.inject(graph, state)
