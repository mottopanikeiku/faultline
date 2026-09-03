"""Mutable numeric state for a compiled factory graph."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from faultline.env.graph import FactoryGraph

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(slots=True)
class FactoryState:
    """One environment state; arrays are copied explicitly when branching worlds."""

    inputs: FloatArray
    outputs: FloatArray
    node_enabled: BoolArray
    edge_enabled: BoolArray
    last_edge_flow: FloatArray
    tick: int = 0
    injected_total: float = 0.0
    delivered_total: float = 0.0
    initial_material: float = 0.0

    @classmethod
    def healthy(cls, graph: FactoryGraph) -> FactoryState:
        """Create an independent healthy state from immutable graph initial values."""
        return cls(
            inputs=graph.initial_inputs.copy(),
            outputs=graph.initial_outputs.copy(),
            node_enabled=np.ones(graph.node_count, dtype=np.bool_),
            edge_enabled=np.ones(graph.edge_count, dtype=np.bool_),
            last_edge_flow=np.zeros(graph.edge_count, dtype=np.float64),
            initial_material=float(
                graph.initial_inputs.sum(dtype=np.float64)
                + graph.initial_outputs.sum(dtype=np.float64)
            ),
        )

    def clone(self) -> FactoryState:
        """Deep-copy state for deterministic counterfactual simulation."""
        return FactoryState(
            inputs=self.inputs.copy(),
            outputs=self.outputs.copy(),
            node_enabled=self.node_enabled.copy(),
            edge_enabled=self.edge_enabled.copy(),
            last_edge_flow=self.last_edge_flow.copy(),
            tick=self.tick,
            injected_total=self.injected_total,
            delivered_total=self.delivered_total,
            initial_material=self.initial_material,
        )

    @property
    def material_held(self) -> float:
        """Material currently present in all node input and output buffers."""
        return float(self.inputs.sum(dtype=np.float64) + self.outputs.sum(dtype=np.float64))

    @property
    def conservation_residual(self) -> float:
        """Initial plus injected material minus delivered and currently held material."""
        return (
            self.initial_material
            + self.injected_total
            - self.delivered_total
            - self.material_held
        )
