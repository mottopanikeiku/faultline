"""Operational environment wrapper around deterministic factory dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from faultline.env.actions import (
    Action,
    ActionKind,
    Advance,
    ClearBlockage,
    Inspect,
    Isolate,
    MeasureFlow,
    Replace,
    Toggle,
)
from faultline.env.dynamics import advance
from faultline.env.graph import FactoryGraph
from faultline.env.observation import (
    PublicObservation,
    inspect_node,
    measure_edge,
    observe_status,
)
from faultline.env.state import FactoryState
from faultline.faults.base import LatentFault, inject_fault


class ActionError(StrEnum):
    UNKNOWN_COMPONENT = "unknown_component"


@dataclass(frozen=True, slots=True)
class ActionResult:
    """Public result; repair outcomes intentionally omit whether a fault changed."""

    kind: ActionKind
    accepted: bool
    observation: PublicObservation
    error: ActionError | None = None


@dataclass(frozen=True, slots=True)
class Interaction:
    action: Action
    result: ActionResult


@dataclass(slots=True)
class FactoryEnv:
    """A deterministic interactive episode with public-only history."""

    graph: FactoryGraph
    state: FactoryState
    check_invariants: bool = False
    history: list[Interaction] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        graph: FactoryGraph,
        fault: LatentFault | None = None,
        *,
        check_invariants: bool = False,
    ) -> FactoryEnv:
        state = FactoryState.healthy(graph)
        if fault is not None:
            inject_fault(graph, state, fault)
        return cls(graph=graph, state=state, check_invariants=check_invariants)

    def observe(self) -> PublicObservation:
        return observe_status(self.state)

    def act(self, action: Action) -> ActionResult:
        """Execute one typed action and append exactly its public result to history."""
        if isinstance(action, Inspect):
            result = self._inspect(action)
        elif isinstance(action, MeasureFlow):
            result = self._measure_flow(action)
        elif isinstance(action, Isolate):
            result = self._isolate(action)
        elif isinstance(action, Toggle):
            result = self._toggle(action)
        elif isinstance(action, Replace):
            result = self._replace(action)
        elif isinstance(action, ClearBlockage):
            result = self._clear_blockage(action)
        elif isinstance(action, Advance):
            result = self._advance(action)
        else:
            raise TypeError(f"unsupported action type: {type(action).__name__}")
        self.history.append(Interaction(action=action, result=result))
        return result

    def _inspect(self, action: Inspect) -> ActionResult:
        node_index = self.graph.node_index.get(action.node)
        if node_index is None:
            return self._unknown(action.kind, action.node)
        return ActionResult(action.kind, True, inspect_node(self.graph, self.state, node_index))

    def _measure_flow(self, action: MeasureFlow) -> ActionResult:
        edge_index = self.graph.edge_index.get(action.edge)
        if edge_index is None:
            return self._unknown(action.kind, action.edge)
        return ActionResult(action.kind, True, measure_edge(self.graph, self.state, edge_index))

    def _isolate(self, action: Isolate) -> ActionResult:
        edge_index = self.graph.edge_index.get(action.edge)
        if edge_index is None:
            return self._unknown(action.kind, action.edge)
        self.state.edge_enabled[edge_index] = not action.isolated
        return ActionResult(
            action.kind,
            True,
            {
                "event": action.kind.value,
                "edge": action.edge,
                "isolated": action.isolated,
            },
        )

    def _toggle(self, action: Toggle) -> ActionResult:
        node_index = self.graph.node_index.get(action.node)
        if node_index is None:
            return self._unknown(action.kind, action.node)
        self.state.node_enabled[node_index] = not self.state.node_enabled[node_index]
        return ActionResult(
            action.kind,
            True,
            {
                "event": action.kind.value,
                "node": action.node,
                "enabled": bool(self.state.node_enabled[node_index]),
            },
        )

    def _replace(self, action: Replace) -> ActionResult:
        node_index = self.graph.node_index.get(action.node)
        if node_index is None:
            return self._unknown(action.kind, action.node)
        self.state.node_failed[node_index] = False
        self.state.node_backpressured[node_index] = False
        return ActionResult(
            action.kind,
            True,
            {"event": action.kind.value, "node": action.node, "maintenance": "completed"},
        )

    def _clear_blockage(self, action: ClearBlockage) -> ActionResult:
        edge_index = self.graph.edge_index.get(action.edge)
        if edge_index is None:
            return self._unknown(action.kind, action.edge)
        self.state.edge_blocked[edge_index] = False
        return ActionResult(
            action.kind,
            True,
            {"event": action.kind.value, "edge": action.edge, "maintenance": "completed"},
        )

    def _advance(self, action: Advance) -> ActionResult:
        transition = advance(
            self.graph,
            self.state,
            action.ticks,
            check_invariants=self.check_invariants,
        )
        observation = observe_status(self.state)
        observation["event"] = action.kind.value
        observation["ticks_advanced"] = transition.ticks
        return ActionResult(action.kind, True, observation)

    @staticmethod
    def _unknown(kind: ActionKind, component: str) -> ActionResult:
        return ActionResult(
            kind=kind,
            accepted=False,
            observation={"event": kind.value, "component": component, "error": "unknown_component"},
            error=ActionError.UNKNOWN_COMPONENT,
        )
