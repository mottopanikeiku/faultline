"""Typed public action vocabulary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias


class ActionKind(StrEnum):
    INSPECT = "inspect"
    MEASURE_FLOW = "measure_flow"
    ISOLATE = "isolate"
    TOGGLE = "toggle"
    REPLACE = "replace"
    CLEAR_BLOCKAGE = "clear_blockage"
    ADVANCE = "advance"


@dataclass(frozen=True, slots=True)
class Inspect:
    node: str
    kind = ActionKind.INSPECT


@dataclass(frozen=True, slots=True)
class MeasureFlow:
    edge: str
    kind = ActionKind.MEASURE_FLOW


@dataclass(frozen=True, slots=True)
class Isolate:
    edge: str
    isolated: bool = True
    kind = ActionKind.ISOLATE


@dataclass(frozen=True, slots=True)
class Toggle:
    node: str
    kind = ActionKind.TOGGLE


@dataclass(frozen=True, slots=True)
class Replace:
    node: str
    kind = ActionKind.REPLACE


@dataclass(frozen=True, slots=True)
class ClearBlockage:
    edge: str
    kind = ActionKind.CLEAR_BLOCKAGE


@dataclass(frozen=True, slots=True)
class Advance:
    ticks: int = 1
    kind = ActionKind.ADVANCE

    def __post_init__(self) -> None:
        if self.ticks <= 0:
            raise ValueError("advance requires at least one tick")


Action: TypeAlias = Inspect | MeasureFlow | Isolate | Toggle | Replace | ClearBlockage | Advance
