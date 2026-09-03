"""Versioned, non-overlapping procedural seed ranges."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from faultline.generation.automated import GENERATOR_VERSION

SPLIT_VERSION = "factory-pairs-v0"


@dataclass(frozen=True, slots=True)
class SplitDefinition:
    name: str
    seed_start: int
    count: int
    generator_version: str = GENERATOR_VERSION
    split_version: str = SPLIT_VERSION

    def __post_init__(self) -> None:
        if not self.name or self.seed_start < 0 or self.count <= 0:
            raise ValueError("split requires a name, non-negative seed start, and positive count")

    @property
    def seed_stop(self) -> int:
        return self.seed_start + self.count

    def seed_at(self, index: int) -> int:
        if not 0 <= index < self.count:
            raise IndexError("split index out of range")
        return self.seed_start + index


SPLITS: Mapping[str, SplitDefinition] = MappingProxyType(
    {
        "train": SplitDefinition("train", seed_start=0, count=100_000),
        "validation": SplitDefinition("validation", seed_start=1_000_000, count=10_000),
        "test": SplitDefinition("test", seed_start=2_000_000, count=10_000),
    }
)


def get_split(name: str) -> SplitDefinition:
    try:
        return SPLITS[name]
    except KeyError as error:
        raise ValueError(f"unknown split {name!r}") from error
