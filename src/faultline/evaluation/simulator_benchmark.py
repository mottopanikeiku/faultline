"""Reproducible throughput benchmark for the NumPy simulation kernel."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median
from time import perf_counter
from typing import Any

from faultline.env.vectorized import VectorizedState, step_batch
from faultline.generation.layouts import chain_factory

METRIC_VERSION = "simulator-throughput-v1"


@dataclass(frozen=True, slots=True)
class SimulatorBenchmarkConfig:
    batch_sizes: tuple[int, ...] = (1_000, 4_000, 16_000)
    node_count: int = 16
    ticks: int = 200
    warmup_ticks: int = 10
    repeats: int = 3

    def __post_init__(self) -> None:
        if not self.batch_sizes or any(size <= 0 for size in self.batch_sizes):
            raise ValueError("batch sizes must be non-empty and positive")
        if self.node_count < 3:
            raise ValueError("node count must be at least three")
        if self.ticks <= 0 or self.warmup_ticks < 0 or self.repeats <= 0:
            raise ValueError("ticks and repeats must be positive; warmup may be zero")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["batch_sizes"] = list(self.batch_sizes)
        return result


def run_simulator_benchmark(config: SimulatorBenchmarkConfig) -> dict[str, Any]:
    """Measure environment transitions per wall-clock second for each batch size."""
    graph = chain_factory(config.node_count)
    batches: list[dict[str, Any]] = []
    for batch_size in config.batch_sizes:
        samples: list[float] = []
        for _ in range(config.repeats):
            state = VectorizedState.healthy(graph, batch_size)
            for _ in range(config.warmup_ticks):
                step_batch(graph, state)
            started = perf_counter()
            for _ in range(config.ticks):
                step_batch(graph, state)
            elapsed = perf_counter() - started
            samples.append(batch_size * config.ticks / elapsed)
        batches.append(
            {
                "batch_size": batch_size,
                "environment_steps": batch_size * config.ticks,
                "steps_per_second_median": median(samples),
                "steps_per_second_samples": samples,
            }
        )
    return {
        "metric_version": METRIC_VERSION,
        "node_count": graph.node_count,
        "edge_count": graph.edge_count,
        "ticks_per_repeat": config.ticks,
        "repeats": config.repeats,
        "batches": batches,
    }
