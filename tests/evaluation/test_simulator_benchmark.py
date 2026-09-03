from __future__ import annotations

from faultline.evaluation import SimulatorBenchmarkConfig, run_simulator_benchmark


def test_simulator_benchmark_reports_measured_batch_rates() -> None:
    config = SimulatorBenchmarkConfig(
        batch_sizes=(8, 16),
        node_count=5,
        ticks=3,
        warmup_ticks=1,
        repeats=2,
    )

    metrics = run_simulator_benchmark(config)

    assert metrics["node_count"] == 5
    assert metrics["edge_count"] == 4
    assert metrics["metric_version"] == "simulator-throughput-v1"
    batches = metrics["batches"]
    assert [batch["batch_size"] for batch in batches] == [8, 16]
    assert [batch["environment_steps"] for batch in batches] == [24, 48]
    for batch in batches:
        assert batch["steps_per_second_median"] > 0.0
        assert len(batch["steps_per_second_samples"]) == 2
