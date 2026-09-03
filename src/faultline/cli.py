"""Faultline command-line interface."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from faultline import __version__
from faultline.artifacts import (
    build_manifest,
    canonical_sha256,
    git_state,
    repository_root,
    utc_timestamp,
    write_manifest,
)
from faultline.env import Advance, ClearBlockage, FactoryEnv, Replace
from faultline.evaluation import (
    METRIC_VERSION,
    SimulatorBenchmarkConfig,
    run_simulator_benchmark,
)
from faultline.generation import (
    build_manual_diagnostic_pair,
    chain_factory,
    create_world_env,
    diagnostic_evidence,
    evaluate_repair,
    full_passive_snapshot,
    run_contingent_active_policy,
)
from faultline.visualization import render_factory, render_timeline


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="faultline",
        description="Epistemic environment design for active diagnosis.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command")

    demo = commands.add_parser("demo", help="render a deterministic episode")
    demos = demo.add_subparsers(dest="demo")
    healthy = demos.add_parser("healthy", help="run and render a healthy chain")
    healthy.add_argument("--nodes", type=_positive_int, default=8)
    healthy.add_argument("--ticks", type=_positive_int, default=10)
    healthy.add_argument("--debug", action="store_true")
    diagnostic_pair = demos.add_parser(
        "diagnostic-pair",
        help="demonstrate two confusable latent worlds",
    )
    diagnostic_pair.add_argument("--seed", type=_nonnegative_int, default=42)

    benchmark = commands.add_parser("benchmark", help="run measured local benchmarks")
    benchmarks = benchmark.add_subparsers(dest="benchmark")
    simulator = benchmarks.add_parser("simulator", help="benchmark batched simulator throughput")
    simulator.add_argument(
        "--batch-sizes",
        type=_positive_int,
        nargs="+",
        default=[1_000, 4_000, 16_000],
    )
    simulator.add_argument("--nodes", type=_positive_int, default=16)
    simulator.add_argument("--ticks", type=_positive_int, default=200)
    simulator.add_argument("--warmup-ticks", type=_nonnegative_int, default=10)
    simulator.add_argument("--repeats", type=_positive_int, default=3)
    simulator.add_argument("--run-id")
    simulator.add_argument("--output", type=Path)
    return parser


def _run_healthy_demo(args: argparse.Namespace) -> int:
    graph = chain_factory(args.nodes)
    env = FactoryEnv.create(graph, check_invariants=True)
    env.act(Advance(args.ticks))
    print(render_factory(graph, env.state, debug=args.debug))
    print()
    print(render_timeline(env.history))
    return 0


def _format_repair(repair: ClearBlockage | Replace) -> str:
    if isinstance(repair, ClearBlockage):
        return f"clear_blockage({repair.edge})"
    return f"replace({repair.node})"


def _run_diagnostic_pair_demo(args: argparse.Namespace) -> int:
    pair = build_manual_diagnostic_pair(args.seed)
    snapshots = [
        full_passive_snapshot(create_world_env(pair, world)) for world in pair.worlds
    ]
    candidates = (ClearBlockage("delivery"), Replace("processor"))
    passive_values = [
        sum(evaluate_repair(pair, world, repair).total_return for world in pair.worlds)
        / len(pair.worlds)
        for repair in candidates
    ]
    active = [run_contingent_active_policy(pair, world) for world in pair.worlds]
    evidence = [dict(diagnostic_evidence(pair, world)) for world in pair.worlds]

    print(f"FAULTLINE DIAGNOSTIC PAIR seed={pair.seed}")
    comparison = "IDENTICAL" if snapshots[0] == snapshots[1] else "DIFFERENT"
    print(f"Initial complete passive observations: {comparison}")
    print(f"snapshot_sha256={canonical_sha256(asdict(snapshots[0]))}")
    print()
    for index, world in enumerate(pair.worlds):
        print(
            f"WORLD {world.label}: latent={world.fault.kind.value}({world.fault.component}) "
            f"optimal_repair={_format_repair(world.correct_repair)}"
        )
        print(
            f"  after isolate({pair.intervention_edge}) + advance(1): "
            f"{pair.evidence_node}.input={evidence[index]['input_buffer']:.2f} "
            f"output={evidence[index]['output_buffer']:.2f}"
        )
        print(
            f"  contingent_repair={_format_repair(active[index].selected_repair)} "
            f"recovered={active[index].recovered} return={active[index].total_return:.2f}"
        )
    print()
    print("Best passive repair success: 50%")
    print(f"Best passive expected return: {max(passive_values):.2f}")
    print(
        "Evidence-contingent success: "
        f"{sum(result.recovered for result in active) / len(active):.0%}"
    )
    print(
        "Evidence-contingent expected return: "
        f"{sum(result.total_return for result in active) / len(active):.2f}"
    )
    return 0


def _run_simulator_benchmark(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    repo = repository_root(Path.cwd())
    _, dirty = git_state(repo)
    if dirty:
        parser.error("simulator benchmarks require a clean Git worktree")
    config = SimulatorBenchmarkConfig(
        batch_sizes=tuple(args.batch_sizes),
        node_count=args.nodes,
        ticks=args.ticks,
        warmup_ticks=args.warmup_ticks,
        repeats=args.repeats,
    )
    run_id = args.run_id or datetime.now(UTC).strftime("simulator-throughput-%Y%m%dT%H%M%SZ")
    started_at = utc_timestamp()
    metrics = run_simulator_benchmark(config)
    completed_at = utc_timestamp()
    manifest = build_manifest(
        repo=repo,
        run_id=run_id,
        experiment="simulator-throughput",
        config=config.to_dict(),
        metrics=metrics,
        started_at=started_at,
        completed_at=completed_at,
        metric_version=METRIC_VERSION,
    )
    output = args.output or repo / "artifacts" / "manifests" / f"{run_id}.json"
    write_manifest(output, manifest)
    for batch in metrics["batches"]:
        print(
            f"batch={batch['batch_size']:>6} "
            f"steps/s={batch['steps_per_second_median']:>12,.0f}"
        )
    print(f"manifest={output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "demo" and args.demo == "healthy":
        return _run_healthy_demo(args)
    if args.command == "demo" and args.demo == "diagnostic-pair":
        return _run_diagnostic_pair_demo(args)
    if args.command == "benchmark" and args.benchmark == "simulator":
        return _run_simulator_benchmark(args, parser)
    parser.print_help()
    return 0
