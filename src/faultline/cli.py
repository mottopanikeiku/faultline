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
from faultline.env import Action, Advance, ClearBlockage, FactoryEnv, Replace
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
    full_passive_snapshot,
    run_contingent_active_policy,
)
from faultline.oracle import (
    analyze_actions,
    manual_pair_problem,
    solve_active,
    solve_passive,
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


def _diagnostic_depth(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 6:
        raise argparse.ArgumentTypeError("diagnostic depth must be between zero and six")
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

    oracle = commands.add_parser("oracle", help="solve bounded diagnostic problems")
    oracle_commands = oracle.add_subparsers(dest="oracle")
    solve = oracle_commands.add_parser("solve", help="solve the exact manual two-world problem")
    solve.add_argument("--seed", type=_nonnegative_int, default=42)
    solve.add_argument("--depth", type=_diagnostic_depth, default=3)
    solve.add_argument("--run-id")
    solve.add_argument("--output", type=Path)

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
    problem = manual_pair_problem(pair)
    passive_oracle = solve_passive(problem)
    active_oracle = solve_active(problem, max_diagnostic_actions=2)
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
    print(f"Best passive repair success: {passive_oracle.recovery_probability:.0%}")
    print(f"Best passive expected return: {passive_oracle.expected_return:.2f}")
    print(
        "Evidence-contingent success: "
        f"{sum(result.recovered for result in active) / len(active):.0%}"
    )
    print(
        "Evidence-contingent expected return: "
        f"{sum(result.total_return for result in active) / len(active):.2f}"
    )
    print(
        f"Exact active oracle: success={active_oracle.recovery_probability:.0%} "
        f"return={active_oracle.expected_return:.2f} "
        f"EP={active_oracle.expected_return - passive_oracle.expected_return:.2f}"
    )
    first_action = active_oracle.diagnostic_action
    second_action = (
        active_oracle.outcomes[0].decision.diagnostic_action
        if active_oracle.outcomes
        else None
    )
    if first_action is not None and second_action is not None:
        print(
            f"Optimal diagnostic sequence: {first_action.kind.value} "
            f"-> {second_action.kind.value}"
        )
    return 0


def _require_clean_repository(parser: argparse.ArgumentParser) -> Path:
    repo = repository_root(Path.cwd())
    _, dirty = git_state(repo)
    if dirty:
        parser.error("measured runs require a clean Git worktree")
    return repo


def _action_record(action: Action) -> dict[str, object]:
    return {"kind": action.kind.value, **asdict(action)}


def _run_oracle_solve(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    repo = _require_clean_repository(parser)
    pair = build_manual_diagnostic_pair(args.seed)
    problem = manual_pair_problem(pair)
    started_at = utc_timestamp()
    passive = solve_passive(problem)
    active = solve_active(problem, args.depth)
    action_values = analyze_actions(problem)
    completed_at = utc_timestamp()
    snapshot = full_passive_snapshot(create_world_env(pair, pair.worlds[0]))
    metrics = {
        "passive_expected_return": passive.expected_return,
        "passive_recovery_probability": passive.recovery_probability,
        "passive_plan": passive.plan.name,
        "active_expected_return": active.expected_return,
        "active_recovery_probability": active.recovery_probability,
        "ep": active.expected_return - passive.expected_return,
        "root_decision": active.kind.value,
        "root_action": (
            _action_record(active.diagnostic_action)
            if active.diagnostic_action is not None
            else None
        ),
        "initial_snapshot_sha256": canonical_sha256(asdict(snapshot)),
        "action_values": [
            {
                "action": _action_record(value.action),
                "expected_return": value.expected_return,
                "recovery_probability": value.recovery_probability,
                "decision_value": value.decision_value,
                "information_gain_bits": value.information_gain_bits,
                "outcome_count": value.outcome_count,
            }
            for value in action_values
        ],
    }
    config = {
        "pair_seed": pair.seed,
        "pair_version": "manual-v1",
        "max_diagnostic_actions": args.depth,
    }
    run_id = args.run_id or datetime.now(UTC).strftime("two-world-oracle-%Y%m%dT%H%M%SZ")
    manifest = build_manifest(
        repo=repo,
        run_id=run_id,
        experiment="exact-two-world-oracle",
        config=config,
        metrics=metrics,
        started_at=started_at,
        completed_at=completed_at,
        metric_version="exact-two-world-oracle-v1",
        split="manual-construction",
        seed=pair.seed,
        generator_version="manual-v1",
        policy={"name": "exact-enumeration", "diagnostic_depth": args.depth},
        reward=asdict(pair.reward),
    )
    output = args.output or repo / "artifacts" / "manifests" / f"{run_id}.json"
    write_manifest(output, manifest)
    print(
        f"passive_return={passive.expected_return:.2f} "
        f"active_return={active.expected_return:.2f} "
        f"ep={active.expected_return - passive.expected_return:.2f}"
    )
    print(
        f"passive_success={passive.recovery_probability:.0%} "
        f"active_success={active.recovery_probability:.0%}"
    )
    print(f"manifest={output}")
    return 0


def _run_simulator_benchmark(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    repo = _require_clean_repository(parser)
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
    if args.command == "oracle" and args.oracle == "solve":
        return _run_oracle_solve(args, parser)
    if args.command == "benchmark" and args.benchmark == "simulator":
        return _run_simulator_benchmark(args, parser)
    parser.print_help()
    return 0
