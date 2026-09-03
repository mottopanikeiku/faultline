"""Faultline command-line interface."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from faultline import __version__
from faultline.artifacts import (
    build_manifest,
    canonical_sha256,
    git_state,
    repository_root,
    utc_timestamp,
    write_json_artifact,
    write_manifest,
    write_text_artifact,
)
from faultline.env import Action, Advance, ClearBlockage, FactoryEnv, Replace
from faultline.evaluation import (
    EP_ANALYSIS_VERSION,
    METRIC_VERSION,
    SimulatorBenchmarkConfig,
    analyze_ep_distribution,
    run_simulator_benchmark,
)
from faultline.generation import (
    GENERATOR_VERSION,
    build_manual_diagnostic_pair,
    build_pair_dataset,
    chain_factory,
    create_world_env,
    diagnostic_evidence,
    full_passive_snapshot,
    generate_validated_pairs,
    get_split,
    run_contingent_active_policy,
    summarize_generation,
)
from faultline.oracle import (
    analyze_actions,
    diagnostic_pair_problem,
    solve_active,
    solve_passive,
)
from faultline.visualization import (
    render_factory,
    render_histogram_svg,
    render_paired_values_svg,
    render_timeline,
)


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


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
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

    environment = commands.add_parser("env", help="generate and inspect factory tasks")
    environment_commands = environment.add_subparsers(dest="env")
    generate = environment_commands.add_parser(
        "generate",
        help="generate exact validated diagnostic pairs",
    )
    generate.add_argument("--count", type=_positive_int, default=100)
    generate.add_argument("--split", choices=("train", "validation"), default="train")
    generate.add_argument("--offset", type=_nonnegative_int, default=0)
    generate.add_argument("--max-attempts", type=_positive_int)
    generate.add_argument("--minimum-ep", type=_nonnegative_float, default=1e-9)
    generate.add_argument("--run-id")
    generate.add_argument("--output", type=Path)
    generate.add_argument("--manifest", type=Path)

    oracle = commands.add_parser("oracle", help="solve bounded diagnostic problems")
    oracle_commands = oracle.add_subparsers(dest="oracle")
    solve = oracle_commands.add_parser("solve", help="solve the exact manual two-world problem")
    solve.add_argument("--seed", type=_nonnegative_int, default=42)
    solve.add_argument("--depth", type=_diagnostic_depth, default=3)
    solve.add_argument("--run-id")
    solve.add_argument("--output", type=Path)

    curriculum = commands.add_parser("curriculum", help="analyze task-selection signals")
    curriculum_commands = curriculum.add_subparsers(dest="curriculum")
    analyze = curriculum_commands.add_parser(
        "analyze",
        help="analyze EP and intervention values in a saved pair dataset",
    )
    analyze.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "artifacts/results/gate1-diagnostic-pairs-v0.2-20260903.json"
        ),
    )
    analyze.add_argument("--run-id")
    analyze.add_argument("--output", type=Path)
    analyze.add_argument("--ep-plot", type=Path)
    analyze.add_argument("--intervention-plot", type=Path)
    analyze.add_argument("--manifest", type=Path)

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
    problem = diagnostic_pair_problem(pair)
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


def _run_pair_generation(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    repo = _require_clean_repository(parser)
    split = get_split(args.split)
    if args.offset + args.count > split.count:
        parser.error("requested range exceeds the immutable split")
    run_id = args.run_id or datetime.now(UTC).strftime("diagnostic-pairs-%Y%m%dT%H%M%SZ")
    config = {
        "count": args.count,
        "split": split.name,
        "split_version": split.split_version,
        "offset": args.offset,
        "seed_start": split.seed_start + args.offset,
        "max_attempts": args.max_attempts,
        "minimum_ep": args.minimum_ep,
    }
    started_at = utc_timestamp()
    batch = generate_validated_pairs(
        args.count,
        seed_start=split.seed_start + args.offset,
        max_attempts=args.max_attempts,
        minimum_ep=args.minimum_ep,
    )
    dataset = build_pair_dataset(batch, split, offset=args.offset)
    dataset["dataset_id"] = run_id
    completed_at = utc_timestamp()
    result_output = args.output or repo / "artifacts" / "results" / f"{run_id}.json"
    manifest_output = (
        args.manifest or repo / "artifacts" / "manifests" / f"{run_id}.json"
    )
    metrics = summarize_generation(batch)
    metrics["result_path"] = str(result_output)
    metrics["result_sha256"] = canonical_sha256(dataset)
    manifest = build_manifest(
        repo=repo,
        run_id=run_id,
        experiment="diagnostic-pair-generation",
        config=config,
        metrics=metrics,
        started_at=started_at,
        completed_at=completed_at,
        metric_version="diagnostic-pair-validation-v1",
        split=split.name,
        seed=split.seed_start + args.offset,
        generator_version=split.generator_version,
        reward={"varies_by_task": True},
    )
    write_json_artifact(result_output, dataset)
    write_manifest(manifest_output, manifest)
    print(
        f"validated={len(batch.pairs)} attempts={batch.attempts} "
        f"acceptance={batch.acceptance_rate:.1%}"
    )
    print(
        f"ep_min={metrics['ep_min']:.3f} ep_mean={metrics['ep_mean']:.3f} "
        f"ep_max={metrics['ep_max']:.3f}"
    )
    print(f"result={result_output}")
    print(f"manifest={manifest_output}")
    return 0


def _artifact_reference(path: Path, repo: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo.resolve()))
    except ValueError:
        return str(resolved)


def _run_ep_analysis(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    repo = _require_clean_repository(parser)
    dataset_path = args.dataset if args.dataset.is_absolute() else repo / args.dataset
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if dataset["generator_version"] != GENERATOR_VERSION:
        parser.error("dataset generator version does not match this checkout")
    seeds = [int(task["seed"]) for task in dataset["tasks"]]
    if len(seeds) != dataset["count"]:
        parser.error("dataset task count is inconsistent")
    run_id = args.run_id or datetime.now(UTC).strftime("ep-analysis-%Y%m%dT%H%M%SZ")
    started_at = utc_timestamp()
    timer_started = perf_counter()
    analysis = analyze_ep_distribution(seeds)
    elapsed = perf_counter() - timer_started
    analysis["source_dataset"] = _artifact_reference(dataset_path, repo)
    analysis["source_dataset_sha256"] = canonical_sha256(dataset)
    completed_at = utc_timestamp()

    result_output = args.output or repo / "artifacts" / "results" / f"{run_id}.json"
    ep_plot_output = (
        args.ep_plot or repo / "artifacts" / "results" / f"{run_id}-ep.svg"
    )
    intervention_plot_output = (
        args.intervention_plot
        or repo / "artifacts" / "results" / f"{run_id}-interventions.svg"
    )
    manifest_output = (
        args.manifest or repo / "artifacts" / "manifests" / f"{run_id}.json"
    )
    rows = analysis["rows"]
    ep_svg = render_histogram_svg(
        [float(row["epistemic_pressure"]) for row in rows],
        title="Epistemic pressure across validated pairs",
        x_label="active expected return - passive expected return",
    )
    intervention_svg = render_paired_values_svg(
        [float(row["immediate_advance_decision_value"]) for row in rows],
        [float(row["post_advance_inspect_decision_value"]) for row in rows],
        title="Decision value before and after diagnostic dynamics",
        before_label="advance now",
        after_label="inspect after advance",
        y_label="one-step decision value",
    )
    metrics = {
        key: value
        for key, value in analysis.items()
        if key not in {"rows", "seeds"}
    }
    metrics.update(
        {
            "elapsed_seconds": elapsed,
            "pair_analyses_per_second": len(seeds) / elapsed,
            "result_path": _artifact_reference(result_output, repo),
            "result_sha256": canonical_sha256(analysis),
            "ep_plot_path": _artifact_reference(ep_plot_output, repo),
            "ep_plot_sha256": hashlib.sha256(ep_svg.encode("utf-8")).hexdigest(),
            "intervention_plot_path": _artifact_reference(
                intervention_plot_output,
                repo,
            ),
            "intervention_plot_sha256": hashlib.sha256(
                intervention_svg.encode("utf-8")
            ).hexdigest(),
        }
    )
    config = {
        "source_dataset": _artifact_reference(dataset_path, repo),
        "source_dataset_sha256": canonical_sha256(dataset),
        "pair_count": len(seeds),
        "oracle_depth": 2,
    }
    manifest = build_manifest(
        repo=repo,
        run_id=run_id,
        experiment="ep-distribution-analysis",
        config=config,
        metrics=metrics,
        started_at=started_at,
        completed_at=completed_at,
        metric_version=EP_ANALYSIS_VERSION,
        split=str(dataset["split"]),
        seed=seeds[0],
        generator_version=GENERATOR_VERSION,
        policy={"name": "exact-enumeration", "diagnostic_depth": 2},
        reward={"varies_by_task": True},
    )
    write_json_artifact(result_output, analysis)
    write_text_artifact(ep_plot_output, ep_svg)
    write_text_artifact(intervention_plot_output, intervention_svg)
    write_manifest(manifest_output, manifest)
    ep = analysis["epistemic_pressure"]
    print(
        f"pairs={len(seeds)} ep_min={ep['min']:.3f} "
        f"ep_mean={ep['mean']:.3f} ep_max={ep['max']:.3f}"
    )
    print(
        f"analysis_rate={len(seeds) / elapsed:.1f} pairs/s "
        f"result={result_output}"
    )
    print(f"ep_plot={ep_plot_output}")
    print(f"intervention_plot={intervention_plot_output}")
    print(f"manifest={manifest_output}")
    return 0


def _run_oracle_solve(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    repo = _require_clean_repository(parser)
    pair = build_manual_diagnostic_pair(args.seed)
    problem = diagnostic_pair_problem(pair)
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
    if args.command == "env" and args.env == "generate":
        return _run_pair_generation(args, parser)
    if args.command == "curriculum" and args.curriculum == "analyze":
        return _run_ep_analysis(args, parser)
    if args.command == "oracle" and args.oracle == "solve":
        return _run_oracle_solve(args, parser)
    if args.command == "benchmark" and args.benchmark == "simulator":
        return _run_simulator_benchmark(args, parser)
    parser.print_help()
    return 0
