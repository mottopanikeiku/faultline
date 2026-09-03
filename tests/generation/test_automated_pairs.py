from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np

from faultline.generation import (
    GENERATOR_VERSION,
    SPLITS,
    build_generated_diagnostic_pair,
    build_pair_dataset,
    create_world_env,
    full_passive_snapshot,
    generate_validated_pairs,
    summarize_generation,
    validate_diagnostic_pair,
)


def test_generated_pair_is_deterministic_from_seed() -> None:
    left = build_generated_diagnostic_pair(12345)
    right = build_generated_diagnostic_pair(12345)

    assert left.pair_id == right.pair_id
    assert left.generator_version == GENERATOR_VERSION
    assert left.parameters == right.parameters
    assert left.worlds == right.worlds
    assert left.reward == right.reward
    assert left.graph.node_names == right.graph.node_names
    assert left.graph.edge_names == right.graph.edge_names
    np.testing.assert_array_equal(left.graph.node_types, right.graph.node_types)
    np.testing.assert_array_equal(left.graph.rates, right.graph.rates)
    np.testing.assert_array_equal(left.graph.edge_capacities, right.graph.edge_capacities)


def test_generated_pairs_are_exactly_ambiguous_and_oracle_validated() -> None:
    batch = generate_validated_pairs(12, seed_start=100)

    assert batch.attempts == 12
    assert batch.acceptance_rate == 1.0
    assert not batch.rejection_counts
    for pair, validation in zip(batch.pairs, batch.validations, strict=True):
        snapshots = tuple(
            full_passive_snapshot(create_world_env(pair, world)) for world in pair.worlds
        )
        assert snapshots[0] == snapshots[1]
        assert validation.valid
        assert validation.passive_snapshots_identical
        assert validation.repairs_differ
        assert validation.unique_repair_success
        assert validation.diagnostic_outcome_count == 2
        assert validation.epistemic_pressure > 0.0
        assert validation.passive_recovery_probability == 0.5
        assert validation.active_recovery_probability == 1.0


def test_generator_varies_structure_and_cost_parameters() -> None:
    pairs = [build_generated_diagnostic_pair(seed) for seed in range(32)]
    parameter_maps = [dict(pair.parameters) for pair in pairs]

    assert len({parameters["node_count"] for parameters in parameter_maps}) >= 5
    assert len({parameters["fault_position"] for parameters in parameter_maps}) >= 3
    assert len({parameters["rate"] for parameters in parameter_maps}) >= 4
    assert len({parameters["diagnostic_cost"] for parameters in parameter_maps}) == 3
    assert {pair.worlds[0].fault.kind.value for pair in pairs} == {
        "blocked_edge",
        "failed_processor",
    }


def test_individual_validator_reports_positive_repair_margin() -> None:
    validation = validate_diagnostic_pair(build_generated_diagnostic_pair(77))

    assert validation.valid
    assert validation.repair_margin > 0.0
    assert validation.active_expected_return > validation.passive_expected_return


def test_dataset_contains_reconstructable_tasks_and_validation_summary() -> None:
    batch = generate_validated_pairs(3, seed_start=5)
    dataset = build_pair_dataset(batch, SPLITS["train"], offset=5)
    summary = summarize_generation(batch)

    assert dataset["schema_version"] == 1
    assert dataset["seed_start"] == 5
    assert dataset["count"] == 3
    assert [task["seed"] for task in dataset["tasks"]] == [5, 6, 7]
    assert all(task["validation"]["valid"] for task in dataset["tasks"])
    assert all(task["graph"]["node_names"] for task in dataset["tasks"])
    assert summary["validated_count"] == 3
    assert summary["ep_min"] > 0.0


def test_versioned_split_ranges_are_disjoint_and_match_config() -> None:
    ordered = sorted(SPLITS.values(), key=lambda split: split.seed_start)
    for left, right in pairwise(ordered):
        assert left.seed_stop <= right.seed_start

    repo = Path(__file__).parents[2]
    config = json.loads(
        (repo / "configs/splits/factory-pairs-v0.json").read_text(encoding="utf-8")
    )
    assert config["generator_version"] == GENERATOR_VERSION
    for name, split in SPLITS.items():
        assert config["splits"][name] == {
            "seed_start": split.seed_start,
            "count": split.count,
        }
