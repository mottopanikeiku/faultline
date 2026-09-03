"""Machine-readable generated diagnostic-pair datasets."""

from __future__ import annotations

from dataclasses import asdict
from statistics import fmean
from typing import Any

from faultline.env import ClearBlockage
from faultline.generation.diagnostic_pairs import DiagnosticPair, RepairAction
from faultline.generation.splits import SplitDefinition
from faultline.generation.validation import GenerationBatch

PAIR_DATASET_SCHEMA_VERSION = 1


def _repair_record(repair: RepairAction) -> dict[str, object]:
    if isinstance(repair, ClearBlockage):
        return {"kind": repair.kind.value, "edge": repair.edge}
    return {"kind": repair.kind.value, "node": repair.node}


def pair_record(pair: DiagnosticPair) -> dict[str, Any]:
    graph = pair.graph
    return {
        "pair_id": pair.pair_id,
        "seed": pair.seed,
        "generator_version": pair.generator_version,
        "parameters": dict(pair.parameters),
        "graph": {
            "node_names": list(graph.node_names),
            "node_types": graph.node_types.tolist(),
            "rates": graph.rates.tolist(),
            "input_capacities": graph.input_capacities.tolist(),
            "output_capacities": graph.output_capacities.tolist(),
            "initial_inputs": graph.initial_inputs.tolist(),
            "initial_outputs": graph.initial_outputs.tolist(),
            "edge_names": list(graph.edge_names),
            "edge_sources": graph.edge_sources.tolist(),
            "edge_targets": graph.edge_targets.tolist(),
            "edge_capacities": graph.edge_capacities.tolist(),
        },
        "worlds": [
            {
                "label": world.label,
                "fault": {
                    "kind": world.fault.kind.value,
                    "component": world.fault.component,
                },
                "correct_repair": _repair_record(world.correct_repair),
            }
            for world in pair.worlds
        ],
        "intervention_edge": pair.intervention_edge,
        "evidence_node": pair.evidence_node,
        "reward": asdict(pair.reward),
    }


def build_pair_dataset(
    batch: GenerationBatch,
    split: SplitDefinition,
    *,
    offset: int,
) -> dict[str, Any]:
    if offset < 0 or offset + len(batch.pairs) > split.count:
        raise ValueError("dataset range exceeds split definition")
    return {
        "schema_version": PAIR_DATASET_SCHEMA_VERSION,
        "generator_version": split.generator_version,
        "split_version": split.split_version,
        "split": split.name,
        "offset": offset,
        "seed_start": split.seed_start + offset,
        "count": len(batch.pairs),
        "attempts": batch.attempts,
        "rejection_counts": dict(batch.rejection_counts),
        "tasks": [
            {
                **pair_record(pair),
                "validation": asdict(validation),
            }
            for pair, validation in zip(batch.pairs, batch.validations, strict=True)
        ],
    }


def summarize_generation(batch: GenerationBatch) -> dict[str, Any]:
    pressures = [validation.epistemic_pressure for validation in batch.validations]
    margins = [validation.repair_margin for validation in batch.validations]
    return {
        "requested_count": len(batch.pairs),
        "validated_count": sum(validation.valid for validation in batch.validations),
        "attempts": batch.attempts,
        "acceptance_rate": batch.acceptance_rate,
        "rejection_counts": dict(batch.rejection_counts),
        "ep_min": min(pressures),
        "ep_mean": fmean(pressures),
        "ep_max": max(pressures),
        "repair_margin_min": min(margins),
        "passive_recovery_mean": fmean(
            validation.passive_recovery_probability for validation in batch.validations
        ),
        "active_recovery_mean": fmean(
            validation.active_recovery_probability for validation in batch.validations
        ),
    }
