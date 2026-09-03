"""Procedural factory and epistemic-task generation."""

from faultline.generation.automated import (
    GENERATOR_VERSION,
    build_generated_diagnostic_pair,
)
from faultline.generation.dataset import (
    PAIR_DATASET_SCHEMA_VERSION,
    build_pair_dataset,
    pair_record,
    summarize_generation,
)
from faultline.generation.diagnostic_pairs import (
    ActiveEvaluation,
    DiagnosticPair,
    DiagnosticWorld,
    PassiveSnapshot,
    RepairEvaluation,
    advance_inspect_evidence,
    build_manual_diagnostic_pair,
    create_world_env,
    diagnostic_evidence,
    evaluate_repair,
    full_passive_snapshot,
    run_contingent_active_policy,
)
from faultline.generation.layouts import chain_factory
from faultline.generation.splits import SPLIT_VERSION, SPLITS, SplitDefinition, get_split
from faultline.generation.validation import (
    GenerationBatch,
    PairValidation,
    generate_validated_pairs,
    validate_diagnostic_pair,
)

__all__ = [
    "GENERATOR_VERSION",
    "PAIR_DATASET_SCHEMA_VERSION",
    "SPLITS",
    "SPLIT_VERSION",
    "ActiveEvaluation",
    "DiagnosticPair",
    "DiagnosticWorld",
    "GenerationBatch",
    "PairValidation",
    "PassiveSnapshot",
    "RepairEvaluation",
    "SplitDefinition",
    "advance_inspect_evidence",
    "build_generated_diagnostic_pair",
    "build_manual_diagnostic_pair",
    "build_pair_dataset",
    "chain_factory",
    "create_world_env",
    "diagnostic_evidence",
    "evaluate_repair",
    "full_passive_snapshot",
    "generate_validated_pairs",
    "get_split",
    "pair_record",
    "run_contingent_active_policy",
    "summarize_generation",
    "validate_diagnostic_pair",
]
