"""Procedural factory and epistemic-task generation."""

from faultline.generation.diagnostic_pairs import (
    ActiveEvaluation,
    DiagnosticPair,
    DiagnosticWorld,
    PassiveSnapshot,
    RepairEvaluation,
    build_manual_diagnostic_pair,
    create_world_env,
    diagnostic_evidence,
    evaluate_repair,
    full_passive_snapshot,
    run_contingent_active_policy,
)
from faultline.generation.layouts import chain_factory

__all__ = [
    "ActiveEvaluation",
    "DiagnosticPair",
    "DiagnosticWorld",
    "PassiveSnapshot",
    "RepairEvaluation",
    "build_manual_diagnostic_pair",
    "chain_factory",
    "create_world_env",
    "diagnostic_evidence",
    "evaluate_repair",
    "full_passive_snapshot",
    "run_contingent_active_policy",
]
