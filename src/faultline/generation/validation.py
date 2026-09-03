"""Semantic validation and bounded acceptance for generated diagnostic pairs."""

from __future__ import annotations

from dataclasses import dataclass

from faultline.generation.automated import build_generated_diagnostic_pair
from faultline.generation.diagnostic_pairs import (
    DiagnosticPair,
    advance_inspect_evidence,
    create_world_env,
    evaluate_repair,
    full_passive_snapshot,
)
from faultline.oracle import diagnostic_pair_problem, solve_active, solve_passive


@dataclass(frozen=True, slots=True)
class PairValidation:
    pair_id: str
    valid: bool
    rejection_reasons: tuple[str, ...]
    passive_snapshots_identical: bool
    repairs_differ: bool
    unique_repair_success: bool
    diagnostic_outcome_count: int
    repair_margin: float
    passive_expected_return: float
    active_expected_return: float
    epistemic_pressure: float
    passive_recovery_probability: float
    active_recovery_probability: float


@dataclass(frozen=True, slots=True)
class GenerationBatch:
    pairs: tuple[DiagnosticPair, ...]
    validations: tuple[PairValidation, ...]
    attempts: int
    rejection_counts: tuple[tuple[str, int], ...]

    @property
    def acceptance_rate(self) -> float:
        return len(self.pairs) / self.attempts


def validate_diagnostic_pair(
    pair: DiagnosticPair,
    *,
    minimum_ep: float = 1e-9,
) -> PairValidation:
    """Verify ambiguity, repair disagreement, intervention separation, and positive EP."""
    snapshots = tuple(
        full_passive_snapshot(create_world_env(pair, world)) for world in pair.worlds
    )
    snapshots_identical = snapshots[0] == snapshots[1]
    repairs = tuple(dict.fromkeys(world.correct_repair for world in pair.worlds))
    repairs_differ = len(repairs) == len(pair.worlds)

    repair_margin = float("inf")
    unique_repair_success = repairs_differ
    for world in pair.worlds:
        evaluations = tuple(evaluate_repair(pair, world, repair) for repair in repairs)
        successful = tuple(evaluation for evaluation in evaluations if evaluation.recovered)
        if len(successful) != 1 or successful[0].repair != world.correct_repair:
            unique_repair_success = False
            repair_margin = float("-inf")
            continue
        wrong_values = tuple(
            evaluation.total_return
            for evaluation in evaluations
            if evaluation.repair != world.correct_repair
        )
        if wrong_values:
            repair_margin = min(
                repair_margin,
                successful[0].total_return - max(wrong_values),
            )

    evidence = tuple(advance_inspect_evidence(pair, world) for world in pair.worlds)
    diagnostic_outcome_count = len(set(evidence))
    problem = diagnostic_pair_problem(pair)
    passive = solve_passive(problem)
    active = solve_active(problem, max_diagnostic_actions=2)
    epistemic_pressure = active.expected_return - passive.expected_return

    reasons: list[str] = []
    if not snapshots_identical:
        reasons.append("passive_observation_mismatch")
    if not repairs_differ:
        reasons.append("repairs_do_not_differ")
    if not unique_repair_success:
        reasons.append("repair_not_unique")
    if diagnostic_outcome_count < 2:
        reasons.append("no_discriminating_intervention")
    if epistemic_pressure < minimum_ep:
        reasons.append("nonpositive_ep")
    if active.recovery_probability <= passive.recovery_probability:
        reasons.append("no_active_success_advantage")

    return PairValidation(
        pair_id=pair.pair_id,
        valid=not reasons,
        rejection_reasons=tuple(reasons),
        passive_snapshots_identical=snapshots_identical,
        repairs_differ=repairs_differ,
        unique_repair_success=unique_repair_success,
        diagnostic_outcome_count=diagnostic_outcome_count,
        repair_margin=repair_margin,
        passive_expected_return=passive.expected_return,
        active_expected_return=active.expected_return,
        epistemic_pressure=epistemic_pressure,
        passive_recovery_probability=passive.recovery_probability,
        active_recovery_probability=active.recovery_probability,
    )


def generate_validated_pairs(
    count: int,
    *,
    seed_start: int = 0,
    max_attempts: int | None = None,
    minimum_ep: float = 1e-9,
) -> GenerationBatch:
    """Generate exactly ``count`` valid pairs or fail without silently shrinking scope."""
    if count <= 0:
        raise ValueError("pair count must be positive")
    if seed_start < 0:
        raise ValueError("seed start must be non-negative")
    attempt_limit = max_attempts if max_attempts is not None else count * 10
    if attempt_limit < count:
        raise ValueError("max attempts cannot be smaller than requested count")

    accepted_pairs: list[DiagnosticPair] = []
    accepted_validations: list[PairValidation] = []
    rejection_counts: dict[str, int] = {}
    attempts = 0
    while len(accepted_pairs) < count and attempts < attempt_limit:
        pair = build_generated_diagnostic_pair(seed_start + attempts)
        validation = validate_diagnostic_pair(pair, minimum_ep=minimum_ep)
        attempts += 1
        if validation.valid:
            accepted_pairs.append(pair)
            accepted_validations.append(validation)
        else:
            for reason in validation.rejection_reasons:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    if len(accepted_pairs) != count:
        raise RuntimeError(
            f"generated {len(accepted_pairs)} valid pairs in {attempts} attempts; "
            f"rejections={dict(sorted(rejection_counts.items()))}"
        )
    return GenerationBatch(
        pairs=tuple(accepted_pairs),
        validations=tuple(accepted_validations),
        attempts=attempts,
        rejection_counts=tuple(sorted(rejection_counts.items())),
    )
