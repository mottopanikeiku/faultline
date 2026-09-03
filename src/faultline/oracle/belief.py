"""Exact finite-hypothesis belief utilities."""

from __future__ import annotations

from math import fsum, log2


def normalize(weights: tuple[float, ...]) -> tuple[float, ...]:
    total = fsum(weights)
    if total <= 0.0:
        raise ValueError("evidence has zero probability under the prior")
    if any(weight < 0.0 for weight in weights):
        raise ValueError("belief weights must be non-negative")
    return tuple(weight / total for weight in weights)


def bayes_update(
    prior: tuple[float, ...],
    likelihood: tuple[float, ...],
) -> tuple[float, ...]:
    """Apply Bayes' rule to a finite latent-world distribution."""
    if len(prior) != len(likelihood) or not prior:
        raise ValueError("prior and likelihood must be non-empty and aligned")
    if any(probability < 0.0 for probability in prior):
        raise ValueError("prior probabilities must be non-negative")
    if any(value < 0.0 for value in likelihood):
        raise ValueError("likelihood values must be non-negative")
    return normalize(
        tuple(
            probability * evidence_likelihood
            for probability, evidence_likelihood in zip(prior, likelihood, strict=True)
        )
    )


def entropy_bits(probabilities: tuple[float, ...]) -> float:
    """Shannon entropy in bits, with exact zero-mass handling."""
    normalized = normalize(probabilities)
    return -fsum(probability * log2(probability) for probability in normalized if probability > 0.0)


def best_shared_decision_mass(
    prior: tuple[float, ...],
    successful_worlds_by_decision: tuple[frozenset[int], ...],
) -> float:
    """Posterior-mass ceiling for a policy forced to choose one shared decision."""
    normalized = normalize(prior)
    world_count = len(normalized)
    if not successful_worlds_by_decision:
        raise ValueError("at least one terminal decision is required")
    values: list[float] = []
    for successful_worlds in successful_worlds_by_decision:
        if any(index < 0 or index >= world_count for index in successful_worlds):
            raise ValueError("decision success set contains an invalid world index")
        values.append(fsum(normalized[index] for index in successful_worlds))
    return max(values)
