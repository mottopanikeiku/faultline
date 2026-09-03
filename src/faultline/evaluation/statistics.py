"""Training-seed-level paired bootstrap summaries."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

import numpy as np


@dataclass(frozen=True, slots=True)
class BootstrapEstimate:
    estimate: float
    confidence_level: float
    lower: float
    upper: float
    resamples: int
    unit_count: int


def bootstrap_mean(
    values: list[float],
    *,
    confidence_level: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> BootstrapEstimate:
    """Bootstrap a mean over independent training-seed units."""
    if not values:
        raise ValueError("bootstrap requires at least one value")
    if not 0.0 < confidence_level < 1.0 or resamples <= 0:
        raise ValueError("confidence level and resample count are invalid")
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(resamples, array.size))
    bootstrap_means = array[indices].mean(axis=1)
    alpha = 1.0 - confidence_level
    return BootstrapEstimate(
        estimate=float(fmean(values)),
        confidence_level=confidence_level,
        lower=float(np.quantile(bootstrap_means, alpha / 2.0)),
        upper=float(np.quantile(bootstrap_means, 1.0 - alpha / 2.0)),
        resamples=resamples,
        unit_count=len(values),
    )


def paired_bootstrap_difference(
    left: list[float],
    right: list[float],
    *,
    confidence_level: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> BootstrapEstimate:
    """Bootstrap paired training-seed differences ``left - right``."""
    if len(left) != len(right):
        raise ValueError("paired bootstrap inputs must have equal length")
    return bootstrap_mean(
        [left_value - right_value for left_value, right_value in zip(left, right, strict=True)],
        confidence_level=confidence_level,
        resamples=resamples,
        seed=seed,
    )
