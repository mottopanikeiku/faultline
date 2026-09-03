from __future__ import annotations

import pytest

from faultline.evaluation.statistics import bootstrap_mean, paired_bootstrap_difference


def test_constant_bootstrap_has_exact_interval() -> None:
    estimate = bootstrap_mean([0.5] * 8, resamples=1000, seed=7)

    assert estimate.estimate == 0.5
    assert estimate.lower == 0.5
    assert estimate.upper == 0.5
    assert estimate.unit_count == 8


def test_paired_bootstrap_uses_within_seed_differences() -> None:
    estimate = paired_bootstrap_difference(
        [0.9, 0.8, 0.7, 0.6],
        [0.7, 0.6, 0.5, 0.4],
        resamples=1000,
        seed=3,
    )

    assert estimate.estimate == pytest.approx(0.2)
    assert estimate.lower == pytest.approx(0.2)
    assert estimate.upper == pytest.approx(0.2)


def test_paired_bootstrap_rejects_unaligned_units() -> None:
    with pytest.raises(ValueError, match="equal length"):
        paired_bootstrap_difference([1.0], [1.0, 2.0])
