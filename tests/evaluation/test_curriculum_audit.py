from __future__ import annotations

import pytest

from faultline.evaluation import CONTROL_AUDIT_VERSION, analyze_matched_ep_controls


def test_control_audit_changes_joint_dependence_with_exactly_matched_marginals() -> None:
    audit = analyze_matched_ep_controls([0, 1, 2, 3])

    assert audit["analysis_version"] == CONTROL_AUDIT_VERSION
    assert audit["count"] == 4
    assert audit["all_fault_marginals_equal"]
    assert audit["all_cue_marginals_equal"]
    assert audit["all_cue_fault_joints_differ"]
    assert audit["max_abs_nuisance_standardized_mean_difference"] == 0.0
    assert set(audit["nuisance_standardized_mean_differences"].values()) == {0.0}
    assert audit["ambiguous_ep"]["min"] > 0.0
    assert audit["revealed_ep"]["min"] == pytest.approx(0.0, abs=1e-12)
    assert audit["revealed_ep"]["max"] == pytest.approx(0.0, abs=1e-12)
    assert audit["ambiguous_passive_recovery_mean"] == 0.5
    assert audit["ambiguous_active_recovery_mean"] == 1.0
    assert audit["revealed_passive_recovery_mean"] == 1.0
    assert audit["revealed_active_recovery_mean"] == 1.0
