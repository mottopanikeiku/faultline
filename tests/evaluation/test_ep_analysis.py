from __future__ import annotations

from faultline.evaluation import EP_ANALYSIS_VERSION, analyze_ep_distribution


def test_ep_analysis_separates_immediate_information_from_decision_value() -> None:
    analysis = analyze_ep_distribution([0, 1, 2, 3, 4])

    assert analysis["analysis_version"] == EP_ANALYSIS_VERSION
    assert analysis["count"] == 5
    assert analysis["epistemic_pressure"]["min"] > 0.0
    assert 0.0 < analysis["normalized_ep"]["min"] <= analysis["normalized_ep"]["max"] <= 1.0
    assert analysis["immediate_advance_information_gain_bits"] == {
        "min": 0.0,
        "q25": 0.0,
        "median": 0.0,
        "mean": 0.0,
        "q75": 0.0,
        "max": 0.0,
        "std_population": 0.0,
    }
    assert analysis["post_advance_inspect_information_gain_bits"]["mean"] == 1.0
    assert analysis["post_advance_inspect_decision_value"]["min"] > 0.0
    assert len(analysis["rows"]) == 5
    assert set(analysis["correlations_with_ep"]) >= {
        "node_count",
        "passive_difficulty",
        "diagnostic_cost",
    }
    assert set(analysis["correlations_with_normalized_ep"]) == set(
        analysis["correlations_with_ep"]
    )
    for row in analysis["rows"]:
        assert row["normalized_ep"] == (
            row["epistemic_pressure"] / row["passive_decision_regret"]
        )
