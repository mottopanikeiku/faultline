from __future__ import annotations

import pytest

from faultline.visualization import render_histogram_svg, render_paired_values_svg


def test_histogram_svg_contains_observed_counts_and_labels() -> None:
    svg = render_histogram_svg(
        [1.0, 1.5, 2.0, 2.5],
        title="Observed EP",
        x_label="EP value",
        bins=2,
    )

    assert svg.startswith("<svg")
    assert "Observed EP" in svg
    assert "EP value" in svg
    assert svg.count('<rect class="bar"') == 2
    assert ">2</text>" in svg
    assert svg.endswith("</svg>\n")


def test_paired_plot_preserves_each_task_point() -> None:
    svg = render_paired_values_svg(
        [-0.1, -0.2, -0.3],
        [4.0, 5.0, 6.0],
        title="Intervention values",
        before_label="before",
        after_label="after",
        y_label="decision value",
    )

    assert svg.count('<line class="link"') == 3
    assert svg.count('<circle class="point"') == 6
    assert "decision value" in svg


def test_plot_inputs_are_validated() -> None:
    with pytest.raises(ValueError):
        render_histogram_svg([], title="empty", x_label="x")
    with pytest.raises(ValueError):
        render_paired_values_svg(
            [1.0],
            [],
            title="bad",
            before_label="a",
            after_label="b",
            y_label="y",
        )
