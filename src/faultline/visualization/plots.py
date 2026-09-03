"""Small dependency-free SVG plots for immutable research artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape

import numpy as np


def _svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        "<style>text{font-family:ui-monospace,monospace;fill:#172033}"
        ".axis{stroke:#526078;stroke-width:1}.grid{stroke:#dce2eb;stroke-width:1}"
        ".bar{fill:#3767d6}.point{fill:#d04b40}.link{stroke:#8793a8;stroke-opacity:.35}</style>",
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-size="20">'
        f"{escape(title)}</text>",
    ]


def render_histogram_svg(
    values: Sequence[float],
    *,
    title: str,
    x_label: str,
    bins: int = 10,
    width: int = 800,
    height: int = 480,
) -> str:
    """Render a labelled histogram using only observed numeric values."""
    if not values or bins <= 0:
        raise ValueError("histogram requires values and positive bin count")
    counts, edges = np.histogram(np.asarray(values, dtype=np.float64), bins=bins)
    left, right, top, bottom = 75.0, width - 30.0, 55.0, height - 65.0
    plot_width = right - left
    plot_height = bottom - top
    max_count = max(int(counts.max()), 1)
    lines = _svg_header(width, height, title)
    lines.extend(
        [
            f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
        ]
    )
    bar_width = plot_width / bins
    for index, count in enumerate(counts):
        bar_height = plot_height * int(count) / max_count
        x = left + index * bar_width + 1.0
        y = bottom - bar_height
        lines.append(
            f'<rect class="bar" x="{x:.2f}" y="{y:.2f}" '
            f'width="{max(bar_width - 2.0, 1.0):.2f}" height="{bar_height:.2f}"/>'
        )
        lines.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{y - 5:.2f}" '
            f'text-anchor="middle" font-size="11">{int(count)}</text>'
        )
    lines.extend(
        [
            f'<text x="{left}" y="{bottom + 22}" text-anchor="middle" font-size="12">'
            f"{edges[0]:.2f}</text>",
            f'<text x="{right}" y="{bottom + 22}" text-anchor="middle" font-size="12">'
            f"{edges[-1]:.2f}</text>",
            f'<text x="{(left + right) / 2:.1f}" y="{height - 18}" '
            f'text-anchor="middle" font-size="14">{escape(x_label)}</text>',
            f'<text x="18" y="{(top + bottom) / 2:.1f}" text-anchor="middle" '
            f'font-size="14" transform="rotate(-90 18 {(top + bottom) / 2:.1f})">count</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def render_paired_values_svg(
    before: Sequence[float],
    after: Sequence[float],
    *,
    title: str,
    before_label: str,
    after_label: str,
    y_label: str,
    width: int = 800,
    height: int = 480,
) -> str:
    """Render paired action values, preserving every task-level point."""
    if not before or len(before) != len(after):
        raise ValueError("paired plot requires non-empty aligned values")
    all_values = np.asarray([*before, *after, 0.0], dtype=np.float64)
    value_min = float(all_values.min())
    value_max = float(all_values.max())
    if value_min == value_max:
        value_min -= 1.0
        value_max += 1.0
    padding = 0.08 * (value_max - value_min)
    value_min -= padding
    value_max += padding
    left, right, top, bottom = 95.0, width - 55.0, 55.0, height - 65.0
    x_before = left + 0.25 * (right - left)
    x_after = left + 0.75 * (right - left)

    def y_coordinate(value: float) -> float:
        return bottom - (value - value_min) / (value_max - value_min) * (bottom - top)

    lines = _svg_header(width, height, title)
    zero_y = y_coordinate(0.0)
    lines.append(
        f'<line class="grid" x1="{left}" y1="{zero_y:.2f}" x2="{right}" y2="{zero_y:.2f}"/>'
    )
    lines.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>')
    for first, second in zip(before, after, strict=True):
        first_y = y_coordinate(first)
        second_y = y_coordinate(second)
        lines.append(
            f'<line class="link" x1="{x_before:.2f}" y1="{first_y:.2f}" '
            f'x2="{x_after:.2f}" y2="{second_y:.2f}"/>'
        )
        lines.append(f'<circle class="point" cx="{x_before:.2f}" cy="{first_y:.2f}" r="2.4"/>')
        lines.append(f'<circle class="point" cx="{x_after:.2f}" cy="{second_y:.2f}" r="2.4"/>')
    lines.extend(
        [
            f'<text x="{x_before:.2f}" y="{bottom + 25}" text-anchor="middle" font-size="13">'
            f"{escape(before_label)}</text>",
            f'<text x="{x_after:.2f}" y="{bottom + 25}" text-anchor="middle" font-size="13">'
            f"{escape(after_label)}</text>",
            f'<text x="22" y="{(top + bottom) / 2:.1f}" text-anchor="middle" font-size="14" '
            f'transform="rotate(-90 22 {(top + bottom) / 2:.1f})">{escape(y_label)}</text>',
            f'<text x="{left - 8}" y="{top + 4}" text-anchor="end" font-size="11">'
            f"{value_max:.2f}</text>",
            f'<text x="{left - 8}" y="{bottom}" text-anchor="end" font-size="11">'
            f"{value_min:.2f}</text>",
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def render_group_values_svg(
    groups: Mapping[str, Sequence[float]],
    *,
    title: str,
    y_label: str,
    width: int = 800,
    height: int = 480,
) -> str:
    """Render every replicate and its mean for two or more named groups."""
    if len(groups) < 2 or any(not values for values in groups.values()):
        raise ValueError("group plot requires at least two non-empty groups")
    flattened = [float(value) for values in groups.values() for value in values]
    value_min = min(0.0, min(flattened))
    value_max = max(1.0, max(flattened))
    padding = 0.06 * max(value_max - value_min, 1e-9)
    value_min -= padding
    value_max += padding
    left, right, top, bottom = 90.0, width - 40.0, 55.0, height - 65.0

    def y_coordinate(value: float) -> float:
        return bottom - (value - value_min) / (value_max - value_min) * (bottom - top)

    lines = _svg_header(width, height, title)
    lines.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>')
    labels = tuple(groups)
    spacing = (right - left) / len(labels)
    for group_index, label in enumerate(labels):
        x = left + spacing * (group_index + 0.5)
        values = groups[label]
        for value_index, value in enumerate(values):
            jitter = ((value_index % 5) - 2) * 3.0
            lines.append(
                f'<circle class="point" cx="{x + jitter:.2f}" '
                f'cy="{y_coordinate(float(value)):.2f}" r="3.2"/>'
            )
        mean = sum(float(value) for value in values) / len(values)
        mean_y = y_coordinate(mean)
        lines.append(
            f'<line x1="{x - 20:.2f}" y1="{mean_y:.2f}" x2="{x + 20:.2f}" '
            f'y2="{mean_y:.2f}" stroke="#172033" stroke-width="3"/>'
        )
        lines.append(
            f'<text x="{x:.2f}" y="{bottom + 25}" text-anchor="middle" font-size="13">'
            f"{escape(label)}</text>"
        )
    lines.extend(
        [
            f'<text x="20" y="{(top + bottom) / 2:.1f}" text-anchor="middle" font-size="14" '
            f'transform="rotate(-90 20 {(top + bottom) / 2:.1f})">{escape(y_label)}</text>',
            f'<text x="{left - 8}" y="{top + 4}" text-anchor="end" font-size="11">'
            f"{value_max:.2f}</text>",
            f'<text x="{left - 8}" y="{bottom}" text-anchor="end" font-size="11">'
            f"{value_min:.2f}</text>",
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"
