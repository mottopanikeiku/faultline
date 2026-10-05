#!/usr/bin/env python3
"""Plot paired training seeds from committed evaluation rows; no learning dependencies."""

from __future__ import annotations

import argparse
import json
import tomllib
from html import escape
from pathlib import Path
from statistics import fmean

COLORS = ("#0072b2", "#d55e00", "#009e73", "#cc79a7",
          "#8c510a", "#6a51a3", "#59636e", "#b59b00")


def diagnostic_success(rows: list[dict]) -> float:
    """Match evaluation.policy._aggregate, restricted to ambiguous episodes."""
    return fmean(
        row["advance_count"] > 0
        and row["informative_inspection"]
        and row["correct_repair"]
        for row in rows
        if row["condition"] == "ambiguous"
    )


def load_seed_values(repo: Path) -> dict[str, dict[int, float]]:
    with (repo / "configs/evaluation/small-kill-v1.toml").open("rb") as source:
        protocol = tomllib.load(source)
    groups = {}
    for arm in protocol["arms"]:
        values = {}
        for seed in protocol["training_seeds"]:
            run_id = protocol["run_id_template"].format(arm=arm, seed=seed)
            path = repo / "artifacts/results" / f"{run_id}.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            values[seed] = diagnostic_success(result["evaluation"]["rows"])
        groups[arm] = values
    return groups


def render_seed_svg(groups: dict[str, dict[int, float]]) -> str:
    """Show every seed, including coincident points, with matched-seed lines."""
    width, height = 860, 520
    left, right, top, bottom = 95, 800, 70, 395
    arms = list(groups)
    seeds = list(groups[arms[0]])
    spacing = (right - left) / len(arms)

    def x(arm_index: int, seed_index: int) -> float:
        return left + spacing * (arm_index + 0.5) + (seed_index - (len(seeds) - 1) / 2) * 7

    def y(value: float) -> float:
        return bottom - value * (bottom - top)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
        '<title id="title">Diagnostic success by curriculum and training seed</title>',
        '<desc id="description">Each point is recomputed from ambiguous evaluation episode '
        'rows in artifacts/results/small-kill-v1-*-seed-*.json. Colored lines connect the '
        'same training seed across curricula, not a trajectory over time. Black bars are '
        'means. Diagnostic success requires advance, informative inspection and correct repair. '
        'These are validation results, not new training runs.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="sans-serif" fill="#172033">',
        '<text x="430" y="30" text-anchor="middle" font-size="20">'
        'Diagnostic success by training seed</text>',
        '<text x="430" y="52" text-anchor="middle" font-size="13">'
        'Validation ambiguous tasks · matched seeds · black bars: means</text>',
    ]
    for tick in range(6):
        value = tick / 5
        lines.extend([
            f'<line x1="{left}" y1="{y(value):.2f}" x2="{right}" '
            f'y2="{y(value):.2f}" stroke="#e0e4e8"/>',
            f'<text x="{left - 12}" y="{y(value) + 4:.2f}" '
            f'text-anchor="end" font-size="12">{value:.1f}</text>',
        ])
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" '
                 'stroke="#172033"/>')
    for seed_index, seed in enumerate(seeds):
        color = COLORS[seed_index % len(COLORS)]
        points = " ".join(f"{x(i, seed_index):.2f},{y(groups[arm][seed]):.2f}"
                          for i, arm in enumerate(arms))
        lines.append(f'<polyline points="{points}" fill="none" '
                     f'stroke="{color}" stroke-width="1.2" stroke-opacity="0.45"/>')
        for arm_index, arm in enumerate(arms):
            value = groups[arm][seed]
            lines.append(
                f'<circle cx="{x(arm_index, seed_index):.2f}" cy="{y(value):.2f}" '
                f'r="4" fill="{color}"><title>{escape(arm)} seed {seed}: '
                f'{value:.8f}</title></circle>'
            )
    for arm_index, arm in enumerate(arms):
        center = left + spacing * (arm_index + 0.5)
        mean = fmean(groups[arm].values())
        lines.extend([
            f'<line x1="{center - 35:.2f}" y1="{y(mean):.2f}" '
            f'x2="{center + 35:.2f}" y2="{y(mean):.2f}" '
            'stroke="#172033" stroke-width="3"/>',
            f'<text x="{center:.2f}" y="{bottom + 24}" text-anchor="middle" '
            f'font-size="14">{escape(arm.title())}</text>',
            f'<text x="{center:.2f}" y="{bottom + 43}" text-anchor="middle" '
            f'font-size="12">mean {mean:.3f}</text>',
        ])
    for index, seed in enumerate(seeds):
        legend_x = 135 + (index % 4) * 180
        legend_y = 468 + (index // 4) * 25
        lines.extend([
            f'<circle cx="{legend_x}" cy="{legend_y}" r="4" '
            f'fill="{COLORS[index % len(COLORS)]}"/>',
            f'<text x="{legend_x + 12}" y="{legend_y + 4}" font-size="12">seed {seed}</text>',
        ])
    lines.extend([
        '<text x="25" y="233" text-anchor="middle" font-size="14" '
        'transform="rotate(-90 25 233)">Diagnostic success fraction</text>',
        '</g></svg>',
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    groups = load_seed_values(args.repo)
    output = args.output or args.repo / "artifacts/results/small-kill-v1-seeds.svg"
    output.write_text(render_seed_svg(groups), encoding="utf-8")
    for arm, values in groups.items():
        print(f"{arm}: {len(values)} seeds, mean diagnostic success {fmean(values.values()):.8f}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
