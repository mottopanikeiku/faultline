"""Recompute the fixed matched-seed comparison from its compressed raw archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tarfile
import tempfile
import tomllib
from html import escape
from pathlib import Path
from statistics import fmean

from faultline.artifacts.manifest import canonical_sha256
from faultline.evaluation.study import analyze_kill_test, load_kill_test_protocol


def classify(comparisons: dict, margin: float) -> str:
    if all(item["estimate"] >= margin and item["lower"] > 0 for item in comparisons.values()):
        return "supports_advantage_over_both"
    if any(item["upper"] < margin for item in comparisons.values()):
        return "rules_out_target_advantage_over_both"
    return "inconclusive_about_target_advantage"


def verify_run(result: dict, manifest: dict) -> None:
    if canonical_sha256(result) != manifest["metrics"]["result_sha256"]:
        raise ValueError("result digest mismatch")
    if result["checkpoint"]["sha256"] != manifest["metrics"]["checkpoint_sha256"]:
        raise ValueError("checkpoint digest references disagree")
    if canonical_sha256(result["resolved_config"]) != manifest["config_sha256"]:
        raise ValueError("configuration digest mismatch")
    if result["resolved_config"] != manifest["config"]:
        raise ValueError("result and manifest configurations disagree")
    if result["decision_steps"] < result["resolved_config"]["ppo"]["total_decision_steps"]:
        raise ValueError("training stopped before its target budget")
    rows = [row for row in result["evaluation"]["rows"] if row["condition"] == "ambiguous"]
    score = fmean(row["advance_count"] > 0 and row["informative_inspection"]
                  and row["correct_repair"] for row in rows)
    if score != result["evaluation"]["ambiguous"]["experiment_then_correct_repair_rate"]:
        raise ValueError("primary score disagrees with episode traces")


def analyze(repo: Path) -> dict:
    protocol = load_kill_test_protocol(repo / "configs/evaluation/seed-comparison.toml")
    with (repo / protocol.training_config).open("rb") as source:
        training_config = tomllib.load(source)
    archive_path = repo / "artifacts/results/seed-comparison-runs.tar.gz"
    with tempfile.TemporaryDirectory() as directory, tarfile.open(archive_path) as archive:
        extracted = Path(directory)
        for arm in protocol.arms:
            for seed in protocol.training_seeds:
                run_id = protocol.run_id_template.format(arm=arm, seed=seed)
                loaded = []
                for kind in ("results", "manifests"):
                    relative = f"artifacts/{kind}/{run_id}.json"
                    member = archive.extractfile(relative)
                    if member is None:
                        raise ValueError(f"not a regular archive member: {relative}")
                    content = member.read()
                    destination = extracted / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(content)
                    loaded.append(json.loads(content))
                verify_run(*loaded)
                expected_config = {
                    **training_config,
                    "curriculum": {
                        **training_config["curriculum"], "kind": arm, "training_seed": seed,
                    },
                }
                if loaded[0]["resolved_config"] != expected_config:
                    raise ValueError(f"training configuration differs from plan: {run_id}")
                if len(loaded[0]["evaluation"]["rows"]) != 6 * protocol.evaluation_base_pair_count:
                    raise ValueError(f"evaluation episode count differs from plan: {run_id}")
        analysis = analyze_kill_test(extracted, protocol)
    analysis["target_decision"] = classify(analysis["paired_comparisons"],
                                          protocol.minimum_relevant_effect)
    analysis["raw_archive"] = {"path": str(archive_path.relative_to(repo)),
                               "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest()}
    return analysis


def render_svg(analysis: dict) -> str:
    arms = analysis["arms"]
    comparisons = analysis["paired_comparisons"]
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 550" role="img" '
        'aria-labelledby="title desc">',
        '<title id="title">Matched training-seed curriculum comparison</title>',
        '<desc id="desc">Every point is one training seed. Black bars show means. '
        'The lower panel shows paired differences and 95% training-seed bootstrap intervals; '
        'the dashed line marks the five-percentage-point target.</desc>',
        '<rect width="900" height="550" fill="white"/>',
        '<g font-family="sans-serif" fill="#172033">',
        '<text x="450" y="30" text-anchor="middle" font-size="20">'
        'Does ambiguous-only training improve diagnosis?</text>',
        f'<text x="450" y="54" text-anchor="middle" font-size="13">'
        f'{len(next(iter(arms.values()))["individual_seeds"])} matched training seeds '
        'per curriculum · validation tasks</text>',
    ]
    for tick in range(6):
        y = 270 - tick * 36
        elements.append(f'<path d="M80 {y} H850" stroke="#e4e8ec"/>')
        elements.append(f'<text x="65" y="{y + 4}" text-anchor="end">{tick * 20}%</text>')
    for index, (arm, summary) in enumerate(arms.items()):
        center = 210 + index * 250
        for seed_index, value in enumerate(summary["individual_seeds"]):
            x = center + ((seed_index % 16) - 7.5) * 6
            y = 270 - value["value"] * 180
            elements.append(f'<circle cx="{x}" cy="{y:.3f}" r="2.8" '
                            f'fill="#0072b2" fill-opacity="0.5"><title>'
                            f'{escape(arm)} seed {value["seed"]}: '
                            f'{value["value"]:.6f}</title></circle>')
        mean = summary["primary"]["estimate"]
        y = 270 - mean * 180
        elements.extend([
            f'<path d="M{center - 55} {y:.3f} H{center + 55}" stroke="#172033" '
            'stroke-width="3"/>',
            f'<text x="{center}" y="295" text-anchor="middle">{escape(arm.title())}</text>',
            f'<text x="{center}" y="316" text-anchor="middle">mean {mean:.1%}</text>',
        ])
    elements.append('<text x="80" y="358" font-size="15">'
                    'Ambiguous-only minus baseline: paired difference (percentage points)</text>')
    span = max(0.1, *(abs(item[bound]) for item in comparisons.values()
                      for bound in ("lower", "upper")))
    limit = math.ceil(span * 10) * 10

    def x(value: float) -> float:
        return 250 + (value * 100 + limit) / (2 * limit) * 580

    for tick in range(-limit, limit + 1, max(10, limit // 3)):
        elements.append(f'<text x="{x(tick / 100):.3f}" y="506" text-anchor="middle">'
                        f'{tick:+d}</text>')
    elements.append(f'<path d="M{x(0):.3f} 370 V483" stroke="#9ba5b1"/>')
    elements.append(f'<path d="M{x(0.05):.3f} 370 V483" stroke="#d55e00" '
                    'stroke-dasharray="5 4"/>')
    for index, (name, interval) in enumerate(comparisons.items()):
        y = 400 + index * 60
        elements.extend([
            f'<text x="80" y="{y + 4}">{escape(name.removeprefix("epistemic-"))}</text>',
            f'<path d="M{x(interval["lower"]):.3f} {y} H{x(interval["upper"]):.3f}" '
            'stroke="#0072b2" stroke-width="3"/>',
            f'<circle cx="{x(interval["estimate"]):.3f}" cy="{y}" r="5" fill="#172033"/>',
            f'<text x="830" y="{y - 14}" text-anchor="end" font-size="12">'
            f'{interval["estimate"] * 100:+.1f} '
            f'[{interval["lower"] * 100:+.1f}, {interval["upper"] * 100:+.1f}]</text>',
        ])
    elements.extend(['<text x="450" y="537" text-anchor="middle" font-size="12">'
                     'Dots may overlap; individual values are in the analysis JSON.</text>',
                     '</g></svg>'])
    return "\n".join(elements) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    analysis = analyze(args.repo)
    directory = args.repo / "artifacts/results"
    (directory / "seed-comparison-analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (directory / "seed-comparison.svg").write_text(render_svg(analysis), encoding="utf-8")
    print(json.dumps({"decision": analysis["target_decision"],
                      "comparisons": analysis["paired_comparisons"]}, indent=2))


if __name__ == "__main__":
    main()
