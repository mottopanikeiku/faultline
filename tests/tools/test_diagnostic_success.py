"""Check that the published seed figure uses episode-level diagnostic behavior."""

import json
import runpy
from pathlib import Path
from statistics import StatisticsError
from xml.etree import ElementTree

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = runpy.run_path(str(REPO / "tools/plot_diagnostic_success.py"))
diagnostic_success = SCRIPT["diagnostic_success"]
load_seed_values = SCRIPT["load_seed_values"]
render_seed_svg = SCRIPT["render_seed_svg"]


def test_diagnostic_success_requires_all_three_events() -> None:
    rows = [
        {"condition": "ambiguous", "advance_count": 1,
         "informative_inspection": True, "correct_repair": True},
        {"condition": "ambiguous", "advance_count": 0,
         "informative_inspection": True, "correct_repair": True},
        {"condition": "ambiguous", "advance_count": 1,
         "informative_inspection": False, "correct_repair": True},
        {"condition": "ambiguous", "advance_count": 1,
         "informative_inspection": True, "correct_repair": False},
        {"condition": "revealed", "advance_count": 1,
         "informative_inspection": True, "correct_repair": True},
    ]
    assert diagnostic_success(rows) == 0.25


def test_no_ambiguous_rows_is_not_zero_success() -> None:
    with pytest.raises(StatisticsError):
        diagnostic_success([])


def test_committed_episode_traces_match_every_published_seed() -> None:
    groups = load_seed_values(REPO)
    analysis = json.loads((REPO / "artifacts/results/small-kill-v1-analysis.json").read_text())
    assert set(groups) == set(analysis["arms"])
    for arm, values in groups.items():
        assert values == {
            item["seed"]: item["value"] for item in analysis["arms"][arm]["individual_seeds"]
        }


def test_svg_keeps_all_seeds_and_pairing() -> None:
    groups = load_seed_values(REPO)
    svg = render_seed_svg(groups)
    root = ElementTree.fromstring(svg)
    ns = {"svg": "http://www.w3.org/2000/svg"}
    points = root.findall(".//svg:circle[svg:title]", ns)
    assert len(points) == 24
    assert len(root.findall(".//svg:polyline", ns)) == 8
    titles = {point.find("svg:title", ns).text for point in points}
    for arm, values in groups.items():
        for seed, value in values.items():
            assert f"{arm} seed {seed}: {value:.8f}" in titles
    assert render_seed_svg(groups) == svg
