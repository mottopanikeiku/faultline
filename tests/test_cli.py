import pytest

from faultline import __version__
from faultline.cli import main


def test_package_version_is_exposed() -> None:
    assert __version__ == "0.1.0"


def test_empty_cli_succeeds() -> None:
    assert main([]) == 0


def test_healthy_demo_renders_actual_episode(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["demo", "healthy", "--nodes", "4", "--ticks", "3"]) == 0
    output = capsys.readouterr().out
    assert "FACTORY tick=3" in output
    assert "NODES" in output
    assert "EDGES" in output
    assert "TIMELINE" in output


def test_diagnostic_pair_demo_reports_executed_evidence(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["demo", "diagnostic-pair", "--seed", "42"]) == 0
    output = capsys.readouterr().out
    assert "Initial complete passive observations: IDENTICAL" in output
    assert "Best passive repair success: 50%" in output
    assert "Evidence-contingent success: 100%" in output
    assert "Exact active oracle: success=100% return=8.58 EP=7.33" in output
    assert "Optimal diagnostic sequence: advance -> inspect" in output
    assert "processor.input=0.00 output=2.00" in output
    assert "processor.input=2.00 output=0.00" in output


def test_training_dry_run_resolves_without_writing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "train",
                "--curriculum",
                "epistemic",
                "--seed",
                "3",
                "--max-steps",
                "77",
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert '"kind": "epistemic"' in output
    assert '"training_seed": 3' in output
    assert '"total_decision_steps": 77' in output
