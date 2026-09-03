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
