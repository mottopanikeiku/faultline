from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "statement",
    [
        "from faultline.faults import BlockedEdge",
        "import faultline.oracle",
        "import faultline.generation",
    ],
)
def test_package_imports_in_a_fresh_interpreter(statement: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", statement],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
