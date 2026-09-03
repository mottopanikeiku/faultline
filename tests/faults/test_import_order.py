from __future__ import annotations

import subprocess
import sys


def test_fault_package_imports_before_environment_package() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "from faultline.faults import BlockedEdge"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
