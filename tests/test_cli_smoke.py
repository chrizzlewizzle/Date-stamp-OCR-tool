from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_cli_help_runs() -> None:
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "datestamp.py", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode == 0
    assert "calibrate" in proc.stdout
    assert "glyphs" in proc.stdout
    assert "scan" in proc.stdout
    assert "apply" in proc.stdout
