"""Stage 0 calibration (scaffold)."""

from __future__ import annotations

from pathlib import Path


def run_calibrate(photo_dir: Path) -> int:
    """Placeholder for calibration implementation (Milestone 1)."""
    if not photo_dir.exists():
        print(f"calibrate: photo directory not found: {photo_dir}")
        return 2
    if not photo_dir.is_dir():
        print(f"calibrate: not a directory: {photo_dir}")
        return 2
    print("calibrate: not implemented yet (Milestone 1).")
    print(f"  photo_dir={photo_dir}")
    return 1
