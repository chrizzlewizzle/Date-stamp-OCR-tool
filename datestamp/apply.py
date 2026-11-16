"""Stage 6 metadata apply (scaffold)."""

from __future__ import annotations

from pathlib import Path


def run_apply(photo_dir: Path, results_dir: Path, do_apply: bool, also_file_times: bool) -> int:
    """Placeholder for metadata apply implementation (Milestone 8)."""
    if not photo_dir.exists() or not photo_dir.is_dir():
        print(f"apply: photo directory not found or not a directory: {photo_dir}")
        return 2
    if not results_dir.exists() or not results_dir.is_dir():
        print(f"apply: results directory not found or not a directory: {results_dir}")
        return 2

    mode = "APPLY" if do_apply else "DRY-RUN"
    print(f"apply: {mode} not implemented yet (Milestone 8).")
    print(f"  photo_dir={photo_dir}")
    print(f"  results_dir={results_dir}")
    print(f"  also_file_times={also_file_times}")
    if not do_apply:
        print("apply: default is dry-run; no files will be modified.")
    return 1 if do_apply else 0
