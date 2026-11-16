"""Stage 2 glyph bootstrap/cache (scaffold)."""

from __future__ import annotations

from pathlib import Path


def run_glyphs(photo_dir: Path) -> int:
    """Placeholder for glyphs implementation (Milestone 3)."""
    if not photo_dir.exists():
        print(f"glyphs: photo directory not found: {photo_dir}")
        return 2
    if not photo_dir.is_dir():
        print(f"glyphs: not a directory: {photo_dir}")
        return 2
    print("glyphs: not implemented yet (Milestone 3).")
    print(f"  photo_dir={photo_dir}")
    return 1
