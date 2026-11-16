"""I/O and Windows-safety utilities."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

from .config import SUPPORTED_IMAGE_EXTENSIONS


def configure_windows_utf8() -> None:
    """Best-effort UTF-8 console configuration for Windows consoles/pipes."""
    try:
        # Python 3.7+: reconfigure available on text IO wrappers.
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        # Never fail CLI startup due to console encoding quirks.
        pass
    # Helps some Windows environments with UTF-8 output.
    os.environ.setdefault("PYTHONUTF8", "1")


def resolve_path(path: str | Path) -> Path:
    """Resolve a path robustly (spaces/non-ASCII safe via pathlib)."""
    return Path(path).expanduser().resolve()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def discover_images(photo_dir: str | Path, recursive: bool = False, limit: Optional[int] = None) -> list[Path]:
    """Discover supported image files in deterministic order.

    Default is non-recursive as a safe starting point; can be revisited if needed.
    """
    root = Path(photo_dir)
    if not root.exists():
        raise FileNotFoundError(f"Photo directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    iterator: Iterator[Path]
    if recursive:
        iterator = (p for p in root.rglob("*") if is_supported_image(p))
    else:
        iterator = (p for p in root.iterdir() if is_supported_image(p))

    files = sorted(iterator, key=lambda p: str(p.relative_to(root)).lower())
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be >= 0")
        files = files[:limit]
    return files


def read_json(path: str | Path) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any, *, indent: int = 2) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
        f.write("\n")


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def write_csv_rows(path: str | Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fieldnames})


def format_console_path(path: str | Path) -> str:
    """Return a display-friendly path string without risking encoding crashes."""
    try:
        return str(Path(path))
    except Exception:
        return repr(path)
