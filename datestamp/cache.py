"""Scan cache and resumability helpers.

Stores per-image scan results as JSON files under <out_dir>/cache/items/ and
writes updates atomically so interrupted runs can resume.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .io_utils import ensure_dir, read_json, write_json


def safe_cache_key(filename: str) -> str:
    base = Path(filename).name
    stem = Path(base).stem
    # filesystem-safe stem + short hash for uniqueness/collision resistance
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "image"
    h = hashlib.sha1(base.encode("utf-8", errors="replace")).hexdigest()[:10]
    return f"{safe}__{h}"


@dataclass
class ScanCache:
    out_dir: Path
    cache_dir: Path
    items_dir: Path
    crops_dir: Path
    manifest_path: Path

    @classmethod
    def for_output_dir(cls, out_dir: str | Path) -> "ScanCache":
        out = Path(out_dir)
        cache_dir = out / "cache"
        items_dir = cache_dir / "items"
        crops_dir = cache_dir / "crops"
        manifest_path = cache_dir / "manifest.json"
        return cls(out_dir=out, cache_dir=cache_dir, items_dir=items_dir, crops_dir=crops_dir, manifest_path=manifest_path)

    def ensure_dirs(self) -> None:
        ensure_dir(self.out_dir)
        ensure_dir(self.cache_dir)
        ensure_dir(self.items_dir)
        ensure_dir(self.crops_dir)

    def item_path(self, filename: str) -> Path:
        return self.items_dir / f"{safe_cache_key(filename)}.json"

    def crop_path(self, filename: str) -> Path:
        # store as png for easy embedding/inspection
        return self.crops_dir / f"{safe_cache_key(filename)}.png"

    def load_item(self, filename: str) -> Optional[dict[str, Any]]:
        path = self.item_path(filename)
        if not path.exists():
            return None
        try:
            data = read_json(path)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def save_item(self, filename: str, data: dict[str, Any]) -> Path:
        self.ensure_dirs()
        path = self.item_path(filename)
        tmp = path.with_suffix(path.suffix + ".tmp")
        # write_json creates parent; but we want atomic replace
        write_json(tmp, data, indent=2)
        os.replace(tmp, path)
        return path

    def has_complete_item(self, filename: str) -> bool:
        item = self.load_item(filename)
        if not item:
            return False
        # Consider complete if pipeline produced a classification (or explicit no-stamp/failure marker)
        return bool(item.get("scan_complete")) or ("classification" in item and "overall_confidence" in item)

    def iter_items(self) -> Iterable[dict[str, Any]]:
        if not self.items_dir.exists():
            return []
        items = []
        for p in sorted(self.items_dir.glob("*.json")):
            try:
                data = read_json(p)
                if isinstance(data, dict):
                    items.append(data)
            except Exception:
                continue
        return items

    def load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        try:
            data = read_json(self.manifest_path)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def save_manifest(self, data: dict[str, Any]) -> None:
        self.ensure_dirs()
        tmp = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        write_json(tmp, data, indent=2)
        os.replace(tmp, self.manifest_path)

