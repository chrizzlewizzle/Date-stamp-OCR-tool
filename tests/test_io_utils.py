from __future__ import annotations

import json
from pathlib import Path

import pytest

from datestamp import io_utils


def test_resolve_and_format_console_path_handles_unicode(tmp_path: Path) -> None:
    p = tmp_path / "Crème brûlée 扫描"
    p.mkdir()
    resolved = io_utils.resolve_path(p)
    assert resolved.exists()
    s = io_utils.format_console_path(resolved)
    assert "Crème brûlée" in s or "Cr" in s  # representation safe
    assert isinstance(s, str)


def test_discover_images_non_recursive_sorted_and_limited(tmp_path: Path) -> None:
    (tmp_path / "b.JPG").write_bytes(b"x")
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("no", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "z.png").write_bytes(b"x")

    files = io_utils.discover_images(tmp_path)
    assert [f.name for f in files] == ["a.png", "b.JPG"]

    limited = io_utils.discover_images(tmp_path, limit=1)
    assert [f.name for f in limited] == ["a.png"]


def test_discover_images_errors(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        io_utils.discover_images(missing)
    f = tmp_path / "file.txt"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        io_utils.discover_images(f)
    with pytest.raises(ValueError):
        io_utils.discover_images(tmp_path, limit=-1)


def test_json_roundtrip_utf8(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "calibration.json"
    data = {"note": "Café 東京", "values": [1, 2, 3], "nested": {"ok": True}}
    io_utils.write_json(path, data)
    loaded = io_utils.read_json(path)
    assert loaded == data
    # ensure utf-8 written (no escaped-only requirement, but readable)
    text = path.read_text(encoding="utf-8")
    assert "Café" in text


def test_csv_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "out" / "results.csv"
    fields = ["filename", "raw_digits_read", "parsed_date", "notes"]
    rows = [
        {"filename": "a.png", "raw_digits_read": "821224", "parsed_date": "1982-12-24", "notes": "ok"},
        {"filename": "b é.png", "raw_digits_read": "", "parsed_date": "", "notes": "no stamp"},
    ]
    io_utils.write_csv_rows(path, fields, rows)
    loaded = io_utils.read_csv_rows(path)
    assert len(loaded) == 2
    assert loaded[0]["filename"] == "a.png"
    assert loaded[1]["filename"] == "b é.png"
    assert loaded[1]["raw_digits_read"] == ""
