from __future__ import annotations

from pathlib import Path

from datestamp.cache import ScanCache
from datestamp.review import normalize_result_record, write_corrections_stub, write_results_csv


def test_scan_cache_roundtrip_and_complete(tmp_path: Path) -> None:
    cache = ScanCache.for_output_dir(tmp_path / "out")
    assert not cache.has_complete_item("img 1.png")

    payload = {
        "filename": "img 1.png",
        "classification": "needs_review",
        "overall_confidence": 0.4,
        "scan_complete": True,
        "reading": {"raw_digits": "123", "per_digit_scores": [0.9, 0.8, 0.7]},
    }
    cache.save_item("img 1.png", payload)
    assert cache.item_path("img 1.png").exists()
    assert cache.has_complete_item("img 1.png")
    loaded = cache.load_item("img 1.png")
    assert loaded is not None
    assert loaded["filename"] == "img 1.png"
    assert loaded["reading"]["raw_digits"] == "123"


def test_review_writers_emit_expected_files(tmp_path: Path) -> None:
    records = [
        {
            "filename": "a.jpg",
            "reading": {"raw_digits": "821224", "per_digit_scores": [0.9, 0.8, 0.7, 0.6, 0.9, 0.8]},
            "parsed_date": "1982-12-24",
            "overall_confidence": 0.95,
            "classification": "confident",
            "notes": ["ok"],
        },
        {
            "filename": "b.jpg",
            "reading": {"raw_digits": "12", "per_digit_scores": [0.4, 0.3]},
            "parsed_date": "",
            "overall_confidence": 0.2,
            "classification": "failed",
            "notes": ["bad", "parse: no"],
        },
    ]

    results_path = write_results_csv(tmp_path, records)
    corr_path = write_corrections_stub(tmp_path, records)
    assert results_path.exists()
    assert corr_path.exists()

    lines = results_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("filename,raw_digits_read,parsed_date")
    assert any("a.jpg" in l and "0.9000;0.8000;0.7000;0.6000;0.9000;0.8000" in l for l in lines[1:])

    row = normalize_result_record(records[1])
    assert row["per_digit_scores"] == "0.4000;0.3000"
    assert row["notes"] == "bad | parse: no"

    corr_lines = corr_path.read_text(encoding="utf-8").splitlines()
    assert corr_lines[0] == "filename,corrected_date,corrected_digits,notes"
    # failed row should be prefilled for attention
    assert any(line.startswith("b.jpg,") for line in corr_lines[1:])
    assert not any(line.startswith("a.jpg,") for line in corr_lines[1:])
