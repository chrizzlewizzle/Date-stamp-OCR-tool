from __future__ import annotations

from pathlib import Path

from datestamp.apply import plan_apply
from datestamp.io_utils import write_csv_rows


RESULT_FIELDS = [
    "filename",
    "raw_digits_read",
    "parsed_date",
    "per_digit_scores",
    "overall_confidence",
    "classification",
    "notes",
]


def _write_results(path: Path, rows: list[dict[str, str]]) -> None:
    write_csv_rows(path, RESULT_FIELDS, rows)


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def test_apply_plan_selects_confident_and_corrections(tmp_path: Path) -> None:
    photo_dir = tmp_path / "photos"
    results_dir = tmp_path / "results"
    photo_dir.mkdir()
    results_dir.mkdir()

    for name in ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]:
        _touch(photo_dir / name)

    _write_results(
        results_dir / "results.csv",
        [
            {"filename": "a.jpg", "parsed_date": "1982-12-24", "classification": "confident"},
            {"filename": "b.jpg", "parsed_date": "1990-01-01", "classification": "needs_review"},
            {"filename": "c.jpg", "parsed_date": "", "classification": "failed"},
            {"filename": "d.jpg", "parsed_date": "2000-01-02", "classification": "confident"},
        ],
    )
    write_csv_rows(
        results_dir / "corrections.csv",
        ["filename", "corrected_date", "corrected_digits", "notes"],
        [
            {"filename": "b.jpg", "corrected_date": "1991-02-03", "corrected_digits": "", "notes": "manual"},
            {"filename": "c.jpg", "corrected_date": "", "corrected_digits": "821224", "notes": "digits fix"},
            {"filename": "d.jpg", "corrected_date": "2001-01-02", "corrected_digits": "", "notes": "override confident"},
        ],
    )

    items, skipped, meta = plan_apply(photo_dir, results_dir)
    by_file = {i.filename: i for i in items}

    assert meta["to_apply_count"] == 4
    assert "a.jpg" in by_file and by_file["a.jpg"].date_yyyy_mm_dd == "1982-12-24"
    assert by_file["a.jpg"].source == "confident"
    assert by_file["b.jpg"].date_yyyy_mm_dd == "1991-02-03" and by_file["b.jpg"].source == "correction"
    assert by_file["c.jpg"].date_yyyy_mm_dd == "1982-12-24"  # parsed from corrected digits
    assert by_file["c.jpg"].source == "correction_digits"
    # correction overrides confident scan result
    assert by_file["d.jpg"].date_yyyy_mm_dd == "2001-01-02"
    assert skipped == []


def test_apply_plan_skips_ineligible_and_invalid(tmp_path: Path) -> None:
    photo_dir = tmp_path / "photos"
    results_dir = tmp_path / "results"
    photo_dir.mkdir()
    results_dir.mkdir()
    _touch(photo_dir / "ok.jpg")

    _write_results(
        results_dir / "results.csv",
        [
            {"filename": "ok.jpg", "parsed_date": "", "classification": "needs_review"},
            {"filename": "missing.jpg", "parsed_date": "1980-01-01", "classification": "confident"},
            {"filename": "badcorr.jpg", "parsed_date": "1980-01-01", "classification": "failed"},
        ],
    )
    # create badcorr photo and invalid correction
    _touch(photo_dir / "badcorr.jpg")
    write_csv_rows(
        results_dir / "corrections.csv",
        ["filename", "corrected_date", "corrected_digits", "notes"],
        [
            {"filename": "badcorr.jpg", "corrected_date": "1980/01/01", "corrected_digits": "", "notes": "bad format"},
        ],
    )

    items, skipped, _ = plan_apply(photo_dir, results_dir)
    assert items == []
    reasons = {s.filename: s.reason for s in skipped}
    assert "missing.jpg" in reasons and "not found" in reasons["missing.jpg"]
    assert "ok.jpg" in reasons and "not eligible" in reasons["ok.jpg"]
    assert "badcorr.jpg" in reasons and "invalid corrected_date" in reasons["badcorr.jpg"]
