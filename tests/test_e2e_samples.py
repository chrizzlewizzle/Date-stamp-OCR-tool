from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from datestamp.io_utils import read_csv_rows
from datestamp.recognize import glyph_coverage_warning
from datestamp.scan import run_scan


def _read_results(results_csv: Path) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(results_csv)
    return {r.get("filename", ""): r for r in rows if r.get("filename")}


def _is_detected_nothing(row: dict[str, str]) -> bool:
    cls = (row.get("classification") or "").strip()
    raw = (row.get("raw_digits_read") or "").strip()
    parsed = (row.get("parsed_date") or "").strip()
    # Spec: "Rows marked no_stamp are passes only if the tool detects nothing."
    # Classification no_stamp_found is the direct signal; also require no parsed/read output.
    return cls == "no_stamp_found" and raw == "" and parsed == ""


def _is_declined_or_low_confidence(row: dict[str, str]) -> bool:
    cls = (row.get("classification") or "").strip()
    # Unreadable rows must not be returned as a confident answer.
    return cls in {"needs_review", "failed", "no_stamp_found"}


def _normalize_status(raw_status: str, raw_date: str) -> str:
    s = (raw_status or "").strip().lower().replace("-", "_").replace(" ", "_")
    d = (raw_date or "").strip().lower()
    if "no_stamp" in s or "nostamp" in s or ("no" in s and "stamp" in s):
        return "no_stamp"
    if d in {"no_stamp", "no stamp", "nostamp"}:
        return "no_stamp"
    if "unreadable" in s:
        return "unreadable"
    if "partial" in s:
        return "partial"
    if "ok" in s or s == "":
        return "ok"
    return s


def _normalize_gt_date(value: str, status: str) -> str:
    v = (value or "").strip()
    if status == "no_stamp":
        return ""
    if v.lower() in {"no_stamp", "no stamp", "nostamp", "unreadable", "n/a", "na"}:
        return ""
    return v


def _normalize_gt_digits(value: str, status: str) -> str:
    v = (value or "").strip()
    if status == "no_stamp":
        return ""
    if v.lower() in {"no_stamp", "no stamp", "nostamp"}:
        return ""
    return v


def _grade_rows(gt_rows: list[dict[str, str]], results: dict[str, dict[str, str]]) -> dict[str, Any]:
    per_row = []
    digits_total = digits_correct = 0
    date_total = date_correct = 0
    spec_pass = spec_total = 0
    missing_in_results = 0

    for gt in gt_rows:
        filename = (gt.get("filename") or "").strip()
        raw_status = (gt.get("status") or "").strip()
        status = _normalize_status(raw_status, gt.get("date") or "")
        gt_digits = _normalize_gt_digits(gt.get("digits") or "", status)
        gt_date = _normalize_gt_date(gt.get("date") or "", status)
        res = results.get(filename)
        if not res:
            missing_in_results += 1
            per_row.append(
                {
                    "filename": filename,
                    "status": status,
                    "result": "MISSING",
                    "digits_ok": None,
                    "date_ok": None,
                    "spec_ok": False,
                    "note": "no scan result row produced",
                }
            )
            spec_total += 1
            continue

        cls = (res.get("classification") or "").strip()
        raw = (res.get("raw_digits_read") or "").strip()
        parsed = (res.get("parsed_date") or "").strip()

        digits_ok: bool | None = None
        date_ok: bool | None = None
        spec_ok = True
        note_parts: list[str] = []

        if status == "no_stamp":
            spec_total += 1
            digits_ok = _is_detected_nothing(res)
            date_ok = _is_detected_nothing(res)
            spec_ok = bool(digits_ok and date_ok)
            if spec_ok:
                spec_pass += 1
            else:
                note_parts.append("expected no detection")
        elif status == "unreadable":
            # Not failures if declined/low-confidence; failures if confident answer.
            spec_total += 1
            spec_ok = _is_declined_or_low_confidence(res)
            if spec_ok:
                spec_pass += 1
            else:
                note_parts.append("confident answer on unreadable row")
            # Grade against printed stamp: if GT provides digits/date, score them separately.
            if gt_digits != "":
                digits_total += 1
                digits_ok = raw == gt_digits
                if digits_ok:
                    digits_correct += 1
            if gt_date != "":
                date_total += 1
                date_ok = parsed == gt_date
                if date_ok:
                    date_correct += 1
        elif status == "partial":
            # Partial: grade available printed info separately where provided.
            if gt_digits != "":
                digits_total += 1
                digits_ok = raw == gt_digits
                if digits_ok:
                    digits_correct += 1
            if gt_date != "":
                date_total += 1
                date_ok = parsed == gt_date
                if date_ok:
                    date_correct += 1
        else:  # ok (or unknown -> treat as ok-like grading targets if present)
            if gt_digits != "":
                digits_total += 1
                digits_ok = raw == gt_digits
                if digits_ok:
                    digits_correct += 1
            if gt_date != "":
                date_total += 1
                date_ok = parsed == gt_date
                if date_ok:
                    date_correct += 1

        per_row.append(
            {
                "filename": filename,
                "status": status,
                "classification": cls,
                "gt_digits": gt_digits,
                "raw_digits": raw,
                "digits_ok": digits_ok,
                "gt_date": gt_date,
                "parsed_date": parsed,
                "date_ok": date_ok,
                "spec_ok": spec_ok,
                "note": "; ".join(note_parts),
            }
        )

    return {
        "per_row": per_row,
        "digits_total": digits_total,
        "digits_correct": digits_correct,
        "date_total": date_total,
        "date_correct": date_correct,
        "spec_total": spec_total,
        "spec_pass": spec_pass,
        "missing_in_results": missing_in_results,
    }


def _format_scorecard(report: dict[str, Any], cov: dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("E2E SCORECARD (samples/ground_truth.csv)")
    lines.append("=" * 72)
    d_tot = report["digits_total"]
    d_ok = report["digits_correct"]
    t_tot = report["date_total"]
    t_ok = report["date_correct"]
    s_tot = report["spec_total"]
    s_ok = report["spec_pass"]
    d_pct = (100.0 * d_ok / d_tot) if d_tot else 0.0
    t_pct = (100.0 * t_ok / t_tot) if t_tot else 0.0
    s_pct = (100.0 * s_ok / s_tot) if s_tot else 0.0
    lines.append(f"Digits (recogniser): {d_ok}/{d_tot} correct ({d_pct:.1f}%)")
    lines.append(f"Date (parse/pivot):  {t_ok}/{t_tot} correct ({t_pct:.1f}%)")
    lines.append(f"Spec pass rules:     {s_ok}/{s_tot} ({s_pct:.1f}%) [no_stamp/unreadable handling]")
    if report["missing_in_results"]:
        lines.append(f"Missing result rows: {report['missing_in_results']}")

    present = cov.get("digits_present") or []
    missing = cov.get("digits_missing") or []
    lines.append("")
    lines.append("Glyph coverage (reference set):")
    lines.append(f"  present: {', '.join(present) if present else '(none)'}")
    lines.append(f"  missing: {', '.join(missing) if missing else '(none)'}")
    if cov.get("warning"):
        lines.append(f"  WARNING: {cov.get('warning')}")

    lines.append("")
    lines.append("Per-file results:")
    header = f"{'file':<8} {'status':<10} {'class':<14} {'digits':<7} {'gt_d':<8} {'date':<7} {'gt_date':<12} {'spec':<5} note"
    lines.append(header)
    lines.append("-" * len(header))
    for r in report["per_row"]:

        def mark(v: bool | None) -> str:
            if v is None:
                return "-"
            return "OK" if v else "MISS"

        lines.append(
            f"{r.get('filename',''):<8} {r.get('status',''):<10} {str(r.get('classification','')):<14} "
            f"{mark(r.get('digits_ok')):<7} {str(r.get('gt_digits','')):<8} {mark(r.get('date_ok')):<7} "
            f"{str(r.get('gt_date','')):<12} {mark(r.get('spec_ok')):<5} {r.get('note','')}"
        )
    lines.append("=" * 72)
    return "\n".join(lines)


def test_e2e_samples_scorecard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run pipeline on samples and score against ground truth (do not modify GT).

    Produces a clear scorecard with separate digits/date metrics and glyph coverage.
    Enforces spec pass conditions for no_stamp/unreadable rows; reports OK-row gaps.
    """
    repo_root = Path(__file__).resolve().parents[1]
    photo_dir = repo_root / "samples"
    gt_path = photo_dir / "ground_truth.csv"
    assert photo_dir.is_dir(), f"samples directory not found: {photo_dir}"
    assert gt_path.is_file(), f"ground truth not found: {gt_path}"

    # scan uses calibration.json/glyphs from current working directory
    monkeypatch.chdir(repo_root)

    out_dir = tmp_path / "results"
    rc = run_scan(photo_dir, out_dir, field_order="YMD")
    assert rc == 0, f"scan failed with exit code {rc}"

    results_csv = out_dir / "results.csv"
    review_html = out_dir / "review.html"
    corrections_csv = out_dir / "corrections.csv"
    assert results_csv.is_file(), "results.csv not produced"
    assert review_html.is_file(), "review.html not produced"
    assert corrections_csv.is_file(), "corrections.csv not produced"

    gt_rows = read_csv_rows(gt_path)
    results = _read_results(results_csv)
    assert results, "no results rows parsed"

    cov = glyph_coverage_warning(repo_root / "glyphs")
    report = _grade_rows(gt_rows, results)
    scorecard = _format_scorecard(report, cov)
    print(scorecard)

    # All ground-truth files should be present in scan output
    assert report["missing_in_results"] == 0, f"missing scan results for {report['missing_in_results']} GT rows"
    assert len(report["per_row"]) == len(gt_rows)

    # Glyph coverage must be reported and include 0-9 status (present/missing lists computed)
    assert "digits_present" in cov and "digits_missing" in cov
    # Warn condition is informational; report must surface it
    if cov.get("digits_missing"):
        assert "missing" in scorecard.lower()

    # Spec pass conditions (must hold): no_stamp and unreadable handling rules
    assert report["spec_total"] >= 1
    assert report["spec_pass"] == report["spec_total"], (
        "Spec pass-condition failures detected (no_stamp/unreadable rules). See scorecard above."
    )

    # Sanity: scorecard includes separate metrics
    assert "Digits (recogniser):" in scorecard
    assert "Date (parse/pivot):" in scorecard
    assert "Glyph coverage" in scorecard
