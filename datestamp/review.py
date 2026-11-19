"""Review outputs and console summary for scan results."""

from __future__ import annotations

import base64
import html
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from .config import (
    CORRECTIONS_CSV_FILENAME,
    RESULT_CSV_FIELDS,
    RESULTS_CSV_FILENAME,
    REVIEW_HTML_FILENAME,
    Classification,
)
from .io_utils import ensure_dir, write_csv_rows


CLASS_ORDER = {
    "needs_review": 0,
    "failed": 1,
    "no_stamp_found": 2,
    "confident": 3,
}


def _as_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


def normalize_result_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Normalize a cached/scan record into the canonical results.csv row shape."""
    filename = _as_str(rec.get("filename"))
    reading = rec.get("reading") if isinstance(rec.get("reading"), dict) else {}
    raw_digits = rec.get("raw_digits_read")
    if raw_digits is None:
        raw_digits = reading.get("raw_digits", "")
    per_digit_scores = rec.get("per_digit_scores")
    if per_digit_scores is None:
        per_digit_scores = reading.get("per_digit_scores", [])
    if isinstance(per_digit_scores, list):
        per_digit_scores_s = ";".join(f"{float(s):.4f}" for s in per_digit_scores)
    else:
        per_digit_scores_s = _as_str(per_digit_scores)
    overall = rec.get("overall_confidence", 0.0)
    try:
        overall_s = f"{float(overall):.4f}"
    except Exception:
        overall_s = _as_str(overall)
    classification = rec.get("classification")
    if isinstance(classification, Classification):
        classification = classification.value
    classification = _as_str(classification or "")
    notes = rec.get("notes", [])
    if isinstance(notes, list):
        notes_s = " | ".join(_as_str(n) for n in notes if _as_str(n))
    else:
        notes_s = _as_str(notes)
    return {
        "filename": filename,
        "raw_digits_read": _as_str(raw_digits),
        "parsed_date": _as_str(rec.get("parsed_date", "")),
        "per_digit_scores": per_digit_scores_s,
        "overall_confidence": overall_s,
        "classification": classification,
        "notes": notes_s,
    }


def write_results_csv(out_dir: str | Path, records: Sequence[dict[str, Any]]) -> Path:
    path = Path(out_dir) / RESULTS_CSV_FILENAME
    rows = [normalize_result_record(r) for r in records]
    write_csv_rows(path, RESULT_CSV_FIELDS, rows)
    return path


def write_corrections_stub(out_dir: str | Path, records: Sequence[dict[str, Any]]) -> Path:
    path = Path(out_dir) / CORRECTIONS_CSV_FILENAME
    fields = ["filename", "corrected_date", "corrected_digits", "notes"]
    rows = []
    for r in records:
        cls = _as_str(r.get("classification"))
        # Pre-fill rows likely needing attention; keep file easy to hand-edit for all cases.
        if cls in {"needs_review", "failed", "no_stamp_found"}:
            rows.append({
                "filename": _as_str(r.get("filename")),
                "corrected_date": "",
                "corrected_digits": "",
                "notes": "",
            })
    # If nothing to prefill, still provide at least header + maybe one example row comment via empty file? keep header only.
    # But ensure deterministic; header-only is fine.
    write_csv_rows(path, fields, rows)
    return path


def _embed_image_base64(path: Optional[str | Path]) -> tuple[str, str]:
    if not path:
        return "", ""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return "", ""
    try:
        data = p.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        suffix = p.suffix.lower().lstrip(".") or "png"
        mime = "jpeg" if suffix in {"jpg", "jpeg"} else ("png" if suffix == "png" else "png")
        return mime, b64
    except Exception:
        return "", ""


def _confidence_sort_key(rec: dict[str, Any]) -> tuple[int, float, str]:
    cls = _as_str(rec.get("classification"))
    order = CLASS_ORDER.get(cls, 9)
    try:
        conf = float(rec.get("overall_confidence", 0.0) or 0.0)
    except Exception:
        conf = 0.0
    return (order, conf, _as_str(rec.get("filename")).lower())


def _class_color(cls: str) -> str:
    return {
        "confident": "#1a7f37",
        "needs_review": "#9a6700",
        "failed": "#cf222e",
        "no_stamp_found": "#57606a",
    }.get(cls, "#57606a")


def write_review_html(out_dir: str | Path, records: Sequence[dict[str, Any]]) -> Path:
    path = Path(out_dir) / REVIEW_HTML_FILENAME
    sorted_recs = sorted(list(records), key=_confidence_sort_key)

    counts = Counter(_as_str(r.get("classification")) for r in records)
    total = len(records)

    def esc(s: Any) -> str:
        return html.escape(_as_str(s), quote=True)

    rows_html = []
    for rec in sorted_recs:
        filename = esc(rec.get("filename"))
        raw = esc((rec.get("reading") or {}).get("raw_digits", rec.get("raw_digits_read", "")) if isinstance(rec.get("reading"), dict) else rec.get("raw_digits_read", ""))
        parsed = esc(rec.get("parsed_date", ""))
        cls = _as_str(rec.get("classification"))
        conf = rec.get("overall_confidence", 0.0)
        try:
            conf_f = float(conf)
            conf_s = f"{conf_f:.3f}"
        except Exception:
            conf_s = esc(conf)
        notes = rec.get("notes", [])
        if isinstance(notes, list):
            notes_s = esc(" | ".join(_as_str(n) for n in notes))
        else:
            notes_s = esc(notes)
        color = _class_color(cls)

        crop_path = rec.get("crop_image_path") or rec.get("cache_crop_path")
        mime, b64 = _embed_image_base64(crop_path)
        if b64:
            img_html = f'<img src="data:image/{mime};base64,{b64}" alt="crop" style="max-width:360px;max-height:180px;border:1px solid #d0d7de;background:#fff;" />'
        else:
            img_html = '<div style="color:#57606a;font-size:12px;">(no crop)</div>'

        per_digit = (rec.get("reading") or {}).get("per_digit_scores") if isinstance(rec.get("reading"), dict) else rec.get("per_digit_scores")
        if isinstance(per_digit, list) and per_digit:
            try:
                per_digit_s = ", ".join(f"{float(x):.2f}" for x in per_digit)
            except Exception:
                per_digit_s = esc(per_digit)
        else:
            per_digit_s = ""

        rows_html.append(
            f"""
<tr class="row {esc(cls)}">
  <td class="idx"></td>
  <td class="file"><div class="filename">{filename}</div><div class="meta">per-digit: {esc(per_digit_s) if per_digit_s else '-'}</div></td>
  <td class="crop">{img_html}</td>
  <td class="read"><div><strong>Raw:</strong> {raw or '&nbsp;'}</div><div><strong>Date:</strong> {parsed or '&nbsp;'}</div></td>
  <td class="conf"><div class="badge" style="background:{color}22;color:{color};border:1px solid {color}55;">{esc(cls or 'unknown')}</div><div class="score">{conf_s}</div></td>
  <td class="notes">{notes_s or '&nbsp;'}</td>
</tr>"""
        )

    summary_counts = " ".join(
        f'<span class="pill {k}"><strong>{k}</strong>: {counts.get(k,0)}</span>'
        for k in ["confident", "needs_review", "failed", "no_stamp_found"]
    )

    html_doc = f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>Datestamp Scan Review</title>
<style>
  body {{ font-family: Segoe UI, Arial, sans-serif; margin: 16px; color: #24292f; background: #fff; }}
  h1 {{ margin: 0 0 8px 0; font-size: 20px; }}
  .sub {{ color:#57606a; margin-bottom: 12px; }}
  .summary {{ display:flex; flex-wrap:wrap; gap:8px; margin: 8px 0 12px; }}
  .pill {{ padding:4px 8px; border-radius:999px; border:1px solid #d0d7de; background:#f6f8fa; font-size:12px; }}
  .pill.confident {{ border-color:#1a7f3744; }}
  .pill.needs_review {{ border-color:#9a670044; }}
  .pill.failed {{ border-color:#cf222e44; }}
  .pill.no_stamp_found {{ border-color:#57606a44; }}
  table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
  th, td {{ border: 1px solid #d0d7de; padding: 8px; vertical-align: top; }}
  th {{ background:#f6f8fa; position: sticky; top: 0; z-index: 1; text-align: left; }}
  td.idx {{ width: 42px; color:#57606a; text-align:right; }}
  td.file {{ width: 18%; word-break: break-all; }}
  td.crop {{ width: 380px; }}
  td.read {{ width: 16%; }}
  td.conf {{ width: 140px; }}
  td.notes {{ width: auto; word-break: break-word; }}
  .filename {{ font-weight: 600; }}
  .meta {{ color:#57606a; font-size: 12px; margin-top:4px; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600; margin-bottom:4px; }}
  .score {{ font-family: Consolas, monospace; font-size: 13px; }}
  tr.confident {{ background: #f6ffed; }}
  tr.needs_review {{ background: #fff8e1; }}
  tr.failed {{ background: #fff1f0; }}
  tr.no_stamp_found {{ background: #f6f8fa; }}
</style>
</head>
<body>
  <h1>Datestamp Scan Review</h1>
  <div class=\"sub\">Sorted lowest confidence / highest review priority first. Total: {total}</div>
  <div class=\"summary\">{summary_counts}</div>
  <table>
    <thead>
      <tr>
        <th>#</th><th>File</th><th>Stamp Crop</th><th>Read / Parsed</th><th>Confidence</th><th>Notes</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
<script>
  // add row numbers after load for stable display
  document.querySelectorAll('tbody tr td.idx').forEach((td, i) => td.textContent = String(i+1));
</script>
</body>
</html>
"""
    ensure_dir(path.parent)
    path.write_text(html_doc, encoding="utf-8")
    return path


def build_summary(
    records: Sequence[dict[str, Any]],
    *,
    glyph_coverage: Optional[dict[str, Any]] = None,
    layout_report: Optional[dict[str, Any]] = None,
    layout_message: str = "",
) -> dict[str, Any]:
    counts = Counter(_as_str(r.get("classification")) for r in records)
    year_counts: Counter[str] = Counter()
    for r in records:
        d = _as_str(r.get("parsed_date"))
        if len(d) >= 4 and d[:4].isdigit():
            year_counts[d[:4]] += 1
    summary = {
        "total": len(records),
        "counts_by_classification": {k: counts.get(k, 0) for k in ["confident", "needs_review", "failed", "no_stamp_found"]},
        "year_distribution": dict(sorted(year_counts.items())),
        "glyph_coverage": glyph_coverage or {},
        "layout_report": layout_report or {},
        "layout_message": layout_message or (layout_report or {}).get("message", ""),
    }
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    layout_msg = _as_str(summary.get("layout_message"))
    layout_report = summary.get("layout_report") or {}
    is_loud = bool(layout_report.get("is_loud_violation")) or layout_msg.upper().startswith("LAYOUT WARNING")
    if layout_msg and is_loud:
        print("=" * 80)
        print(layout_msg)
        print("=" * 80)
    elif layout_msg:
        print(f"Layout: {layout_msg}")

    print("Scan summary")
    counts = summary.get("counts_by_classification") or {}
    for k in ["confident", "needs_review", "failed", "no_stamp_found"]:
        print(f"  {k:14s}: {counts.get(k, 0)}")
    print(f"  {'total':14s}: {summary.get('total', 0)}")

    years = summary.get("year_distribution") or {}
    if years:
        print("  year distribution:")
        for y, n in years.items():
            print(f"    {y}: {n}")
    else:
        print("  year distribution: (none)")

    cov = summary.get("glyph_coverage") or {}
    missing = cov.get("digits_missing") or []
    present = cov.get("digits_present") or []
    if missing:
        print(f"  WARNING missing glyph digits 0-9: {', '.join(missing)}")
    elif present:
        print(f"  glyph coverage: digits present {', '.join(present)}")
    else:
        print("  glyph coverage: (unknown)")
    if cov.get("warning"):
        print(f"  {cov.get('warning')}")

    if layout_msg and not is_loud:
        # already printed above
        pass

