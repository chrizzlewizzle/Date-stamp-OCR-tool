"""Scan pipeline orchestration (detect -> segment -> recognize -> parse -> confidence)."""

from __future__ import annotations

import datetime as _dt
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import cv2

from .cache import ScanCache
from .calibration import run_calibrate
from .config import CALIBRATION_FILENAME, GLYPHS_DIRNAME, Classification
from .confidence import PlausibilityContext, score_photo
from .detect import detect_stamp, load_calibration_data
from .glyphs import run_glyphs
from .io_utils import discover_images, ensure_dir
from .parse import check_layout_crosscheck, infer_batch_pivot, parse_date
from .recognize import GlyphTemplates, glyph_coverage_warning, recognize_components
from .review import build_summary, print_summary, write_corrections_stub, write_results_csv, write_review_html
from .segment import segment_stamp_crop


def _derive_stage_notes(record: dict[str, Any]) -> list[str]:
    """Best-effort extraction of non-final notes from older cached records."""
    notes = list(record.get("stage_notes") or [])
    if notes:
        return notes
    raw_notes = list(record.get("notes") or [])
    # Drop obvious parse/confidence/scoring lines to avoid duplication on recompute.
    drop_prefixes = ("parse:", "penalty:")
    drop_contains = (
        "overall score=",
        "worst per-digit score=",
        "detector found stamp",
        "no stamp detected",
        "digit count ",
        "spacing ",
        "valid calendar date",
        "plausibility checks passed",
        "parse confidence=",
        "date could not be parsed",
        "downgraded from confident",
        "missing or invalid parsed date",
        "unreadable marker",
        "fallback",
        "date in the future",
        "after scan/file date",
        "below observed archive range",
        "above observed archive range",
        "year before plausible minimum",
        "at least one digit has very low match score",
    )
    for n in raw_notes:
        s = str(n)
        if s.startswith(drop_prefixes):
            continue
        if any(tok in s for tok in drop_contains):
            continue
        notes.append(s)
    return notes


def _best_effort_scan_date(path: Path) -> Optional[_dt.date]:
    # Try EXIF datetime fields via Pillow if available
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as im:
            exif = im.getexif() if hasattr(im, "getexif") else None
            if exif:
                # Common tags: DateTimeOriginal(36867), DateTimeDigitized(36868), DateTime(306)
                for tag in (36867, 36868, 306):
                    val = exif.get(tag)
                    if not val:
                        continue
                    s = str(val).strip()
                    # formats like 'YYYY:MM:DD HH:MM:SS'
                    if len(s) >= 10:
                        try:
                            return _dt.date(int(s[0:4]), int(s[5:7]), int(s[8:10]))
                        except Exception:
                            pass
    except Exception:
        pass
    # Fallback: file mtime
    try:
        ts = path.stat().st_mtime
        return _dt.datetime.fromtimestamp(ts).date()
    except Exception:
        return None


def _detection_to_dict(det: Any) -> dict[str, Any]:
    if det is None:
        return {"found": False, "method": "none", "corner": None, "bbox": None, "confidence": None, "notes": []}
    return {
        "found": bool(getattr(det, "found", False)),
        "method": getattr(det, "method", "none"),
        "corner": getattr(det, "corner", None),
        "bbox": list(getattr(det, "bbox", None)) if getattr(det, "bbox", None) is not None else None,
        "confidence": getattr(det, "confidence", None),
        "notes": list(getattr(det, "notes", []) or []),
    }


def _reading_to_dict(reading: Any) -> dict[str, Any]:
    if reading is None:
        return {"raw_digits": "", "per_digit_scores": [], "digit_details": [], "notes": []}
    return {
        "raw_digits": getattr(reading, "raw_digits", "") or "",
        "per_digit_scores": [float(s) for s in (getattr(reading, "per_digit_scores", []) or [])],
        "digit_details": list(getattr(reading, "digit_details", []) or []),
        "notes": list(getattr(reading, "notes", []) or []),
    }


def _dict_to_detection(d: dict[str, Any]):
    from .config import DetectionResult

    bbox = d.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        bbox_t = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    else:
        bbox_t = None
    conf = d.get("confidence")
    try:
        conf = float(conf) if conf is not None else None
    except Exception:
        conf = None
    return DetectionResult(
        found=bool(d.get("found", False)),
        method=str(d.get("method", "none")),
        corner=d.get("corner"),
        bbox=bbox_t,
        confidence=conf,
        notes=list(d.get("notes", []) or []),
    )


def _dict_to_reading(d: dict[str, Any]):
    from .config import DigitReading

    scores = []
    for s in (d.get("per_digit_scores") or []):
        try:
            scores.append(float(s))
        except Exception:
            pass
    return DigitReading(
        raw_digits=str(d.get("raw_digits", "") or ""),
        per_digit_scores=scores,
        digit_details=list(d.get("digit_details", []) or []),
        notes=list(d.get("notes", []) or []),
    )


def _ensure_calibration_and_glyphs(photo_dir: Path) -> int:
    cal_path = Path(CALIBRATION_FILENAME)
    glyphs_dir = Path(GLYPHS_DIRNAME)
    if not cal_path.exists():
        print(f"scan: calibration.json missing; running calibrate on {photo_dir} ...")
        rc = run_calibrate(photo_dir)
        if rc != 0:
            return rc
    if not glyphs_dir.exists() or not any(glyphs_dir.iterdir()):
        print(f"scan: glyphs/ missing or empty; running glyphs on {photo_dir} ...")
        rc = run_glyphs(photo_dir)
        if rc != 0:
            return rc
    return 0


def run_scan(photo_dir: Path, out_dir: Path, field_order: str = "YMD", limit: Optional[int] = None) -> int:
    """Run end-to-end scan and write non-destructive review outputs."""
    if not photo_dir.exists() or not photo_dir.is_dir():
        print(f"scan: photo directory not found or not a directory: {photo_dir}")
        return 2

    rc = _ensure_calibration_and_glyphs(photo_dir)
    if rc != 0:
        return rc

    try:
        images = discover_images(photo_dir, limit=limit)
    except Exception as e:
        print(f"scan: failed to discover images in {photo_dir}: {e}")
        return 2
    if not images:
        print(f"scan: no supported images found in {photo_dir}")
        return 1

    ensure_dir(out_dir)
    cache = ScanCache.for_output_dir(out_dir)
    cache.ensure_dirs()

    try:
        cal = load_calibration_data(CALIBRATION_FILENAME)
    except Exception as e:
        print(f"scan: failed to load calibration: {e}")
        return 1
    try:
        templates = GlyphTemplates.load(GLYPHS_DIRNAME)
    except Exception as e:
        print(f"scan: failed to load glyph templates: {e}")
        return 1
    glyph_cov = glyph_coverage_warning(GLYPHS_DIRNAME)

    print(f"scan: processing {len(images)} image(s) from {photo_dir} -> {out_dir}")
    records: list[dict[str, Any]] = []
    reused = 0
    processed = 0

    # Pass 1: detect/segment/recognize (cached/resumable)
    for idx, path in enumerate(images, start=1):
        filename = path.name
        cached = cache.load_item(filename)
        if cached and cache.has_complete_item(filename) and cached.get("reading") is not None:
            # ensure crop path present in record for review if cached
            if not cached.get("crop_image_path") and cached.get("cache_crop_path"):
                cached["crop_image_path"] = cached.get("cache_crop_path")
            cached["filename"] = filename
            # Backward-compat: older cache items may have accumulated final notes; derive stage notes once.
            if "stage_notes" not in cached:
                cached["stage_notes"] = _derive_stage_notes(cached)
            records.append(cached)
            reused += 1
            continue

        rec: dict[str, Any] = {
            "filename": filename,
            "source_path": str(path),
            "scan_date": (_best_effort_scan_date(path).isoformat() if _best_effort_scan_date(path) else ""),
            "detection": None,
            "reading": None,
            "parsed_date": "",
            "overall_confidence": 0.0,
            "classification": Classification.FAILED.value,
            "stage_notes": [],
            "notes": [],
            "crop_image_path": "",
            "cache_crop_path": "",
            "scan_complete": False,
        }
        try:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                rec["stage_notes"].append("image unreadable")
                rec["detection"] = _detection_to_dict(None)
                rec["reading"] = _reading_to_dict(None)
            else:
                det, crop = detect_stamp(img, cal)
                rec["detection"] = _detection_to_dict(det)
                crop_saved = None
                if crop is not None and crop.size > 0:
                    crop_path = cache.crop_path(filename)
                    try:
                        cv2.imwrite(str(crop_path), crop)
                        crop_saved = str(crop_path)
                    except Exception:
                        crop_saved = None
                rec["crop_image_path"] = crop_saved or ""
                rec["cache_crop_path"] = crop_saved or ""

                if det and det.found and crop is not None:
                    seg = segment_stamp_crop(crop, cal, expected_count_hint=None)
                    reading = recognize_components(seg.digit_components, templates)
                    # keep segmentation/reading diagnostics inside reading notes; do not duplicate into stage notes
                    if seg.notes:
                        reading.notes.extend([f"segmentation: {n}" for n in seg.notes[:3]])
                    rec["reading"] = _reading_to_dict(reading)
                else:
                    rec["reading"] = _reading_to_dict(None)
                    if det and getattr(det, "notes", None):
                        rec["stage_notes"].extend([f"detection: {n}" for n in det.notes[:3]])
        except Exception as e:
            rec["stage_notes"].append(f"scan error: {e}")
            if rec.get("detection") is None:
                rec["detection"] = _detection_to_dict(None)
            if rec.get("reading") is None:
                rec["reading"] = _reading_to_dict(None)

        rec["notes"] = list(rec.get("stage_notes") or [])
        rec["stage_complete"] = True
        cache.save_item(filename, rec)
        records.append(rec)
        processed += 1
        if idx % 10 == 0 or idx == len(images):
            print(f"  pass1: {idx}/{len(images)} (processed={processed}, reused={reused})")

    # Pass 2: batch pivot/layout, parse, confidence/classification
    raw_strings = []
    for r in records:
        reading = r.get("reading") or {}
        raw_strings.append(str(reading.get("raw_digits", "") or ""))

    pivot_info = infer_batch_pivot(raw_strings, assumed_field_order=field_order)
    pivot = pivot_info.pivot
    print(f"scan: {pivot_info.rationale}")

    # Preliminary parse to estimate archive year range and layout diagnostics
    prelim_results = []
    parsed_years = []
    for r in records:
        reading = r.get("reading") or {}
        raw = str(reading.get("raw_digits", "") or "")
        pr = parse_date(raw, field_order=field_order, pivot=pivot, auto_detect=True)
        prelim_results.append(pr)
        if pr.ok and pr.year is not None:
            parsed_years.append(int(pr.year))

    layout_report = check_layout_crosscheck(raw_strings, assumed_field_order=field_order, pivot=pivot)
    layout_dict = layout_report.to_dict() if layout_report else {}
    layout_message = layout_report.message if layout_report else ""

    archive_year_min = min(parsed_years) if parsed_years else None
    archive_year_max = max(parsed_years) if parsed_years else None
    if archive_year_min is not None and archive_year_max is not None:
        print(f"scan: observed parsed year range (preliminary): {archive_year_min}-{archive_year_max}")

    # Final scoring pass
    for i, r in enumerate(records):
        reading_dict = r.get("reading") or {}
        raw = str(reading_dict.get("raw_digits", "") or "")
        pr = prelim_results[i]
        r["parsed_date"] = pr.parsed_date if pr.ok else ""
        r["pivot_used"] = pivot
        r["field_order_effective"] = pr.field_order or field_order
        r["parse_confidence"] = float(getattr(pr, "confidence", 0.0) or 0.0)

        stage_notes = list(r.get("stage_notes") or [])
        # include reading-level diagnostics as stage context (deduped)
        for n in (reading_dict.get("notes") or []):
            if n not in stage_notes:
                stage_notes.append(n)

        final_notes = list(stage_notes)
        for n in (pr.notes or []):
            note = f"parse: {n}"
            if note not in final_notes:
                final_notes.append(note)

        det = _dict_to_detection(r.get("detection") or {})
        reading = _dict_to_reading(reading_dict)
        scan_date_s = r.get("scan_date") or ""
        scan_date = None
        if scan_date_s:
            try:
                scan_date = _dt.date.fromisoformat(scan_date_s)
            except Exception:
                scan_date = None
        ctx = PlausibilityContext(
            scan_date=scan_date,
            archive_year_min=archive_year_min,
            archive_year_max=archive_year_max,
        )
        breakdown = score_photo(det, reading, r.get("parsed_date", ""), parse_confidence=r.get("parse_confidence"), plausibility_context=ctx)
        r["overall_confidence"] = float(breakdown.overall_score)
        r["classification"] = breakdown.classification.value
        for n in breakdown.to_notes():
            if n not in final_notes:
                final_notes.append(n)
        r["notes"] = final_notes
        r["score_breakdown"] = {
            "detector_score": breakdown.detector_score,
            "worst_digit_score": breakdown.worst_digit_score,
            "digit_quality_score": breakdown.digit_quality_score,
            "parse_plausibility_score": breakdown.parse_plausibility_score,
            "overall_score": breakdown.overall_score,
            "details": breakdown.details,
        }
        r["scan_complete"] = True
        cache.save_item(r["filename"], r)

    # Sort records deterministically by filename for CSV/cache manifest; review sorts by confidence itself
    records.sort(key=lambda x: str(x.get("filename", "")).lower())

    results_path = write_results_csv(out_dir, records)
    review_path = write_review_html(out_dir, records)
    corrections_path = write_corrections_stub(out_dir, records)

    summary = build_summary(records, glyph_coverage=glyph_cov, layout_report=layout_dict, layout_message=layout_message)
    summary["pivot"] = pivot
    summary["pivot_rationale"] = pivot_info.rationale
    summary["output_files"] = {
        "results_csv": str(results_path),
        "review_html": str(review_path),
        "corrections_csv": str(corrections_path),
        "cache_dir": str(cache.cache_dir),
    }

    # Save run summary in cache for reference/resume context
    cache.save_manifest({
        "version": 1,
        "photo_dir": str(photo_dir),
        "out_dir": str(out_dir),
        "field_order": field_order,
        "limit": limit,
        "pivot": pivot,
        "pivot_rationale": pivot_info.rationale,
        "summary": summary,
    })

    # Console summary (layout loud warning first if applicable)
    print_summary(summary)
    print(f"scan: wrote {results_path}")
    print(f"scan: wrote {review_path}")
    print(f"scan: wrote {corrections_path}")
    print(f"scan: cache in {cache.cache_dir} (reused {reused} cached item(s), processed {processed})")
    return 0
