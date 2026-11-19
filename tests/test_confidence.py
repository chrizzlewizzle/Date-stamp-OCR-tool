from __future__ import annotations

import datetime as dt

from datestamp.config import Classification, DetectionResult, DigitReading, PhotoResult
from datestamp.confidence import (
    PlausibilityContext,
    assess_plausibility,
    classify_photo_result,
    classify_score,
    get_worst_digit_score,
    score_digit_count_and_spacing,
    score_photo,
)


def _reading(raw: str, scores: list[float], bboxes: list[tuple[int, int, int, int]] | None = None) -> DigitReading:
    details = []
    bboxes = bboxes or []
    for i, ch in enumerate(raw):
        d = {"index": i, "kind": "digit", "recognized": ch, "score": scores[i] if i < len(scores) else 0.0}
        if i < len(bboxes):
            d["bbox"] = bboxes[i]
        details.append(d)
    return DigitReading(raw_digits=raw, per_digit_scores=scores, digit_details=details)


def test_no_detection_is_classified_no_stamp_found() -> None:
    det = DetectionResult(found=False, method="none")
    reading = DigitReading()
    breakdown = score_photo(det, reading, "")
    assert breakdown.classification == Classification.NO_STAMP_FOUND
    assert breakdown.overall_score == 0.0
    assert any("no stamp detected" in r.lower() for r in breakdown.reasons)


def test_worst_digit_score_dominates_over_average() -> None:
    det = DetectionResult(found=True, method="hsv_geometry", confidence=0.9)
    high_avg_one_bad = _reading("821224", [0.99, 0.99, 0.99, 0.99, 0.99, 0.20])
    all_good = _reading("821224", [0.98, 0.97, 0.99, 0.98, 0.97, 0.96])
    ctx = PlausibilityContext(scan_date="2001-01-01", today=dt.date(2001, 1, 1))

    bad = score_photo(det, high_avg_one_bad, "1982-12-24", parse_confidence=1.0, plausibility_context=ctx)
    good = score_photo(det, all_good, "1982-12-24", parse_confidence=1.0, plausibility_context=ctx)

    assert get_worst_digit_score(high_avg_one_bad.per_digit_scores) == 0.20
    assert bad.worst_digit_score == 0.20
    assert bad.overall_score < good.overall_score
    assert bad.classification != Classification.CONFIDENT
    assert any("worst per-digit" in r.lower() for r in bad.reasons)


def test_digit_count_and_spacing_scoring() -> None:
    # plausible count and regular spacing
    bboxes = [(0, 0, 10, 20), (12, 0, 10, 20), (24, 0, 10, 20), (36, 0, 10, 20), (48, 0, 10, 20), (60, 0, 10, 20)]
    good = _reading("821224", [0.9] * 6, bboxes=bboxes)
    s_good, notes_good, details_good = score_digit_count_and_spacing(good)
    assert details_good["digit_count"] == 6
    assert s_good >= 0.9
    assert any("digit count plausible" in n.lower() for n in notes_good)

    # unusual/implausible count
    short = _reading("12", [0.9, 0.9], bboxes=[(0, 0, 10, 20), (30, 0, 10, 20)])
    s_short, notes_short, details_short = score_digit_count_and_spacing(short)
    assert details_short["digit_count"] == 2
    assert s_short < s_good
    assert any("implausible" in n.lower() or "unusual" in n.lower() for n in notes_short)

    empty = DigitReading(raw_digits="", per_digit_scores=[])
    s_empty, notes_empty, _ = score_digit_count_and_spacing(empty)
    assert s_empty == 0.0
    assert any("no digits" in n.lower() for n in notes_empty)


def test_classification_thresholds() -> None:
    assert classify_score(0.95, confident_min=0.9, needs_review_min=0.6) == Classification.CONFIDENT
    assert classify_score(0.90, confident_min=0.9, needs_review_min=0.6) == Classification.CONFIDENT
    assert classify_score(0.89, confident_min=0.9, needs_review_min=0.6) == Classification.NEEDS_REVIEW
    assert classify_score(0.60, confident_min=0.9, needs_review_min=0.6) == Classification.NEEDS_REVIEW
    assert classify_score(0.59, confident_min=0.9, needs_review_min=0.6) == Classification.FAILED


def test_confident_when_strong_evidence_and_plausible() -> None:
    det = DetectionResult(found=True, method="hsv_geometry", confidence=0.85)
    bboxes = [(i * 12, 0, 10, 20) for i in range(6)]
    reading = _reading("821224", [0.99, 0.98, 0.99, 0.97, 0.98, 0.99], bboxes=bboxes)
    ctx = PlausibilityContext(scan_date="2001-01-01", today=dt.date(2001, 1, 1), archive_year_min=1980, archive_year_max=1999)
    breakdown = score_photo(det, reading, "1982-12-24", parse_confidence=1.0, plausibility_context=ctx)
    assert breakdown.classification == Classification.CONFIDENT
    assert breakdown.overall_score >= 0.9
    assert breakdown.worst_digit_score >= 0.97


def test_unparsed_or_implausible_date_not_confident() -> None:
    det = DetectionResult(found=True, method="hsv_geometry", confidence=0.9)
    bboxes = [(i * 12, 0, 10, 20) for i in range(6)]
    reading = _reading("821224", [0.99] * 6, bboxes=bboxes)
    ctx = PlausibilityContext(scan_date="2001-01-01", today=dt.date(2001, 1, 1))

    unparsed = score_photo(det, reading, "", parse_confidence=0.0, plausibility_context=ctx)
    assert unparsed.classification != Classification.CONFIDENT
    assert any("could not be parsed" in p.lower() or "downgraded" in p.lower() for p in unparsed.penalties)

    future_ctx = PlausibilityContext(scan_date="2001-01-01", today=dt.date(1990, 1, 1), plausible_year_max_future_years=0)
    future = score_photo(det, reading, "1995-01-01", parse_confidence=1.0, plausibility_context=future_ctx)
    assert future.classification != Classification.CONFIDENT
    assert any("future" in p.lower() or "downgraded" in p.lower() or "plausibility" in p.lower() for p in future.penalties)


def test_unreadable_marker_prevents_confident_classification() -> None:
    det = DetectionResult(found=True, method="hsv_geometry", confidence=0.9)
    reading = _reading("82?224", [0.99, 0.99, 0.4, 0.99, 0.99, 0.99])
    ctx = PlausibilityContext(scan_date="2001-01-01", today=dt.date(2001, 1, 1))
    breakdown = score_photo(det, reading, "1982-12-24", parse_confidence=0.8, plausibility_context=ctx)
    assert breakdown.classification != Classification.CONFIDENT
    assert any("unreadable marker" in p.lower() for p in breakdown.penalties)


def test_fallback_detector_confidence_is_recorded() -> None:
    det = DetectionResult(found=True, method="yolo_onnx", confidence=0.42)
    reading = _reading("821224", [0.95] * 6)
    ctx = PlausibilityContext(scan_date="2001-01-01", today=dt.date(2001, 1, 1))
    breakdown = score_photo(det, reading, "1982-12-24", parse_confidence=1.0, plausibility_context=ctx)
    assert any("fallback" in r.lower() for r in breakdown.reasons)
    assert any("low fallback detector confidence" in p.lower() for p in breakdown.penalties)


def test_classify_photo_result_can_apply_to_photo_result() -> None:
    det = DetectionResult(found=True, method="hsv_geometry", confidence=0.85)
    bboxes = [(i * 12, 0, 10, 20) for i in range(6)]
    reading = _reading("821224", [0.99, 0.98, 0.99, 0.97, 0.98, 0.99], bboxes=bboxes)
    pr = PhotoResult(filename="x.png", detection=det, reading=reading, parsed_date="1982-12-24")
    ctx = PlausibilityContext(scan_date="2001-01-01", today=dt.date(2001, 1, 1))

    breakdown = classify_photo_result(pr, parse_confidence=1.0, plausibility_context=ctx, apply_to_result=True)
    assert pr.classification == breakdown.classification
    assert abs(pr.overall_confidence - breakdown.overall_score) < 1e-9
    assert pr.classification == Classification.CONFIDENT
    assert len(pr.notes) > 0
