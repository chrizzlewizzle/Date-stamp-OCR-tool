"""Confidence scoring, plausibility checks, and classification.

Configurable scoring/classification constants live at the top of this module
(single source of truth for confidence logic), with shared defaults in
`datestamp.config` referenced where appropriate.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .config import (
    CONFIDENT_MIN_OVERALL_SCORE,
    NEEDS_REVIEW_MIN_OVERALL_SCORE,
    Classification,
    DetectionResult,
    DigitReading,
    PLAUSIBLE_YEAR_MAX_FUTURE_YEARS,
    PLAUSIBLE_YEAR_MIN,
)

# --- Configurable constants (confidence/plausibility) ---
# Overall score weights/components (kept simple and explainable)
WEIGHT_WORST_DIGIT = 0.55
WEIGHT_DIGIT_QUALITY = 0.20
WEIGHT_PARSE_PLAUSIBILITY = 0.20
WEIGHT_DETECTOR = 0.05  # mainly informative; emphasized when fallback detector used

# Digit count/spacing expectations
EXPECTED_DIGIT_COUNTS = {4, 5, 6}
MIN_DIGIT_COUNT = 4
MAX_DIGIT_COUNT = 6
SPACING_CV_GOOD_MAX = 0.35  # coefficient of variation threshold for "regular" spacing
SPACING_CV_BAD_MIN = 0.80

# Detector handling
DETECTOR_CONFIDENCE_FALLBACK_METHODS = {"yolo_onnx"}
DETECTOR_CONFIDENCE_STRONG_MIN = 0.60

# Plausibility bounds/defaults (can be overridden via function args/context)
DEFAULT_PLAUSIBLE_YEAR_MIN = PLAUSIBLE_YEAR_MIN
DEFAULT_PLAUSIBLE_YEAR_MAX_FUTURE_YEARS = PLAUSIBLE_YEAR_MAX_FUTURE_YEARS
ARCHIVE_YEAR_RANGE_MARGIN = 2  # tolerance around observed archive range

# Classification thresholds (canonical defaults from config)
DEFAULT_CONFIDENT_MIN_OVERALL_SCORE = CONFIDENT_MIN_OVERALL_SCORE
DEFAULT_NEEDS_REVIEW_MIN_OVERALL_SCORE = NEEDS_REVIEW_MIN_OVERALL_SCORE


@dataclass
class PlausibilityContext:
    """Context for plausibility checks.

    `scan_date` may be YYYY-MM-DD, EXIF-like datetime string, datetime/date, or None.
    `archive_year_min/max` represent the observed range for the rest of the archive.
    """

    scan_date: Optional[Any] = None
    today: Optional[_dt.date] = None
    archive_year_min: Optional[int] = None
    archive_year_max: Optional[int] = None
    plausible_year_min: int = DEFAULT_PLAUSIBLE_YEAR_MIN
    plausible_year_max_future_years: int = DEFAULT_PLAUSIBLE_YEAR_MAX_FUTURE_YEARS
    archive_year_margin: int = ARCHIVE_YEAR_RANGE_MARGIN


@dataclass
class PlausibilityAssessment:
    ok: bool
    score: float  # 0..1 contribution factor
    reasons: list[str] = field(default_factory=list)
    penalties: list[str] = field(default_factory=list)
    parsed_date: Optional[_dt.date] = None
    scan_date: Optional[_dt.date] = None


@dataclass
class ScoreBreakdown:
    detector_score: float
    worst_digit_score: float
    digit_quality_score: float
    parse_plausibility_score: float
    overall_score: float
    classification: Classification
    reasons: list[str] = field(default_factory=list)
    penalties: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_notes(self) -> list[str]:
        notes: list[str] = []
        notes.extend(self.reasons)
        notes.extend([f"penalty: {p}" for p in self.penalties])
        return notes


def _clamp01(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _parse_date_like(value: Any) -> Optional[_dt.date]:
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    s = str(value).strip()
    if not s:
        return None
    # common formats: YYYY-MM-DD, YYYY:MM:DD HH:MM:SS, YYYY-MM-DDTHH...
    m = re.match(r"^(\d{4})[-:](\d{2})[-:](\d{2})", s)
    if not m:
        return None
    try:
        return _dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except Exception:
        return None


def get_worst_digit_score(per_digit_scores: Sequence[float] | None) -> float:
    if not per_digit_scores:
        return 0.0
    vals = []
    for s in per_digit_scores:
        try:
            vals.append(float(s))
        except Exception:
            vals.append(0.0)
    if not vals:
        return 0.0
    return _clamp01(min(vals))


def get_average_digit_score(per_digit_scores: Sequence[float] | None) -> float:
    if not per_digit_scores:
        return 0.0
    vals = []
    for s in per_digit_scores:
        try:
            vals.append(float(s))
        except Exception:
            vals.append(0.0)
    if not vals:
        return 0.0
    return _clamp01(sum(vals) / len(vals))


def _extract_digit_bboxes(reading: DigitReading | None) -> list[tuple[int, int, int, int]]:
    if reading is None:
        return []
    bboxes: list[tuple[int, int, int, int]] = []
    for d in reading.digit_details or []:
        if not isinstance(d, dict):
            continue
        if d.get("kind") not in (None, "digit"):
            continue
        bbox = d.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                x, y, w, h = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                bboxes.append((x, y, w, h))
            except Exception:
                continue
    bboxes.sort(key=lambda b: b[0])
    return bboxes


def _spacing_score_from_bboxes(bboxes: Sequence[tuple[int, int, int, int]]) -> tuple[float, Optional[float], list[str]]:
    notes: list[str] = []
    if len(bboxes) < 2:
        return 0.7, None, ["spacing not directly assessable (<2 digit boxes)"]
    centers = [x + w / 2.0 for (x, _y, w, _h) in bboxes]
    gaps = [centers[i] - centers[i - 1] for i in range(1, len(centers))]
    gaps = [g for g in gaps if g > 0]
    if not gaps:
        return 0.5, None, ["spacing not assessable (non-positive gaps)"]
    mean_gap = sum(gaps) / len(gaps)
    if mean_gap <= 1e-6:
        return 0.3, None, ["spacing irregular (degenerate gaps)"]
    var = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
    std = var ** 0.5
    cv = std / mean_gap
    if cv <= SPACING_CV_GOOD_MAX:
        score = 1.0
        notes.append(f"spacing regular (cv={cv:.2f})")
    elif cv >= SPACING_CV_BAD_MIN:
        score = 0.25
        notes.append(f"spacing irregular (cv={cv:.2f})")
    else:
        # linear interpolation between good/bad thresholds
        t = (cv - SPACING_CV_GOOD_MAX) / max(1e-9, (SPACING_CV_BAD_MIN - SPACING_CV_GOOD_MAX))
        score = 1.0 - 0.75 * max(0.0, min(1.0, t))
        notes.append(f"spacing moderately irregular (cv={cv:.2f})")
    return _clamp01(score), float(cv), notes


def score_digit_count_and_spacing(
    reading: DigitReading | None,
    *,
    expected_counts: set[int] | None = None,
) -> tuple[float, list[str], dict[str, Any]]:
    expected_counts = expected_counts or set(EXPECTED_DIGIT_COUNTS)
    notes: list[str] = []
    details: dict[str, Any] = {}
    raw = (reading.raw_digits if reading else "") or ""
    # count recognized digit-like chars
    digit_chars = [c for c in raw if c.isdigit()]
    count = len(digit_chars)
    details["digit_count"] = count
    details["raw_digits"] = raw

    if count == 0:
        notes.append("no digits recognized")
        return 0.0, notes, details

    if count in expected_counts:
        count_score = 1.0
        notes.append(f"digit count plausible ({count})")
    elif MIN_DIGIT_COUNT <= count <= MAX_DIGIT_COUNT:
        count_score = 0.8
        notes.append(f"digit count acceptable ({count})")
    elif count == 3 or count == 7:
        count_score = 0.45
        notes.append(f"digit count unusual ({count})")
    else:
        count_score = 0.15
        notes.append(f"digit count implausible ({count})")

    bboxes = _extract_digit_bboxes(reading)
    spacing_score, spacing_cv, spacing_notes = _spacing_score_from_bboxes(bboxes)
    notes.extend(spacing_notes)
    details["spacing_cv"] = spacing_cv
    details["bbox_count"] = len(bboxes)

    # weight count more than spacing (spacing may be unavailable/noisy)
    score = 0.65 * count_score + 0.35 * spacing_score
    details["count_score"] = count_score
    details["spacing_score"] = spacing_score
    return _clamp01(score), notes, details


def assess_plausibility(
    parsed_date: Any,
    *,
    context: PlausibilityContext | None = None,
) -> PlausibilityAssessment:
    ctx = context or PlausibilityContext()
    today = ctx.today or _dt.date.today()
    parsed = _parse_date_like(parsed_date)
    scan = _parse_date_like(ctx.scan_date)
    reasons: list[str] = []
    penalties: list[str] = []

    if parsed is None:
        penalties.append("missing or invalid parsed date")
        return PlausibilityAssessment(ok=False, score=0.0, reasons=reasons, penalties=penalties, parsed_date=None, scan_date=scan)

    score = 1.0
    reasons.append(f"valid calendar date ({parsed.isoformat()})")

    # not before plausible minimum
    if parsed.year < ctx.plausible_year_min:
        score *= 0.1
        penalties.append(f"year before plausible minimum ({parsed.year} < {ctx.plausible_year_min})")

    # not in the future (relative to today, allowing configured future years)
    latest_future = today.replace(year=today.year + max(0, int(ctx.plausible_year_max_future_years)))
    # handle Feb 29 edge case for replace
    try:
        if (today.month, today.day) == (2, 29):
            latest_future = _dt.date(today.year + max(0, int(ctx.plausible_year_max_future_years)), 2, 28)
    except Exception:
        latest_future = today
    if parsed > latest_future:
        score *= 0.05
        penalties.append(f"date in the future ({parsed.isoformat()} > {latest_future.isoformat()})")

    # not after file's own scan date
    if scan is not None and parsed > scan:
        score *= 0.05
        penalties.append(f"date after scan/file date ({parsed.isoformat()} > {scan.isoformat()})")

    # within archive-observed range (if known)
    if ctx.archive_year_min is not None:
        lo = int(ctx.archive_year_min) - int(ctx.archive_year_margin)
        if parsed.year < lo:
            score *= 0.7
            penalties.append(f"year below observed archive range ({parsed.year} < {lo})")
    if ctx.archive_year_max is not None:
        hi = int(ctx.archive_year_max) + int(ctx.archive_year_margin)
        if parsed.year > hi:
            score *= 0.7
            penalties.append(f"year above observed archive range ({parsed.year} > {hi})")

    ok = len(penalties) == 0
    if ok:
        reasons.append("plausibility checks passed")
    return PlausibilityAssessment(
        ok=ok,
        score=_clamp01(score),
        reasons=reasons,
        penalties=penalties,
        parsed_date=parsed,
        scan_date=scan,
    )


def classify_score(
    overall_score: float,
    *,
    confident_min: float = DEFAULT_CONFIDENT_MIN_OVERALL_SCORE,
    needs_review_min: float = DEFAULT_NEEDS_REVIEW_MIN_OVERALL_SCORE,
) -> Classification:
    s = _clamp01(overall_score)
    if s >= confident_min:
        return Classification.CONFIDENT
    if s >= needs_review_min:
        return Classification.NEEDS_REVIEW
    return Classification.FAILED


def score_photo(
    detection: DetectionResult | None,
    reading: DigitReading | None,
    parsed_date: Any = "",
    *,
    parse_confidence: Optional[float] = None,
    plausibility_context: PlausibilityContext | None = None,
    confident_min: float = DEFAULT_CONFIDENT_MIN_OVERALL_SCORE,
    needs_review_min: float = DEFAULT_NEEDS_REVIEW_MIN_OVERALL_SCORE,
) -> ScoreBreakdown:
    """Compute overall confidence and classification for one photo result.

    - `no_stamp_found` is assigned only when detection is absent/empty.
    - Overall score emphasizes the worst per-digit score (not average).
    """
    reasons: list[str] = []
    penalties: list[str] = []
    details: dict[str, Any] = {}

    if detection is None or not detection.found:
        reasons.append("no stamp detected")
        return ScoreBreakdown(
            detector_score=0.0,
            worst_digit_score=0.0,
            digit_quality_score=0.0,
            parse_plausibility_score=0.0,
            overall_score=0.0,
            classification=Classification.NO_STAMP_FOUND,
            reasons=reasons,
            penalties=penalties,
            details={"detection_found": False},
        )

    # Detector component
    det_conf = detection.confidence if detection.confidence is not None else 0.5
    detector_score = _clamp01(float(det_conf))
    details["detection_method"] = detection.method
    details["detection_confidence"] = detector_score
    if detection.method in DETECTOR_CONFIDENCE_FALLBACK_METHODS:
        reasons.append(f"detector fallback used ({detection.method}, conf={detector_score:.2f})")
        if detector_score < DETECTOR_CONFIDENCE_STRONG_MIN:
            penalties.append("low fallback detector confidence")
    else:
        reasons.append(f"detector found stamp ({detection.method or 'unknown'}, conf={detector_score:.2f})")

    # Worst per-digit score (primary recognizer signal)
    per_digit_scores = (reading.per_digit_scores if reading else []) or []
    worst_digit = get_worst_digit_score(per_digit_scores)
    avg_digit = get_average_digit_score(per_digit_scores)
    details["worst_digit_score"] = worst_digit
    details["average_digit_score"] = avg_digit
    details["per_digit_scores"] = list(per_digit_scores)
    if per_digit_scores:
        reasons.append(f"worst per-digit score={worst_digit:.3f} (avg={avg_digit:.3f})")
        if worst_digit < 0.5:
            penalties.append("at least one digit has very low match score")
    else:
        penalties.append("no per-digit recognition scores")

    # Digit count and spacing quality
    digit_quality, digit_notes, digit_details = score_digit_count_and_spacing(reading)
    reasons.extend(digit_notes)
    details.update({f"digit_{k}": v for k, v in digit_details.items()})

    # Parse/plausibility component
    plaus = assess_plausibility(parsed_date, context=plausibility_context)
    reasons.extend(plaus.reasons)
    penalties.extend(plaus.penalties)
    details["plausibility_score"] = plaus.score
    details["parsed_date_assessed"] = plaus.parsed_date.isoformat() if plaus.parsed_date else ""
    details["scan_date_assessed"] = plaus.scan_date.isoformat() if plaus.scan_date else ""

    parse_conf = _clamp01(parse_confidence) if parse_confidence is not None else (1.0 if plaus.parsed_date else 0.0)
    details["parse_confidence"] = parse_conf
    if parse_confidence is not None:
        reasons.append(f"parse confidence={parse_conf:.2f}")

    # combine parse validity confidence with plausibility
    parse_plausibility_score = _clamp01(0.7 * plaus.score + 0.3 * parse_conf)
    details["parse_plausibility_component"] = parse_plausibility_score
    if plaus.parsed_date is None:
        penalties.append("date could not be parsed")

    # Weighted overall score (worst-digit dominates)
    overall = (
        WEIGHT_WORST_DIGIT * worst_digit
        + WEIGHT_DIGIT_QUALITY * digit_quality
        + WEIGHT_PARSE_PLAUSIBILITY * parse_plausibility_score
        + WEIGHT_DETECTOR * detector_score
    )

    # Hard failure guards for clearly unusable results
    if not per_digit_scores:
        overall = min(overall, 0.40)
    if plaus.parsed_date is None:
        overall = min(overall, 0.35)
    raw = (reading.raw_digits if reading else "") or ""
    if "?" in raw:
        penalties.append("raw reading contains unreadable marker '?'")
        overall = min(overall, 0.45)

    overall = _clamp01(overall)
    classification = classify_score(overall, confident_min=confident_min, needs_review_min=needs_review_min)

    # If detected but unusable, prefer explicit failed/review semantics
    if classification == Classification.CONFIDENT and (plaus.parsed_date is None or worst_digit < 0.75):
        classification = Classification.NEEDS_REVIEW
        penalties.append("downgraded from confident due to weak parse/recognition evidence")
    if classification == Classification.CONFIDENT and not plaus.ok:
        classification = Classification.NEEDS_REVIEW
        penalties.append("downgraded from confident due to plausibility concerns")

    reasons.append(f"overall score={overall:.3f} -> {classification.value}")

    return ScoreBreakdown(
        detector_score=detector_score,
        worst_digit_score=worst_digit,
        digit_quality_score=digit_quality,
        parse_plausibility_score=parse_plausibility_score,
        overall_score=overall,
        classification=classification,
        reasons=reasons,
        penalties=penalties,
        details=details,
    )


def classify_photo_result(
    result: Any,
    *,
    parse_confidence: Optional[float] = None,
    plausibility_context: PlausibilityContext | None = None,
    confident_min: float = DEFAULT_CONFIDENT_MIN_OVERALL_SCORE,
    needs_review_min: float = DEFAULT_NEEDS_REVIEW_MIN_OVERALL_SCORE,
    apply_to_result: bool = False,
) -> ScoreBreakdown:
    """Convenience wrapper for `PhotoResult`-like objects."""
    detection = getattr(result, "detection", None)
    reading = getattr(result, "reading", None)
    parsed_date = getattr(result, "parsed_date", "")
    breakdown = score_photo(
        detection,
        reading,
        parsed_date,
        parse_confidence=parse_confidence,
        plausibility_context=plausibility_context,
        confident_min=confident_min,
        needs_review_min=needs_review_min,
    )
    if apply_to_result and result is not None:
        try:
            result.overall_confidence = breakdown.overall_score
            result.classification = breakdown.classification
            # append explainability notes
            existing = list(getattr(result, "notes", []) or [])
            result.notes = existing + breakdown.to_notes()
        except Exception:
            pass
    return breakdown

