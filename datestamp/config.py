"""Configuration, defaults, thresholds, and shared data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


# --- Classification thresholds and constants (single source of truth) ---
# Values are initial placeholders; refine as pipeline is calibrated.
CONFIDENT_MIN_OVERALL_SCORE = 0.90
NEEDS_REVIEW_MIN_OVERALL_SCORE = 0.60

# Plausibility defaults (to be refined)
PLAUSIBLE_YEAR_MIN = 1970
PLAUSIBLE_YEAR_MAX_FUTURE_YEARS = 0  # not in future relative to scan/file context

# File/discovery defaults
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
DEFAULT_OUTPUT_DIRNAME = "results"
CALIBRATION_FILENAME = "calibration.json"
GLYPHS_DIRNAME = "glyphs"
CACHE_DIRNAME = "cache"
RESULTS_CSV_FILENAME = "results.csv"
REVIEW_HTML_FILENAME = "review.html"
CORRECTIONS_CSV_FILENAME = "corrections.csv"

# Result CSV schema (canonical)
RESULT_CSV_FIELDS = [
    "filename",
    "raw_digits_read",
    "parsed_date",
    "per_digit_scores",
    "overall_confidence",
    "classification",
    "notes",
]


class Classification(str, Enum):
    CONFIDENT = "confident"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"
    NO_STAMP_FOUND = "no_stamp_found"


@dataclass
class Calibration:
    """Runtime calibration values loaded from calibration.json."""

    version: int = 1
    photo_dir: Optional[str] = None
    notes: list[str] = field(default_factory=list)
    image_stats: dict[str, Any] = field(default_factory=dict)
    hsv_ranges: list[dict[str, Any]] = field(default_factory=list)
    corner_regions: list[dict[str, Any]] = field(default_factory=list)
    stamp_geometry: dict[str, Any] = field(default_factory=dict)
    font_hints: dict[str, Any] = field(default_factory=dict)
    orientation: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Calibration":
        known = {
            "version": data.get("version", 1),
            "photo_dir": data.get("photo_dir"),
            "notes": data.get("notes", []) or [],
            "image_stats": data.get("image_stats", {}) or {},
            "hsv_ranges": data.get("hsv_ranges", []) or [],
            "corner_regions": data.get("corner_regions", []) or [],
            "stamp_geometry": data.get("stamp_geometry", {}) or {},
            "font_hints": data.get("font_hints", {}) or {},
            "orientation": data.get("orientation", {}) or {},
            "raw": data,
        }
        return cls(**known)

    def to_dict(self) -> dict[str, Any]:
        if self.raw:
            # Preserve unknown keys from loaded calibration while syncing known fields.
            out = dict(self.raw)
        else:
            out = {}
        out.update(
            {
                "version": self.version,
                "photo_dir": self.photo_dir,
                "notes": self.notes,
                "image_stats": self.image_stats,
                "hsv_ranges": self.hsv_ranges,
                "corner_regions": self.corner_regions,
                "stamp_geometry": self.stamp_geometry,
                "font_hints": self.font_hints,
                "orientation": self.orientation,
            }
        )
        return out


@dataclass
class DetectionResult:
    found: bool
    method: str = "none"  # e.g. hsv_geometry, yolo_onnx, none
    corner: Optional[str] = None
    bbox: Optional[tuple[int, int, int, int]] = None  # x, y, w, h
    confidence: Optional[float] = None
    notes: list[str] = field(default_factory=list)


@dataclass
class DigitReading:
    raw_digits: str = ""
    per_digit_scores: list[float] = field(default_factory=list)
    digit_details: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class PhotoResult:
    filename: str
    source_path: Optional[Path] = None
    detection: DetectionResult = field(default_factory=lambda: DetectionResult(found=False))
    reading: DigitReading = field(default_factory=DigitReading)
    parsed_date: str = ""
    overall_confidence: float = 0.0
    classification: Classification = Classification.FAILED
    notes: list[str] = field(default_factory=list)
    crop_image_path: Optional[Path] = None

    def to_csv_row(self) -> dict[str, Any]:
        per_digit = ";".join(f"{s:.4f}" for s in self.reading.per_digit_scores)
        return {
            "filename": self.filename,
            "raw_digits_read": self.reading.raw_digits,
            "parsed_date": self.parsed_date,
            "per_digit_scores": per_digit,
            "overall_confidence": f"{self.overall_confidence:.4f}",
            "classification": self.classification.value,
            "notes": " | ".join(self.notes),
        }
