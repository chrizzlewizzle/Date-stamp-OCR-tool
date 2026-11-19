"""Template matching recogniser.

Loads cached glyph templates from `glyphs/` and recognizes segmented digit crops
by template matching, producing per-digit scores and sequence readings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import cv2
import numpy as np

from .config import DigitReading, GLYPHS_DIRNAME
from .io_utils import read_json
from .segment import DigitComponent


@dataclass
class GlyphTemplates:
    glyphs_dir: Path
    # digit -> list of template images (uint8, normalized)
    templates: dict[str, list[np.ndarray]] = field(default_factory=dict)
    template_sources: dict[str, list[str]] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    missing_digits: list[str] = field(default_factory=list)
    present_digits: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, glyphs_dir: str | Path = GLYPHS_DIRNAME) -> "GlyphTemplates":
        gdir = Path(glyphs_dir)
        obj = cls(glyphs_dir=gdir)
        manifest_path = gdir / "manifest.json"
        if manifest_path.exists():
            try:
                data = read_json(manifest_path)
                if isinstance(data, dict):
                    obj.manifest = data
            except Exception as e:
                obj.notes.append(f"failed to read manifest.json: {e}")

        # Prefer directly-labeled exemplar images saved under glyphs/digits/
        digits_dir = gdir / "digits"
        loaded_any = False
        if digits_dir.exists() and digits_dir.is_dir():
            by_digit: dict[str, list[tuple[str, np.ndarray]]] = {str(d): [] for d in range(10)}
            for p in sorted(digits_dir.glob("digit_*_*.png")):
                name = p.name
                # expected: digit_{d}_{k}_{crop_id}.png
                parts = name.split("_")
                if len(parts) < 3 or parts[0] != "digit":
                    continue
                d = parts[1]
                if d not in by_digit:
                    continue
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                by_digit[d].append((str(p), img))
                loaded_any = True
            for d, items in by_digit.items():
                if items:
                    obj.templates[d] = [im for _, im in items]
                    obj.template_sources[d] = [src for src, _ in items]

        # Fallback: manifest digit_exemplars paths (may point to digits/ images)
        if not loaded_any and isinstance(obj.manifest.get("digit_exemplars"), dict):
            for d, info in obj.manifest.get("digit_exemplars", {}).items():
                if not str(d).isdigit():
                    continue
                paths = []
                if isinstance(info, dict):
                    paths = info.get("exemplar_images") or []
                imgs = []
                srcs = []
                for rel in paths:
                    p = gdir / str(rel)
                    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    imgs.append(img)
                    srcs.append(str(p))
                if imgs:
                    obj.templates[str(d)] = imgs
                    obj.template_sources[str(d)] = srcs
                    loaded_any = True

        # Additional fallback: use cluster representatives for clusters assigned to digits
        if not loaded_any and isinstance(obj.manifest.get("clusters"), dict):
            for _, cinfo in obj.manifest.get("clusters", {}).items():
                if not isinstance(cinfo, dict):
                    continue
                label = cinfo.get("assigned_label")
                if label is None:
                    continue
                d = str(label)
                if not d.isdigit():
                    continue
                rel = cinfo.get("representative_saved_as") or cinfo.get("representative_image")
                if not rel:
                    continue
                p = gdir / str(rel)
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                obj.templates.setdefault(d, []).append(img)
                obj.template_sources.setdefault(d, []).append(str(p))
                loaded_any = True

        obj.present_digits = sorted([d for d, imgs in obj.templates.items() if imgs], key=lambda x: int(x))
        obj.missing_digits = [str(d) for d in range(10) if str(d) not in obj.templates]
        if not obj.present_digits:
            obj.notes.append("no digit templates loaded from glyph cache")
        if obj.missing_digits:
            obj.notes.append(f"missing glyph templates for digits: {','.join(obj.missing_digits)}")
        return obj

    def coverage_report(self) -> dict[str, Any]:
        return {
            "glyphs_dir": str(self.glyphs_dir),
            "digits_present": list(self.present_digits),
            "digits_missing": list(self.missing_digits),
            "num_templates_per_digit": {d: len(v) for d, v in sorted(self.templates.items(), key=lambda kv: int(kv[0]))},
            "notes": list(self.notes),
        }


def _prepare_image(img: np.ndarray, size: Optional[tuple[int, int]] = None) -> Optional[np.ndarray]:
    if img is None or img.size == 0:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = img
    if size is not None and (out.shape[0] != size[0] or out.shape[1] != size[1]):
        out = cv2.resize(out, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    # binarize/normalize for stable comparison
    _, out = cv2.threshold(out, 127, 255, cv2.THRESH_BINARY)
    return out


def _score_pair(a: np.ndarray, b: np.ndarray) -> float:
    # both uint8 binary/grayscale, same shape
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        _, b = cv2.threshold(b, 127, 255, cv2.THRESH_BINARY)
    fa = (a.astype(np.float32) / 255.0).reshape(-1)
    fb = (b.astype(np.float32) / 255.0).reshape(-1)
    # normalized correlation-like score mapped to [0,1]
    na = np.linalg.norm(fa)
    nb = np.linalg.norm(fb)
    if na < 1e-6 and nb < 1e-6:
        return 1.0
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    cos = float(np.dot(fa, fb) / (na * nb))
    cos = max(-1.0, min(1.0, cos))
    # also consider pixel agreement (IoU-like on binary)
    aa = fa > 0.5
    bb = fb > 0.5
    inter = float(np.logical_and(aa, bb).sum())
    union = float(np.logical_or(aa, bb).sum())
    iou = (inter / union) if union > 0 else 1.0
    score = 0.65 * ((cos + 1.0) / 2.0) + 0.35 * iou
    return float(max(0.0, min(1.0, score)))


def recognize_digit_image(
    img: np.ndarray,
    templates: GlyphTemplates,
    top_k: int = 3,
) -> dict[str, Any]:
    """Recognize one normalized digit image against cached templates.

    Returns a dict with best digit, score, alternates, and per-class best scores.
    """
    if img is None or img.size == 0:
        return {
            "digit": None,
            "score": 0.0,
            "alternates": [],
            "class_scores": {},
            "notes": ["empty digit image"],
        }
    # determine reference size from first available template
    ref_size = (img.shape[0], img.shape[1])
    for imgs in templates.templates.values():
        if imgs:
            ref_size = (imgs[0].shape[0], imgs[0].shape[1])
            break
    prep = _prepare_image(img, size=ref_size)
    if prep is None:
        return {"digit": None, "score": 0.0, "alternates": [], "class_scores": {}, "notes": ["unreadable digit image"]}

    class_scores: dict[str, float] = {}
    class_best_src: dict[str, str] = {}
    for d, timgs in templates.templates.items():
        best_s = -1.0
        best_src = ""
        for j, t in enumerate(timgs):
            tt = _prepare_image(t, size=ref_size)
            if tt is None:
                continue
            s = _score_pair(prep, tt)
            if s > best_s:
                best_s = s
                srcs = templates.template_sources.get(d, [])
                best_src = srcs[j] if j < len(srcs) else ""
        if best_s >= 0:
            class_scores[d] = float(best_s)
            if best_src:
                class_best_src[d] = best_src

    if not class_scores:
        return {
            "digit": None,
            "score": 0.0,
            "alternates": [],
            "class_scores": {},
            "notes": ["no templates available for matching"],
        }

    ranked = sorted(class_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    best_digit, best_score = ranked[0]
    alts = []
    for d, s in ranked[1 : max(1, top_k)]:
        alts.append({"digit": d, "score": float(s)})
    margin = float(best_score - ranked[1][1]) if len(ranked) > 1 else float(best_score)
    return {
        "digit": best_digit,
        "score": float(best_score),
        "margin": margin,
        "alternates": alts,
        "class_scores": {k: float(v) for k, v in class_scores.items()},
        "best_template": class_best_src.get(best_digit),
        "notes": [],
    }


def recognize_components(
    components: Sequence[DigitComponent] | Sequence[np.ndarray],
    templates: GlyphTemplates,
    top_k: int = 3,
) -> DigitReading:
    """Recognize an ordered sequence of digit components/images into a DigitReading."""
    reading = DigitReading()
    if not components:
        reading.notes.append("no digit components to recognize")
        return reading

    if templates.missing_digits:
        reading.notes.append(f"glyph coverage warning: missing digits {','.join(templates.missing_digits)}")

    chars: list[str] = []
    scores: list[float] = []
    details: list[dict[str, Any]] = []
    for idx, comp in enumerate(components):
        if isinstance(comp, DigitComponent):
            img = comp.image
            meta = {
                "index": comp.index,
                "bbox": comp.bbox,
                "kind": comp.kind,
            }
        else:
            img = comp  # type: ignore[assignment]
            meta = {"index": idx}
        rec = recognize_digit_image(img, templates, top_k=top_k)
        digit = rec.get("digit")
        score = float(rec.get("score") or 0.0)
        ch = str(digit) if digit is not None else "?"
        chars.append(ch)
        scores.append(score)
        detail = {
            **meta,
            "recognized": ch,
            "score": score,
            "margin": rec.get("margin"),
            "alternates": rec.get("alternates", []),
        }
        if rec.get("best_template"):
            detail["best_template"] = rec.get("best_template")
        if rec.get("notes"):
            detail["notes"] = rec.get("notes")
        details.append(detail)
        if digit is None:
            reading.notes.append(f"digit {idx}: unrecognized")

    reading.raw_digits = "".join(chars)
    reading.per_digit_scores = scores
    reading.digit_details = details
    return reading


def glyph_coverage_warning(glyphs_dir: str | Path = GLYPHS_DIRNAME) -> dict[str, Any]:
    """Check cached glyph set and report which digits 0-9 are missing."""
    templates = GlyphTemplates.load(glyphs_dir)
    report = templates.coverage_report()
    missing = report.get("digits_missing") or []
    if missing:
        report["warning"] = f"Missing reference glyphs for digits: {', '.join(missing)}"
    else:
        report["warning"] = ""
    return report


def format_coverage_warning(report: dict[str, Any]) -> str:
    missing = report.get("digits_missing") or []
    if missing:
        return f"WARNING: missing reference glyphs for digits 0-9: {', '.join(missing)}"
    present = report.get("digits_present") or []
    if present:
        return f"Glyph coverage OK (digits present: {', '.join(present)})"
    return "WARNING: no glyph templates found"
