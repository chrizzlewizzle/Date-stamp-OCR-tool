"""Stage 0 calibration: measure sample properties and persist calibration.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .config import CALIBRATION_FILENAME, Calibration
from .io_utils import discover_images, read_json, write_json


CORNER_NAMES = ("top_left", "top_right", "bottom_left", "bottom_right")


@dataclass
class ImageMeasurement:
    filename: str
    width: int
    height: int
    orientation: Optional[int]
    dpi: Optional[tuple[float, float]]
    corner_hits: dict[str, dict[str, Any]]
    best_corner: Optional[str]
    best_bbox_frac: Optional[dict[str, float]]
    hsv_samples: list[list[int]]
    font_hint: Optional[str]


def _corner_slices(h: int, w: int, y_frac: float = 0.35, x_frac: float = 0.35) -> dict[str, tuple[slice, slice]]:
    yh = max(1, int(h * y_frac))
    xw = max(1, int(w * x_frac))
    return {
        "top_left": (slice(0, yh), slice(0, xw)),
        "top_right": (slice(0, yh), slice(w - xw, w)),
        "bottom_left": (slice(h - yh, h), slice(0, xw)),
        "bottom_right": (slice(h - yh, h), slice(w - xw, w)),
    }


def _red_orange_mask(hsv: np.ndarray) -> np.ndarray:
    # Broad initial red/orange bands (OpenCV hue 0-179). Tuned from typical film stamps.
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    h = hsv[:, :, 0]
    sat_ok = s >= 80
    val_ok = v >= 80
    # red/orange/yellowish-red wrap: [0,35] and [160,179]
    hue_ok = ((h <= 35) | (h >= 160))
    mask = (hue_ok & sat_ok & val_ok).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _analyze_corner(region_bgr: np.ndarray, region_mask: np.ndarray) -> dict[str, Any]:
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    mask = _red_orange_mask(hsv) & (region_mask if region_mask is not None else 255)
    ys, xs = np.where(mask > 0)
    pixel_count = int(len(xs))
    info: dict[str, Any] = {"pixel_count": pixel_count}
    if pixel_count < 30:
        return info

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    area = bw * bh
    info.update({
        "bbox": [x0, y0, bw, bh],
        "bbox_area": int(area),
        "fill_ratio": float(pixel_count) / float(area) if area else 0.0,
        "aspect": float(bw) / float(bh) if bh else 0.0,
    })

    # Component stats inside bbox for digit-row plausibility
    sub = mask[y0:y1 + 1, x0:x1 + 1]
    num, labels, stats, centroids = cv2.connectedComponentsWithStats((sub > 0).astype(np.uint8), connectivity=8)
    comp_areas = []
    comp_boxes = []
    for i in range(1, num):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a >= 15:  # ignore specks
            comp_areas.append(a)
            comp_boxes.append([
                int(stats[i, cv2.CC_STAT_LEFT]),
                int(stats[i, cv2.CC_STAT_TOP]),
                int(stats[i, cv2.CC_STAT_WIDTH]),
                int(stats[i, cv2.CC_STAT_HEIGHT]),
            ])
    info["component_count"] = len(comp_areas)
    info["component_areas"] = comp_areas[:50]
    # Plausibility heuristic: date stamp tends to be a horizontal-ish row of small components
    if comp_boxes:
        widths = [b[2] for b in comp_boxes]
        heights = [b[3] for b in comp_boxes]
        info["median_comp_w"] = float(np.median(widths))
        info["median_comp_h"] = float(np.median(heights))
    info["row_like"] = bool(1 <= len(comp_areas) <= 40 and bw >= bh * 1.2)
    # Score favors compact, row-like clusters with moderate pixel counts
    score = pixel_count
    if info["row_like"]:
        score *= 1.5
    if 0.5 <= info.get("aspect", 0) <= 12:
        score *= 1.2
    info["score"] = float(score)

    # Sample HSV values for measured pixels (cap for memory)
    pts = np.column_stack((ys, xs))
    if len(pts) > 5000:
        idx = np.random.default_rng(0).choice(len(pts), 5000, replace=False)
        pts = pts[idx]
    info["hsv_samples"] = hsv[pts[:, 0], pts[:, 1]].astype(int).tolist()

    # Rough font hint: many thin/small components can indicate segmented strokes
    thin_parts = 0
    for _, _, cw, ch in comp_boxes:
        if cw <= 0 or ch <= 0:
            continue
        ar = max(cw, ch) / max(1, min(cw, ch))
        if ar >= 2.5 or min(cw, ch) <= max(2, int(0.25 * max(cw, ch))):
            thin_parts += 1
    if comp_boxes:
        info["thin_component_ratio"] = thin_parts / len(comp_boxes)
    return info


def _get_pil_meta(path: Path) -> tuple[Optional[int], Optional[tuple[float, float]]]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            exif = im.getexif() if hasattr(im, "getexif") else None
            orient = exif.get(274) if exif else None
            dpi = im.info.get("dpi")
            if dpi and isinstance(dpi, tuple) and len(dpi) >= 2:
                dpi = (float(dpi[0]), float(dpi[1]))
            else:
                dpi = None
            return orient, dpi
    except Exception:
        return None, None


def _font_hint_from_ratio(ratios: list[float]) -> dict[str, Any]:
    if not ratios:
        return {"likely_styles": [], "note": "insufficient data"}
    avg = float(np.mean(ratios))
    # Heuristic only; real determination refined during glyph/bootstrap stages.
    styles = []
    if avg >= 0.35:
        styles.append("possible_seven_segment_or_segmented_strokes")
    if avg <= 0.5:
        styles.append("possible_solid_strokes")
    if not styles:
        styles = ["unknown"]
    return {
        "avg_thin_component_ratio": avg,
        "likely_styles": styles,
        "note": "Heuristic from calibration masks; confirm during glyph inspection (samples may contain multiple camera fonts).",
    }


def measure_samples(photo_dir: Path, limit: Optional[int] = None) -> tuple[list[ImageMeasurement], dict[str, Any]]:
    images = discover_images(photo_dir, recursive=False, limit=limit)
    measurements: list[ImageMeasurement] = []
    all_hsv: list[list[int]] = []
    chosen_corners: list[str] = []
    bbox_fracs: list[dict[str, float]] = []
    thin_ratios: list[float] = []
    dims: list[tuple[int, int]] = []
    dpis: list[tuple[float, float]] = []
    orientations: list[Optional[int]] = []

    for path in images:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        dims.append((w, h))
        orient, dpi = _get_pil_meta(path)
        orientations.append(orient)
        if dpi:
            dpis.append(dpi)

        slices = _corner_slices(h, w)
        corner_hits: dict[str, dict[str, Any]] = {}
        best_corner = None
        best_score = -1.0
        best_bbox_frac = None
        hsv_samples: list[list[int]] = []
        img_thin = None

        for cname, (yslc, xslc) in slices.items():
            region = img[yslc, xslc]
            info = _analyze_corner(region, None)
            # convert bbox to full-image fractional coords if present
            if "bbox" in info:
                bx, by, bw, bh = info["bbox"]
                # offset by region origin
                y_off = 0 if yslc.start is None else yslc.start
                x_off = 0 if xslc.start is None else xslc.start
                fx = (x_off + bx) / float(w)
                fy = (y_off + by) / float(h)
                fw = bw / float(w)
                fh = bh / float(h)
                info["bbox_frac"] = {"x": fx, "y": fy, "w": fw, "h": fh}
            corner_hits[cname] = {k: v for k, v in info.items() if k != "hsv_samples"}
            score = float(info.get("score", 0.0))
            if score > best_score and info.get("pixel_count", 0) >= 30:
                best_score = score
                best_corner = cname
                best_bbox_frac = info.get("bbox_frac")
                hsv_samples = info.get("hsv_samples", [])
                if "thin_component_ratio" in info:
                    img_thin = float(info["thin_component_ratio"])

        if best_corner:
            chosen_corners.append(best_corner)
        if best_bbox_frac:
            bbox_fracs.append(best_bbox_frac)
        if hsv_samples:
            all_hsv.extend(hsv_samples)
        if img_thin is not None:
            thin_ratios.append(img_thin)

        measurements.append(
            ImageMeasurement(
                filename=path.name,
                width=w,
                height=h,
                orientation=orient,
                dpi=dpi,
                corner_hits=corner_hits,
                best_corner=best_corner,
                best_bbox_frac=best_bbox_frac,
                hsv_samples=hsv_samples,
                font_hint=None,
            )
        )

    summary = _summarize(measurements, all_hsv, chosen_corners, bbox_fracs, thin_ratios, dims, dpis, orientations)
    return measurements, summary


def _summarize(
    measurements: list[ImageMeasurement],
    all_hsv: list[list[int]],
    chosen_corners: list[str],
    bbox_fracs: list[dict[str, float]],
    thin_ratios: list[float],
    dims: list[tuple[int, int]],
    dpis: list[tuple[float, float]],
    orientations: list[Optional[int]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    summary["num_images_measured"] = len(measurements)
    summary["num_images_with_stamp_candidate"] = sum(1 for m in measurements if m.best_corner)

    if dims:
        ws = [d[0] for d in dims]
        hs = [d[1] for d in dims]
        summary["image_dimensions"] = {
            "width": {"min": int(min(ws)), "max": int(max(ws)), "median": float(np.median(ws))},
            "height": {"min": int(min(hs)), "max": int(max(hs)), "median": float(np.median(hs))},
        }

    if dpis:
        summary["dpi"] = {
            "x_median": float(np.median([d[0] for d in dpis])),
            "y_median": float(np.median([d[1] for d in dpis])),
            "unique_approx": sorted({(round(d[0], 1), round(d[1], 1)) for d in dpis}),
        }
    else:
        summary["dpi"] = {"note": "no DPI info found"}

    orient_counts: dict[str, int] = {}
    for o in orientations:
        k = str(o) if o is not None else "none"
        orient_counts[k] = orient_counts.get(k, 0) + 1
    summary["exif_orientation_tags"] = orient_counts

    if chosen_corners:
        counts: dict[str, int] = {}
        for c in chosen_corners:
            counts[c] = counts.get(c, 0) + 1
        summary["stamp_corner_counts"] = counts
        summary["likely_stamp_corners"] = sorted(counts, key=counts.get, reverse=True)

    if bbox_fracs:
        summary["stamp_bbox_fraction_median"] = {
            "x": float(np.median([b["x"] for b in bbox_fracs])),
            "y": float(np.median([b["y"] for b in bbox_fracs])),
            "w": float(np.median([b["w"] for b in bbox_fracs])),
            "h": float(np.median([b["h"] for b in bbox_fracs])),
        }

    if all_hsv:
        arr = np.array(all_hsv, dtype=np.uint8)
        # Percentile-based robust range; split hue into low/high red wrap bands if needed
        h = arr[:, 0]
        s = arr[:, 1]
        v = arr[:, 2]
        s_low, s_high = np.percentile(s, [5, 99]).astype(int).tolist()
        v_low, v_high = np.percentile(v, [5, 99]).astype(int).tolist()
        low_red = h[h <= 90]
        high_red = h[h >= 90]
        ranges = []
        if len(low_red) > 0:
            hl0, hl1 = np.percentile(low_red, [1, 99]).astype(int).tolist()
            ranges.append({"h_min": int(max(0, hl0 - 3)), "h_max": int(min(179, hl1 + 3)), "s_min": int(max(0, s_low - 10)), "s_max": 255, "v_min": int(max(0, v_low - 10)), "v_max": 255})
        if len(high_red) > 0:
            hh0, hh1 = np.percentile(high_red, [1, 99]).astype(int).tolist()
            ranges.append({"h_min": int(max(0, hh0 - 3)), "h_max": int(min(179, hh1 + 3)), "s_min": int(max(0, s_low - 10)), "s_max": 255, "v_min": int(max(0, v_low - 10)), "v_max": 255})
        if not ranges:
            ranges = [{"h_min": 0, "h_max": 35, "s_min": 80, "s_max": 255, "v_min": 80, "v_max": 255},
                      {"h_min": 160, "h_max": 179, "s_min": 80, "s_max": 255, "v_min": 80, "v_max": 255}]
        summary["hsv_ranges"] = ranges
        summary["hsv_percentiles"] = {
            "h": np.percentile(h, [1, 5, 50, 95, 99]).astype(int).tolist(),
            "s": np.percentile(s, [1, 5, 50, 95, 99]).astype(int).tolist(),
            "v": np.percentile(v, [1, 5, 50, 95, 99]).astype(int).tolist(),
        }
    else:
        summary["hsv_ranges"] = [
            {"h_min": 0, "h_max": 35, "s_min": 80, "s_max": 255, "v_min": 80, "v_max": 255},
            {"h_min": 160, "h_max": 179, "s_min": 80, "s_max": 255, "v_min": 80, "v_max": 255},
        ]

    summary["font_hints"] = _font_hint_from_ratio(thin_ratios)
    summary["notes"] = [
        "Measured from actual images using HSV red/orange masking in four corner regions.",
        "HSV ranges and geometry are initial calibrated defaults; refine if detection misses stamps.",
        "Samples may include multiple cameras/fonts and variable contrast; inspect glyphs/ and review outputs during later stages.",
    ]
    return summary


def build_calibration(photo_dir: Path, measurements: list[ImageMeasurement], summary: dict[str, Any]) -> dict[str, Any]:
    corners = summary.get("likely_stamp_corners", list(CORNER_NAMES))
    corner_regions = []
    for c in CORNER_NAMES:
        corner_regions.append({
            "name": c,
            "enabled": c in corners[:4],
            "x_frac": 0.35,
            "y_frac": 0.35,
            "priority": (corners.index(c) + 1) if c in corners else 99,
        })
    corner_regions.sort(key=lambda x: x["priority"])

    cal = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "photo_dir": str(photo_dir),
        "measurement_method": {
            "description": "HSV red/orange thresholding in four corner regions with connected-component row heuristics.",
            "corner_region_fraction": {"x": 0.35, "y": 0.35},
            "base_thresholds": {"hue": [[0, 35], [160, 179]], "s_min": 80, "v_min": 80},
            "morphology": {"open_kernel": [3, 3], "close_kernel": [3, 3], "close_iterations": 2},
        },
        "image_stats": {
            "num_images_measured": summary.get("num_images_measured", 0),
            "num_images_with_stamp_candidate": summary.get("num_images_with_stamp_candidate", 0),
            "dimensions": summary.get("image_dimensions"),
            "dpi": summary.get("dpi"),
            "exif_orientation_tags": summary.get("exif_orientation_tags", {}),
        },
        "hsv_ranges": summary.get("hsv_ranges", []),
        "corner_regions": corner_regions,
        "stamp_geometry": {
            "bbox_fraction_median": summary.get("stamp_bbox_fraction_median"),
            "stamp_corner_counts": summary.get("stamp_corner_counts", {}),
            "expected_digit_row": {"orientation": "horizontal", "typical_digit_count_guess": 6, "note": "Ground truth shows variable-length digit strings; segmentation must be flexible."},
        },
        "font_hints": summary.get("font_hints", {}),
        "orientation": {
            "check_all_four_corners": True,
            "exif_orientation_observed": summary.get("exif_orientation_tags", {}),
            "note": "No EXIF orientation tags observed in sampled files; detector should still check all four corners for rotated scans.",
        },
        "per_image": [
            {
                "filename": m.filename,
                "width": m.width,
                "height": m.height,
                "orientation": m.orientation,
                "dpi": m.dpi,
                "best_corner": m.best_corner,
                "best_bbox_frac": m.best_bbox_frac,
            }
            for m in measurements
        ],
        "notes": summary.get("notes", []),
    }
    return cal


def load_calibration(path: str | Path = CALIBRATION_FILENAME) -> Calibration:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid calibration format in {path}")
    return Calibration.from_dict(data)


def save_calibration(data: dict[str, Any], path: str | Path = CALIBRATION_FILENAME) -> None:
    write_json(path, data, indent=2)


def run_calibrate(photo_dir: Path) -> int:
    """Measure sample properties and write calibration.json."""
    if not photo_dir.exists():
        print(f"calibrate: photo directory not found: {photo_dir}")
        return 2
    if not photo_dir.is_dir():
        print(f"calibrate: not a directory: {photo_dir}")
        return 2

    try:
        images = discover_images(photo_dir)
    except Exception as e:
        print(f"calibrate: failed to discover images in {photo_dir}: {e}")
        return 2
    if not images:
        print(f"calibrate: no supported images found in {photo_dir}")
        return 2

    print(f"Calibrating from {len(images)} image(s) in: {photo_dir}")
    measurements, summary = measure_samples(photo_dir)
    cal = build_calibration(photo_dir, measurements, summary)
    out_path = Path(CALIBRATION_FILENAME)
    save_calibration(cal, out_path)

    # Human-readable log
    print(f"Wrote calibration: {out_path.resolve()}")
    print(f"  Images measured: {summary.get('num_images_measured', 0)}")
    print(f"  Stamp candidates found: {summary.get('num_images_with_stamp_candidate', 0)}")
    dims = summary.get("image_dimensions")
    if dims:
        print(f"  Dimensions (w x h median): {dims['width']['median']:.0f} x {dims['height']['median']:.0f}")
    dpi = summary.get("dpi")
    if dpi and "x_median" in dpi:
        print(f"  DPI median: {dpi['x_median']:.1f} x {dpi['y_median']:.1f}")
    print(f"  EXIF orientation tags: {summary.get('exif_orientation_tags', {})}")
    corners = summary.get("stamp_corner_counts", {})
    if corners:
        print(f"  Likely stamp corners (counts): {corners}")
    bbox = summary.get("stamp_bbox_fraction_median")
    if bbox:
        print(f"  Median stamp bbox fraction: x={bbox['x']:.3f}, y={bbox['y']:.3f}, w={bbox['w']:.3f}, h={bbox['h']:.3f}")
    hsv_ranges = summary.get("hsv_ranges", [])
    print(f"  HSV ranges ({len(hsv_ranges)}): {json.dumps(hsv_ranges)}")
    fh = summary.get("font_hints", {})
    if fh:
        print(f"  Font hints: avg_thin_component_ratio={fh.get('avg_thin_component_ratio')} styles={fh.get('likely_styles')}")
    for note in summary.get("notes", []):
        print(f"  Note: {note}")
    return 0

