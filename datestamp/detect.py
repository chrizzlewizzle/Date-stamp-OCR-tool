"""Stage 1 stamp detection: HSV+geometry primary path with optional ONNX fallback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol

import cv2
import numpy as np

from .config import CALIBRATION_FILENAME, DetectionResult
from .io_utils import discover_images, read_csv_rows, read_json


# --- Fallback detector interface (pluggable) ---
class FallbackStampDetector(Protocol):
    name: str

    def detect(self, image_bgr: np.ndarray) -> DetectionResult:
        ...


@dataclass
class Candidate:
    corner: str
    bbox: tuple[int, int, int, int]  # x,y,w,h in full image
    score: float
    pixel_count: int
    component_count: int
    row_like: bool
    details: dict[str, Any]


class OnnxYoloStampDetector:
    """Optional fallback using pike00/yolo-date-stamp-detector exported to ONNX.

    Runtime is offline: the model file must be present locally (setup-time acquisition/export).
    """

    name = "yolo_onnx"

    def __init__(self, model_path: str | Path, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._session = None
        self._input_name = None
        self._input_shape = None

    def _ensure_loaded(self) -> bool:
        if self._session is not None:
            return True
        if not self.model_path.exists():
            return False
        try:
            import onnxruntime as ort  # type: ignore

            self._session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
            inp = self._session.get_inputs()[0]
            self._input_name = inp.name
            self._input_shape = inp.shape
            return True
        except Exception:
            self._session = None
            return False

    def detect(self, image_bgr: np.ndarray) -> DetectionResult:
        if not self._ensure_loaded():
            return DetectionResult(found=False, method="none", notes=["yolo_onnx model not available"])

        h0, w0 = image_bgr.shape[:2]
        # Common YOLO ONNX expects 640x640; infer if shape fixed, else default.
        size = 640
        if self._input_shape and len(self._input_shape) >= 4:
            try:
                if isinstance(self._input_shape[2], int) and self._input_shape[2] > 0:
                    size = int(self._input_shape[2])
            except Exception:
                pass
        img = cv2.resize(image_bgr, (size, size))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blob = img_rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        try:
            outputs = self._session.run(None, {self._input_name: blob})  # type: ignore[arg-type]
        except Exception as e:
            return DetectionResult(found=False, method="none", notes=[f"yolo_onnx inference failed: {e}"])

        boxes, scores = _parse_yolo_outputs(outputs, conf_threshold=self.conf_threshold, iou_threshold=self.iou_threshold)
        if not boxes:
            return DetectionResult(found=False, method=self.name, confidence=0.0, notes=["yolo_onnx found no stamp"])
        # pick highest score
        best_i = int(np.argmax(scores))
        x1, y1, x2, y2, conf = boxes[best_i]
        # scale back
        sx = w0 / float(size)
        sy = h0 / float(size)
        bx = max(0, int(x1 * sx))
        by = max(0, int(y1 * sy))
        bw = max(1, int((x2 - x1) * sx))
        bh = max(1, int((y2 - y1) * sy))
        bw = min(bw, w0 - bx)
        bh = min(bh, h0 - by)
        return DetectionResult(found=True, method=self.name, corner=None, bbox=(bx, by, bw, bh), confidence=float(conf), notes=["yolo_onnx fallback detection"])


def _parse_yolo_outputs(outputs: list[np.ndarray], conf_threshold: float, iou_threshold: float) -> tuple[list[tuple[float, float, float, float, float]], list[float]]:
    """Best-effort parser for common YOLO ONNX output layouts."""
    if not outputs:
        return [], []
    out = outputs[0]
    # Common shapes: (1,84,8400) or (1,8400,84/85) or (1,N,6)
    arr = np.array(out)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        return [], []
    if arr.shape[0] < arr.shape[1] and arr.shape[0] <= 128:
        arr = arr.T  # (N,C)
    n, c = arr.shape
    boxes_xyxy = []
    scores = []
    if c >= 6 and c <= 8:
        # [x1,y1,x2,y2,conf,(class)] or similar
        for row in arr:
            conf = float(row[4])
            if c > 6:
                # class score/product style
                cls_score = float(np.max(row[5:])) if c > 5 else 1.0
                conf = conf * cls_score if conf <= 1.5 else cls_score
            if conf < conf_threshold:
                continue
            x1, y1, x2, y2 = map(float, row[:4])
            boxes_xyxy.append([x1, y1, x2, y2])
            scores.append(conf)
    elif c >= 5:
        # YOLOv8 style: [cx,cy,w,h,class_scores...]
        cls = arr[:, 4:]
        if cls.size == 0:
            return [], []
        cls_scores = cls.max(axis=1)
        cls_ids = cls.argmax(axis=1) if cls.shape[1] > 1 else np.zeros((n,), dtype=int)
        confs = cls_scores
        for i in range(n):
            conf = float(confs[i])
            if conf < conf_threshold:
                continue
            cx, cy, w, h = map(float, arr[i, :4])
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            boxes_xyxy.append([x1, y1, x2, y2])
            scores.append(conf)
    else:
        return [], []

    if not boxes_xyxy:
        return [], []
    b = np.array(boxes_xyxy, dtype=np.float32)
    s = np.array(scores, dtype=np.float32)
    # NMS
    try:
        idxs = cv2.dnn.NMSBoxes(b.tolist(), s.tolist(), conf_threshold, iou_threshold)
    except Exception:
        idxs = list(range(len(b)))
    if idxs is None:
        return [], []
    if isinstance(idxs, np.ndarray):
        idxs = idxs.flatten().tolist()
    elif isinstance(idxs, (list, tuple)):
        idxs = [int(i[0]) if isinstance(i, (list, tuple, np.ndarray)) else int(i) for i in idxs]
    else:
        idxs = [int(idxs)]
    out_boxes = [(float(b[i,0]), float(b[i,1]), float(b[i,2]), float(b[i,3]), float(s[i])) for i in idxs]
    out_scores = [float(s[i]) for i in idxs]
    return out_boxes, out_scores


def load_calibration_data(path: str | Path = CALIBRATION_FILENAME) -> dict[str, Any]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid calibration data: {path}")
    return data


def _get_hsv_ranges(cal: dict[str, Any]) -> list[dict[str, int]]:
    ranges = cal.get("hsv_ranges") or []
    out = []
    for r in ranges:
        if not isinstance(r, dict):
            continue
        out.append({
            "h_min": int(r.get("h_min", 0)),
            "h_max": int(r.get("h_max", 179)),
            "s_min": int(r.get("s_min", 80)),
            "s_max": int(r.get("s_max", 255)),
            "v_min": int(r.get("v_min", 80)),
            "v_max": int(r.get("v_max", 255)),
        })
    if not out:
        out = [
            {"h_min": 0, "h_max": 35, "s_min": 80, "s_max": 255, "v_min": 80, "v_max": 255},
            {"h_min": 160, "h_max": 179, "s_min": 80, "s_max": 255, "v_min": 80, "v_max": 255},
        ]
    return out


def _corner_slices(h: int, w: int, y_frac: float, x_frac: float) -> dict[str, tuple[slice, slice, tuple[int, int]]]:
    yh = max(1, int(h * y_frac))
    xw = max(1, int(w * x_frac))
    return {
        "top_left": (slice(0, yh), slice(0, xw), (0, 0)),
        "top_right": (slice(0, yh), slice(w - xw, w), (w - xw, 0)),
        "bottom_left": (slice(h - yh, h), slice(0, xw), (0, h - yh)),
        "bottom_right": (slice(h - yh, h), slice(w - xw, w), (w - xw, h - yh)),
    }


def build_mask_from_ranges(hsv: np.ndarray, ranges: list[dict[str, int]]) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for r in ranges:
        lo = np.array([r["h_min"], r["s_min"], r["v_min"]], dtype=np.uint8)
        hi = np.array([r["h_max"], r["s_max"], r["v_max"]], dtype=np.uint8)
        m = cv2.inRange(hsv, lo, hi)
        mask = cv2.bitwise_or(mask, m)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _candidate_from_mask(mask: np.ndarray, corner: str, offset: tuple[int, int], img_shape: tuple[int, int, int]) -> Optional[Candidate]:
    h_img, w_img = img_shape[:2]
    ys, xs = np.where(mask > 0)
    if len(xs) < 60:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    if bw < 25 or bh < 10:
        return None
    sub = (mask[y0:y1 + 1, x0:x1 + 1] > 0).astype(np.uint8)
    num, _, stats, centroids = cv2.connectedComponentsWithStats(sub, connectivity=8)
    comp_count = 0
    comp_areas = []
    comp_centroids = []
    comp_hs = []
    comp_ws = []
    for i in range(1, num):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a >= 10:
            comp_count += 1
            comp_areas.append(a)
            comp_centroids.append((float(centroids[i][0]), float(centroids[i][1])))
            comp_hs.append(int(stats[i, cv2.CC_STAT_HEIGHT]))
            comp_ws.append(int(stats[i, cv2.CC_STAT_WIDTH]))
    if comp_count < 2:
        return None
    aspect = bw / float(bh) if bh else 0.0
    pixel_count = int(len(xs))
    fill_ratio = pixel_count / float(bw * bh) if bw * bh else 0.0

    # Row alignment heuristic: digit stamps form a mostly horizontal row/cluster
    row_like = False
    if comp_centroids and len(comp_centroids) >= 2:
        ys_c = np.array([c[1] for c in comp_centroids], dtype=np.float32)
        xs_c = np.array([c[0] for c in comp_centroids], dtype=np.float32)
        y_spread = float(np.std(ys_c))
        x_spread = float(np.max(xs_c) - np.min(xs_c)) if len(xs_c) else 0.0
        med_h = float(np.median(comp_hs)) if comp_hs else bh
        med_w = float(np.median(comp_ws)) if comp_ws else 1.0
        # components should have meaningful size relative to bbox (not tiny specks spread out)
        size_ok = med_h >= max(4.0, bh * 0.12) or med_w >= max(4.0, bw * 0.03)
        row_like = (
            size_ok
            and (x_spread >= max(bw * 0.25, med_h * 0.8))
            and (y_spread <= max(0.45 * med_h, 0.22 * bh))
        )
    # also require generally horizontal-ish extent
    if aspect < 0.8:
        row_like = False

    # Reject obvious non-stamp blobs (large flat regions, scattered noise)
    if fill_ratio > 0.80 and comp_count < 4:
        return None
    if bw > 0.85 * w_img or bh > 0.60 * h_img:
        return None

    score = float(pixel_count) * (1.0 + 0.12 * min(comp_count, 20))
    if row_like:
        score *= 1.6
    else:
        score *= 0.6
    if 1.0 <= aspect <= 25:
        score *= 1.15
    # prefer moderate component counts typical of date stamps (digits + separators/segments)
    if 3 <= comp_count <= 40:
        score *= 1.1

    ox, oy = offset
    fx, fy, fw, fh = ox + x0, oy + y0, bw, bh
    # pad for segmentation context
    pad_x = max(2, int(0.08 * fw))
    pad_y = max(2, int(0.20 * fh))
    fx = max(0, fx - pad_x)
    fy = max(0, fy - pad_y)
    fw = min(w_img - fx, fw + 2 * pad_x)
    fh = min(h_img - fy, fh + 2 * pad_y)
    details = {
        "aspect": aspect,
        "fill_ratio": fill_ratio,
        "component_areas_sample": comp_areas[:20],
        "component_count_raw": comp_count,
        "local_bbox": [int(x0), int(y0), int(bw), int(bh)],
    }
    return Candidate(corner=corner, bbox=(int(fx), int(fy), int(fw), int(fh)), score=score,
                     pixel_count=pixel_count, component_count=comp_count, row_like=row_like, details=details)


def detect_stamp_hsv(image_bgr: np.ndarray, cal: dict[str, Any], debug_dir: Optional[Path] = None, debug_name: Optional[str] = None) -> tuple[DetectionResult, Optional[np.ndarray]]:
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    ranges = _get_hsv_ranges(cal)

    # corner config
    y_frac = 0.35
    x_frac = 0.35
    cm = cal.get("measurement_method", {}).get("corner_region_fraction", {}) if isinstance(cal.get("measurement_method"), dict) else {}
    try:
        x_frac = float(cm.get("x", x_frac))
        y_frac = float(cm.get("y", y_frac))
    except Exception:
        pass
    # widen a bit for robustness vs calibration medians
    x_frac = min(0.5, max(0.2, x_frac * 1.15))
    y_frac = min(0.5, max(0.2, y_frac * 1.15))

    enabled = {c["name"] for c in cal.get("corner_regions", []) if c.get("enabled", True)} or set(CORNER_NAMES_DEFAULT := ("top_left","top_right","bottom_left","bottom_right"))
    slices = _corner_slices(h, w, y_frac, x_frac)
    candidates: list[Candidate] = []
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for corner, (yslc, xslc, offset) in slices.items():
        if corner not in enabled:
            continue
        region_hsv = hsv[yslc, xslc]
        mask = build_mask_from_ranges(region_hsv, ranges)
        # place into combined for debug
        combined_mask[yslc, xslc] = cv2.bitwise_or(combined_mask[yslc, xslc], mask)
        cand = _candidate_from_mask(mask, corner, offset, image_bgr.shape)
        if cand:
            candidates.append(cand)

    if debug_dir and debug_name:
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / f"{debug_name}.mask.png"), combined_mask)
        except Exception:
            pass

    if not candidates:
        return DetectionResult(found=False, method="hsv_geometry", notes=["no plausible stamp candidate found by HSV+geometry"]), None

    # prefer row-like plausible candidates, then score
    candidates.sort(key=lambda c: (1 if c.row_like else 0, c.component_count >= 3, c.score), reverse=True)
    best = candidates[0]
    # implausibility guards (avoid false positives on colored non-stamp regions)
    if not best.row_like and best.component_count < 4:
        return DetectionResult(found=False, method="hsv_geometry", notes=["HSV candidate not row-like/plausible"], confidence=0.0), None
    if best.pixel_count < 80:
        return DetectionResult(found=False, method="hsv_geometry", notes=["HSV candidate too small"], confidence=0.0), None

    conf = min(1.0, 0.15 + 0.85 * (min(best.score, 60000.0) / 60000.0))
    if not best.row_like:
        conf *= 0.5
    if best.component_count < 3:
        conf *= 0.6
    # Final acceptance threshold: prefer returning no detection over weak false positives.
    # Fallback detector (if available/configured) can be tried by caller when confidence is low.
    if conf < 0.35:
        return DetectionResult(found=False, method="hsv_geometry", corner=best.corner, bbox=best.bbox,
                               confidence=conf, notes=["HSV candidate rejected as low-confidence", f"components={best.component_count}", f"pixels={best.pixel_count}", f"row_like={best.row_like}"]), None

    x, y, bw, bh = best.bbox
    crop = image_bgr[y:y+bh, x:x+bw].copy() if bw > 0 and bh > 0 else None
    if debug_dir and debug_name and crop is not None:
        try:
            cv2.imwrite(str(debug_dir / f"{debug_name}.crop.png"), crop)
        except Exception:
            pass
    notes = [f"components={best.component_count}", f"pixels={best.pixel_count}", f"row_like={best.row_like}"]
    return DetectionResult(found=True, method="hsv_geometry", corner=best.corner, bbox=best.bbox, confidence=conf, notes=notes), crop


def detect_stamp(image_bgr: np.ndarray, cal: Optional[dict[str, Any]] = None, fallback: Optional[FallbackStampDetector] = None,
                 debug_dir: Optional[Path] = None, debug_name: Optional[str] = None) -> tuple[DetectionResult, Optional[np.ndarray]]:
    """Detect stamp using primary HSV+geometry, falling back if needed."""
    if cal is None:
        cal = load_calibration_data(CALIBRATION_FILENAME)
    result, crop = detect_stamp_hsv(image_bgr, cal, debug_dir=debug_dir, debug_name=debug_name)

    need_fallback = False
    if not result.found:
        need_fallback = True
    else:
        # implausible/weak result triggers fallback
        if result.confidence is not None and result.confidence < 0.45:
            need_fallback = True
        if result.bbox is not None:
            _, _, bw, bh = result.bbox
            if bw < 15 or bh < 8:
                need_fallback = True

    if need_fallback and fallback is not None:
        fb = fallback.detect(image_bgr)
        if fb.found and fb.bbox is not None:
            x, y, bw, bh = fb.bbox
            h, w = image_bgr.shape[:2]
            x = max(0, min(x, w - 1)); y = max(0, min(y, h - 1))
            bw = max(1, min(bw, w - x)); bh = max(1, min(bh, h - y))
            fb.bbox = (x, y, bw, bh)
            crop = image_bgr[y:y+bh, x:x+bw].copy()
            fb.notes = list(fb.notes or []) + ["fallback_used_after_hsv_miss_or_implausible"]
            return fb, crop
        # if fallback explicitly says no stamp, that's informative
        if fb.method == getattr(fallback, 'name', 'fallback'):
            result.notes = list(result.notes or []) + [f"fallback_{getattr(fallback,'name','model')}_no_result"]
    return result, crop


def _load_ground_truth_map(samples_dir: Path) -> dict[str, dict[str, str]]:
    gt_path = samples_dir / "ground_truth.csv"
    if not gt_path.exists():
        return {}
    try:
        rows = read_csv_rows(gt_path)
        return {r.get("filename", ""): r for r in rows}
    except Exception:
        return {}


def _is_gt_no_stamp(row: dict[str, str]) -> bool:
    status = (row.get("status") or "").strip().lower()
    date = (row.get("date") or "").strip().lower()
    digits = (row.get("digits") or "").strip().lower()
    return (
        status == "no_stamp"
        or "no_stamp" in status
        or "no stamp" in status
        or date in {"no_stamp", "no stamp"}
        or digits in {"no_stamp", "no stamp"}
    )


def evaluate_detection(photo_dir: Path, calibration_path: str | Path = CALIBRATION_FILENAME, debug_dir: Optional[Path] = None,
                       limit: Optional[int] = None, use_fallback: bool = True, model_path: Optional[str | Path] = None) -> dict[str, Any]:
    cal = load_calibration_data(calibration_path)
    images = discover_images(photo_dir, limit=limit)
    gt = _load_ground_truth_map(photo_dir)

    fallback = None
    if use_fallback:
        mp = Path(model_path) if model_path else Path("models/yolo-date-stamp-detector.onnx")
        det = OnnxYoloStampDetector(mp)
        if det._ensure_loaded():
            fallback = det
        else:
            fallback = det  # keep to report unavailability

    rows = []
    counts = {"found": 0, "not_found": 0}
    method_counts: dict[str, int] = {}
    corner_counts: dict[str, int] = {}
    no_stamp_cases = {"total": 0, "correct_no_detection": 0}
    for path in images:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            rows.append({"filename": path.name, "error": "unreadable"})
            continue
        name = path.stem if debug_dir else path.stem
        result, _ = detect_stamp(img, cal=cal, fallback=fallback, debug_dir=debug_dir, debug_name=name)
        method_counts[result.method] = method_counts.get(result.method, 0) + 1
        if result.found:
            counts["found"] += 1
            if result.corner:
                corner_counts[result.corner] = corner_counts.get(result.corner, 0) + 1
        else:
            counts["not_found"] += 1
        gt_row = gt.get(path.name, {})
        gt_status = gt_row.get("status", "")
        gt_no_stamp = _is_gt_no_stamp(gt_row)
        if gt_no_stamp:
            no_stamp_cases["total"] += 1
            if not result.found:
                no_stamp_cases["correct_no_detection"] += 1
        rows.append({
            "filename": path.name,
            "found": result.found,
            "method": result.method,
            "corner": result.corner,
            "bbox": result.bbox,
            "confidence": result.confidence,
            "notes": result.notes,
            "gt_status": gt_status,
            "gt_no_stamp": gt_no_stamp,
        })

    fallback_available = False
    if use_fallback:
        mp = Path(model_path) if model_path else Path("models/yolo-date-stamp-detector.onnx")
        fallback_available = OnnxYoloStampDetector(mp)._ensure_loaded()

    summary = {
        "photo_dir": str(photo_dir),
        "num_images": len(images),
        "counts": counts,
        "method_counts": method_counts,
        "corner_counts": corner_counts,
        "fallback_requested": use_fallback,
        "fallback_available": fallback_available,
        "no_stamp_cases": no_stamp_cases,
        "rows": rows,
    }
    return summary


def print_detection_eval(summary: dict[str, Any]) -> None:
    print(f"Detection evaluation: {summary.get('photo_dir')}")
    print(f"  Images: {summary.get('num_images', 0)}")
    c = summary.get('counts', {})
    print(f"  Found: {c.get('found', 0)}  Not found: {c.get('not_found', 0)}")
    print(f"  Methods: {summary.get('method_counts', {})}")
    print(f"  Corners: {summary.get('corner_counts', {})}")
    fb_req = summary.get('fallback_requested')
    fb_av = summary.get('fallback_available')
    print(f"  Fallback requested={fb_req} available={fb_av}")
    ns = summary.get('no_stamp_cases', {})
    print(f"  Ground-truth no_stamp: {ns.get('correct_no_detection', 0)}/{ns.get('total', 0)} correctly not detected")
    for r in summary.get('rows', []):
        if 'error' in r:
            print(f"  - {r['filename']}: ERROR {r['error']}")
            continue
        bbox = r.get('bbox')
        bbox_s = f"{bbox}" if bbox else "-"
        conf = r.get('confidence')
        conf_s = f"{conf:.3f}" if isinstance(conf, (int,float)) and conf is not None else "-"
        gt_no_stamp = bool(r.get('gt_no_stamp'))
        mark = ''
        if gt_no_stamp and r.get('found'):
            mark = ' [GT no_stamp but detected]'
        if gt_no_stamp and not r.get('found'):
            mark = ' [GT no_stamp ok]'
        print(f"  - {r['filename']}: found={r['found']} method={r['method']} corner={r.get('corner')} conf={conf_s} bbox={bbox_s}{mark}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate stamp detection on a photo directory")
    ap.add_argument("photo_dir")
    ap.add_argument("--calibration", default=CALIBRATION_FILENAME)
    ap.add_argument("--debug-dir", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-fallback", action="store_true")
    ap.add_argument("--model-path", default=None)
    args = ap.parse_args()
    summary = evaluate_detection(Path(args.photo_dir), calibration_path=args.calibration,
                                 debug_dir=Path(args.debug_dir) if args.debug_dir else None,
                                 limit=args.limit, use_fallback=not args.no_fallback, model_path=args.model_path)
    print_detection_eval(summary)

