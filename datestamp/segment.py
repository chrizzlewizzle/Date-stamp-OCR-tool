"""Digit segmentation utilities.

Segments a detected stamp crop into ordered digit glyph candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

from .detect import build_mask_from_ranges, load_calibration_data, _get_hsv_ranges


@dataclass
class DigitComponent:
    """A single ordered glyph candidate extracted from a stamp crop."""

    index: int
    bbox: tuple[int, int, int, int]  # x, y, w, h in crop coordinates
    image: np.ndarray  # normalized grayscale/binary image (fixed size)
    mask: np.ndarray  # normalized mask image (fixed size)
    kind: str = "digit"  # digit | separator | unknown
    confidence: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentationResult:
    components: list[DigitComponent] = field(default_factory=list)
    digit_components: list[DigitComponent] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def digit_string(self) -> str:
        # placeholder; real labeling happens later
        return ""


def _normalize_glyph(binary: np.ndarray, out_h: int = 64, out_w: int = 40, pad: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a binary glyph (foreground=255) to a fixed-size canvas."""
    if binary.size == 0:
        canvas = np.zeros((out_h, out_w), dtype=np.uint8)
        return canvas, canvas.copy()
    ys, xs = np.where(binary > 0)
    if len(xs) == 0 or len(ys) == 0:
        canvas = np.zeros((out_h, out_w), dtype=np.uint8)
        return canvas, canvas.copy()
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    glyph = binary[y0 : y1 + 1, x0 : x1 + 1]
    gh, gw = glyph.shape[:2]
    target_h = max(1, out_h - 2 * pad)
    target_w = max(1, out_w - 2 * pad)
    scale = min(target_w / max(1, gw), target_h / max(1, gh))
    nw = max(1, int(round(gw * scale)))
    nh = max(1, int(round(gh * scale)))
    resized = cv2.resize(glyph, (nw, nh), interpolation=cv2.INTER_NEAREST if scale >= 1 else cv2.INTER_AREA)
    # binarize again after resize
    _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    canvas = np.zeros((out_h, out_w), dtype=np.uint8)
    y_off = (out_h - nh) // 2
    x_off = (out_w - nw) // 2
    canvas[y_off : y_off + nh, x_off : x_off + nw] = resized
    return canvas, canvas.copy()


def _group_components_into_glyphs(
    comps: list[dict[str, Any]], max_gap: Optional[float] = None
) -> list[list[dict[str, Any]]]:
    """Group connected components (possibly bars of seven-segment digits) into glyph clusters by x proximity."""
    if not comps:
        return []
    comps = sorted(comps, key=lambda c: (c["x"], c["cx"]))
    heights = [c["h"] for c in comps]
    med_h = float(np.median(heights)) if heights else 1.0
    if max_gap is None:
        max_gap = max(3.0, 0.35 * med_h)
    groups: list[list[dict[str, Any]]] = []
    cur = [comps[0]]
    cur_right = comps[0]["x"] + comps[0]["w"]
    for c in comps[1:]:
        # overlap or small horizontal gap relative to digit height -> same glyph group
        if c["x"] <= cur_right + max_gap:
            cur.append(c)
            cur_right = max(cur_right, c["x"] + c["w"])
        else:
            groups.append(cur)
            cur = [c]
            cur_right = c["x"] + c["w"]
    groups.append(cur)
    return groups


def segment_stamp_crop(
    crop_bgr: np.ndarray,
    cal: Optional[dict[str, Any]] = None,
    expected_count_hint: Optional[int] = None,
) -> SegmentationResult:
    """Segment a stamp crop into ordered glyph components.

    The input crop is typically coarse (from detection); this function re-masks
    and extracts a plausible horizontal row of digit-like components.
    """
    result = SegmentationResult()
    if crop_bgr is None or crop_bgr.size == 0:
        result.notes.append("empty crop")
        return result
    if cal is None:
        try:
            cal = load_calibration_data()
        except Exception:
            cal = {}

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    ranges = _get_hsv_ranges(cal)
    mask = build_mask_from_ranges(hsv, ranges)
    result.debug["mask_pixels"] = int(np.count_nonzero(mask))
    if result.debug["mask_pixels"] < 20:
        result.notes.append("insufficient mask pixels in crop")
        return result

    num, labels, stats, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    h_img, w_img = mask.shape[:2]
    raw_comps: list[dict[str, Any]] = []
    for i in range(1, num):
        x, y, w, h, area = [int(v) for v in stats[i]]
        if area < 10:
            continue
        # reject components spanning almost entire crop (background regions)
        if w >= 0.95 * w_img and h >= 0.4 * h_img:
            continue
        if h >= 0.95 * h_img and w >= 0.4 * w_img:
            continue
        if area >= 0.35 * h_img * w_img:
            continue
        # reject very large solid-ish blobs unlikely to be single characters/bars
        if w > 0.6 * w_img or h > 0.6 * h_img:
            continue
        raw_comps.append(
            {
                "label": i,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area": area,
                "cx": float(centroids[i][0]),
                "cy": float(centroids[i][1]),
            }
        )

    if not raw_comps:
        result.notes.append("no connected components after filtering")
        return result

    hs_all = np.array([c["h"] for c in raw_comps], dtype=np.float32)
    global_med_h = float(np.median(hs_all)) if len(hs_all) else 1.0
    candidates = [c for c in raw_comps if c["h"] >= max(3, 0.08 * global_med_h) and c["w"] >= 2 and c["h"] >= 3]
    if len(candidates) < 2:
        candidates = list(raw_comps)

    def v_overlap_ratio(a: dict[str, Any], b: dict[str, Any]) -> float:
        ay0, ay1 = a["y"], a["y"] + a["h"]
        by0, by1 = b["y"], b["y"] + b["h"]
        inter = max(0, min(ay1, by1) - max(ay0, by0))
        return inter / float(max(1, min(a["h"], b["h"])))

    def h_gap(a: dict[str, Any], b: dict[str, Any]) -> int:
        if a["x"] > b["x"]:
            a, b = b, a
        return max(0, b["x"] - (a["x"] + a["w"]))

    def components_close(a: dict[str, Any], b: dict[str, Any], ref_h: float) -> bool:
        gap = h_gap(a, b)
        vo = v_overlap_ratio(a, b)
        cy_d = abs(a["cy"] - b["cy"])
        # allow chaining across a row: either overlap vertically or share baseline band
        aligned = vo >= 0.2 or cy_d <= max(6.0, 0.8 * ref_h)
        # horizontal proximity: digits/bars within a stamp row are fairly close
        near = gap <= max(10.0, 2.2 * ref_h, 0.5 * max(a["w"], b["w"]))
        return bool(aligned and near)

    def build_clusters(comps: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        comps_sorted = sorted(comps, key=lambda c: (c["cx"], c["cy"]))
        used = [False] * len(comps_sorted)
        clusters: list[list[dict[str, Any]]] = []
        for i in range(len(comps_sorted)):
            if used[i]:
                continue
            stack = [i]
            used[i] = True
            cluster_idx: list[int] = []
            while stack:
                j = stack.pop()
                cluster_idx.append(j)
                cluster = [comps_sorted[k] for k in cluster_idx]
                ref_h = float(np.median([c["h"] for c in cluster])) if cluster else global_med_h
                cj = comps_sorted[j]
                for k in range(len(comps_sorted)):
                    if used[k]:
                        continue
                    ck = comps_sorted[k]
                    if components_close(cj, ck, ref_h) or components_close(ck, cj, ref_h):
                        # quick extra check to avoid chaining distant parallel noise
                        ccy = float(np.median([c["cy"] for c in cluster]))
                        if abs(ck["cy"] - ccy) <= max(10.0, 1.4 * ref_h):
                            used[k] = True
                            stack.append(k)
            clusters.append([comps_sorted[k] for k in cluster_idx])
        return clusters

    clusters = build_clusters(candidates)
    if not clusters and candidates is not raw_comps:
        clusters = build_clusters(raw_comps)

    def cluster_score(cluster: list[dict[str, Any]]) -> tuple:
        if not cluster:
            return (-1e9,)
        xs = [c["x"] for c in cluster]
        ys = [c["y"] for c in cluster]
        xe = [c["x"] + c["w"] for c in cluster]
        ye = [c["y"] + c["h"] for c in cluster]
        bw = max(1, max(xe) - min(xs))
        bh = max(1, max(ye) - min(ys))
        hs = np.array([c["h"] for c in cluster], dtype=np.float32)
        ws = np.array([c["w"] for c in cluster], dtype=np.float32)
        areas = np.array([c["area"] for c in cluster], dtype=np.float32)
        cys = np.array([c["cy"] for c in cluster], dtype=np.float32)
        med_h = float(np.median(hs))
        med_w = float(np.median(ws))
        count = len(cluster)
        crop_area = float(h_img * w_img)
        area_frac = (bw * bh) / crop_area
        aspect_row = bw / float(bh)

        # prefer moderate number of components (digits + segments/bars), not 1 giant and not 100 noise
        count_ok = 1.0
        if count < 3:
            count_ok = 0.25
        elif count > 80:
            count_ok = 0.2
        elif count > 40:
            count_ok = 0.6

        # row-like: wider than tall, limited vertical spread
        row_like = 1.0 if aspect_row >= 1.2 else 0.4
        y_spread = float(np.std(cys)) / (med_h + 1e-6)
        align_ok = max(0.1, 1.0 - min(1.0, y_spread / 1.5))

        # component sizes should be in a reasonable range (digit/bar sized)
        med_area = float(np.median(areas))
        # penalize if cluster bbox is huge portion of crop unless very text-like
        area_pen = 1.0
        if area_frac > 0.7:
            area_pen = 0.15
        elif area_frac > 0.4:
            area_pen = 0.5

        # exclude clusters dominated by tiny specks
        tiny_frac = float(np.mean((hs < max(4, 0.25 * med_h)) & (areas < max(20, 0.2 * med_area))))
        tiny_pen = max(0.2, 1.0 - 0.8 * tiny_frac)

        score = (1.5 * min(count, 30) + 2.0 * align_ok + 1.0 * row_like) * area_pen * tiny_pen * count_ok
        # prefer clusters not at extreme tiny sizes
        return (score, -area_frac, -y_spread, count)

    best_cluster: list[dict[str, Any]] = []
    if clusters:
        clusters_sorted = sorted(clusters, key=cluster_score, reverse=True)
        best_cluster = list(clusters_sorted[0])
        # try a better one if best is obviously bad (e.g. too big)
        if best_cluster:
            bx0 = min(c["x"] for c in best_cluster); by0 = min(c["y"] for c in best_cluster)
            bx1 = max(c["x"]+c["w"] for c in best_cluster); by1 = max(c["y"]+c["h"] for c in best_cluster)
            if (bx1-bx0)*(by1-by0) > 0.75*h_img*w_img and len(clusters_sorted) > 1:
                # pick first that is not huge
                for cl in clusters_sorted[1:5]:
                    xs=[c["x"] for c in cl]; ys=[c["y"] for c in cl]; xe=[c["x"]+c["w"] for c in cl]; ye=[c["y"]+c["h"] for c in cl]
                    if (max(xe)-min(xs))*(max(ye)-min(ys)) <= 0.75*h_img*w_img and len(cl) >= 2:
                        best_cluster = list(cl)
                        break
    if not best_cluster:
        result.notes.append("no plausible component cluster found")
        return result

    # tighten to likely digit row: drop outlier components far in y from main band
    cys = np.array([c["cy"] for c in best_cluster], dtype=np.float32)
    hs = np.array([c["h"] for c in best_cluster], dtype=np.float32)
    med_h = float(np.median(hs)) if len(hs) else 1.0
    med_cy = float(np.median(cys)) if len(cys) else h_img / 2.0
    refined = [c for c in best_cluster if abs(c["cy"] - med_cy) <= max(8.0, 1.3 * med_h)]
    if len(refined) >= 2:
        best_cluster = refined
    # drop extreme size outliers (very large blobs relative to rest)
    if len(best_cluster) >= 3:
        areas = np.array([c["area"] for c in best_cluster], dtype=np.float32)
        med_a = float(np.median(areas))
        kept = [c for c in best_cluster if c["area"] <= max(med_a * 25, med_a + 20000)]
        if len(kept) >= 2:
            best_cluster = kept

    groups = _group_components_into_glyphs(best_cluster)
    result.debug["num_raw_components"] = len(raw_comps)
    result.debug["num_selected_components"] = len(best_cluster)
    result.debug["num_groups"] = len(groups)

    components: list[DigitComponent] = []
    for gi, group in enumerate(groups):
        gx0 = min(c["x"] for c in group)
        gy0 = min(c["y"] for c in group)
        gx1 = max(c["x"] + c["w"] for c in group)
        gy1 = max(c["y"] + c["h"] for c in group)
        gw = max(1, gx1 - gx0)
        gh = max(1, gy1 - gy0)
        glyph_mask = np.zeros((gh, gw), dtype=np.uint8)
        for c in group:
            comp_mask = ((labels[c["y"] : c["y"] + c["h"], c["x"] : c["x"] + c["w"]] == c["label"]).astype(np.uint8) * 255)
            ox = c["x"] - gx0
            oy = c["y"] - gy0
            glyph_mask[oy : oy + c["h"], ox : ox + c["w"]] = cv2.bitwise_or(
                glyph_mask[oy : oy + c["h"], ox : ox + c["w"]], comp_mask
            )
        norm_img, norm_mask = _normalize_glyph(glyph_mask, out_h=64, out_w=40, pad=4)

        # classify as digit-like vs separator-like
        aspect = gw / float(gh) if gh else 0.0
        area = int(np.count_nonzero(glyph_mask))
        group_h_med = float(np.median([c["h"] for c in group]))
        kind = "digit"
        # separators often very thin/wide or very small relative to row height
        if gh < 0.28 * band_ref_h(best_cluster) and aspect > 1.2:
            kind = "separator"
        elif gw < 0.12 * group_h_med and gh < 0.5 * group_h_med:
            kind = "separator"
        elif aspect > 2.8 and gh < 0.6 * group_h_med:
            kind = "separator"

        comp = DigitComponent(
            index=gi,
            bbox=(int(gx0), int(gy0), int(gw), int(gh)),
            image=norm_img,
            mask=norm_mask,
            kind=kind,
            confidence=1.0,
            meta={
                "group_size": len(group),
                "area": area,
                "aspect": aspect,
                "raw_bbox": [int(gx0), int(gy0), int(gw), int(gh)],
            },
        )
        components.append(comp)

    # post-process: if we got way too many groups, try to keep digit-like ones by size
    components.sort(key=lambda c: c.bbox[0])
    for i, c in enumerate(components):
        c.index = i
    result.components = components
    result.digit_components = [c for c in components if c.kind == "digit"]

    # Fallback: projection-based segmentation if result looks poor or doesn't match expected count.
    need_projection = False
    if not result.digit_components:
        need_projection = True
    elif expected_count_hint is not None and len(result.digit_components) != expected_count_hint:
        need_projection = True
    elif len(result.digit_components) <= 1 or len(result.digit_components) > 12:
        need_projection = True
    if need_projection:
        proj = _segment_by_projection(crop_bgr, mask, labels, best_cluster, expected_count_hint=expected_count_hint)
        if proj and proj.digit_components:
            cur_n = len(result.digit_components)
            proj_n = len(proj.digit_components)
            choose_proj = False
            if cur_n == 0:
                choose_proj = True
            elif expected_count_hint is not None:
                if abs(proj_n - expected_count_hint) < abs(cur_n - expected_count_hint):
                    choose_proj = True
                elif abs(proj_n - expected_count_hint) == abs(cur_n - expected_count_hint) and 2 <= proj_n <= 10:
                    choose_proj = True
            else:
                if (cur_n <= 1 or cur_n > 12) and 2 <= proj_n <= 10:
                    choose_proj = True
            if choose_proj:
                proj.notes = result.notes + ["used projection-based segmentation fallback"] + proj.notes
                proj.debug = {**result.debug, **{f"proj_{k}": v for k, v in proj.debug.items()}}
                result = proj

    # if expected count provided and separators present, refine
    if expected_count_hint and len(result.digit_components) != expected_count_hint and result.components:
        result.notes.append(
            f"digit component count {len(result.digit_components)} differs from hint {expected_count_hint}"
        )
    if not result.digit_components and result.components:
        # fallback: treat all as digits
        for c in result.components:
            c.kind = "digit"
        result.digit_components = list(result.components)
        result.notes.append("no digit-classified components; treated all components as digits")
    if not result.notes:
        result.notes.append(f"segmented {len(result.digit_components)} digit components from {len(groups)} groups")
    elif not any("segmented" in n for n in result.notes):
        result.notes.append(f"segmented {len(result.digit_components)} digit components")
    return result


def _segment_by_projection(
    crop_bgr: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    seed_cluster: list[dict[str, Any]],
    expected_count_hint: Optional[int] = None,
) -> Optional[SegmentationResult]:
    """Fallback segmentation by tightening to ink band and splitting on x-projection valleys."""
    h_img, w_img = mask.shape[:2]
    res = SegmentationResult()
    if mask is None or mask.size == 0:
        return None

    # Determine ROI: prefer seed cluster bbox, else mask bbox
    if seed_cluster:
        xs = [c["x"] for c in seed_cluster]
        ys = [c["y"] for c in seed_cluster]
        xe = [c["x"] + c["w"] for c in seed_cluster]
        ye = [c["y"] + c["h"] for c in seed_cluster]
        rx0, ry0, rx1, ry1 = min(xs), min(ys), max(xe), max(ye)
        # pad
        pad_x = max(2, int(0.15 * (rx1 - rx0 + 1)))
        pad_y = max(2, int(0.35 * (ry1 - ry0 + 1)))
        rx0 = max(0, rx0 - pad_x); ry0 = max(0, ry0 - pad_y)
        rx1 = min(w_img, rx1 + pad_x); ry1 = min(h_img, ry1 + pad_y)
    else:
        ys_m, xs_m = np.where(mask > 0)
        if len(xs_m) == 0:
            return None
        rx0, rx1 = int(xs_m.min()), int(xs_m.max()) + 1
        ry0, ry1 = int(ys_m.min()), int(ys_m.max()) + 1

    roi_mask = (mask[ry0:ry1, rx0:rx1] > 0).astype(np.uint8)
    rh, rw = roi_mask.shape[:2]
    if rh < 5 or rw < 5:
        return None

    # Clean ROI mask by removing components that are too large/small extremes? keep as-is mostly
    # Find tighter vertical band using y projection
    y_proj = roi_mask.sum(axis=1).astype(np.float32)
    if y_proj.max() <= 0:
        return None
    y_thresh = max(1.0, 0.08 * float(y_proj.max()))
    y_idx = np.where(y_proj >= y_thresh)[0]
    if len(y_idx) == 0:
        return None
    y0b, y1b = int(y_idx[0]), int(y_idx[-1]) + 1
    # add small padding
    y0b = max(0, y0b - 1); y1b = min(rh, y1b + 2)
    band = roi_mask[y0b:y1b, :]
    bh, bw = band.shape[:2]
    if bh < 3 or bw < 3:
        return None

    # X projection and split
    x_proj = band.sum(axis=0).astype(np.float32)
    # smooth
    k = max(3, int(round(bh * 0.08)) * 2 + 1)
    if k >= 3 and bw >= k:
        kernel = np.ones(k, dtype=np.float32) / k
        x_smooth = np.convolve(x_proj, kernel, mode="same")
    else:
        x_smooth = x_proj
    max_x = float(x_smooth.max()) if x_smooth.size else 0.0
    if max_x <= 0:
        return None

    # threshold for ink presence; valleys below this can be splits
    ink_thresh = max(1.0, 0.12 * max_x)
    ink = x_smooth >= ink_thresh
    segments: list[tuple[int, int]] = []
    in_seg = False
    s = 0
    for i, v in enumerate(ink):
        if v and not in_seg:
            in_seg = True; s = i
        elif not v and in_seg:
            in_seg = False; segments.append((s, i))
    if in_seg:
        segments.append((s, bw))

    # merge very small gaps/segments heuristically
    if segments:
        merged = [segments[0]]
        min_gap = max(1, int(0.06 * bw), int(0.15 * bh))
        min_seg_w = max(1, int(0.04 * bw), 2)
        for a, b in segments[1:]:
            pa, pb = merged[-1]
            gap = a - pb
            if gap < min_gap:
                merged[-1] = (pa, b)
            else:
                merged.append((a, b))
        segments = [(a, b) for a, b in merged if (b - a) >= min_seg_w]

    # if still far from expected, try alternative threshold
    def try_split(thr_ratio: float, min_gap_ratio: float) -> list[tuple[int, int]]:
        ink_t = max(1.0, thr_ratio * max_x)
        ink2 = x_smooth >= ink_t
        segs: list[tuple[int, int]] = []
        in_s = False; st = 0
        for i, v in enumerate(ink2):
            if v and not in_s:
                in_s = True; st = i
            elif not v and in_s:
                in_s = False; segs.append((st, i))
        if in_s:
            segs.append((st, bw))
        if not segs:
            return []
        mg = max(1, int(min_gap_ratio * bw), int(0.12 * bh))
        msw = max(1, int(0.03 * bw), 2)
        out = [segs[0]]
        for a, b in segs[1:]:
            pa, pb = out[-1]
            if a - pb < mg:
                out[-1] = (pa, b)
            else:
                out.append((a, b))
        return [(a, b) for a, b in out if (b - a) >= msw]

    if expected_count_hint and segments:
        if len(segments) != expected_count_hint:
            candidates = []
            for tr in (0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.28, 0.35):
                for gr in (0.02, 0.03, 0.05, 0.07, 0.10, 0.14):
                    segs = try_split(tr, gr)
                    if segs:
                        candidates.append(segs)
            if candidates:
                def seg_score(segs):
                    n = len(segs)
                    diff = abs(n - expected_count_hint)
                    widths = np.array([b-a for a, b in segs], dtype=np.float32)
                    mean_w = float(np.mean(widths)) if len(widths) else 1.0
                    cv = float(np.std(widths) / (mean_w + 1e-6)) if len(widths) else 9.0
                    # prefer exact match strongly
                    exact_penalty = 0 if diff == 0 else (1 + diff)
                    return (exact_penalty, diff, cv)
                segments = sorted(candidates, key=seg_score)[0]

    if not segments:
        return None

    components: list[DigitComponent] = []
    for gi, (sx0, sx1) in enumerate(segments):
        # glyph bbox in full crop coords
        gx0 = rx0 + sx0
        gx1 = rx0 + sx1
        gy0 = ry0 + y0b
        gy1 = ry0 + y1b
        # tighten vertically within segment
        seg_mask = (mask[gy0:gy1, gx0:gx1] > 0).astype(np.uint8)
        ys_s, xs_s = np.where(seg_mask > 0)
        if len(xs_s) == 0:
            continue
        ly0, ly1 = int(ys_s.min()), int(ys_s.max()) + 1
        lx0, lx1 = int(xs_s.min()), int(xs_s.max()) + 1
        fgx0 = gx0 + lx0; fgy0 = gy0 + ly0; fgx1 = gx0 + lx1; fgy1 = gy0 + ly1
        gw = max(1, fgx1 - fgx0); gh = max(1, fgy1 - fgy0)
        glyph_mask = (mask[fgy0:fgy1, fgx0:fgx1] > 0).astype(np.uint8) * 255
        norm_img, norm_mask = _normalize_glyph(glyph_mask, out_h=64, out_w=40, pad=4)
        comp = DigitComponent(
            index=gi,
            bbox=(int(fgx0), int(fgy0), int(gw), int(gh)),
            image=norm_img,
            mask=norm_mask,
            kind="digit",
            confidence=0.8,
            meta={"method": "projection", "area": int(np.count_nonzero(glyph_mask))},
        )
        components.append(comp)

    components.sort(key=lambda c: c.bbox[0])
    for i, c in enumerate(components):
        c.index = i
    res.components = components
    res.digit_components = [c for c in components if c.kind == "digit"]
    res.notes.append(f"projection segmented {len(res.digit_components)} components")
    res.debug = {"proj_segments": len(segments), "roi": [rx0, ry0, rx1, ry1], "band": [int(y0b), int(y1b)]}
    return res if res.digit_components else None


def band_ref_h(comps: list[dict[str, Any]]) -> float:
    if not comps:
        return 1.0
    return float(np.median([c["h"] for c in comps]))
