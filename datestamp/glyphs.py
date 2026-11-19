"""Stage 2 glyph bootstrap/cache.

Builds a reference digit glyph set automatically from sample photos:
- detect + segment digit crops
- cluster crops
- auto-label clusters using samples/ground_truth.csv alignment (best-effort) and
  positional sanity checks; report unlabeled/missing cases
- cache inspectable glyph images and manifest/report under glyphs/
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .config import CALIBRATION_FILENAME, GLYPHS_DIRNAME
from .detect import detect_stamp, load_calibration_data
from .io_utils import discover_images, ensure_dir, read_csv_rows, write_json
from .segment import DigitComponent, SegmentationResult, segment_stamp_crop


@dataclass
class CropRecord:
    crop_id: str
    filename: str
    image_stem: str
    index: int
    bbox: tuple[int, int, int, int]
    image_path: str
    gt_digits: str
    direct_label: Optional[str]
    alignment: str  # exact | mismatch | none
    seg_count: int
    gt_len: int
    notes: list[str]


def _load_ground_truth(photo_dir: Path) -> dict[str, dict[str, str]]:
    gt_path = photo_dir / "ground_truth.csv"
    if not gt_path.exists():
        return {}
    try:
        return {r.get("filename", ""): r for r in read_csv_rows(gt_path)}
    except Exception:
        return {}


def _is_no_stamp_row(row: dict[str, str]) -> bool:
    status = (row.get("status") or "").strip().lower()
    date = (row.get("date") or "").strip().lower()
    digits = (row.get("digits") or "").strip().lower()
    return (
        "no_stamp" in status
        or "no stamp" in status
        or date in {"no_stamp", "no stamp", ""}
        and ("no stamp" in status or status == "no_stamp")
        or digits in {"no_stamp", "no stamp"}
    )


def _feature_vector(img: np.ndarray, size: tuple[int, int] = (16, 10)) -> np.ndarray:
    # img expected uint8 0/255 or 0..255
    if img is None or img.size == 0:
        return np.zeros((size[0] * size[1],), dtype=np.float32)
    small = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    arr = small.astype(np.float32) / 255.0
    return arr.reshape(-1)


def _representative_index(indices: list[int], features: np.ndarray) -> int:
    if len(indices) == 1:
        return indices[0]
    sub = features[indices]
    centroid = sub.mean(axis=0, keepdims=True)
    dists = np.linalg.norm(sub - centroid, axis=1)
    return indices[int(np.argmin(dists))]


def _cluster_crops(features: np.ndarray, n_hint: int = 10) -> np.ndarray:
    n = features.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)
    # choose number of clusters: around 10, but adapt to data; allow style variants
    k = min(max(2, n_hint), n)
    # if many crops, allow a few extra clusters for style/separator variance
    if n >= 20:
        k = min(n, max(n_hint, 12))
    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(features)
        return labels.astype(int)
    except Exception:
        # fallback: trivial single cluster
        return np.zeros((n,), dtype=int)


def _positional_sanity_warnings(aligned_records: list[CropRecord]) -> list[str]:
    """Best-effort sanity checks; only strong when 6-digit alignment is unambiguous."""
    warnings: list[str] = []
    # group by filename for 6-char exact alignments
    by_file: dict[str, list[CropRecord]] = defaultdict(list)
    for r in aligned_records:
        if r.alignment == "exact" and r.direct_label and r.direct_label.isdigit():
            by_file[r.filename].append(r)
    for fname, recs in sorted(by_file.items()):
        recs_sorted = sorted(recs, key=lambda x: x.index)
        labels = [r.direct_label for r in recs_sorted if r.direct_label]
        if len(labels) == 6:
            # assume YY MM DD
            if labels[2] not in {"0", "1"}:
                warnings.append(f"{fname}: 6-digit exact alignment gives month tens '{labels[2]}' (expected 0 or 1)")
            if labels[4] not in {"0", "1", "2", "3"}:
                warnings.append(f"{fname}: 6-digit exact alignment gives day tens '{labels[4]}' (expected 0-3)")
        # generic: first char of month in common 4/5/6 forms sometimes 0/1 but ambiguous; skip
    return warnings


def run_glyphs(photo_dir: Path, output_dir: Optional[Path] = None) -> int:
    """Build glyph cache from photos in photo_dir and write to glyphs/ (or output_dir)."""
    if not photo_dir.exists():
        print(f"glyphs: photo directory not found: {photo_dir}")
        return 2
    if not photo_dir.is_dir():
        print(f"glyphs: not a directory: {photo_dir}")
        return 2

    glyphs_dir = Path(output_dir) if output_dir else Path(GLYPHS_DIRNAME)
    crops_dir = ensure_dir(glyphs_dir / "crops")
    ensure_dir(glyphs_dir)

    try:
        cal = load_calibration_data(CALIBRATION_FILENAME)
    except Exception as e:
        print(f"glyphs: failed to load calibration ({CALIBRATION_FILENAME}): {e}")
        return 1

    gt_map = _load_ground_truth(photo_dir)
    images = discover_images(photo_dir)
    if not images:
        print(f"glyphs: no supported images found in {photo_dir}")
        return 1

    print(f"glyphs: scanning {len(images)} image(s) in {photo_dir} ...")

    crop_records: list[CropRecord] = []
    crop_images: list[np.ndarray] = []
    per_image_reports: list[dict[str, Any]] = []

    for path in images:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            per_image_reports.append({"filename": path.name, "error": "unreadable"})
            continue
        gt_row = gt_map.get(path.name, {})
        gt_digits = (gt_row.get("digits") or "").strip()
        gt_status = (gt_row.get("status") or "").strip()
        gt_date = (gt_row.get("date") or "").strip()
        is_no_stamp = ("no stamp" in gt_status.lower()) or ("no stamp" in gt_date.lower()) or (gt_status.lower() == "no_stamp")

        det, crop = detect_stamp(img, cal=cal)
        if not det.found or crop is None:
            per_image_reports.append({
                "filename": path.name,
                "detected": False,
                "gt_digits": gt_digits,
                "gt_status": gt_status,
                "notes": det.notes,
            })
            continue

        # use GT length hint when available to improve segmentation/alignment for bootstrap
        hint = len(gt_digits) if (gt_digits and gt_digits.isdigit()) else None
        seg = segment_stamp_crop(crop, cal=cal, expected_count_hint=hint)
        digit_comps = seg.digit_components
        seg_count = len(digit_comps)

        alignment = "none"
        direct_labels: list[Optional[str]] = [None] * seg_count
        usable_gt = bool(gt_digits) and gt_digits.isdigit() and not is_no_stamp
        if usable_gt and seg_count == len(gt_digits):
            alignment = "exact"
            direct_labels = list(gt_digits)
        elif usable_gt and seg_count > 0:
            alignment = "mismatch"

        img_rep = {
            "filename": path.name,
            "detected": True,
            "corner": det.corner,
            "bbox": det.bbox,
            "segmented_digits": seg_count,
            "gt_digits": gt_digits,
            "alignment": alignment,
            "notes": seg.notes[:5],
        }
        per_image_reports.append(img_rep)

        for comp in digit_comps:
            crop_id = f"{path.stem}_d{comp.index:02d}"
            img_name = f"{crop_id}.png"
            img_path = crops_dir / img_name
            # save normalized glyph image for inspection
            try:
                cv2.imwrite(str(img_path), comp.image)
            except Exception:
                pass
            label = direct_labels[comp.index] if comp.index < len(direct_labels) else None
            rec = CropRecord(
                crop_id=crop_id,
                filename=path.name,
                image_stem=path.stem,
                index=comp.index,
                bbox=tuple(int(x) for x in comp.bbox),
                image_path=str(Path("crops") / img_name),
                gt_digits=gt_digits,
                direct_label=label if (label and label.isdigit()) else None,
                alignment=alignment,
                seg_count=seg_count,
                gt_len=len(gt_digits) if gt_digits else 0,
                notes=[],
            )
            crop_records.append(rec)
            crop_images.append(comp.image.copy())

    if not crop_records:
        report = {
            "version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "photo_dir": str(photo_dir),
            "error": "no digit crops collected",
            "per_image": per_image_reports,
        }
        write_json(glyphs_dir / "report.json", report)
        print("glyphs: no digit crops collected; wrote report.json")
        return 1

    features = np.stack([_feature_vector(im) for im in crop_images], axis=0)
    cluster_ids = _cluster_crops(features, n_hint=10)

    # cluster analysis and labeling
    clusters: dict[int, dict[str, Any]] = {}
    for cid in sorted(set(int(x) for x in cluster_ids)):
        idxs = [i for i, c in enumerate(cluster_ids) if int(c) == cid]
        label_counts = Counter(
            crop_records[i].direct_label for i in idxs if crop_records[i].direct_label and crop_records[i].direct_label.isdigit()
        )
        assigned = None
        assigned_conf = 0.0
        if label_counts:
            assigned, top = label_counts.most_common(1)[0]
            total_labeled = sum(label_counts.values())
            assigned_conf = top / float(total_labeled) if total_labeled else 0.0
        rep_i = _representative_index(idxs, features)
        rep = crop_records[rep_i]
        clusters[cid] = {
            "cluster_id": int(cid),
            "size": len(idxs),
            "assigned_label": assigned,
            "label_confidence": assigned_conf,
            "label_votes": dict(label_counts),
            "representative_crop_id": rep.crop_id,
            "representative_image": rep.image_path,
            "member_crop_ids": [crop_records[i].crop_id for i in idxs],
        }

    # attach cluster labels back to records
    for i, rec in enumerate(crop_records):
        cid = int(cluster_ids[i])
        rec_cluster = clusters.get(cid, {})
        # store in notes-like via report; CropRecord dataclass has no cluster field, keep mapping separately
    record_to_cluster = {crop_records[i].crop_id: int(cluster_ids[i]) for i in range(len(crop_records))}

    # save representative glyph images per cluster
    reps_dir = ensure_dir(glyphs_dir / "representatives")
    for cid, info in clusters.items():
        rep_id = info["representative_crop_id"]
        # find image
        try:
            idx = next(i for i, r in enumerate(crop_records) if r.crop_id == rep_id)
            label = info.get("assigned_label")
            label_part = str(label) if label is not None else "unlabeled"
            out_name = f"cluster_{cid:02d}_digit_{label_part}.png"
            cv2.imwrite(str(reps_dir / out_name), crop_images[idx])
            info["representative_saved_as"] = str(Path("representatives") / out_name)
        except Exception as e:
            info["representative_save_error"] = str(e)

    # also save best-effort per-digit exemplars (from labeled clusters/records)
    by_digit_dir = ensure_dir(glyphs_dir / "digits")
    digit_exemplars: dict[str, list[str]] = defaultdict(list)
    # prefer exact-aligned crops
    for i, rec in enumerate(crop_records):
        if rec.direct_label and rec.direct_label.isdigit():
            digit_exemplars[rec.direct_label].append(rec.crop_id)
    # choose up to a few exemplars per digit and copy/save
    digit_manifest: dict[str, Any] = {}
    for d in sorted(digit_exemplars.keys(), key=lambda x: int(x)):
        ids = digit_exemplars[d]
        # pick diverse-ish by taking first/middle/last after stable sort
        pick = []
        for j in [0, len(ids)//2, len(ids)-1]:
            if 0 <= j < len(ids) and ids[j] not in pick:
                pick.append(ids[j])
        saved = []
        for k, cid_ in enumerate(pick[:3]):
            idx = next((i for i, r in enumerate(crop_records) if r.crop_id == cid_), None)
            if idx is None:
                continue
            out_name = f"digit_{d}_{k+1}_{crop_records[idx].crop_id}.png"
            try:
                cv2.imwrite(str(by_digit_dir / out_name), crop_images[idx])
                saved.append(str(Path("digits") / out_name))
            except Exception:
                pass
        digit_manifest[d] = {"count_labeled_crops": len(ids), "exemplar_images": saved}

    # unlabeled clusters and coverage
    unlabeled_clusters = [c for c in clusters.values() if c.get("assigned_label") is None]
    present_digits = sorted([d for d in digit_manifest.keys()], key=lambda x: int(x))
    missing_digits = [str(d) for d in range(10) if str(d) not in digit_manifest]

    # positional sanity warnings (best-effort)
    sanity_warnings = _positional_sanity_warnings(crop_records)

    # build manifest
    manifest = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "photo_dir": str(photo_dir),
        "calibration_file": str(CALIBRATION_FILENAME),
        "glyphs_dir": str(glyphs_dir),
        "num_images_scanned": len(images),
        "num_crops": len(crop_records),
        "num_clusters": len(clusters),
        "clustering": {
            "method": "kmeans_or_fallback",
            "feature": "normalized_64x40_resized_to_16x10_flattened",
            "n_hint": 10,
        },
        "digits_present": present_digits,
        "digits_missing": missing_digits,
        "digit_exemplars": digit_manifest,
        "clusters": {str(cid): info for cid, info in sorted(clusters.items())},
        "crop_cluster_map": record_to_cluster,
        "sanity_warnings": sanity_warnings,
        "notes": [
            "Auto-labeled from ground_truth.csv exact segment-count alignments when available.",
            "No external OCR fallback was available in this environment; unlabeled clusters should be inspected.",
            "Ground truth digit lengths vary; alignment is best-effort and reported per image.",
        ],
    }
    write_json(glyphs_dir / "manifest.json", manifest)

    # detailed report
    report = {
        "version": 1,
        "generated_at": manifest["generated_at"],
        "photo_dir": str(photo_dir),
        "summary": {
            "num_images_scanned": len(images),
            "num_crops": len(crop_records),
            "num_clusters": len(clusters),
            "num_unlabeled_clusters": len(unlabeled_clusters),
            "digits_present": present_digits,
            "digits_missing": missing_digits,
            "exact_aligned_images": sum(1 for r in per_image_reports if r.get("alignment") == "exact"),
            "mismatch_aligned_images": sum(1 for r in per_image_reports if r.get("alignment") == "mismatch"),
            "detected_images": sum(1 for r in per_image_reports if r.get("detected")),
        },
        "sanity_warnings": sanity_warnings,
        "unlabeled_clusters": unlabeled_clusters,
        "per_image": per_image_reports,
        "crops": [asdict(r) for r in crop_records],
    }
    write_json(glyphs_dir / "report.json", report)

    # console summary
    summary = report["summary"]
    print(f"glyphs: wrote cache to {glyphs_dir}")
    print(f"  images scanned:      {summary['num_images_scanned']}")
    print(f"  detected:            {summary['detected_images']}")
    print(f"  digit crops:         {summary['num_crops']}")
    print(f"  clusters:            {summary['num_clusters']} (unlabeled: {summary['num_unlabeled_clusters']})")
    print(f"  exact alignments:    {summary['exact_aligned_images']}")
    print(f"  mismatch alignments:  {summary['mismatch_aligned_images']}")
    print(f"  digits present:      {', '.join(present_digits) if present_digits else '(none)'}")
    if missing_digits:
        print(f"  WARNING missing digits from glyph set: {', '.join(missing_digits)}")
    else:
        print("  digit coverage:      all digits 0-9 present")
    if sanity_warnings:
        print(f"  sanity warnings:     {len(sanity_warnings)} (see glyphs/report.json)")
        for w in sanity_warnings[:10]:
            print(f"    - {w}")
    print("  artifacts: manifest.json, report.json, crops/, representatives/, digits/")

    return 0

