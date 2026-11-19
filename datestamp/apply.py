"""Stage 6: apply photographed dates to EXIF metadata (gated, dry-run by default)."""

from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .config import CORRECTIONS_CSV_FILENAME, RESULTS_CSV_FILENAME
from .io_utils import read_csv_rows
from .parse import normalize_digits, parse_date


@dataclass
class ApplyItem:
    filename: str
    source_path: Path
    date_yyyy_mm_dd: str
    source: str  # correction | confident
    reason: str
    corrected_digits: str = ""


@dataclass
class SkipItem:
    filename: str
    reason: str
    classification: str = ""
    parsed_date: str = ""


def _find_exiftool() -> Optional[str]:
    for name in ("exiftool", "exiftool.exe"):
        p = shutil.which(name)
        if p:
            return p
    return None


def _print_exiftool_missing() -> None:
    print("ERROR: exiftool was not found on PATH.")
    print("ExifTool is required for the 'apply' command (metadata writing). Scan/review do not require it.")
    print("Install instructions (Windows):")
    print("  1. Download ExifTool for Windows from https://exiftool.org/")
    print("  2. Extract the archive.")
    print("  3. Rename 'exiftool(-k).exe' to 'exiftool.exe'.")
    print("  4. Place exiftool.exe in a folder on your PATH (or add that folder to PATH).")
    print("  5. Open a new terminal and verify with: exiftool -ver")
    print("Then re-run: datestamp apply <photo_dir> --results <results_dir> [--apply]")


def _parse_iso_date(s: str) -> Optional[str]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        dt = datetime.strptime(s, "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def _to_exif_datetime(date_yyyy_mm_dd: str) -> str:
    # Fill time as 12:00:00 per spec
    return date_yyyy_mm_dd.replace("-", ":") + " 12:00:00"


def _build_photo_index(photo_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    # non-recursive scan behavior matches discovery; index basenames
    try:
        for p in photo_dir.iterdir():
            if p.is_file():
                key = p.name.lower()
                # keep first deterministically if duplicates by case-collisions unlikely; prefer exact
                if key not in index:
                    index[key] = p
    except Exception:
        pass
    return index


def _load_corrections(results_dir: Path) -> dict[str, dict[str, str]]:
    path = results_dir / CORRECTIONS_CSV_FILENAME
    if not path.exists():
        return {}
    rows = read_csv_rows(path)
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        filename = (row.get("filename") or "").strip()
        if not filename:
            continue
        out[filename] = {k: (v if v is not None else "") for k, v in row.items()}
        # also allow case-insensitive lookup fallback via lower key stored separately by caller if needed
    return out


def _load_results(results_dir: Path) -> list[dict[str, str]]:
    path = results_dir / RESULTS_CSV_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"results.csv not found in {results_dir}")
    return read_csv_rows(path)


def _resolve_source_path(filename: str, photo_dir: Path, photo_index: dict[str, Path]) -> Optional[Path]:
    # try exact path/name first
    direct = photo_dir / filename
    if direct.exists() and direct.is_file():
        return direct
    return photo_index.get(filename.lower())


def plan_apply(photo_dir: Path, results_dir: Path) -> tuple[list[ApplyItem], list[SkipItem], dict[str, Any]]:
    results = _load_results(results_dir)
    corrections = _load_corrections(results_dir)
    photo_index = _build_photo_index(photo_dir)

    # index corrections case-insensitively too
    corr_ci: dict[str, dict[str, str]] = {k.lower(): v for k, v in corrections.items()}

    to_apply: list[ApplyItem] = []
    skipped: list[SkipItem] = []

    for row in results:
        filename = (row.get("filename") or "").strip()
        if not filename:
            continue
        classification = (row.get("classification") or "").strip()
        parsed_date = (row.get("parsed_date") or "").strip()
        corr = corrections.get(filename) or corr_ci.get(filename.lower())
        src_path = _resolve_source_path(filename, photo_dir, photo_index)
        if src_path is None:
            skipped.append(SkipItem(filename=filename, reason="photo file not found in photo_dir", classification=classification, parsed_date=parsed_date))
            continue

        chosen_date: Optional[str] = None
        chosen_source = ""
        chosen_digits = ""
        correction_reason = ""

        if corr:
            corrected_date = (corr.get("corrected_date") or "").strip()
            corrected_digits = (corr.get("corrected_digits") or "").strip()
            if corrected_date or corrected_digits:
                chosen_digits = corrected_digits
                if corrected_date:
                    chosen_date = _parse_iso_date(corrected_date)
                    if chosen_date:
                        chosen_source = "correction"
                    else:
                        correction_reason = f"correction has invalid corrected_date '{corrected_date}' (expected YYYY-MM-DD)"
                elif corrected_digits:
                    # try parse corrected digits as fallback
                    nd = normalize_digits(corrected_digits)
                    pr = parse_date(nd, field_order="YMD", auto_detect=True)
                    if pr.ok and pr.parsed_date:
                        chosen_date = pr.parsed_date
                        chosen_source = "correction_digits"
                    else:
                        correction_reason = "correction provided corrected_digits but no valid date could be parsed; provide corrected_date (YYYY-MM-DD)"
            # if correction row exists but empty, ignore it

        if chosen_date and chosen_source:
            to_apply.append(ApplyItem(filename=filename, source_path=src_path, date_yyyy_mm_dd=chosen_date, source=chosen_source, reason="authoritative correction", corrected_digits=chosen_digits))
            continue

        if correction_reason:
            skipped.append(SkipItem(filename=filename, reason=correction_reason, classification=classification, parsed_date=parsed_date))
            continue

        # no authoritative correction -> only confident
        if classification == "confident":
            dt = _parse_iso_date(parsed_date)
            if dt:
                to_apply.append(ApplyItem(filename=filename, source_path=src_path, date_yyyy_mm_dd=dt, source="confident", reason="confident scan result"))
            else:
                skipped.append(SkipItem(filename=filename, reason="classified confident but parsed_date is missing/invalid", classification=classification, parsed_date=parsed_date))
        else:
            skipped.append(SkipItem(filename=filename, reason=f"not eligible (classification={classification or 'unknown'}; no correction)", classification=classification, parsed_date=parsed_date))

    meta = {
        "results_count": len(results),
        "corrections_file_exists": (results_dir / CORRECTIONS_CSV_FILENAME).exists(),
        "to_apply_count": len(to_apply),
        "skipped_count": len(skipped),
    }
    return to_apply, skipped, meta


def _chunked(items: list[ApplyItem], size: int) -> list[list[ApplyItem]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _write_exiftool_csv(csv_path: Path, items: list[ApplyItem], also_file_times: bool) -> None:
    fieldnames = ["SourceFile", "DateTimeOriginal"]
    if also_file_times:
        fieldnames.append("FileModifyDate")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            dt = _to_exif_datetime(item.date_yyyy_mm_dd)
            row = {"SourceFile": str(item.source_path), "DateTimeOriginal": dt}
            if also_file_times:
                row["FileModifyDate"] = dt
            writer.writerow(row)


def _run_exiftool_batch(exiftool_path: str, csv_path: Path) -> tuple[bool, str]:
    # Preserve _original backups by default (do NOT pass -overwrite_original).
    cmd = [exiftool_path, "-charset", "filename=utf8", "-csv=" + str(csv_path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except Exception as e:
        return False, f"failed to execute exiftool: {e}"
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        return False, out.strip() or f"exiftool exited with code {proc.returncode}"
    return True, out.strip()


def _print_plan(items: list[ApplyItem], skipped: list[SkipItem], meta: dict[str, Any], *, do_apply: bool, also_file_times: bool) -> None:
    mode = "APPLY" if do_apply else "DRY-RUN"
    print(f"apply ({mode})")
    print(f"  results rows: {meta.get('results_count', 0)}")
    print(f"  corrections.csv present: {meta.get('corrections_file_exists', False)}")
    print(f"  eligible to write: {len(items)}")
    print(f"  skipped: {len(skipped)}")
    if also_file_times:
        print("  also_file_times: enabled (will set FileModifyDate)")
    else:
        print("  also_file_times: disabled")
    print("  backups: exiftool _original backups will be kept by default (no -overwrite_original).")

    if items:
        print("Planned writes:")
        for it in items:
            dt = _to_exif_datetime(it.date_yyyy_mm_dd)
            extra = f", digits={it.corrected_digits}" if it.corrected_digits else ""
            print(f"  - {it.filename}: DateTimeOriginal={dt} (source={it.source}{extra}) -> {it.source_path}")
    else:
        print("Planned writes: (none)")

    if skipped:
        print("Skipped:")
        for sk in skipped:
            detail = f", classification={sk.classification}" if sk.classification else ""
            if sk.parsed_date:
                detail += f", parsed_date={sk.parsed_date}"
            print(f"  - {sk.filename}: {sk.reason}{detail}")


def _print_reversal_instructions(photo_dir: Path) -> None:
    print("Reversal / undo:")
    print("  - ExifTool keeps original backups as *_original files by default.")
    print("  - To restore originals in a directory (PowerShell):")
    print(f'      exiftool -restore_original "{photo_dir}"')
    print("  - To restore recursively:")
    print(f'      exiftool -restore_original -r "{photo_dir}"')
    print("  - If you need to remove backup files after verifying restores:")
    print(f'      exiftool -delete_original "{photo_dir}"')
    print("  - Do not disable backups unless you are certain; this tool does not pass -overwrite_original.")


def run_apply(photo_dir: Path, results_dir: Path, do_apply: bool, also_file_times: bool) -> int:
    if not photo_dir.exists() or not photo_dir.is_dir():
        print(f"apply: photo directory not found or not a directory: {photo_dir}")
        return 2
    if not results_dir.exists() or not results_dir.is_dir():
        print(f"apply: results directory not found or not a directory: {results_dir}")
        return 2

    exiftool_path = _find_exiftool()
    if not exiftool_path:
        _print_exiftool_missing()
        return 2

    try:
        items, skipped, meta = plan_apply(photo_dir, results_dir)
    except FileNotFoundError as e:
        print(f"apply: {e}")
        print("Run 'datestamp scan <photo_dir> --out <results_dir>' first to generate results.csv.")
        return 2
    except Exception as e:
        print(f"apply: failed to read/plan from results: {e}")
        return 2

    _print_plan(items, skipped, meta, do_apply=do_apply, also_file_times=also_file_times)
    print(f"apply: using exiftool: {exiftool_path}")
    _print_reversal_instructions(photo_dir)

    if not do_apply:
        print("apply: dry-run complete (no files modified). Use --apply to write metadata.")
        return 0

    if not items:
        print("apply: nothing eligible to write.")
        return 0

    # Apply in batches via exiftool -csv=
    batch_size = 100
    batches = _chunked(items, batch_size)
    failures = 0
    successes = 0
    for bi, batch in enumerate(batches, start=1):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8", newline="") as tf:
            tmp_csv = Path(tf.name)
        try:
            _write_exiftool_csv(tmp_csv, batch, also_file_times=also_file_times)
            ok, msg = _run_exiftool_batch(exiftool_path, tmp_csv)
            if ok:
                successes += len(batch)
                print(f"apply: batch {bi}/{len(batches)} wrote {len(batch)} file(s).")
                if msg:
                    # exiftool often prints summary lines; keep concise
                    for line in msg.splitlines()[:5]:
                        if line.strip():
                            print(f"  exiftool: {line.strip()}")
            else:
                failures += len(batch)
                print(f"apply: batch {bi}/{len(batches)} FAILED for {len(batch)} file(s).")
                for line in (msg or "").splitlines()[:20]:
                    if line.strip():
                        print(f"  exiftool: {line.strip()}")
                print("  Continuing with remaining batches (partial failure handling).")
        finally:
            try:
                tmp_csv.unlink(missing_ok=True)
            except Exception:
                pass

    print(f"apply: completed. successful files: {successes}, failed files: {failures}, skipped (not attempted): {len(skipped)}")
    if failures:
        return 1
    return 0

