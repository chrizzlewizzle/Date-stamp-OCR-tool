# Datestamp OCR

Local/offline Python tool to read orange/red analog date stamps burned into scanned family photos (film camera era, ~1980s–2000s) and write the photographed date into EXIF metadata.

- **Windows 11**, **Python 3.11+**, **CPU-only** (no CUDA), no WSL/Docker
- **No runtime network calls**, no paid APIs/cloud OCR, **no LLM** in the pipeline
- **Correctness and reviewability over speed** (designed for archives in the hundreds to low thousands)
- Originals are **never modified** except the final metadata write step, which is gated behind explicit `--apply`

## Project status

Milestones 0–9 are implemented (CLI, calibration, detection, segmentation/glyph bootstrap, recognition, parsing/pivot/layout checks, confidence/classification, scan/review outputs with caching, gated EXIF apply, and tests including e2e scoring against `samples/ground_truth.csv`).

Current sample-run recognition quality is limited (see **Final sample run report** below), but the pipeline is inspectable and review/correction-driven: use `review.html` and `corrections.csv` to verify and fix reads before applying metadata.

## Setup (Windows, uv)

1. Install Python 3.11+.
2. Install [uv](https://docs.astral.sh/uv/), for example:
   ```powershell
   winget install --id=astral-sh.uv -e
   ```
3. From the project root, create/sync the environment and install dependencies:
   ```powershell
   uv sync --extra dev
   ```
   - Uses `pyproject.toml` (and `uv.lock` when present) as the source of truth.
   - `--extra dev` installs `pytest` for tests; runtime-only install can use `uv sync`.
4. (Optional, required for `apply` only) Install **ExifTool**:
   - Download from https://exiftool.org/
   - Extract and rename `exiftool(-k).exe` to `exiftool.exe`
   - Place it on your `PATH` (or add its folder to `PATH`)
   - Verify:
     ```powershell
     exiftool -ver
     ```
   - `scan`, `calibrate`, and `glyphs` do **not** require ExifTool. `apply` performs a preflight check and prints clear Windows install instructions if ExifTool is missing.

## CLI usage

Run via `uv` (no manual venv activation required):

```powershell
# 1) Measure sample properties and write calibration.json (re-runnable)
uv run datestamp calibrate <photo_dir>

# 2) Build/rebuild reference digit glyph cache from samples
uv run datestamp glyphs <photo_dir>

# 3) Scan photos and produce review outputs (auto-runs calibrate/glyphs if missing)
uv run datestamp scan <photo_dir> --out results --field-order YMD --limit N

# 4) Dry-run metadata apply plan (default, no file changes)
uv run datestamp apply <photo_dir> --results results

# 5) Actually write metadata (explicit gate)
uv run datestamp apply <photo_dir> --results results --apply [--also-file-times]
```

Notes:
- Convenience entrypoint: `uv run python datestamp.py ...` (same subcommands)
- `--field-order` defaults to `YMD` and is used as an assumption/override; parsing also performs auto-detection/cross-checks and reports layout concerns.
- `--limit` is useful for quick test runs.

### Detection evaluation utility (development/validation)

```powershell
uv run python -m datestamp.detect samples --no-fallback
# optional: --debug-dir results/debug-detect --limit N --model-path models/yolo-date-stamp-detector.onnx
```
Reports per-image detection results, expected corner per orientation, and checks `samples/ground_truth.csv` `no_stamp` cases. Runtime remains offline.

## Outputs and review workflow

`scan` writes non-destructive outputs to the `--out` directory (default `results/`):

- `results.csv` — per-image results: `filename, raw_digits_read, parsed_date, per_digit_scores, overall_confidence, classification, notes`
- `review.html` — self-contained single-file review page (stamp crops embedded as base64), **sorted lowest confidence / highest review priority first**, with color-coded classifications
- `corrections.csv` — hand-editable corrections stub (`filename,corrected_date,corrected_digits,notes`)
- `cache/` — resumable per-image scan cache and cropped stamp images (safe to delete to force recompute, but rescanning is slower)

### How to read `review.html`

- Open `review.html` in a browser and review rows from top to bottom (most uncertain/problematic first).
- For each photo, compare the **stamp crop** against **Raw / Date**, **Confidence/classification**, and **Notes**.
- Classifications:
  - `confident`: strong evidence and plausible parsed date
  - `needs_review`: plausible but not strong enough to auto-apply
  - `failed`: unreadable/implausible/unparsed or weak evidence
  - `no_stamp_found`: no stamp detected
- Treat `notes` as explainability hints (detection/segmentation/parse/plausibility/penalty reasons), not ground truth.

### How to fill `corrections.csv`

- `corrections.csv` is authoritative for `apply` when a row contains a correction.
- Columns:
  - `filename`: exact image filename (as in `results.csv`)
  - `corrected_date`: preferred, format `YYYY-MM-DD`
  - `corrected_digits`: optional fallback (digits as printed); used only if `corrected_date` is empty and the digits can be parsed
  - `notes`: freeform (for your tracking)
- Typical workflow: add/fill rows for misreads or anything you want to force, save the CSV, then run `apply` in dry-run mode to verify the plan.

## Metadata apply (safe, gated)

`apply` is the **only** step that can modify photo files.

- **Dry-run by default**: prints exactly what would be written and what would be skipped (no file changes).
- **Explicit gate**: pass `--apply` to write metadata.
- **Eligibility**: only `confident` scan results and rows with authoritative corrections in `corrections.csv` are written. Everything else is skipped and listed with reasons.
- **EXIF behavior**:
  - Sets `DateTimeOriginal` to the stamp date at **12:00:00** (day known, time unknown; midday reduces timezone date shifts)
  - Leaves `CreateDate` unchanged (scan/digitization date preserved)
  - Optional `--also-file-times` sets `FileModifyDate` to the same datetime (helps Windows Explorer sorting)
- **Write method**: batched `exiftool -csv=` (not one process per file)
- **Backups**: ExifTool `_original` backups are kept by default (the tool does **not** pass `-overwrite_original`)
- **Failure handling**: batch failures are reported; remaining batches continue (partial-failure tolerant) and the command exits non-zero on failures

### Reversal / undo (ExifTool backups)

If backups were kept (default), you can restore originals:

```powershell
# Restore originals in a directory
exiftool -restore_original "<photo_dir>"

# Restore recursively (if needed)
exiftool -restore_original -r "<photo_dir>"

# After verifying restores, optionally remove backup files
exiftool -delete_original "<photo_dir>"
```

Always verify on a copy/small subset first for large archive runs.

## Project structure

- `datestamp.py` — convenience CLI entrypoint
- `datestamp/` — pipeline modules and CLI implementation
  - `calibration.py` (Stage 0), `detect.py` (Stage 1), `segment.py`/`glyphs.py` (Stage 2 bootstrap), `recognize.py` (Stage 2 recognizer), `parse.py` (Stage 3), `confidence.py` (Stage 4), `scan.py`/`cache.py`/`review.py` (Stage 5), `apply.py` (Stage 6)
- `pyproject.toml`, `uv.lock` — dependencies and tooling (`uv`, console script, pytest config)
- `calibration.json` — committed runtime calibration defaults (generated/refined from samples)
- `glyphs/` — inspectable cached glyph crops, representatives, per-digit exemplars, and manifests/reports
- `samples/` — local sample photos and `ground_truth.csv` (**authoritative, do not overwrite**; directory is gitignored locally)
- `results/` — scan outputs and cache (`results.csv`, `review.html`, `corrections.csv`, `cache/`)
- `tests/` — unit tests (photo-free where applicable) and e2e sample scoring

## Optional ONNX fallback detector (offline after setup)

- Primary detector: HSV red/orange masking + geometry/connected-component heuristics (calibration-driven).
- Optional fallback: `OnnxYoloStampDetector` for `pike00/yolo-date-stamp-detector` (CPU ONNX Runtime), used on HSV miss/weak/implausible cases when a local model is available.
- **Default model path:** `models/yolo-date-stamp-detector.onnx` (not bundled).
- **Setup-time only**: obtain/export the model to ONNX once on a machine where you can download it, then copy the `.onnx` locally. Runtime performs **no network calls**.
- If the model is missing/unloadable, the tool continues with HSV+geometry only.

## Decision log (measured from `samples/`, Milestones 0–10)

- **Calibration (`calibration.json`, measured from 20 sample images):** HSV red/orange ranges `[0–34]` and `[156–179]` with `s/v >= ~73/74`; corner search regions `x_frac=0.35`, `y_frac=0.35`; EXIF orientation tags observed as `none`. Median dimensions ~3380×2340, DPI median 600×600. Median bbox fraction recorded (`x≈0.409, y≈0.650, w≈0.343, h≈0.337`).
- **Detection corner policy:** analog stamps are corner-anchored by frame orientation. Landscape frames are searched only at `bottom_right`; portrait frames (rotated landscape frames) are searched at `bottom_left` and `top_right`, and their crops are rotated upright before segmentation. Nearly square frames fall back to checking all four corners. A bright/low-saturation fallback range catches faint overexposed stamps (e.g. pale pink on snow) without relaxing the primary mask.
- **Detection:** HSV+geometry is the primary path; optional CPU ONNX fallback is pluggable/offline-local. Detection crops are often coarse corner-region crops (localization-first), which still affects downstream segmentation.
- **Glyph bootstrap (`glyphs/`, current sample run):** 20 images scanned, 19 detected, 68 digit crops, 12 clusters, **digits 0–9 all present** (no missing glyphs). Auto-labeling uses best-effort alignment to `samples/ground_truth.csv` where segment counts match; 6 exact and 13 mismatch alignments recorded, 0 unlabeled clusters. One sanity warning recorded: `20.png` 6-digit exact alignment gives day tens `8` (expected 0–3), indicating format/segmentation ambiguity. No external OCR fallback tool was used in this environment.
- **Recognition:** Template matching against cached glyph exemplars/representatives; per-digit scores/margins/alternates retained for explainability. Quality depends strongly on segmentation/crop tightness and glyph label quality.
- **Parsing/layout/pivot:** Photo-free deterministic parsing with normalization, variable-length candidate splitting, field-order handling (`--field-order` assumption/override + auto-detect), and two-digit year expansion via batch-inferred pivot. Current sample scan inferred **pivot 29** (`YY<=29 -> 20YY`, else `19YY`) from 14 year samples. Layout cross-check reported a **loud warning** for assumed `YMD` (2/9 violating items, 22.2%; `field2>12: 2`, `field3>31: 2`), suggesting mixed/ambiguous layouts or segmentation/read errors in parts of the batch.
- **Confidence/review:** Configurable scoring/classification (`confident/needs_review/failed/no_stamp_found`) emphasizes worst per-digit score, digit count/spacing, parse validity, and plausibility (calendar validity, future/scan-date/archive/year-range checks). `review.html` is the primary audit surface; `results.csv` and notes capture machine-readable detail.
- **Metadata apply:** Dry-run default, explicit `--apply` gate, corrections authoritative over scan results, writes only eligible items, `DateTimeOriginal` set to stamp date at `12:00:00`, `CreateDate` preserved, optional `FileModifyDate`, batched `exiftool -csv=`, backups preserved by default, reversal instructions printed.
- **Ground truth/format handling:** `samples/ground_truth.csv` contains variable-length `digits` strings and status text variants (e.g. `no stamp` wording in row 16); parsing/scoring/e2e comparison are implemented to handle variable lengths and normalize status/date fields for grading/reporting without modifying the GT file.

## Final sample run report (`samples/`, 20 images)

Commands run on the bundled samples (artifacts in repo state): `calibrate`, `glyphs`, `scan --out results`.

### Scan summary (latest `datestamp scan samples --out results`)

- Total: 20
- Classifications: `confident=1`, `needs_review=4`, `failed=14`, `no_stamp_found=1`
- Parsed year distribution (preliminary parsed results): `1959=1, 1961=1, 1980=1, 1990=1, 2022=1, 2026=1, 2521=1`
- Pivot: `29` (`YY<=29->20YY, else 19YY`) inferred from 14 year samples
- Layout: **warning** for assumed `YMD` (2/9 violating items, 22.2%)
- Glyph coverage: digits `0–9` present (none missing)
- Outputs written: `results/results.csv`, `results/review.html`, `results/corrections.csv`, `results/cache/`

### E2E scoring vs `samples/ground_truth.csv` (Milestone 9 scorer)

- Digits (recogniser): **1/19 correct** (5.3%)
- Date (parse/pivot): **1/19 correct** (5.3%)
- Spec pass rules (`no_stamp`/`unreadable` handling): **1/1** (100.0%)
- Correct match: `9.png` (`9059` → `1990-05-09`)
- Correct `no_stamp` handling: `16.png` detected as `no_stamp_found` with no digits/date

### Key gaps / mismatches (observed)

- Segmentation quality on coarse detection crops is the primary bottleneck (frequent over/under-segmentation: 1–10 components observed), leading to wrong digit strings and parse failures/implausible dates.
- Mixed stamp formats/fonts and variable digit lengths in samples increase ambiguity; layout/pivot logic is in place but depends on correct digit reads.
- Glyph cache currently covers all digits `0–9`, but some exemplar/label assignments may be noisy due to alignment mismatches; inspect `glyphs/manifest.json`, `glyphs/report.json`, and image artifacts when improving segmentation/recognition.

### Detection corner-policy check (`samples/`, 20 images)

`uv run python -m datestamp.detect samples --no-fallback`:

| Metric | All-four-corners (previous) | Orientation-aware (current) |
|---|---|---|
| Stamps found | 19/20 | 19/20 |
| Found stamps in the expected corner | 6/19 | **19/19** |
| `no_stamp` case correctly not detected | 1/1 | 1/1 |
| Faint `2.png` stamp localized tightly | Wrong corner (false-positive crop) | **Bottom-right, tight crop** |

Portrait samples (`6/7/11/12.png`) are now searched only at `bottom_left`/`top_right` and returned upright, so their vertical stamp rows become horizontal crops for segmentation. End-to-end digit recognition is still limited by segmentation/mask quality; this change fixes detection placement, not the remaining recognition bottleneck.

## Operational safeguards (verified)

- Dry-run is the default for `apply`; metadata writes require explicit `--apply`
- No GUI, no file moves/renames
- No LLM in the pipeline
- No runtime network calls (optional ONNX model is local/setup-time only)
- Originals are not modified except gated metadata write; ExifTool backups preserved by default
- `samples/ground_truth.csv` is treated as authoritative test input and is not overwritten by tooling

## Running tests

```powershell
uv run pytest -q
```

Notes:
- Unit tests are photo-free where applicable.
- `tests/test_e2e_samples.py` runs the pipeline on `samples/` and prints a scorecard (digits vs date, spec rules, glyph coverage). It requires the local `samples/` data and generated calibration/glyph artifacts available in the working tree.
