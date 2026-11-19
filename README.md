# Datestamp OCR

Tool to read orange/red analog date stamps burned into scanned family photos (1980s–2000s film cameras) and write the photographed date into EXIF metadata.

**Status:** Milestone 0 scaffold in progress (CLI/contracts/utilities). Pipeline stages are not implemented yet.

## Goals and constraints
- Windows 11, Python 3.11+, CPU-only (no CUDA), no WSL/Docker.
- No runtime network calls, no paid APIs/cloud OCR, no LLM in pipeline.
- Correctness and reviewability over speed.
- Originals are never modified except the final metadata write step with explicit `--apply`.

## Setup (Windows, uv)
1. Install [uv](https://docs.astral.sh/uv/) (e.g. `winget install --id=astral-sh.uv -e` or via the official installer).
2. From the project root, create/sync the environment and install dependencies:
   ```powershell
   uv sync --extra dev
   ```
   This uses `pyproject.toml` (and `uv.lock` when present) to manage the virtual environment and dependencies. Python 3.11+ is required.
3. Install **ExifTool** (required for Stage 6 `apply` only):
   - Download from https://exiftool.org/
   - Place `exiftool(-k).exe` on your `PATH` as `exiftool.exe` (or ensure `exiftool` is callable)
   - Verify: `exiftool -ver`
   - If missing, the tool should print clear instructions instead of failing with a traceback (to be enforced in apply implementation).

## CLI (planned / in progress)
Run commands via `uv` (no manual venv activation required):
```powershell
uv run datestamp calibrate <photo_dir>       # re-measure colour/geometry, rewrite calibration.json
uv run datestamp glyphs    <photo_dir>       # build the reference digit set
uv run datestamp scan      <photo_dir> --out results/ --field-order YMD --limit N
uv run datestamp apply     <photo_dir> --results results/ [--apply] [--also-file-times]
```
A convenience entrypoint is also available: `uv run python datestamp.py --help`.

Current status:
- `calibrate`: implemented (writes `calibration.json`)
- `glyphs`: implemented (Milestone 3 bootstrap/cache; best-effort segmentation + clustering + GT-alignment labeling)
- `recognize` (module): implemented (Milestone 4 template-matching recogniser over cached glyphs; scan CLI wiring pending)
- `scan`: scaffold (pipeline not yet wired; auto-runs missing calibration/glyph steps when implemented)
- `apply`: placeholder (Milestone 8)

### Detection evaluation (Milestone 2 utility)
A detection evaluation harness is available for development/validation:
```powershell
uv run python -m datestamp.detect samples --no-fallback
# optional: --debug-dir results/debug-detect --limit N --model-path models/yolo-date-stamp-detector.onnx
```
This reports per-image found/not-found, method/corner/confidence/bbox, and checks `samples/ground_truth.csv` `no_stamp` cases. Runtime remains offline.

### Glyph bootstrap (Milestone 3)
Builds/rebuilds the reference glyph cache from detected stamp crops:
```powershell
uv run datestamp glyphs <photo_dir>
# convenience: uv run python datestamp.py glyphs samples
```
Outputs (default `glyphs/`):
- `manifest.json`: clustering/labeling metadata, per-digit exemplars, cluster membership, sanity warnings
- `report.json`: per-image detection/segmentation/alignment summary, crop records, missing/unlabeled cluster info
- `crops/`: normalized per-digit crop images (inspectable)
- `representatives/`: representative image per cluster
- `digits/`: exemplar images grouped by labeled digit

Notes: segmentation and GT alignment are best-effort (coarse detections, variable digit counts/formats/fonts). The command prints digit coverage (0–9 present/missing) and any sanity warnings; inspect artifacts when improving later stages. No external OCR fallback tools are assumed/used in this environment.

### Template matching recogniser (Milestone 4)
`datestamp/recognize.py` loads glyph templates from `glyphs/` and recognizes segmented digit crops by template matching:
- per-digit best label, score, margin/alternates, and best-template trace in `DigitReading`
- raw digit string assembly for an ordered component sequence
- glyph coverage utilities to report/warn on missing digits 0–9

Recognition quality currently depends on segmentation and glyph/template quality; scan pipeline integration comes in later milestones.

## Project structure (target)
- `pyproject.toml`: project metadata, dependencies, and `datestamp` console script (managed with `uv`)
- `datestamp.py` / `datestamp/`: CLI and pipeline modules
- `calibration.json`: committed runtime calibration defaults (generated/refined from samples)
- `glyphs/`: cached reference digit glyph images (inspectable)
- `samples/`: sample photos and `ground_truth.csv` (authoritative; do not overwrite; directory is gitignored locally)
- `tests/`: unit tests (no photos required for core logic) and e2e sample scoring
- `results/`: scan outputs (`results.csv`, `review.html`, `corrections.csv`, caches/debug as added)

## Review workflow (to be implemented)
- `scan` produces non-destructive outputs: `results.csv`, self-contained `review.html` (lowest confidence first, crops embedded), and `corrections.csv` stub.
- Review `review.html`, fill corrections in `corrections.csv` where needed.
- `apply` defaults to dry-run and requires `--apply` to write metadata.

## Metadata write & undo (to be implemented)
- Intended writes: `DateTimeOriginal` = stamp date, keep `CreateDate` as scan date, time `12:00:00`, optional `FileModifyDate` via `--also-file-times`.
- Writes via batched `exiftool -csv=`; backups (`_original`) kept by default.
- Undo guidance will be documented with exact `exiftool` restore commands once apply is implemented.

## Decision log
- **2026-xx Milestone 0 (scaffold):** Established package/CLI skeleton, shared data contracts (`config.py`), Windows-safe I/O helpers (`io_utils.py`), pinned baseline dependencies, and placeholder stage modules. No OCR/calibration/glyph/EXIF logic implemented yet.
- **2026-xx Tooling update:** Re-architected project to use **uv** for environment/dependency management and command execution (`pyproject.toml`, `uv sync`, `uv run datestamp ...`, `uv run pytest`). Legacy `requirements.txt` removed in favor of `pyproject.toml` as the source of truth.
- **2026-xx Milestone 1 (calibration):** Implemented `datestamp calibrate` to measure samples via HSV red/orange masking across four corner regions and connected-component heuristics, writing inspectable `calibration.json` (re-runnable). Measured from 20 sample images: 600 DPI median, no EXIF orientation tags observed, stamp candidates detected in all four corners (counts bottom_right=7, bottom_left=6, top_left=4, top_right=3), HSV ranges approx `[0-34]` and `[156-179]` with `s/v >= ~73`, median bbox fraction recorded, and font hints indicating likely solid strokes on average while notes flag multi-camera/font variability (incl. seven-segment cases seen in samples). Values are initial defaults for later detection refinement.
- **2026-xx Milestone 2 (detection):** Implemented Stage 1 in `datestamp/detect.py`: primary HSV mask + geometry detector (calibration-driven HSV ranges, morphology, all four corner search with widened regions, connected-component row-plausibility heuristics, padded crop/bbox output) and a pluggable fallback interface with `OnnxYoloStampDetector` (CPU ONNX Runtime) for `pike00/yolo-date-stamp-detector`. Fallback is attempted on HSV miss/weak or implausible results when a local model is available. Added detection evaluation harness (`uv run python -m datestamp.detect ...`) with per-image report and GT `no_stamp` checking. Current sample eval (HSV-only, no model): **19 found / 1 not found**, and **1/1 ground-truth `no_stamp` correctly not detected** (`16.png` rejected as low-confidence/non-plausible). Bboxes are currently coarse corner-region crops (good for localization; segmentation will refine) and confidence scoring is provisional pending later stages.
- **2026-xx Milestone 3 (glyph bootstrap):** Implemented `datestamp/segment.py` (digit segmentation from detected stamp crops via HSV masking, component clustering/projection-based fallback, normalized glyph extraction) and `datestamp/glyphs.py` (glyph bootstrap/cache pipeline). `uv run datestamp glyphs samples` now runs detection+segmentation, collects ordered digit crops, clusters them (KMeans over normalized glyph features), auto-labels clusters/records using best-effort `samples/ground_truth.csv` alignment (exact when segmented count matches GT digit length; GT length used as a hint), records positional sanity warnings, and writes inspectable artifacts under `glyphs/` (`manifest.json`, `report.json`, `crops/`, `representatives/`, `digits/`). No external/local OCR fallback tool was available in this environment; labeling relies on GT alignment + cluster majority. Current samples run: 20 images scanned, 19 detected, 68 crops, 12 clusters, **all digits 0–9 present**, 6 exact and 13 mismatch alignments, 1 sanity warning (format-position ambiguity case recorded in report).
- **Open (from samples/spec):** `samples/ground_truth.csv` contains variable-length `digits` values while spec references six characters; normalization/alignment for scoring and parsing must be defined during implementation and recorded here. Date formats and fonts vary across samples (including seven-segment examples); calibration/glyph/bootstrap strategies must remain inspectable and handle this variability.
- **Open (segmentation/bootstrap quality):** Segmentation operates on coarse detection crops and currently achieves limited exact count alignment on samples (mismatches remain common). Cluster labels may be noisy where alignment is imperfect; inspect `glyphs/` artifacts and improve segmentation/tightening in later milestones before trusting full-archive recognition.
- **Open:** ONNX detector model acquisition/export is a setup-time activity (runtime must remain offline). Fallback expects a local ONNX model (default path `models/yolo-date-stamp-detector.onnx`). Exact download/export steps are not automated in-runtime; obtain/export the Hugging Face model to ONNX once during setup and place it at that path (or pass `--model-path` to detection eval) when fallback is needed.

## ONNX fallback detector setup (Stage 1, optional)
- **Model:** `pike00/yolo-date-stamp-detector` (YOLO, fine-tuned for orange/red date stamps)
- **Runtime:** ONNX Runtime (CPU), no network calls at runtime
- **Default local path:** `models/yolo-date-stamp-detector.onnx`
- **Setup-time (manual/offline-after-download):** download model artifacts and export to ONNX once (e.g. via Ultralytics export on a machine with the model available), then copy the resulting `.onnx` file into `models/` (create the directory if needed). Keep this as a one-time setup step; do not perform network calls during tool execution.
- **Behavior:** if the model file is missing/unloadable, the tool continues with HSV+geometry detection and reports fallback unavailable in evaluation; if present, fallback is used on HSV misses/weak detections and an empty detection is treated as "no stamp".
