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

## CLI (planned)
Run commands via `uv` (no manual venv activation required):
```powershell
uv run datestamp calibrate <photo_dir>       # re-measure colour/geometry, rewrite calibration.json
uv run datestamp glyphs    <photo_dir>       # build the reference digit set
uv run datestamp scan      <photo_dir> --out results/ --field-order YMD --limit N
uv run datestamp apply     <photo_dir> --results results/ [--apply] [--also-file-times]
```
A convenience entrypoint is also available: `uv run python datestamp.py --help`.

Current scaffold behavior: commands parse and run placeholder handlers; `scan` will auto-run `calibrate`/`glyphs` when outputs are missing once implemented.

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
- **Open (from samples/spec):** `samples/ground_truth.csv` contains variable-length `digits` values while spec references six characters; normalization/alignment for scoring and parsing must be defined during implementation and recorded here. Date formats and fonts vary across samples (including seven-segment examples); calibration/glyph/bootstrap strategies must remain inspectable and handle this variability.
- **Open:** ONNX detector model acquisition/export is a setup-time activity (runtime must remain offline); exact steps to be documented when fallback is implemented.
