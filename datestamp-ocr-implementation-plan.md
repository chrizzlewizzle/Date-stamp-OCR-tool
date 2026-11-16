# Datestamp OCR Implementation Plan (modular, multi-session)

Source spec: `datestamp-ocr-prompt.md`  
Environment: Windows 11, Python 3.11+, CPU-only (AMD GPU, no CUDA), no network calls at runtime, `exiftool` for writes.  
Workspace snapshot: 20 sample images in `samples/` and `samples/ground_truth.csv` present; no implementation code/tests yet.

## Guiding principles (apply in every session)
- **Correctness & reviewability over speed.** Prefer simple, inspectable stages and explicit artifacts over clever end-to-end magic.
- **Autonomy with traceability.** Where samples leave ambiguity, choose a reasonable default, record rationale in `README.md` (Decision Log) and/or `calibration.json` notes.
- **Windows-native.** Use `pathlib`, avoid bash-only assumptions, handle spaces/non-ASCII paths, force UTF-8 console/file output where relevant.
- **No runtime network / no paid APIs / no LLM in pipeline.** Model download is setup-time only (if used); runtime must be local.
- **No mutation of originals except Stage 6 with explicit `--apply`.** Scan/review/glyph/calibrate produce sidecar outputs only.
- **Atomic commits/PR-sized units (conceptually).** Each session should finish with: code + tests (where applicable) + updated docs + a short status note of what’s done and what’s next.

## Observations from current samples / ground truth (to account for in design)
- `samples/ground_truth.csv` columns match spec; statuses include `ok` and `no_stamp` (e.g. `16.png`).
- `digits` values vary in length in samples (e.g. `4599`, `99716`, `821224`) while spec states “six characters…” — plan must define a **normalization/alignment strategy** (leading-zero inference, separator handling, or positional mapping) driven by measured stamp structure, and record it.
- Notes indicate **multiple fonts/styles** (solid and “seven segment font”), variable backgrounds/contrast/blur, and possible red backgrounds — reinforces need for robust calibration, fallback detector, and inspectable glyphs.
- **Do not generate/overwrite `samples/ground_truth.csv`.** Treat as authoritative grading reference.

## Proposed repo/module layout (target)
```text
datestamp.py                  # CLI entrypoint (thin dispatcher) or console_script target
datestamp/                    # package
  __init__.py
  cli.py                      # argparse/subcommands, orchestration
  config.py                   # defaults, thresholds, paths, constants (single source of truth)
  io_utils.py                 # safe path handling, UTF-8 helpers, image load/save, CSV/JSON I/O
  calibration.py              # Stage 0: measure + load/save calibration.json
  detect.py                   # Stage 1: HSV mask+geometry, YOLO/ONNX fallback, corner search
  segment.py                  # digit/component extraction & normalization
  glyphs.py                   # Stage 2: crop collection, clustering, labeling, glyph cache/build
  recognize.py                # template matching, per-digit scores
  parse.py                    # Stage 3: format inference/field-order, YY pivot, layout checks
  confidence.py               # Stage 4: scoring, plausibility, classification
  review.py                   # Stage 5: results.csv, review.html (base64 crops), corrections stub, summary
  apply.py                    # Stage 6: exiftool detection, dry-run/apply, batch CSV write, backups/warnings
  cache.py                    # resumable scan cache / per-image result persistence
  models/                     # (optional) ONNX export/load helpers; no weights committed unless allowed
tests/
  conftest.py
  test_parse.py
  test_pivot.py
  test_layout_crosscheck.py
  test_plausibility.py
  test_confidence.py          # unit-level scoring/classification rules
  test_io_utils.py
  test_e2e_samples.py         # scored against samples/ground_truth.csv
calibration.json              # committed defaults (generated/refined from samples)
glyphs/                       # cached reference glyph images (inspectable)
results/                      # default output dir (gitignored except maybe .gitkeep if desired)
requirements.txt
README.md
.gitignore
```
*Note: exact names can be adjusted early, but keep boundaries stable so sessions don’t thrash.*

---

## Milestone 0 — Repo bootstrap & contracts (Session 1)
**Goal:** establish a runnable skeleton, shared contracts, and reproducible Windows setup so later sessions can focus on one stage at a time.

**Tasks (atomic):**
1. **Create package skeleton & CLI stub**
   - Add `datestamp/` package, `datestamp.py` (or entrypoint) with subcommands: `calibrate`, `glyphs`, `scan`, `apply`.
   - Implement argument parsing only (no heavy logic): required dirs, `--out`, `--field-order`, `--limit`, `--results`, `--apply`, `--also-file-times`.
   - Ensure `python datestamp.py --help` works on Windows.
   - *Deliverable:* runnable CLI scaffold.
2. **Define core data contracts/types**
   - Introduce typed structures/dataclasses (or clear dict schemas) for: `Calibration`, `DetectionResult`, `DigitReading`, `PhotoResult`, `Classification` enum (`confident|needs_review|failed|no_stamp_found`), summary stats.
   - Define canonical result fields matching spec for `results.csv` (filename, raw digits read, parsed date, per-digit scores, overall confidence, classification, notes).
   - *Deliverable:* `config.py`/`models.py` (or equivalent) with documented fields.
3. **Implement I/O and Windows-safety utilities**
   - UTF-8 stdout/stderr configuration helper; safe path resolution for spaces/non-ASCII.
   - Image discovery (recursive? define—likely non-recursive by default unless needed; document) and deterministic ordering.
   - CSV/JSON read-write helpers with stable encoding/newlines.
   - *Deliverable:* `io_utils.py` + small unit tests (`test_io_utils.py`) for path/encoding edge cases (no photos required).
4. **Project metadata**
   - Add `requirements.txt` with **pinned versions** (start minimal: e.g. `numpy`, `opencv-python-headless`/`opencv-python`, `Pillow`, `scikit-learn`/`scipy` if needed for clustering, `onnxruntime` placeholder decision noted; avoid CUDA deps).
   - Add `.gitignore` (venv, `__pycache__`, `results/` outputs, caches, local artifacts as appropriate; be careful not to ignore required `glyphs/`/`calibration.json` if meant committed).
   - Add initial `README.md` skeleton: purpose, Windows/venv setup, exiftool install note placeholder, command overview, decision log section.
   - *Deliverable:* reproducible install baseline.

**Exit criteria:** CLI parses commands; basic utils tested; repo conventions set. No OCR logic required yet.

---

## Milestone 1 — Calibration (Stage 0) (Session 2)
**Goal:** measure real sample properties and persist inspectable runtime defaults.

**Tasks (atomic):**
1. **Sample inspection utilities**
   - Read image dimensions, DPI/metadata if present, EXIF orientation tags, color stats.
   - Report per-image and aggregate findings (corner likelihood, stamp size fraction estimate, red/orange HSV ranges, font style hints).
   - *Deliverable:* measurement functions + console/log output.
2. **Implement `calibrate` command**
   - Input: `<photo_dir>` (default workflow uses `./samples` initially).
   - Compute and write `calibration.json` including: plausible corner region(s)/fractions, HSV thresholds/ranges, morphology/connected-component params, expected digit count/spacing hints, orientation handling flags, detected camera/style variants notes.
   - Make step **re-runnable** and log what/how measured; if no samples or insufficient signal, fail clearly with guidance.
   - *Deliverable:* `calibration.json` (committed defaults after run on samples) + loader in `calibration.py`.
3. **Default behavior wiring**
   - Ensure tool can run with committed `calibration.json` without requiring calibration; `scan` auto-runs calibrate if missing (per spec).
   - *Deliverable:* load/validate/fallback logic with clear errors.
4. **Document decisions**
   - Record measured values and open assumptions in README decision log (and/or calibration notes).

**Exit criteria:** `python datestamp.py calibrate samples` produces/updates `calibration.json` and a human-readable measurement summary.

**Dependencies:** Milestone 0.

---

## Milestone 2 — Stamp detection (Stage 1) (Session 3, can split 3a/3b)
**Goal:** reliably locate the stamp region (or declare none) with inspectable intermediate outputs.

**Tasks (atomic):**
1. **HSV mask + geometry primary detector**
   - Convert to HSV, threshold red/orange high-sat/high-val using calibration ranges (handle red hue wrap-around).
   - Restrict to plausible corner regions; find connected components; group into digit-row candidates by alignment/size/spacing.
   - Return bounding box/rotated crop + mask/confidence signals + debug artifacts (optional save under output/debug for reviewability).
   - *Deliverable:* primary path in `detect.py`.
2. **Four-corner / orientation handling**
   - Check all four corners (and/or rotate candidates) to handle rotated scans; respect EXIF orientation observations from calibration.
   - *Deliverable:* deterministic search order and notes in result.
3. **Fallback detector hook (YOLO/ONNX) — interface-first**
   - Define fallback trigger: mask finds nothing or implausible result (thresholds configurable).
   - Implement adapter boundary: `FallbackDetector` interface so primary flow works even if model unavailable in-session.
   - *Deliverable:* pluggable fallback call site + clear logging when triggered.
4. **Fallback detector implementation (separate subtask/session if needed)**
   - Integrate `pike00/yolo-date-stamp-detector` via **ONNX Runtime** (CPU). Handle one-time model acquisition/export as setup step (documented), not runtime network call.
   - Map detections to same crop/result contract; treat empty result as “no stamp”.
   - *Deliverable:* working fallback path or explicit, documented setup script/instructions if export tooling is separated.
5. **Detection evaluation harness on samples**
   - Run detection over `samples/`, emit a small report (found/not found, chosen corner, method used). Compare `no_stamp` cases against ground truth expectations.
   - *Deliverable:* scriptable check / summary output (can be part of `scan --limit` dry internals).

**Exit criteria:** For most samples, produces a plausible crop/region; `16.png` (no_stamp) yields no detection; failures are explainable from saved debug info.

**Dependencies:** Milestones 0–1.

---

## Milestone 3 — Digit segmentation & glyph bootstrap (Stage 2, part A) (Session 4)
**Goal:** extract per-digit crops and build an inspectable, reusable reference glyph set automatically.

**Tasks (atomic):**
1. **Digit segmentation**
   - From detected stamp region: preprocess (binarize/normalize/denoise/morphology as needed, calibration-driven), segment by connected components into ordered digit candidates.
   - Normalize each to fixed box/canvas preserving aspect; capture per-digit crop images and metadata (source file, index, bbox, quality).
   - Handle variability noted in samples (separators/gaps/6 vs fewer chars) with explicit heuristics and logging.
   - *Deliverable:* `segment.py`.
2. **Collect crops across dataset**
   - `glyphs` command (and scan auto-step) runs detection+segmentation over input dir (or samples) to gather crops.
   - *Deliverable:* crop collection routine with deterministic storage/index.
3. **Clustering into digit classes**
   - Cluster normalized crops (e.g. feature vector / pixel similarity / embedding-free methods suitable for 10 classes, one/few fonts) into ~10 groups; allow diagnostics for outliers/multi-font splits.
   - *Deliverable:* clustering module with saved intermediate assignments for inspection.
4. **Automatic labeling strategy (no manual labeling asked from user)**
   - **Primary:** align detected digit positions with `samples/ground_truth.csv` `digits` where available; majority-vote label per cluster.
   - **Fallback:** local OCR vote over cluster (ssocr for true seven-segment per calibration, else Tesseract `--psm 10` digit whitelist or RapidOCR) — bootstrap only.
   - Apply **positional constraints sanity check** (YY MM DD assumptions): month tens ∈ {0,1}, day tens ∈ {0..3}; if violated, retry/flag and record.
   - *Deliverable:* labeled clusters + confidence/ambiguity report.
5. **Cache glyph set**
   - Save representative/template images to `glyphs/` (and a manifest JSON mapping label→files/params) so user can eyeball; make step one-time/idempotent.
   - *Deliverable:* `glyphs/` artifacts + rebuild behavior.

**Exit criteria:** `python datestamp.py glyphs samples` produces inspectable glyph images and a labeling report; missing/ambiguous digits are clearly flagged.

**Dependencies:** Milestones 0–2.

---

## Milestone 4 — Template matching recogniser (Stage 2, part B) (Session 5)
**Goal:** read digit sequences from crops against cached glyphs with per-digit scores.

**Tasks (atomic):**
1. **Template matching engine**
   - Compare normalized digit crops to cached glyph references (support multiple exemplars/styles per digit if needed).
   - Produce per-digit best match, score, and alternates/margin (useful for confidence and debugging).
   - *Deliverable:* `recognize.py`.
2. **Sequence assembly**
   - Combine ordered digit reads into raw digit string(s); preserve evidence (per-digit scores, crop refs).
   - *Deliverable:* `DigitReading`/`PhotoResult` population for recognition stage.
3. **Glyph coverage warning**
   - Check which digits `0–9` missing from glyph set; surface prominently in scan summary and results notes (spec requirement).
   - *Deliverable:* reusable coverage check function.

**Exit criteria:** Recogniser runs on sample crops end-to-end and emits per-digit scores; missing glyph digits are warned.

**Dependencies:** Milestone 3.

---

## Milestone 5 — Parsing, layout cross-check, and century pivot (Stage 3) (Session 6)
**Goal:** convert raw readings into dates safely, with explicit handling of format/layout ambiguity.

**Tasks (atomic):**
1. **Date parsing core**
   - Implement parsers for candidate formats (YYMMDD, MM-DD-YY, DD/MM/YY, etc.) informed by observed separators/positions; support `--field-order` override (default YMD per spec, but respect auto-detect goal).
   - Normalize raw input (strip separators, handle short readings) with documented rules aligned to ground truth behavior.
   - *Deliverable:* `parse.py`.
2. **Century pivot inference**
   - Infer 19xx/2000s pivot from observed year distribution across batch (and/or calibration hints); log chosen pivot and rationale.
   - Map two-digit years accordingly.
   - *Deliverable:* deterministic pivot function + logged decision.
3. **Batch layout cross-check**
   - Pooled validation: field 2 never > 12, field 3 never > 31 (for assumed order). Detect widespread violations; report loudly at top of summary (not silent wrong dates).
   - *Deliverable:* layout diagnostics in summary/result model.
4. **Unit tests (no photos)**
   - `tests/test_parse.py`, `tests/test_pivot.py`, `tests/test_layout_crosscheck.py` covering valid/invalid dates, separators, ambiguous cases, pivot behavior, and violation reporting logic.
   - *Deliverable:* fast, photo-free tests.

**Exit criteria:** Parsing/pivot/layout logic is deterministic, tested, and explains chosen interpretation.

**Dependencies:** Milestones 0, 4 (for integration), but most logic testable independently.

---

## Milestone 6 — Confidence, plausibility, and classification (Stage 4) (Session 7)
**Goal:** produce trustworthy per-photo scores and categories that drive review and safe writes.

**Tasks (atomic):**
1. **Score components**
   - Combine: detector confidence (if fallback used), **worst per-digit match score** (not average), digit count/spacing sanity, plausibility checks.
   - *Deliverable:* scoring functions in `confidence.py`.
2. **Plausibility rules**
   - Valid calendar date, not future, not before ~1970, not after file’s own scan date, year within archive-observed range (configurable bounds/constants).
   - *Deliverable:* rule set returning reasons/penalties.
3. **Classification thresholds**
   - Map to `confident / needs review / failed / no stamp found` using **configurable constants at top of file/module** (single location).
   - Ensure `no_stamp` only when detection genuinely absent/empty per pipeline rules.
   - *Deliverable:* classifier + notes/reasons attached to each result.
4. **Unit tests (no photos)**
   - `tests/test_plausibility.py`, `tests/test_confidence.py` for thresholds/boundaries, worst-digit behavior, and classification outcomes.
   - *Deliverable:* deterministic tests.

**Exit criteria:** Every processed photo yields a classification and explainable confidence breakdown.

**Dependencies:** Milestones 2, 4, 5.

---

## Milestone 7 — Scan orchestration, caching/resume, and outputs (Stage 5) (Session 8, can split)
**Goal:** deliver the full non-destructive scan workflow and reviewer-friendly artifacts.

**Tasks (atomic):**
1. **`scan` pipeline orchestration**
   - Wire stages: ensure calibration/glyphs exist (auto-run if missing), detect → segment → recognize → parse → confidence for each image.
   - Support `--out`, `--field-order`, `--limit N`; deterministic processing order.
   - *Deliverable:* end-to-end `scan` command (CPU-local).
2. **Caching & resumability**
   - Cache per-image detection/recognition results so re-runs don’t redo completed work; tolerate interruption and resume safely (atomic writes/temp files for cache/state).
   - *Deliverable:* `cache.py` or equivalent, stored under output/cache dir.
3. **`results.csv` writer**
   - Emit required columns and stable formatting; include per-digit scores and notes/reasons.
   - *Deliverable:* CSV artifact.
4. **`review.html` generator**
   - Self-contained single portable file: for each photo show blown-up stamp crop + read value + confidence, **sorted lowest confidence first**, classification color-coded.
   - Embed crops as **base64**; make scannable for ~2000 rows (compact layout, clear highlights/anchors, maybe filters/legend if simple and static—no external deps).
   - *Deliverable:* HTML artifact.
5. **`corrections.csv` stub & console summary**
   - Create stub user can hand-fill (authoritative in apply). 
   - Print summary: counts per classification, year distribution, digit coverage/missing digits warning, and any Stage 3 layout violations **loudly at top** when present.
   - *Deliverable:* stub + summary report.

**Exit criteria:** `datestamp.py scan samples --out results/` produces `results.csv`, `review.html`, `corrections.csv` without modifying photos.

**Dependencies:** Milestones 1–6.

---

## Milestone 8 — Metadata write & safety gates (Stage 6) (Session 9)
**Goal:** implement the only irreversible step with strict safeguards.

**Tasks (atomic):**
1. **`exiftool` discovery & diagnostics**
   - Detect on PATH; if missing, print clear Windows install/usage instructions (no traceback).
   - *Deliverable:* preflight checks in `apply.py`.
2. **Dry-run apply planning**
   - Read `results.csv`/cache + `corrections.csv` (corrections authoritative). 
   - Select only `confident` + corrected entries; list/skip others with reasons.
   - Default mode prints exactly what would be written (no changes).
   - *Deliverable:* dry-run report.
3. **Apply mode (`--apply`)**
   - Set `DateTimeOriginal` to stamp date; keep `CreateDate` as scan date; fill time `12:00:00`.
   - Optional `--also-file-times` sets `FileModifyDate`.
   - Write via `exiftool -csv=` in batches (not per-file process).
   - Preserve `_original` backups by default; if user requests disabling backups, warn loudly. Print reversal instructions.
   - *Deliverable:* safe batch write implementation.
4. **Idempotency & error handling**
   - Handle partial batch failures gracefully with clear logs; do not corrupt unaffected files.

**Exit criteria:** Dry-run is informative and side-effect free; `--apply` only with explicit flag and correct field semantics.

**Dependencies:** Milestone 7 (result/correction formats), Milestone 0.

---

## Milestone 9 — Testing & end-to-end validation (Session 10)
**Goal:** lock behavior with unit tests and grade against provided samples.

**Tasks (atomic):**
1. **Unit test suite completion**
   - Ensure `tests/` covers parsing, century pivot, layout cross-check, plausibility rules (all runnable without photos).
   - *Deliverable:* green `pytest` (or `unittest`) locally on Windows.
2. **End-to-end sample test (`tests/test_e2e_samples.py`)**
   - Run pipeline over `./samples/` and score against **`samples/ground_truth.csv` (do not modify)**.
   - Grade separately: `digits` (recogniser) vs `date` (parsing/pivot). 
   - Pass conditions per spec: `no_stamp` rows pass only if tool detects nothing; `unreadable` rows not failures if declined/low-confidence but failures if confident wrong answer. Grade against printed stamp.
   - *Deliverable:* automated e2e assertions/reporting suitable for iterative improvement.
3. **Digit coverage/missing glyph report check**
   - Verify summary/test output reports which of digits `0–9` missing from reference glyph set.
4. **Performance/sanity (non-optimizing)**
   - Confirm runtime acceptable on CPU for <2000 photos; no premature optimization.

**Exit criteria:** Unit tests pass without photos; e2e produces a clear scorecard vs ground truth highlighting gaps.

**Dependencies:** Milestones 5–7 (and 3–4 for glyph/recognition).

---

## Milestone 10 — Documentation, polish, and handoff (Session 11)
**Goal:** make the tool usable and auditable for a one-off full archive run.

**Tasks (atomic):**
1. **Finalize `README.md`**
   - Windows setup (venv, pinned deps install), **how to install exiftool**, command examples.
   - How to read `review.html`, interpret classifications/confidence, and fill `corrections.csv`.
   - How to undo metadata write (exiftool backups/restore guidance) and warnings about disabling backups.
   - Decision log: measured calibration values, format/layout assumptions, pivot choice, fallback/model setup choices, handling of variable-length `digits` in ground truth, any open risks.
2. **Operational safeguards review**
   - Verify defaults are dry-run, no GUI, no file moves/renames, no LLM, no runtime network calls.
3. **Final sample run report**
   - Run `calibrate`, `glyphs`, `scan` on `samples/` and summarize results, mismatches vs ground truth, and digit coverage.

**Exit criteria:** A new user (on Windows, CPU-only) can follow README to scan, review, correct, and safely apply.

**Dependencies:** All prior milestones.

---

## Cross-cutting concerns (maintain throughout)
- **Configurability:** thresholds, HSV ranges, field order, pivot bounds, paths — centralized constants/config, not magic numbers.
- **Explainability artifacts:** keep intermediate crops/masks/debug outputs easy to enable; `review.html` and CSV must show why a result is low confidence.
- **Error handling:** clear, actionable messages (especially missing exiftool/model/glyph/calibration files); no tracebacks for expected user/setup issues.
- **Determinism:** stable sort order, seeded clustering where applicable, reproducible outputs given same inputs/config.
- **Separation of concerns:** detection vs recognition vs parsing vs I/O so fixes in one session don’t destabilize others.

## Suggested session breakdown (practical)
1. **S1:** Milestone 0 (scaffold, contracts, utils, deps, README skeleton)
2. **S2:** Milestone 1 (calibration on samples)
3. **S3:** Milestone 2a (HSV+geometry detection) *(optionally S3b for ONNX fallback)*
4. **S4:** Milestone 3 (segmentation + glyph bootstrap/cache)
5. **S5:** Milestone 4 (template matching + coverage)
6. **S6:** Milestone 5 (parse/pivot/layout + unit tests)
7. **S7:** Milestone 6 (confidence/plausibility/classification + unit tests)
8. **S8:** Milestone 7 (scan orchestration, cache/resume, CSV/HTML/corrections/summary)
9. **S9:** Milestone 8 (apply/exiftool dry-run & gated write)
10. **S10:** Milestone 9 (e2e vs ground truth, fix gaps surfaced)
11. **S11:** Milestone 10 (README, decision log, final polish)

*If time-constrained, prioritize a vertical slice early:* M0 → M1 → M2 (primary) → M3/M4 minimal → M5/M6 minimal → M7 minimal, then iterate to improve accuracy and add fallback/write polish.

## Key risks / open decisions to resolve early (and log)
- **Variable digit length/format in `ground_truth.csv` vs “six digits” expectation:** define canonical internal representation and how to compare/score `digits` fairly.
- **Auto-detecting date format vs default YYMMDD:** decide detection heuristics and when to trust positional constraints vs separators.
- **Glyph labeling without user input:** ensure alignment to `ground_truth.csv` is robust when segmentation count differs; define fallback when clusters unlabelled.
- **ONNX model availability:** setup-time acquisition/export steps must be documented since runtime network is forbidden.
- **EXIF orientation & rotated scans:** normalize coordinates/crops consistently so segmentation/recognition see upright digits.

## Definition of done (overall)
- `calibrate`, `glyphs`, `scan`, `apply` implemented per spec and Windows-safe.
- `calibration.json` committed with measured defaults; `glyphs/` inspectable; `results.csv`, `review.html`, `corrections.csv`, and console summary generated.
- `apply` defaults to dry-run, requires `--apply`, writes only allowed records, uses batched `exiftool`, and preserves backups by default.
- Unit tests (no photos) and e2e test against `samples/ground_truth.csv` in place; missing glyph digits reported.
- `README.md` documents setup, review workflow, undo procedure, and decision log.