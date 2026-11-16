## Task

Build a Python tool that reads the orange/red analog date stamp burned into scanned family
photos by 1980s–2000s film cameras, and writes that date into each file's EXIF metadata.
The scans currently have metadata reflecting the date they were scanned, not the date the
photo was taken.

Archive size is under 2000 photos. This runs once. **Correctness and reviewability matter
far more than speed.** Do not optimise for throughput.

**Work autonomously.** Sample photos are in `./samples/`. Inspect them, calibrate against
them, build the tool, test it, and report results at the end. Do not stop partway to ask me
questions or request confirmation — make a reasonable decision, write down why in the
README, and keep going.

## Known facts about these photos

These have been verified by hand against the samples. Treat them as given:

- **The date format varies across photos.** While many follow YY MM DD, others may use different formats such as MM-DD-YY or DD/MM/YY. The tool should attempt to auto-detect the format based on the stamp location and digit patterns.
- **The font varies by camera.** Some cameras use solid numerals while others use true seven-segment fonts with separate unconnected bars. The recogniser must handle both styles.
  are either continuous strokes or a small number of separate bars. Do not build a
  dot-matrix path.

## Environment and hard constraints

- **Windows 11.** Everything must run natively on Windows. No WSL, no Docker, no bash-only
  tooling. Handle paths with spaces and non-ASCII characters (Dutch filenames, accented
  names), and set UTF-8 output so the console doesn't mangle them.
- **AMD GPU (RX 9070 XT), no CUDA.** Assume CPU-only execution. Do not add a CUDA
  dependency. If you use a neural detector, prefer ONNX Runtime over PyTorch so the install
  stays small; CPU inference is fine at this volume.
- **Everything runs locally. No paid APIs, no cloud OCR, no network calls at runtime.**
  Downloading a model file once during setup is fine. Calling a hosted LLM or OCR service is
  not.
- Python 3.11+, in a venv, with a `requirements.txt` that pins versions.
- `exiftool` is used for metadata writing. Detect it on PATH, and if it's missing, print
  clear instructions rather than failing with a traceback.

## Pipeline

### Stage 0 — Calibrate against the samples

- Measure, from the actual images in `./samples/`, rather than assuming:

- - Which corner the stamp sits in and roughly what fraction of the frame it occupies
- - The HSV range of the stamp pixels (may vary by camera)
- - Whether the numerals are true seven-segment (separate unconnected bars) or solid strokes (varies by camera)
- - Image dimensions, scan DPI, orientation variation, whether EXIF orientation tags are set
- - Whether all samples look like one camera or several different stamp styles (multiple cameras expected)

Write the measured values into a `calibration.json` that the tool loads at runtime, and log
what was measured and how. Keep the calibration step re-runnable so it can be pointed at a
different batch later, but don't require it — the committed defaults should work on my
archive out of the box.

### Stage 1 — Locate the stamp

Primary path: **colour mask plus geometry.** Convert to HSV, keep high-saturation
high-value pixels in the red/orange band, restrict to the plausible corner region, group
into a digit row via connected components and their alignment. Simple, dependency-light,
fully inspectable.

Fallback for images where the mask finds nothing or finds something implausible:
**`pike00/yolo-date-stamp-detector` on Hugging Face**, a YOLO model fine-tuned specifically
to box orange/red date stamps in scanned photos, trained with negatives so an empty result
genuinely means "no stamp". Export it to ONNX and run it under ONNX Runtime.

Check all four corners, since some scans come in rotated.

### Stage 2 — Read the digits

**Template matching is the primary recogniser.** Segment digits by connected components,
normalise each to a fixed box, compare against ten reference glyphs. The whole archive comes
from one or two cameras with one font, so this should be near-perfect, and every failure is
inspectable. This works whether the numerals turn out seven-segment or solid.

Build the reference glyph set **automatically**, without asking me to label anything:

1. Run detection across the samples and collect every individual digit crop.
2. Cluster the crops — there are only ten classes, and one font, so they separate cleanly.
3. Label each cluster. Prefer the hand-labelled `digits` column in
   `samples/ground_truth.csv` (see Tests below): align each detected digit position with
   the corresponding character and take the majority label per cluster. Where that leaves a
   cluster unlabelled, fall back to a majority vote from a local OCR pass (ssocr if Stage 0
   finds true seven-segment numerals, otherwise Tesseract with `--psm 10` and a digit
   whitelist, or RapidOCR). Per-instance errors don't matter here, because the vote is over
   the whole cluster.
4. Sanity-check the labelling against positional constraints: in YY MM DD, the month's
   tens digit is only ever 0 or 1, and the day's tens digit only 0–3. A labelling that
   violates these is wrong; retry or flag it.
5. Cache the resulting glyphs to `glyphs/` as images so I can eyeball them, and so the step
   is one-time.

Then template-match the full archive against that glyph set.

Note: general-purpose OCR including Tesseract does badly on seven-segment digits, because
the gaps between segments make it split one digit into several fragments. That's why it's
used only for bootstrap labelling of clusters, never as the main recogniser.

### Stage 3 — Parse and cross-check the layout

Parse as YY MM DD. Two-digit years map to 19xx, except values low enough to be plausibly
2000s given the rest of the archive — infer the pivot from the observed year distribution
and log which pivot was chosen.

As a safety net, still run the pooled check across the whole batch: field 2 should never
exceed 12 and field 3 should never exceed 31. If either is violated in more than a trivial
number of photos, the layout assumption is wrong for part of the archive — report this
loudly at the top of the summary rather than silently producing wrong dates. Keep a
`--field-order` flag so the assumption can be overridden.

### Stage 4 — Confidence and validation

Every photo gets a confidence score, composed from:

- Detector confidence (if the model fallback was used)
- The worst per-digit match score across the six digits — not the average, since one bad
  digit ruins the date
- Whether the digit count and spacing look right
- Plausibility: valid calendar date, not in the future, not before ~1970, not after the
  file's own scan date, year within the range the rest of the archive suggests

Classify each photo into `confident` / `needs review` / `failed` / `no stamp found`, with
the thresholds as configurable constants at the top of the file, not scattered magic
numbers.

### Stage 5 — Review output

Produce, without touching any photo files:

- **`results.csv`** — filename, raw digits read, parsed date, per-digit scores, overall
  confidence, classification, notes.
- **`review.html`** — a self-contained page showing, for each photo, the cropped stamp image
  blown up next to what was read and the confidence, **sorted lowest confidence first**,
  with the classification colour-coded. Embed the crops as base64 so it's one portable file.
  This is what I'll actually use to verify the run, so make it genuinely scannable: I want
  to check 2000 rows quickly and have my eye drawn to the wrong ones.
- **`corrections.csv`** — a stub I can hand-fill for anything misread, which the write step
  reads back and takes as authoritative over the detected value.
- A console summary: counts per classification, year distribution, digit coverage, and any
  layout violations from Stage 3.

### Stage 6 — Write metadata

Separate command. Requires an explicit `--apply` flag; default is a dry run that prints what
it would do. This is the one place I do want a gate, because it's the only irreversible step.

- Set **`DateTimeOriginal`** to the stamp date (when the photo was taken).
- **Leave `CreateDate` as the scan date** — in EXIF that field means when the image was
  digitised, so keeping it preserves both facts and makes Apple Photos, Google Photos and
  Lightroom sort correctly.
- Fill the time as **12:00:00**, since only the day is known and midday prevents timezone
  shifts from moving a photo onto the wrong date.
- Optionally also set `FileModifyDate` behind a flag, so Windows Explorer sorts correctly.
- Only write to photos classified `confident`, plus anything I've filled into
  `corrections.csv`. Skip the rest and list them.
- Let exiftool keep its `_original` backups, and warn loudly if I pass the flag that
  disables them. Print how to reverse the whole operation.
- Write via `exiftool -csv=` in batches rather than one process per file.

## CLI

```
datestamp.py calibrate <photo_dir>       # re-measure colour/geometry, rewrite calibration.json
datestamp.py glyphs    <photo_dir>       # build the reference digit set
datestamp.py scan      <photo_dir> [--out results/] [--field-order YMD] [--limit N]
datestamp.py apply     <photo_dir> --results results/ [--apply] [--also-file-times]
```

`scan` should run `calibrate` and `glyphs` automatically if their outputs are missing, so
the normal case is a single command. Cache detection results so re-running `scan` doesn't
redo completed work, and make the whole thing resumable if I interrupt it.

## Tests

- A `tests/` directory with unit tests for date parsing, the century pivot, the layout
  cross-check and the plausibility rules, all runnable without any photos.
- An end-to-end test over `./samples/`, scored against **`samples/ground_truth.csv`, which
  I have filled in by hand.** Do not generate or overwrite this file, and do not substitute
  your own reading of the stamps for it — it is the reference the tool is graded against.
  Its columns are:

  | column | meaning |
  |---|---|
  | `filename` | exactly as on disk, including extension |
  | `digits` | the six characters as literally printed, `?` for an illegible one, blank if no stamp |
  | `date` | resolved date as `YYYY-MM-DD`, blank if unreadable or absent |
  | `status` | `ok`, `no_stamp`, `unreadable`, or `partial` |
  | `notes` | freeform |

  Score the two columns separately: `digits` measures the recogniser, `date` measures
  parsing and the century pivot. Rows marked `no_stamp` are passes only if the tool detects
  nothing. Rows marked `unreadable` are not failures if the tool declines them or flags them
  low-confidence — they are failures if it returns a confident answer. A note saying the
  camera clock was wrong does not mean the stamp was misread; grade against what is printed.
- A check that reports **which of the digits 0–9 are missing** from the reference glyph set.
  Print this as a warning in the summary — if the samples never contained a `7`, I need to
  know that before trusting a full run.

## Non-goals

- Don't build a GUI.
- Don't try to handle photos with no date stamp — classify and skip them.
- Don't modify, move, or rename original files at any point except the final metadata write.
- Don't add a language model to the pipeline. A model that invents a plausible date when the
  digits are ambiguous is worse than one that reports a low score.

## Deliverables

Working code, `requirements.txt`, and a `README.md` covering Windows setup including how to
install exiftool, how to read the review page, how to undo the metadata write, and a short
log of the decisions you made where the samples left something open.
