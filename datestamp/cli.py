"""Command-line interface for datestamp tool."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .io_utils import configure_windows_utf8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datestamp.py",
        description="Read analog date stamps from scanned photos and manage EXIF metadata.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_cal = sub.add_parser("calibrate", help="Measure colour/geometry and write calibration.json")
    p_cal.add_argument("photo_dir", help="Directory containing sample photos")
    p_cal.set_defaults(handler=_cmd_calibrate)

    p_glyphs = sub.add_parser("glyphs", help="Build the reference digit glyph set")
    p_glyphs.add_argument("photo_dir", help="Directory containing photos to derive glyphs from")
    p_glyphs.set_defaults(handler=_cmd_glyphs)

    p_scan = sub.add_parser("scan", help="Scan photos and produce review outputs")
    p_scan.add_argument("photo_dir", help="Directory containing photos to scan")
    p_scan.add_argument("--out", dest="out_dir", default="results", help="Output directory (default: results)")
    p_scan.add_argument("--field-order", dest="field_order", default="YMD", help="Date field order override (default: YMD)")
    p_scan.add_argument("--limit", dest="limit", type=int, default=None, help="Limit number of photos processed")
    p_scan.set_defaults(handler=_cmd_scan)

    p_apply = sub.add_parser("apply", help="Write metadata from scan results (dry-run by default)")
    p_apply.add_argument("photo_dir", help="Directory containing photo files to update")
    p_apply.add_argument("--results", dest="results_dir", required=True, help="Directory containing scan results/corrections")
    p_apply.add_argument("--apply", dest="do_apply", action="store_true", help="Actually write metadata (default: dry-run)")
    p_apply.add_argument(
        "--also-file-times",
        dest="also_file_times",
        action="store_true",
        help="Also set FileModifyDate (behind explicit flag)",
    )
    p_apply.set_defaults(handler=_cmd_apply)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_windows_utf8()
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2
    return int(handler(args))


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from .calibration import run_calibrate

    return run_calibrate(Path(args.photo_dir))


def _cmd_glyphs(args: argparse.Namespace) -> int:
    from .glyphs import run_glyphs

    return run_glyphs(Path(args.photo_dir))


def _cmd_scan(args: argparse.Namespace) -> int:
    from .calibration import run_calibrate
    from .glyphs import run_glyphs

    photo_dir = Path(args.photo_dir)
    out_dir = Path(args.out_dir)
    limit = args.limit
    field_order = args.field_order

    # scan should auto-run calibrate/glyphs if outputs missing (spec) - placeholders for now
    cal_path = Path("calibration.json")
    glyphs_dir = Path("glyphs")
    if not cal_path.exists():
        print(f"calibration.json missing; running calibrate on {photo_dir} ...")
        rc = run_calibrate(photo_dir)
        if rc != 0:
            return rc
    if not glyphs_dir.exists() or not any(glyphs_dir.iterdir()):
        print(f"glyphs/ missing or empty; running glyphs on {photo_dir} ...")
        rc = run_glyphs(photo_dir)
        if rc != 0:
            return rc

    print("scan: pipeline not implemented yet (Milestone 0 scaffold).")
    print(f"  photo_dir={photo_dir}")
    print(f"  out_dir={out_dir}")
    print(f"  field_order={field_order}")
    print(f"  limit={limit}")
    return 0


def _cmd_apply(args: argparse.Namespace) -> int:
    from .apply import run_apply

    return run_apply(
        photo_dir=Path(args.photo_dir),
        results_dir=Path(args.results_dir),
        do_apply=bool(args.do_apply),
        also_file_times=bool(args.also_file_times),
    )
