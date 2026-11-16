"""CLI entrypoint for datestamp OCR tool."""

from __future__ import annotations

import sys

from datestamp.cli import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
