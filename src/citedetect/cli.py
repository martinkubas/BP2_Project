#!/usr/bin/env python3
from __future__ import annotations

import argparse

from .detector import process_pdf


def main() -> None:
    p = argparse.ArgumentParser(description="Detect IEEE-style numeric citations and export cited passages as JSON.")
    p.add_argument("pdf", help="Input PDF path")
    p.add_argument("--out", help="Write JSON output to this path", default="cites.json")
    args = p.parse_args()

    process_pdf(
        pdf_path=args.pdf,
        out_json_path=args.out,
        json_indent=2,
    )


if __name__ == "__main__":
    main()
