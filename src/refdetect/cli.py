#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path
from .detector import extract_references, write_json

def main() -> None:
    ap = argparse.ArgumentParser(description="Extract references from a PDF and write JSON.")
    ap.add_argument("pdf", help="Path to input PDF")
    ap.add_argument("--out", "-o", default="refs.json", help="Output JSON path")
    args = ap.parse_args()

    in_path = Path(args.pdf)
    if not in_path.exists():
        print(f"Error: file not found: {in_path}", file=sys.stderr)
        sys.exit(2)

    refs = extract_references(str(in_path))
    write_json(refs, args.out)

    print(f"Extracted {len(refs)} references -> {args.out}")

if __name__ == "__main__":
    main()
