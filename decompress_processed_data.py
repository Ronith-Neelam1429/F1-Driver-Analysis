#!/usr/bin/env python3
"""Decompress .csv.gz processed data for local inspection (optional)."""

import gzip
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import PROCESSED_DIR


def decompress_file(gz_path: Path) -> Path:
    csv_path = Path(str(gz_path)[:-3])  # strip .gz
    with gzip.open(gz_path, "rb") as src, open(csv_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    print(f"Decompressed {gz_path.name} -> {csv_path.name}")
    return csv_path


def main():
    targets = [Path(p) for p in sys.argv[1:]] if len(sys.argv) > 1 else sorted(PROCESSED_DIR.glob("*.csv.gz"))

    if not targets:
        print(f"No .csv.gz files found in {PROCESSED_DIR}")
        return

    for gz_path in targets:
        if not gz_path.exists():
            print(f"Skip (missing): {gz_path}")
            continue
        decompress_file(gz_path)


if __name__ == "__main__":
    main()
