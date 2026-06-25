#!/usr/bin/env python3
"""Compress processed CSV files for git-friendly storage."""

import gzip
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import PROCESSED_DIR, QUALIFYING_PROCESSED


def compress_file(csv_path: Path) -> Path | None:
    if not csv_path.exists():
        print(f"Skip (missing): {csv_path.name}")
        return None

    gz_path = Path(str(csv_path) + ".gz")
    with open(csv_path, "rb") as src, gzip.open(gz_path, "wb", compresslevel=9) as dst:
        shutil.copyfileobj(src, dst)

    raw_mb = csv_path.stat().st_size / (1024 * 1024)
    gz_mb = gz_path.stat().st_size / (1024 * 1024)
    ratio = (1 - gz_mb / raw_mb) * 100 if raw_mb else 0
    print(f"Compressed {csv_path.name}: {raw_mb:.1f} MB -> {gz_mb:.1f} MB ({ratio:.0f}% smaller)")
    return gz_path


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    targets = [QUALIFYING_PROCESSED]

    for csv in targets:
        compress_file(csv)

    for gz in PROCESSED_DIR.glob("*.csv.gz"):
        print(f"Ready for git: {gz.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
