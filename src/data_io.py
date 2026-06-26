"""Read/write helpers for CSV and compressed CSV data."""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import pandas as pd

SAMPLE_LINE_LIMIT = 1800


def sample_csv_path(path: Path) -> Path:
    """Return the `_sample.csv` path for a given CSV."""
    return path.with_name(f"{path.stem}_sample{path.suffix}")


def resolve_data_path(path: Path) -> Path:
    """Prefer uncompressed CSV; fall back to .csv.gz if present."""
    if path.exists():
        return path
    gz_path = Path(str(path) + ".gz")
    if gz_path.exists():
        return gz_path
    return path


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Load a CSV or gzip-compressed CSV."""
    resolved = resolve_data_path(path)
    return pd.read_csv(resolved, low_memory=False, **kwargs)


def write_sample_csv(path: Path, *, n_lines: int = SAMPLE_LINE_LIMIT) -> Path | None:
    """Write the first `n_lines` lines of a CSV (including header) to a `_sample.csv` file."""
    source = resolve_data_path(path)
    if not source.exists():
        return None

    sample_path = sample_csv_path(path)
    sample_path.parent.mkdir(parents=True, exist_ok=True)

    if str(source).endswith(".gz"):
        src_ctx = gzip.open(source, "rt", encoding="utf-8", newline="")
    else:
        src_ctx = open(source, "r", encoding="utf-8", newline="")

    with src_ctx as src, open(sample_path, "w", encoding="utf-8", newline="") as dst:
        for i, line in enumerate(src):
            if i >= n_lines:
                break
            dst.write(line)

    return sample_path


def write_csv(
    df: pd.DataFrame,
    path: Path,
    *,
    compress: bool = False,
    **kwargs,
) -> Path:
    """Write CSV; optionally also write a gzip companion for version control."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, **kwargs)

    if not compress:
        return path

    gz_path = Path(str(path) + ".gz")
    with open(path, "rb") as src, gzip.open(gz_path, "wb", compresslevel=9) as dst:
        shutil.copyfileobj(src, dst)

    return path
