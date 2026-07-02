"""Project path helpers for portable, machine-independent file access."""

import re
from pathlib import Path


def get_project_root() -> Path:
    """Return the repository root (directory containing requirements.txt)."""
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "requirements.txt").exists():
            return candidate
    return Path.cwd()


def slugify_country(country: str) -> str:
    slug = country.lower().strip()
    slug = re.sub(r'[^a-z0-9]+', '_', slug)
    return slug.strip('_')


def event_prefix(round_num: int, country: str) -> str:
    """Build a unique file prefix, e.g. r01_australia."""
    return f"r{round_num:02d}_{slugify_country(country)}"


PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RAW_FASTF1_DIR = DATA_DIR / "raw_data" / "fastf1"
PROCESSED_DIR = DATA_DIR / "processed"

# Backward-compatible aliases for Australia (round 1)
_AU_PREFIX = event_prefix(1, 'Australia')
QUALIFYING_RAW = RAW_FASTF1_DIR / f"{_AU_PREFIX}_2025_telemetry_qualifying.csv"
RACE_RAW = RAW_FASTF1_DIR / f"{_AU_PREFIX}_2025_telemetry_race.csv"
QUALIFYING_PROCESSED = PROCESSED_DIR / f"{_AU_PREFIX}_2025_quali_telemetry_processed.csv"

QUALIFYING_RAW_GZ = Path(str(QUALIFYING_RAW) + ".gz")
RACE_RAW_GZ = Path(str(RACE_RAW) + ".gz")
QUALIFYING_PROCESSED_GZ = Path(str(QUALIFYING_PROCESSED) + ".gz")


def raw_qualifying_path(round_num: int, country: str) -> Path:
    prefix = event_prefix(round_num, country)
    return RAW_FASTF1_DIR / f"{prefix}_2025_telemetry_qualifying.csv"


def raw_race_path(round_num: int, country: str) -> Path:
    prefix = event_prefix(round_num, country)
    return RAW_FASTF1_DIR / f"{prefix}_2025_telemetry_race.csv"


def processed_qualifying_path(round_num: int, country: str) -> Path:
    prefix = event_prefix(round_num, country)
    return PROCESSED_DIR / f"{prefix}_2025_quali_telemetry_processed.csv"


def season_summary_path(name: str) -> Path:
    return PROCESSED_DIR / name


def iter_processed_qualifying_2025() -> list[tuple[int, str, Path]]:
    """Yield (round_num, country_slug, path) for each 2025 processed qualifying file.

    Prefers full processed CSV / gzip; falls back to ``_sample.csv`` when only previews exist.
    """
    seen: set[int] = set()
    entries: list[tuple[int, str, Path]] = []

    patterns = (
        "r*_2025_quali_telemetry_processed.csv",
        "r*_2025_quali_telemetry_processed.csv.gz",
    )
    for pattern in patterns:
        for path in sorted(PROCESSED_DIR.glob(pattern)):
            if path.name.endswith("_sample.csv"):
                continue
            name = path.name.replace(".gz", "")
            match = re.match(r"r(\d+)_(.+)_2025_quali_telemetry_processed\.csv$", name)
            if not match:
                continue
            round_num = int(match.group(1))
            if round_num in seen:
                continue
            if _resolved_csv(PROCESSED_DIR / name) is not None:
                seen.add(round_num)
                entries.append((round_num, match.group(2), PROCESSED_DIR / name))

    for path in sorted(PROCESSED_DIR.glob("r*_2025_quali_telemetry_processed_sample.csv")):
        match = re.match(r"r(\d+)_(.+)_2025_quali_telemetry_processed_sample\.csv$", path.name)
        if not match:
            continue
        round_num = int(match.group(1))
        if round_num in seen:
            continue
        seen.add(round_num)
        entries.append((round_num, match.group(2), path))

    return sorted(entries, key=lambda x: x[0])


def _resolved_csv(path: Path) -> Path | None:
    if path.exists():
        return path
    gz_path = Path(str(path) + ".gz")
    if gz_path.exists():
        return gz_path
    return None
