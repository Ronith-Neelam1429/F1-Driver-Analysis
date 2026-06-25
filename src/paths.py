"""Project path helpers for portable, machine-independent file access."""

from pathlib import Path


def get_project_root() -> Path:
    """Return the repository root (directory containing requirements.txt)."""
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "requirements.txt").exists():
            return candidate
    return Path.cwd()


PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RAW_FASTF1_DIR = DATA_DIR / "raw_data" / "fastf1"
PROCESSED_DIR = DATA_DIR / "processed"

QUALIFYING_RAW = RAW_FASTF1_DIR / "australia_2025_telemetry_qualifying.csv"
RACE_RAW = RAW_FASTF1_DIR / "australia_2025_telemetry_race.csv"
QUALIFYING_PROCESSED = PROCESSED_DIR / "australia_2025_quali_telemetry_processed.csv"

QUALIFYING_RAW_GZ = Path(str(QUALIFYING_RAW) + ".gz")
RACE_RAW_GZ = Path(str(RACE_RAW) + ".gz")
QUALIFYING_PROCESSED_GZ = Path(str(QUALIFYING_PROCESSED) + ".gz")
