#!/usr/bin/env python3
"""
Process qualifying telemetry with acceleration, deceleration, and corner detection.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from generate_2025_season import process_round_qualifying


def process_qualifying_telemetry(*, compress: bool = True):
    """Process Australia 2025 qualifying telemetry (round 1)."""
    ok = process_round_qualifying(1, 'Australia', compress=compress)
    if not ok:
        raise FileNotFoundError(
            "Qualifying telemetry not found for Australia 2025. "
            "Run: python generate_2025_season.py"
        )


if __name__ == "__main__":
    process_qualifying_telemetry()
