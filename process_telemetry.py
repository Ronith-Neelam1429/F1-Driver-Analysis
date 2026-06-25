#!/usr/bin/env python3
"""
Process qualifying telemetry with acceleration, deceleration, and corner detection.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_io import read_csv, resolve_data_path, write_csv, write_sample_csv, SAMPLE_LINE_LIMIT
from src.paths import QUALIFYING_PROCESSED, QUALIFYING_RAW

# Albert Park corner coordinates (approximate center points)
ALBERT_PARK_CORNERS = {
    'Turn 1': {'x': (-900, -700), 'y': (-1600, -1400)},
    'Turn 2': {'x': (-600, -400), 'y': (-1100, -900)},
    'Turn 3-4': {'x': (-200, 100), 'y': (-600, -400)},
    'Turn 5-6': {'x': (400, 600), 'y': (-200, 100)},
    'Turn 7': {'x': (800, 1000), 'y': (300, 500)},
    'Turn 8': {'x': (1200, 1400), 'y': (900, 1100)},
    'Turn 9-10': {'x': (1500, 1700), 'y': (1300, 1500)},
    'Turn 11': {'x': (1200, 1400), 'y': (1600, 1800)},
    'Turn 12': {'x': (600, 800), 'y': (1700, 1900)},
    'Turn 13': {'x': (0, 200), 'y': (1500, 1700)},
}


def identify_corner(x, y):
    """Identify which corner a position is closest to."""
    if pd.isna(x) or pd.isna(y):
        return None

    min_distance = float('inf')
    closest_corner = None

    for corner_name, bounds in ALBERT_PARK_CORNERS.items():
        corner_x = (bounds['x'][0] + bounds['x'][1]) / 2
        corner_y = (bounds['y'][0] + bounds['y'][1]) / 2
        distance = np.sqrt((x - corner_x) ** 2 + (y - corner_y) ** 2)

        if distance < min_distance:
            min_distance = distance
            closest_corner = corner_name

    return closest_corner if min_distance < 500 else None


def add_acceleration(quali_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate per-lap acceleration aligned to each telemetry row."""
    quali_df = quali_df.sort_values(['DriverNumber', 'LapNumber', 'TimeInLap']).reset_index(drop=True)
    quali_df['Speed_ms'] = quali_df['Speed'] / 3.6
    quali_df['Acceleration'] = np.nan

    for _, group in quali_df.groupby(['DriverNumber', 'LapNumber'], sort=False):
        rows = group.sort_values('TimeInLap')
        prev_time = None
        prev_speed = None

        for idx, row in rows.iterrows():
            if prev_time is not None:
                time_diff = row['TimeInLap'] - prev_time
                speed_diff = row['Speed_ms'] - prev_speed
                if time_diff > 0:
                    quali_df.loc[idx, 'Acceleration'] = np.clip(speed_diff / time_diff, -15, 15)
            prev_time = row['TimeInLap']
            prev_speed = row['Speed_ms']

    quali_df['Accel_Type'] = quali_df['Acceleration'].apply(
        lambda x: (
            'Acceleration' if x > 0.5 else ('Deceleration' if x < -0.5 else 'Neutral')
        )
        if pd.notna(x)
        else None
    )
    return quali_df


def process_qualifying_telemetry(*, compress: bool = True):
    """Process full qualifying telemetry with acceleration and corner data."""
    raw_path = resolve_data_path(QUALIFYING_RAW)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Qualifying telemetry not found at {QUALIFYING_RAW} (or .gz). "
            "Run: python src/data_extraction_fastf1.py or python extract_australia_telemetry.py"
        )

    print(f"Loading qualifying telemetry from {raw_path.name}...")
    quali_df = read_csv(QUALIFYING_RAW)
    print(f"Processing {len(quali_df):,} data points...")

    print("Identifying corners...")
    quali_df['Corner'] = quali_df.apply(
        lambda row: identify_corner(row['X'], row['Y']),
        axis=1,
    )

    print("Calculating acceleration and deceleration...")
    quali_df = add_acceleration(quali_df)

    print("Saving processed telemetry...")
    write_csv(quali_df, QUALIFYING_PROCESSED, index=False, compress=compress)
    sample_path = write_sample_csv(QUALIFYING_PROCESSED)
    print(f"Saved {QUALIFYING_PROCESSED.name}" + (" (+ .gz for git)" if compress else ""))
    if sample_path:
        print(f"Saved {sample_path.name} (first {SAMPLE_LINE_LIMIT} lines)")

    print("\n" + "=" * 70)
    print("QUALIFYING TELEMETRY PROCESSING COMPLETE")
    print("=" * 70)
    print(f"\nTotal records: {len(quali_df):,}")
    print(f"Drivers: {quali_df['Driver'].nunique()}")
    print(f"Corners identified: {quali_df['Corner'].nunique()}")

    return quali_df


if __name__ == "__main__":
    process_qualifying_telemetry()
