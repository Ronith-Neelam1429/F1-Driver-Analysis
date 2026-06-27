"""Shared telemetry processing helpers."""

import numpy as np
import pandas as pd

# Corner bounds keyed by round number (extend per circuit as needed)
CIRCUIT_CORNERS = {
    1: {
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
    },
}


def identify_corner(x, y, corners: dict | None) -> str | None:
    if corners is None or pd.isna(x) or pd.isna(y):
        return None

    min_distance = float('inf')
    closest_corner = None

    for corner_name, bounds in corners.items():
        corner_x = (bounds['x'][0] + bounds['x'][1]) / 2
        corner_y = (bounds['y'][0] + bounds['y'][1]) / 2
        distance = np.sqrt((x - corner_x) ** 2 + (y - corner_y) ** 2)

        if distance < min_distance:
            min_distance = distance
            closest_corner = corner_name

    return closest_corner if min_distance < 500 else None


def add_acceleration(quali_df: pd.DataFrame) -> pd.DataFrame:
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


def process_qualifying_dataframe(quali_df: pd.DataFrame, *, round_num: int | None = None) -> pd.DataFrame:
    corners = CIRCUIT_CORNERS.get(round_num) if round_num is not None else None

    if corners is not None:
        quali_df = quali_df.copy()
        quali_df['Corner'] = quali_df.apply(
            lambda row: identify_corner(row['X'], row['Y'], corners),
            axis=1,
        )
    else:
        quali_df = quali_df.copy()
        quali_df['Corner'] = None

    return add_acceleration(quali_df)
