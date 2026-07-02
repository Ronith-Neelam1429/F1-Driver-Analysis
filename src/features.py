"""Driving-style feature extraction from qualifying telemetry."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis import fastest_lap_telemetry, parse_laptime_seconds, to_numeric_brake

# Minimum consecutive braking time (seconds) to count as a zone.
MIN_BRAKE_ZONE_DURATION = 0.25
# Deceleration threshold (m/s²) when Brake channel is unavailable.
DECEL_THRESHOLD = -2.0
# Lap-distance bins for speed / brake profiles (0–100 %).
LAP_PROFILE_BINS = 10

STYLE_FEATURE_COLUMNS = [
    "Avg_Speed",
    "Max_Speed",
    "Avg_Throttle",
    "Brake_Pct",
    "Avg_RPM",
    "Brake_Zones",
    "Brake_Time_Pct",
    "Mean_Entry_Speed",
    "Mean_Apex_Speed",
    "Mean_Exit_Speed",
    "Mean_Brake_Duration",
    "Std_Brake_Duration",
    "Mean_Peak_Decel",
    "Max_Peak_Decel",
    "Sector1_Ratio",
    "Sector2_Ratio",
    "Sector3_Ratio",
    "Speed_Pct_10",
    "Speed_Pct_20",
    "Speed_Pct_30",
    "Speed_Pct_40",
    "Speed_Pct_50",
    "Speed_Pct_60",
    "Speed_Pct_70",
    "Speed_Pct_80",
    "Speed_Pct_90",
    "Brake_First_Third",
    "Brake_Mid_Third",
    "Brake_Last_Third",
]


def _is_braking(row: pd.Series, brake_num: int) -> bool:
    if brake_num == 1:
        return True
    accel = row.get("Acceleration")
    return pd.notna(accel) and accel < DECEL_THRESHOLD


def add_lap_distance_pct(lap: pd.DataFrame) -> pd.DataFrame:
    """Cumulative track distance normalized to 0–100 % of lap."""
    lap = lap.sort_values("TimeInLap").copy()
    dx = lap["X"].diff() if "X" in lap.columns else pd.Series(0, index=lap.index)
    dy = lap["Y"].diff() if "Y" in lap.columns else pd.Series(0, index=lap.index)
    step = np.sqrt(dx.astype(float) ** 2 + dy.astype(float) ** 2).fillna(0)
    cum = step.cumsum()
    total = cum.iloc[-1] if len(cum) else 0.0
    if total > 0:
        lap["LapPct"] = cum / total * 100.0
    elif len(lap) > 1 and lap["TimeInLap"].iloc[-1] > 0:
        lap["LapPct"] = lap["TimeInLap"] / lap["TimeInLap"].iloc[-1] * 100.0
    else:
        lap["LapPct"] = np.zeros(len(lap))
    return lap


def detect_braking_zones(lap: pd.DataFrame) -> list[pd.DataFrame]:
    """Return contiguous braking-zone slices from a single lap, time-ordered."""
    lap = lap.sort_values("TimeInLap").reset_index(drop=True)
    if lap.empty:
        return []

    brake = to_numeric_brake(lap["Brake"])
    zones: list[pd.DataFrame] = []
    start: int | None = None

    for i in range(len(lap)):
        braking = _is_braking(lap.iloc[i], int(brake.iloc[i]))
        if braking and start is None:
            start = i
        elif not braking and start is not None:
            zone = lap.iloc[start:i]
            duration = zone["TimeInLap"].iloc[-1] - zone["TimeInLap"].iloc[0]
            if duration >= MIN_BRAKE_ZONE_DURATION:
                zones.append(zone)
            start = None

    if start is not None:
        zone = lap.iloc[start:]
        duration = zone["TimeInLap"].iloc[-1] - zone["TimeInLap"].iloc[0]
        if duration >= MIN_BRAKE_ZONE_DURATION:
            zones.append(zone)

    return zones


def _zone_features(zone: pd.DataFrame) -> dict[str, float]:
    speeds = zone["Speed"]
    entry_speed = float(speeds.iloc[0])
    apex_speed = float(speeds.min())
    exit_speed = float(speeds.iloc[-1])
    duration = float(zone["TimeInLap"].iloc[-1] - zone["TimeInLap"].iloc[0])

    decel = zone.loc[zone["Acceleration"] < -0.5, "Acceleration"]
    peak_decel = float(decel.min()) if len(decel) else 0.0

    return {
        "entry_speed": entry_speed,
        "apex_speed": apex_speed,
        "exit_speed": exit_speed,
        "duration": duration,
        "peak_decel": peak_decel,
    }


def _lap_profile_features(lap: pd.DataFrame) -> dict[str, float]:
    lap = add_lap_distance_pct(lap)
    brake = to_numeric_brake(lap["Brake"])
    lap = lap.assign(Brake_num=brake)

    out: dict[str, float] = {}
    pct_points = list(range(10, 100, 10))
    for pct in pct_points:
        window = lap[(lap["LapPct"] >= pct - 5) & (lap["LapPct"] < pct + 5)]
        out[f"Speed_Pct_{pct}"] = float(window["Speed"].mean()) if len(window) else np.nan

    for label, lo, hi in (
        ("Brake_First_Third", 0, 33.3),
        ("Brake_Mid_Third", 33.3, 66.6),
        ("Brake_Last_Third", 66.6, 100.0),
    ):
        window = lap[(lap["LapPct"] >= lo) & (lap["LapPct"] < hi)]
        out[label] = float(window["Brake_num"].mean() * 100) if len(window) else np.nan

    return out


def _sector_ratios(driver_lap: pd.DataFrame, round_df: pd.DataFrame) -> dict[str, float]:
    """Driver sector time relative to the field best (1.0 = session best)."""
    work = round_df.copy()
    work["LapTime_seconds"] = parse_laptime_seconds(work["LapTime"])
    work["S1"] = parse_laptime_seconds(work["Sector1Time"])
    work["S2"] = parse_laptime_seconds(work["Sector2Time"])
    work["S3"] = parse_laptime_seconds(work["Sector3Time"])

    per_lap = (
        work.groupby(["Driver", "LapNumber"], as_index=False)
        .agg(S1=("S1", "first"), S2=("S2", "first"), S3=("S3", "first"))
    )
    best = {
        "S1": per_lap["S1"].min(),
        "S2": per_lap["S2"].min(),
        "S3": per_lap["S3"].min(),
    }

    driver = driver_lap["Driver"].iloc[0]
    driver_rows = per_lap[per_lap["Driver"] == driver]
    if driver_rows.empty:
        return {"Sector1_Ratio": np.nan, "Sector2_Ratio": np.nan, "Sector3_Ratio": np.nan}

    fastest = driver_rows.loc[
        driver_rows[["S1", "S2", "S3"]].sum(axis=1).idxmin()
    ]
    return {
        "Sector1_Ratio": float(fastest["S1"] / best["S1"]) if best["S1"] > 0 else np.nan,
        "Sector2_Ratio": float(fastest["S2"] / best["S2"]) if best["S2"] > 0 else np.nan,
        "Sector3_Ratio": float(fastest["S3"] / best["S3"]) if best["S3"] > 0 else np.nan,
    }


def extract_lap_style_features(lap: pd.DataFrame, round_df: pd.DataFrame) -> dict[str, float]:
    """Style features for a single lap telemetry slice."""
    lap = lap.sort_values("TimeInLap")
    if len(lap) < 20:
        return {}

    brake = to_numeric_brake(lap["Brake"])
    lap_time = lap["TimeInLap"].iloc[-1] - lap["TimeInLap"].iloc[0]

    zones = detect_braking_zones(lap)
    zone_stats = [_zone_features(z) for z in zones]
    brake_time = sum(z["duration"] for z in zone_stats)

    features: dict[str, float] = {
        "Avg_Speed": float(lap["Speed"].mean()),
        "Max_Speed": float(lap["Speed"].max()),
        "Avg_Throttle": float(lap["Throttle"].mean()),
        "Brake_Pct": float(brake.mean() * 100),
        "Avg_RPM": float(lap["RPM"].mean()),
        "Brake_Zones": float(len(zones)),
        "Brake_Time_Pct": float(brake_time / lap_time * 100) if lap_time > 0 else 0.0,
    }

    if zone_stats:
        features.update(
            {
                "Mean_Entry_Speed": float(np.mean([z["entry_speed"] for z in zone_stats])),
                "Mean_Apex_Speed": float(np.mean([z["apex_speed"] for z in zone_stats])),
                "Mean_Exit_Speed": float(np.mean([z["exit_speed"] for z in zone_stats])),
                "Mean_Brake_Duration": float(np.mean([z["duration"] for z in zone_stats])),
                "Std_Brake_Duration": float(np.std([z["duration"] for z in zone_stats])),
                "Mean_Peak_Decel": float(np.mean([z["peak_decel"] for z in zone_stats])),
                "Max_Peak_Decel": float(np.min([z["peak_decel"] for z in zone_stats])),
            }
        )
    else:
        features.update(
            {
                "Mean_Entry_Speed": np.nan,
                "Mean_Apex_Speed": np.nan,
                "Mean_Exit_Speed": np.nan,
                "Mean_Brake_Duration": np.nan,
                "Std_Brake_Duration": np.nan,
                "Mean_Peak_Decel": np.nan,
                "Max_Peak_Decel": np.nan,
            }
        )

    features.update(_sector_ratios(lap, round_df))
    features.update(_lap_profile_features(lap))
    return features


def extract_round_driver_features(round_df: pd.DataFrame) -> pd.DataFrame:
    """Per-driver style features from each driver's fastest qualifying lap."""
    round_num = int(round_df["Round"].iloc[0])
    drivers = sorted(round_df["Driver"].dropna().unique())
    rows: list[dict] = []

    for driver in drivers:
        lap = fastest_lap_telemetry(round_df, driver)
        if len(lap) < 50:
            continue
        feats = extract_lap_style_features(lap, round_df)
        if not feats:
            continue
        feats["Driver"] = driver
        feats["Round"] = round_num
        rows.append(feats)

    if not rows:
        return pd.DataFrame(columns=["Driver", "Round"] + STYLE_FEATURE_COLUMNS)

    out = pd.DataFrame(rows)
    return out[["Driver", "Round"] + STYLE_FEATURE_COLUMNS]


def zscore_within_round(round_features: pd.DataFrame) -> pd.DataFrame:
    """Z-score style columns within a single round so circuit effects are reduced."""
    work = round_features.copy()
    for col in STYLE_FEATURE_COLUMNS:
        if col not in work.columns:
            continue
        vals = work[col]
        std = vals.std()
        if pd.isna(std) or std == 0:
            work[col] = 0.0
        else:
            work[col] = (vals - vals.mean()) / std
    return work


def aggregate_driver_features(per_round: pd.DataFrame) -> pd.DataFrame:
    """Mean of round-normalized features → one row per driver."""
    if per_round.empty:
        return pd.DataFrame(columns=["Driver"] + STYLE_FEATURE_COLUMNS)

    normalized_parts: list[pd.DataFrame] = []
    for round_num, grp in per_round.groupby("Round"):
        normalized_parts.append(zscore_within_round(grp))

    normalized = pd.concat(normalized_parts, ignore_index=True)
    agg = (
        normalized.groupby("Driver")[STYLE_FEATURE_COLUMNS]
        .mean()
        .reset_index()
    )
    return agg


def build_driver_feature_matrix(per_round: pd.DataFrame) -> pd.DataFrame:
    """Driver-level feature matrix ready for clustering."""
    return aggregate_driver_features(per_round)
