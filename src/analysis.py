"""Reusable analytics transforms for the Australia 2025 telemetry dataset.

These functions are deliberately framework-agnostic (plain pandas) so they can be
used from the Streamlit dashboard, the notebook, or scripts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Team / driver colours keep charts readable and roughly on-brand.
DRIVER_COLORS = {
    "VER": "#1B2A4A", "TSU": "#3671C6",          # Red Bull
    "NOR": "#FF8000", "PIA": "#FF8000",          # McLaren
    "LEC": "#E8002D", "HAM": "#E8002D",          # Ferrari
    "RUS": "#27F4D2", "ANT": "#27F4D2",          # Mercedes
    "ALO": "#229971", "STR": "#229971",          # Aston Martin
    "GAS": "#0093CC", "DOO": "#0093CC", "COL": "#0093CC",  # Alpine
    "ALB": "#64C4FF", "SAI": "#64C4FF",          # Williams
    "HUL": "#52E252", "BOR": "#52E252",          # Sauber
    "OCO": "#B6BABD", "BEA": "#B6BABD",          # Haas
    "LAW": "#6692FF", "HAD": "#6692FF",          # Racing Bulls
}

DEFAULT_COLOR = "#9AA0A6"


def driver_color(code: str) -> str:
    """Return a hex colour for a driver abbreviation."""
    return DRIVER_COLORS.get(str(code).upper(), DEFAULT_COLOR)


def to_numeric_brake(series: pd.Series) -> pd.Series:
    """FastF1 stores Brake as a boolean; turn it into 0/1 for averaging."""
    if series.dtype == bool:
        return series.astype(int)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1, "false": 0, "1": 1, "0": 0})
        .fillna(0)
        .astype(int)
    )


def parse_laptime_seconds(series: pd.Series) -> pd.Series:
    """Parse FastF1 timedelta strings ('0 days 00:01:38.807000') to seconds."""
    return pd.to_timedelta(series, errors="coerce").dt.total_seconds()


def best_laps(df: pd.DataFrame) -> pd.DataFrame:
    """One row per driver: their fastest valid lap, sorted quickest first.

    Returns columns: Driver, LapNumber, LapTime_seconds, Compound, Gap.
    """
    work = df.copy()
    work["LapTime_seconds"] = parse_laptime_seconds(work["LapTime"])

    valid = work.dropna(subset=["LapTime_seconds"])
    if valid.empty:
        return pd.DataFrame(columns=["Driver", "LapNumber", "LapTime_seconds", "Compound", "Gap"])

    # Each (Driver, LapNumber) repeats across telemetry rows; collapse to one.
    per_lap = (
        valid.groupby(["Driver", "LapNumber"], as_index=False)
        .agg(LapTime_seconds=("LapTime_seconds", "first"),
             Compound=("Compound", "first"))
    )
    idx = per_lap.groupby("Driver")["LapTime_seconds"].idxmin()
    out = per_lap.loc[idx].sort_values("LapTime_seconds").reset_index(drop=True)
    out["Gap"] = out["LapTime_seconds"] - out["LapTime_seconds"].iloc[0]
    return out


def driver_telemetry_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-driver averages/maxes for the headline telemetry channels."""
    work = df.copy()
    work["Brake_num"] = to_numeric_brake(work["Brake"])
    summary = (
        work.groupby("Driver")
        .agg(
            Avg_Speed=("Speed", "mean"),
            Max_Speed=("Speed", "max"),
            Avg_Throttle=("Throttle", "mean"),
            Brake_Pct=("Brake_num", "mean"),
            Avg_RPM=("RPM", "mean"),
            Max_RPM=("RPM", "max"),
            Laps=("LapNumber", "nunique"),
            Data_Points=("Speed", "size"),
        )
        .reset_index()
    )
    summary["Brake_Pct"] = summary["Brake_Pct"] * 100
    return summary.round(1)


def fastest_lap_telemetry(df: pd.DataFrame, driver: str) -> pd.DataFrame:
    """Return the telemetry samples for a driver's fastest lap, time-ordered."""
    sub = df[df["Driver"] == driver].copy()
    if sub.empty:
        return sub
    sub["LapTime_seconds"] = parse_laptime_seconds(sub["LapTime"])
    sub = sub.dropna(subset=["LapTime_seconds"])
    if sub.empty:
        return sub
    fastest_lap = sub.loc[sub["LapTime_seconds"].idxmin(), "LapNumber"]
    lap = sub[sub["LapNumber"] == fastest_lap].copy()
    return lap.sort_values("TimeInLap").reset_index(drop=True)


def corner_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-driver, per-corner min speed and peak accel/decel.

    Only corners that were actually detected appear here.
    """
    work = df.dropna(subset=["Corner"]).copy()
    if work.empty:
        return pd.DataFrame(
            columns=["Driver", "Corner", "Min_Speed", "Max_Deceleration", "Max_Acceleration_Exit"]
        )

    rows = []
    for (driver, corner), g in work.groupby(["Driver", "Corner"]):
        decel = g[g["Accel_Type"] == "Deceleration"]["Acceleration"]
        accel = g[g["Accel_Type"] == "Acceleration"]["Acceleration"]
        rows.append(
            {
                "Driver": driver,
                "Corner": corner,
                "Min_Speed": g["Speed"].min(),
                "Max_Deceleration": decel.min() if len(decel) else np.nan,
                "Max_Acceleration_Exit": accel.max() if len(accel) else np.nan,
                "Data_Points": len(g),
            }
        )
    return pd.DataFrame(rows).sort_values(["Corner", "Min_Speed"]).reset_index(drop=True)
