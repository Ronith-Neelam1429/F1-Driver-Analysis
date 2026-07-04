"""OpenF1 API telemetry extraction (fallback when FastF1 is unavailable)."""

from __future__ import annotations

import time
import urllib3
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests

from src.paths import RAW_FASTF1_DIR, PROCESSED_DIR, slugify_country, event_prefix

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://api.openf1.org/v1"
REQUEST_DELAY = 0.15


def _get(endpoint: str, params: dict | None = None) -> list | dict:
    time.sleep(REQUEST_DELAY)
    response = requests.get(
        f"{BASE_URL}/{endpoint}",
        params=params or {},
        timeout=120,
        verify=False,
    )
    if response.status_code == 404:
        return []
    response.raise_for_status()
    return response.json()


def load_round_map(schedule_path: Path) -> dict[tuple[str, str], int]:
    """Map (country_name, location) from the season schedule to round number."""
    schedule = pd.read_csv(schedule_path)
    schedule = schedule[schedule["RoundNumber"] > 0]
    mapping: dict[tuple[str, str], int] = {}
    for _, row in schedule.iterrows():
        mapping[(str(row["Country"]), str(row["Location"]))] = int(row["RoundNumber"])
    return mapping


def list_qualifying_sessions(year: int = 2025) -> list[dict]:
    return _get("sessions", {"year": year, "session_name": "Qualifying"})


def _compound_for_lap(stints: list[dict], lap_number: int) -> tuple[str | None, bool | None]:
    for stint in stints:
        if stint["lap_start"] <= lap_number <= stint.get("lap_end", stint["lap_start"]):
            return stint.get("compound"), True
    return None, None


def _assign_laps(
    car_df: pd.DataFrame,
    laps_df: pd.DataFrame,
) -> pd.DataFrame:
    """Tag each telemetry row with a lap number using lap start/end timestamps."""
    car_df = car_df.sort_values("date").copy()
    car_df["LapNumber"] = pd.NA
    laps_sorted = laps_df.sort_values("date_start").reset_index(drop=True)

    for i, lap in laps_sorted.iterrows():
        start = lap["date_start"]
        if pd.notna(lap.get("lap_duration")) and float(lap["lap_duration"]) > 0:
            end = start + timedelta(seconds=float(lap["lap_duration"]))
        elif i + 1 < len(laps_sorted):
            end = laps_sorted.iloc[i + 1]["date_start"]
        else:
            end = start + timedelta(seconds=180)

        mask = (car_df["date"] >= start) & (car_df["date"] <= end)
        car_df.loc[mask, "LapNumber"] = lap["lap_number"]

    return car_df.dropna(subset=["LapNumber"])


def extract_session_telemetry(session: dict, round_num: int) -> pd.DataFrame:
    """Download and normalize one qualifying session to FastF1-compatible columns."""
    session_key = session["session_key"]
    drivers = _get("drivers", {"session_key": session_key})
    all_laps = pd.DataFrame(_get("laps", {"session_key": session_key}))
    all_stints = _get("stints", {"session_key": session_key})

    if all_laps.empty:
        return pd.DataFrame()

    all_laps["date_start"] = pd.to_datetime(all_laps["date_start"], utc=True, format="ISO8601")
    records: list[pd.DataFrame] = []

    for driver in drivers:
        driver_num = driver["driver_number"]
        acronym = driver["name_acronym"]

        car = pd.DataFrame(_get("car_data", {"session_key": session_key, "driver_number": driver_num}))
        loc = pd.DataFrame(_get("location", {"session_key": session_key, "driver_number": driver_num}))
        if car.empty:
            continue

        car["date"] = pd.to_datetime(car["date"], utc=True, format="ISO8601")
        if not loc.empty:
            loc["date"] = pd.to_datetime(loc["date"], utc=True, format="ISO8601")
            car = pd.merge_asof(
                car.sort_values("date"),
                loc[["date", "x", "y", "z"]].sort_values("date"),
                on="date",
                direction="nearest",
                tolerance=pd.Timedelta("200ms"),
            )
        else:
            car["x"] = pd.NA
            car["y"] = pd.NA
            car["z"] = pd.NA

        driver_laps = all_laps[all_laps["driver_number"] == driver_num].copy()
        if driver_laps.empty:
            continue

        car = _assign_laps(car, driver_laps)
        if car.empty:
            continue

        driver_stints = [s for s in all_stints if s["driver_number"] == driver_num]
        lap_meta = driver_laps.set_index("lap_number")

        rows = []
        for lap_num, group in car.groupby("LapNumber"):
            lap_num = int(lap_num)
            if lap_num not in lap_meta.index:
                continue
            meta = lap_meta.loc[lap_num]
            if bool(meta.get("is_pit_out_lap")):
                continue

            lap_start = group["date"].iloc[0]
            compound, fresh = _compound_for_lap(driver_stints, lap_num)
            lap_duration = float(meta["lap_duration"]) if pd.notna(meta["lap_duration"]) else None

            for _, row in group.iterrows():
                rows.append(
                    {
                        "Year": session["year"],
                        "Round": round_num,
                        "SessionType": "Q",
                        "DriverNumber": driver_num,
                        "Driver": acronym,
                        "LapNumber": lap_num,
                        "TimeInLap": (row["date"] - lap_start).total_seconds(),
                        "Time": row["date"],
                        "Throttle": row["throttle"],
                        "Brake": bool(row["brake"]),
                        "DRS": row["drs"],
                        "RPM": row["rpm"],
                        "Speed": row["speed"],
                        "X": row.get("x"),
                        "Y": row.get("y"),
                        "Z": row.get("z"),
                        "Steering": pd.NA,
                        "Gear": row.get("n_gear"),
                        "LapTime": timedelta(seconds=lap_duration) if lap_duration else pd.NA,
                        "Sector1Time": timedelta(seconds=float(meta["duration_sector_1"]))
                        if pd.notna(meta.get("duration_sector_1"))
                        else pd.NA,
                        "Sector2Time": timedelta(seconds=float(meta["duration_sector_2"]))
                        if pd.notna(meta.get("duration_sector_2"))
                        else pd.NA,
                        "Sector3Time": timedelta(seconds=float(meta["duration_sector_3"]))
                        if pd.notna(meta.get("duration_sector_3"))
                        else pd.NA,
                        "Compound": compound,
                        "FreshTyre": fresh,
                    }
                )

        if rows:
            records.append(pd.DataFrame(rows))

    if not records:
        return pd.DataFrame()

    out = pd.concat(records, ignore_index=True)
    out["LapTime"] = out["LapTime"].astype(str)
    out["Sector1Time"] = out["Sector1Time"].astype(str)
    out["Sector2Time"] = out["Sector2Time"].astype(str)
    out["Sector3Time"] = out["Sector3Time"].astype(str)
    out["Time"] = out["Time"].astype(str)
    return out


def save_session_csv(df: pd.DataFrame, round_num: int, country: str) -> Path:
    RAW_FASTF1_DIR.mkdir(parents=True, exist_ok=True)
    prefix = event_prefix(round_num, country)
    path = RAW_FASTF1_DIR / f"{prefix}_2025_telemetry_qualifying.csv"
    df.to_csv(path, index=False)
    return path
