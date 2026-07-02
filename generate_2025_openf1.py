#!/usr/bin/env python3
"""Extract and process 2025 qualifying telemetry via OpenF1 (FastF1 fallback)."""

import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_io import write_csv, write_sample_csv
from src.openf1_telemetry import (
    extract_session_telemetry,
    list_qualifying_sessions,
    load_round_map,
    save_session_csv,
)
from src.paths import PROCESSED_DIR, processed_qualifying_path, season_summary_path
from src.telemetry_processing import process_qualifying_dataframe

SCHEDULE_PATH = season_summary_path("schedule_2025_sample.csv")


def main() -> None:
    if not SCHEDULE_PATH.exists():
        print(f"Schedule file not found: {SCHEDULE_PATH}")
        sys.exit(1)

    round_map = load_round_map(SCHEDULE_PATH)
    sessions = list_qualifying_sessions(2025)
    print(f"Found {len(sessions)} qualifying sessions on OpenF1")

    processed_count = 0
    for session in sessions:
        key = (session["country_name"], session["location"])
        round_num = round_map.get(key)
        if round_num is None:
            print(f"  Skip (no round map): {session['country_name']} / {session['location']}")
            continue

        country = session["country_name"]
        out_path = processed_qualifying_path(round_num, country)
        gz_path = Path(str(out_path) + ".gz")
        if out_path.exists() or gz_path.exists():
            if out_path.exists() or gz_path.stat().st_size > 100_000:
                print(f"\nRound {round_num:02d}: {country} — skip (already processed)")
                processed_count += 1
                continue

        print(f"\nRound {round_num:02d}: {country} ({session['location']})")

        try:
            df = extract_session_telemetry(session, round_num)
        except requests.exceptions.RequestException as exc:
            print(f"  API error: {exc}")
            continue

        if df.empty:
            print("  No telemetry extracted")
            continue

        raw_path = save_session_csv(df, round_num, country)
        print(f"  Saved raw: {raw_path.name} ({len(df):,} rows, {df['Driver'].nunique()} drivers)")

        processed = process_qualifying_dataframe(df, round_num=round_num)
        out_path = processed_qualifying_path(round_num, country)
        write_csv(processed, out_path, index=False, compress=True)
        write_sample_csv(out_path)
        print(f"  Saved processed: {out_path.name}")
        processed_count += 1

    print(f"\nDone. Processed {processed_count} qualifying sessions.")


if __name__ == "__main__":
    main()
