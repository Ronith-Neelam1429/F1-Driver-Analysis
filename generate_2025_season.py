#!/usr/bin/env python3
"""Extract, process, compress, and sample all 2025 season telemetry CSVs."""

import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import fastf1
import pandas as pd

warnings.filterwarnings('ignore')

# Unbuffered progress in long-running jobs
sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_extraction_fastf1 import extract_telemetry_data, extract_2025_data
from src.data_io import read_csv, resolve_data_path, sample_csv_path, write_csv, write_sample_csv, SAMPLE_LINE_LIMIT
from src.paths import (
    PROCESSED_DIR,
    RAW_FASTF1_DIR,
    event_prefix,
    processed_qualifying_path,
    raw_qualifying_path,
    raw_race_path,
    season_summary_path,
)
from src.telemetry_processing import process_qualifying_dataframe

YEAR = 2025

LEGACY_RENAMES = {
    RAW_FASTF1_DIR / 'australia_2025_telemetry_qualifying.csv': raw_qualifying_path(1, 'Australia'),
    RAW_FASTF1_DIR / 'australia_2025_telemetry_race.csv': raw_race_path(1, 'Australia'),
    PROCESSED_DIR / 'australia_2025_quali_telemetry_processed.csv': processed_qualifying_path(1, 'Australia'),
    PROCESSED_DIR / 'australia_2025_quali_telemetry_processed.csv.gz': Path(
        str(processed_qualifying_path(1, 'Australia')) + '.gz'
    ),
    PROCESSED_DIR / 'australia_2025_quali_telemetry_processed_sample.csv': sample_csv_path(
        processed_qualifying_path(1, 'Australia')
    ),
}


def migrate_legacy_filenames() -> None:
    for old_path, new_path in LEGACY_RENAMES.items():
        if old_path.exists() and not new_path.exists():
            new_path.parent.mkdir(parents=True, exist_ok=True)
            old_path.rename(new_path)
            print(f"Renamed {old_path.name} -> {new_path.name}")


def setup_fastf1_cache() -> None:
    cache_dir = os.path.expanduser('~/.cache/fastf1')
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)


def completed_race_rounds(schedule: pd.DataFrame) -> pd.DataFrame:
    today = datetime.now(timezone.utc).date()
    rounds = schedule[schedule['RoundNumber'] > 0].copy()
    rounds = rounds[rounds['EventDate'].notna()]
    rounds = rounds[rounds['EventDate'].dt.date <= today]
    return rounds.sort_values('RoundNumber')


def extract_round_telemetry(round_num: int, country: str, event_name: str, *, skip_existing: bool = True) -> dict:
    prefix = event_prefix(round_num, country)
    quali_file = f"{prefix}_2025_telemetry_qualifying.csv"
    race_file = f"{prefix}_2025_telemetry_race.csv"
    quali_path = RAW_FASTF1_DIR / quali_file
    race_path = RAW_FASTF1_DIR / race_file

    print(f"\n{'=' * 70}")
    print(f"Round {round_num}: {event_name} ({country})")
    print('=' * 70)

    if skip_existing and quali_path.exists() and race_path.exists():
        print(f"  Skipping extraction (files exist): {quali_file}, {race_file}")
        return {'round': round_num, 'country': country, 'quali_ok': True, 'race_ok': True}

    quali_df = None
    race_df = None

    if not (skip_existing and quali_path.exists()):
        quali_df = extract_telemetry_data(YEAR, round_num, 'Q', quali_file)
    else:
        print(f"  Skipping quali (exists): {quali_file}")

    if not (skip_existing and race_path.exists()):
        race_df = extract_telemetry_data(YEAR, round_num, 'R', race_file)
    else:
        print(f"  Skipping race (exists): {race_file}")

    return {
        'round': round_num,
        'country': country,
        'quali_ok': quali_df is not None or quali_path.exists(),
        'race_ok': race_df is not None or race_path.exists(),
    }


def process_round_qualifying(round_num: int, country: str, *, compress: bool = True, skip_existing: bool = True) -> bool:
    raw_path = raw_qualifying_path(round_num, country)
    processed_path = processed_qualifying_path(round_num, country)
    gz_path = Path(str(processed_path) + '.gz')
    sample_path = sample_csv_path(processed_path)

    if skip_existing and processed_path.exists() and gz_path.exists() and sample_path.exists():
        print(f"  Skipping processing (outputs exist): {processed_path.name}")
        return True

    source = resolve_data_path(raw_path)
    if not source.exists():
        print(f"  Skip processing (no raw file): {raw_path.name}")
        return False

    print(f"  Processing {raw_path.name}...")
    quali_df = read_csv(raw_path)
    processed = process_qualifying_dataframe(quali_df, round_num=round_num)

    write_csv(processed, processed_path, index=False, compress=compress)
    write_sample_csv(processed_path)
    print(f"  Saved {processed_path.name}" + (" + .gz" if compress else ""))
    print(f"  Saved {sample_path.name} ({SAMPLE_LINE_LIMIT} lines)")
    return True


def publish_summary_csv(df: pd.DataFrame, filename: str, *, compress: bool = True) -> None:
    if df is None or df.empty:
        return

    path = season_summary_path(filename)
    write_csv(df, path, index=False, compress=compress)
    write_sample_csv(path)
    print(f"  Saved {path.name} (+ .gz + sample)")


def extract_season_telemetry() -> list[dict]:
    setup_fastf1_cache()
    schedule = fastf1.get_event_schedule(YEAR)
    rounds = completed_race_rounds(schedule)

    print(f"Extracting telemetry for {len(rounds)} completed 2025 rounds...")
    results = []
    for _, event in rounds.iterrows():
        result = extract_round_telemetry(
            int(event['RoundNumber']),
            event['Country'],
            event['EventName'],
        )
        results.append(result)
    return results


def process_season_qualifying(*, compress: bool = True) -> int:
    setup_fastf1_cache()
    schedule = fastf1.get_event_schedule(YEAR)
    rounds = completed_race_rounds(schedule)

    processed_count = 0
    for _, event in rounds.iterrows():
        if process_round_qualifying(int(event['RoundNumber']), event['Country'], compress=compress):
            processed_count += 1
    return processed_count


def extract_season_summaries(*, skip_existing: bool = True) -> None:
    """Extract aggregated laps/results/schedule and publish compressed copies."""
    setup_fastf1_cache()
    extract_2025_data()

    for name in ('schedule_2025.csv', 'race_results_2025.csv', 'laps_2025.csv'):
        raw = RAW_FASTF1_DIR / name
        out = season_summary_path(name)
        if skip_existing and out.exists() and Path(str(out) + '.gz').exists():
            print(f"  Skipping summary (exists): {name}")
            continue
        if raw.exists():
            df = pd.read_csv(raw, low_memory=False)
            publish_summary_csv(df, name)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_FASTF1_DIR.mkdir(parents=True, exist_ok=True)
    migrate_legacy_filenames()

    print("STEP 1/3: Season summary CSVs")
    extract_season_summaries()

    print("\nSTEP 2/3: Per-round qualifying & race telemetry extraction")
    extract_season_telemetry()

    print("\nSTEP 3/3: Process qualifying telemetry")
    count = process_season_qualifying(compress=True)
    print(f"\nDone. Processed {count} qualifying sessions.")


if __name__ == "__main__":
    main()
