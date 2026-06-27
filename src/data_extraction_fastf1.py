"""
FastF1 API data extraction for 2025 F1 season.
Fetches all available data from the FastF1 library and saves to CSV files.
"""

import os
import sys
import warnings
from pathlib import Path

import fastf1
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import RAW_FASTF1_DIR

OUTPUT_DIR = str(RAW_FASTF1_DIR)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Enable caching for faster subsequent runs
cache_dir = os.path.expanduser('~/.cache/fastf1')
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

def save_to_csv(data, filename):
    """Save data to CSV file."""
    if data is None or data.empty:
        print(f"No data to save for {filename}")
        return
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    data.to_csv(filepath)
    print(f"✓ Saved {len(data)} records to {filename}")

def extract_telemetry_data(year, round_num, session_type, output_filename):
    """
    Extract detailed telemetry data for a specific session.
    
    Args:
        year (int): Year of the season
        round_num (int): Round number
        session_type (str): 'Q' for qualifying, 'R' for race
        output_filename (str): Output CSV filename
    """
    print(f"   Extracting telemetry for {session_type}...")
    
    try:
        session = fastf1.get_session(year, round_num, session_type)
        session.load()
        
        telemetry_records = []
        
        # Iterate through each driver
        for driver in session.drivers:
            try:
                laps = session.laps.pick_driver(driver)
                
                # Iterate through each lap
                for lap_num, lap in laps.iterrows():
                    try:
                        # Get telemetry data for this lap
                        telemetry = lap.get_telemetry()
                        
                        if telemetry is None or telemetry.empty:
                            continue
                        
                        # Add lap and driver information
                        telemetry['DriverNumber'] = driver
                        telemetry['Driver'] = lap['Driver']
                        telemetry['LapNumber'] = lap['LapNumber']
                        telemetry['LapTime'] = lap['LapTime']
                        telemetry['Sector1Time'] = lap['Sector1Time']
                        telemetry['Sector2Time'] = lap['Sector2Time']
                        telemetry['Sector3Time'] = lap['Sector3Time']
                        telemetry['Compound'] = lap['Compound']
                        telemetry['FreshTyre'] = lap['FreshTyre']
                        telemetry['SessionType'] = session_type
                        telemetry['Round'] = round_num
                        telemetry['Year'] = year
                        
                        # Calculate time in lap (seconds)
                        telemetry['TimeInLap'] = (telemetry['Time'] - telemetry['Time'].iloc[0]).dt.total_seconds()
                        
                        telemetry_records.append(telemetry)
                        
                    except Exception as e:
                        print(f"      ✗ Error processing lap {lap_num} for driver {driver}: {e}")
                        continue
                        
            except Exception as e:
                print(f"      ✗ Error processing driver {driver}: {e}")
                continue
        
        if telemetry_records:
            # Combine all records
            telemetry_df = pd.concat(telemetry_records, ignore_index=True)
            
            # Reorder columns for better readability
            cols = ['Year', 'Round', 'SessionType', 'DriverNumber', 'Driver', 
                   'LapNumber', 'TimeInLap', 'Time',
                   'Position', 'Throttle', 'Brake', 'DRS', 'RPM', 'Speed',
                   'X', 'Y', 'Z',
                   'Steering', 'Gear',
                   'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time',
                   'Compound', 'FreshTyre']
            
            # Only include columns that exist in the data
            existing_cols = [col for col in cols if col in telemetry_df.columns]
            telemetry_df = telemetry_df[existing_cols]
            
            filepath = os.path.join(OUTPUT_DIR, output_filename)
            telemetry_df.to_csv(filepath, index=False)
            print(f"      ✓ Saved {len(telemetry_df)} telemetry records to {output_filename}")
            
            return telemetry_df
        else:
            print(f"      ✗ No telemetry data found")
            return None
            
    except Exception as e:
        print(f"   Error extracting telemetry: {e}")
        return None

def extract_2025_data():
    """Extract all available 2025 season data from FastF1."""
    
    print("=" * 60)
    print("FastF1 Data Extraction - 2025 Season")
    print("=" * 60)
    
    # Get schedule for 2025
    print("\n1. Fetching 2025 F1 schedule...")
    try:
        schedule = fastf1.get_event_schedule(2025)
        save_to_csv(schedule, "schedule_2025.csv")
    except Exception as e:
        print(f"Error fetching schedule: {e}")
        return
    
    # Initialize containers for aggregated data
    all_sessions = []
    all_laps = []
    all_drivers = []
    all_weather = []
    all_results = []
    
    # Iterate through each round
    print("\n2. Processing each round...")
    
    for idx, event in schedule.iterrows():
        round_num = event['RoundNumber']
        gp_name = event['EventName']

        if round_num == 0:
            print(f"   Skipping round {round_num} ({gp_name}) - testing session")
            continue

        # Skip cancelled events
        if event['EventDate'] == 'NaT' or pd.isna(event['EventDate']):
            print(f"   Skipping round {round_num} ({gp_name}) - No date")
            continue
        
        print(f"\n   Round {round_num}: {gp_name}")
        
        try:
            # Get Race session data
            try:
                race = fastf1.get_session(2025, round_num, 'R')
                race.load()
                
                # Save race results
                results = race.results[['DriverNumber', 'Driver', 'Abbreviation', 
                                        'Team', 'ClassifiedFinishingPosition', 'Points']].copy()
                results['Round'] = round_num
                results['EventName'] = gp_name
                all_results.append(results)
                
                # Save race laps
                laps = race.laps[['DriverNumber', 'Driver', 'LapNumber', 'LapTime', 
                                   'Sector1Time', 'Sector2Time', 'Sector3Time',
                                   'Compound', 'FreshTyre']].copy()
                laps['Round'] = round_num
                laps['EventName'] = gp_name
                laps['SessionType'] = 'Race'
                all_laps.append(laps)
                
                print(f"      ✓ Loaded Race session")
                
            except Exception as e:
                print(f"      ✗ Could not load Race: {str(e)[:50]}")
            
            # Get Qualifying session data
            try:
                quali = fastf1.get_session(2025, round_num, 'Q')
                quali.load()
                
                # Save qualifying laps
                laps = quali.laps[['DriverNumber', 'Driver', 'LapNumber', 'LapTime',
                                    'Sector1Time', 'Sector2Time', 'Sector3Time',
                                    'IsPersonalBest']].copy()
                laps['Round'] = round_num
                laps['EventName'] = gp_name
                laps['SessionType'] = 'Qualifying'
                all_laps.append(laps)
                
                print(f"      ✓ Loaded Qualifying session")
                
            except Exception as e:
                print(f"      ✗ Could not load Qualifying: {str(e)[:50]}")
            
            # Get Practice 3 session data (if available)
            try:
                fp3 = fastf1.get_session(2025, round_num, 'FP3')
                fp3.load()
                
                # Save FP3 laps
                laps = fp3.laps[['DriverNumber', 'Driver', 'LapNumber', 'LapTime',
                                  'Sector1Time', 'Sector2Time', 'Sector3Time',
                                  'Compound']].copy()
                laps['Round'] = round_num
                laps['EventName'] = gp_name
                laps['SessionType'] = 'FP3'
                all_laps.append(laps)
                
                print(f"      ✓ Loaded FP3 session")
                
            except Exception as e:
                print(f"      ✗ Could not load FP3: {str(e)[:50]}")
                
        except Exception as e:
            print(f"   Error processing round {round_num}: {e}")
            continue
    
    # Save aggregated data
    print("\n3. Saving aggregated data...")
    
    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        save_to_csv(results_df, "race_results_2025.csv")
    
    if all_laps:
        laps_df = pd.concat(all_laps, ignore_index=True)
        for col in ('LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time'):
            if col in laps_df.columns:
                laps_df[f'{col}_seconds'] = pd.to_timedelta(laps_df[col], errors='coerce').dt.total_seconds()
        save_to_csv(laps_df, "laps_2025.csv")
    
    print("\n" + "=" * 60)
    print("FastF1 data extraction completed!")
    print(f"All files saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)

def extract_australia_2025_telemetry():
    """Extract detailed telemetry data for Australia 2025 qualifying and race."""
    
    print("\n" + "=" * 60)
    print("Extracting Australia 2025 Telemetry Data")
    print("=" * 60)
    
    # Australia is Round 1 in 2025 season
    round_num = 1
    
    print("\nExtracting Qualifying Telemetry...")
    extract_telemetry_data(2025, round_num, 'Q', 'australia_2025_telemetry_qualifying.csv')
    
    print("\nExtracting Race Telemetry...")
    extract_telemetry_data(2025, round_num, 'R', 'australia_2025_telemetry_race.csv')
    
    print("\n" + "=" * 60)
    print("Australia 2025 telemetry extraction completed!")
    print(f"Files saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)

if __name__ == "__main__":
    extract_2025_data()
    extract_australia_2025_telemetry()
