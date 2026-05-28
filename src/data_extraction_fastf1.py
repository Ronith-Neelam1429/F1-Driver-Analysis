"""
FastF1 API data extraction for 2025 F1 season.
Fetches all available data from the FastF1 library and saves to CSV files.
"""

import fastf1
import pandas as pd
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

OUTPUT_DIR = "data/raw_data/fastf1"

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
        
        # Skip cancelled events
        if event['EventDate'] == 'NaT' or pd.isna(event['EventDate']):
            print(f"   Skipping round {round_num} ({gp_name}) - No date")
            continue
        
        print(f"\n   Round {round_num}: {gp_name}")
        
        try:
            # Load all sessions for this round (Practice, Qualifying, Race, etc.)
            sessions_for_round = fastf1.get_session(2025, round_num, identifier=None)
            
            if sessions_for_round is None:
                continue
            
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
        # Convert LapTime to total seconds for easier analysis
        laps_df['LapTime_seconds'] = laps_df['LapTime'].dt.total_seconds()
        laps_df['Sector1Time_seconds'] = laps_df['Sector1Time'].dt.total_seconds()
        laps_df['Sector2Time_seconds'] = laps_df['Sector2Time'].dt.total_seconds()
        laps_df['Sector3Time_seconds'] = laps_df['Sector3Time'].dt.total_seconds()
        save_to_csv(laps_df, "laps_2025.csv")
    
    print("\n" + "=" * 60)
    print("FastF1 data extraction completed!")
    print(f"All files saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)

if __name__ == "__main__":
    extract_2025_data()
