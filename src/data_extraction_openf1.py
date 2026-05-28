"""
OpenF1 API data extraction for 2025 F1 season.
Fetches all available data from the OpenF1 API and saves to CSV files.
"""

import requests
import pandas as pd
import os
from datetime import datetime
import time

BASE_URL = "https://api.openf1.org/v1"
OUTPUT_DIR = "data/raw_data/openf1"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_data(endpoint, params=None):
    """
    Fetch data from OpenF1 API.
    
    Args:
        endpoint (str): API endpoint (e.g., 'meetings', 'sessions', 'drivers')
        params (dict): Query parameters
        
    Returns:
        list: JSON response data
    """
    url = f"{BASE_URL}/{endpoint}"
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {endpoint}: {e}")
        return []

def save_to_csv(data, filename):
    """Save data to CSV file."""
    if not data:
        print(f"No data to save for {filename}")
        return
    
    df = pd.DataFrame(data)
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"✓ Saved {len(df)} records to {filename}")

def extract_2025_data():
    """Extract all available 2025 season data from OpenF1."""
    
    print("=" * 60)
    print("OpenF1 Data Extraction - 2025 Season")
    print("=" * 60)
    
    # 1. Get all meetings (races) for 2025
    print("\n1. Fetching meetings for 2025...")
    meetings = fetch_data("meetings", params={"year": 2025})
    save_to_csv(meetings, "meetings_2025.csv")
    
    if not meetings:
        print("No meetings found for 2025. Exiting.")
        return
    
    # 2. Get all sessions for 2025
    print("\n2. Fetching sessions for 2025...")
    sessions = fetch_data("sessions", params={"year": 2025})
    save_to_csv(sessions, "sessions_2025.csv")
    
    # Extract session keys for detailed data
    session_keys = [session['session_key'] for session in sessions]
    
    # 3. Get drivers data for each session
    print("\n3. Fetching drivers data...")
    all_drivers = []
    for idx, session in enumerate(sessions[:5]):  # Limit to first 5 sessions to avoid rate limiting
        drivers = fetch_data("drivers", params={"session_key": session['session_key']})
        all_drivers.extend(drivers)
        time.sleep(0.5)  # Respectful rate limiting
    
    drivers_df = pd.DataFrame(all_drivers)
    drivers_unique = drivers_df.drop_duplicates(subset=['driver_number'])
    save_to_csv(drivers_unique.to_dict('records'), "drivers_2025.csv")
    
    # 4. Get championship standings (drivers)
    print("\n4. Fetching driver championship standings...")
    championship_drivers = []
    for session in sessions:
        if 'Race' in session.get('session_name', ''):
            data = fetch_data("championship_drivers", params={"session_key": session['session_key']})
            championship_drivers.extend(data)
            time.sleep(0.5)
    save_to_csv(championship_drivers, "championship_drivers_2025.csv")
    
    # 5. Get championship standings (teams)
    print("\n5. Fetching team championship standings...")
    championship_teams = []
    for session in sessions:
        if 'Race' in session.get('session_name', ''):
            data = fetch_data("championship_teams", params={"session_key": session['session_key']})
            championship_teams.extend(data)
            time.sleep(0.5)
    save_to_csv(championship_teams, "championship_teams_2025.csv")
    
    # 6. Get laps data (for a few sessions to avoid excessive requests)
    print("\n6. Fetching laps data (sample from first 3 sessions)...")
    all_laps = []
    for session in sessions[:3]:
        laps = fetch_data("laps", params={"session_key": session['session_key']})
        all_laps.extend(laps)
        time.sleep(1)  # Higher delay for larger dataset
    save_to_csv(all_laps, "laps_sample_2025.csv")
    
    # 7. Get starting grid data
    print("\n7. Fetching starting grid data...")
    all_grids = []
    for session in sessions:
        if session.get('session_type') in ['Qualifying', 'Race']:
            grid = fetch_data("starting_grid", params={"session_key": session['session_key']})
            all_grids.extend(grid)
            time.sleep(0.5)
    save_to_csv(all_grids, "starting_grid_2025.csv")
    
    # 8. Get session results
    print("\n8. Fetching session results...")
    all_results = []
    for session in sessions:
        results = fetch_data("session_result", params={"session_key": session['session_key']})
        all_results.extend(results)
        time.sleep(0.5)
    save_to_csv(all_results, "session_results_2025.csv")
    
    # 9. Get stint data
    print("\n9. Fetching stint data...")
    all_stints = []
    for session in sessions:
        stints = fetch_data("stints", params={"session_key": session['session_key']})
        all_stints.extend(stints)
        time.sleep(0.5)
    save_to_csv(all_stints, "stints_2025.csv")
    
    # 10. Get pit data
    print("\n10. Fetching pit data...")
    all_pits = []
    for session in sessions:
        pits = fetch_data("pit", params={"session_key": session['session_key']})
        all_pits.extend(pits)
        time.sleep(0.5)
    save_to_csv(all_pits, "pit_stops_2025.csv")
    
    # 11. Get overtakes data (races only)
    print("\n11. Fetching overtakes data...")
    all_overtakes = []
    for session in sessions:
        if 'Race' in session.get('session_name', ''):
            overtakes = fetch_data("overtakes", params={"session_key": session['session_key']})
            all_overtakes.extend(overtakes)
            time.sleep(0.5)
    save_to_csv(all_overtakes, "overtakes_2025.csv")
    
    # 12. Get weather data (sample)
    print("\n12. Fetching weather data...")
    all_weather = []
    for meeting in meetings[:5]:  # Sample from first 5 meetings
        weather = fetch_data("weather", params={"meeting_key": meeting['meeting_key']})
        all_weather.extend(weather)
        time.sleep(0.5)
    save_to_csv(all_weather, "weather_sample_2025.csv")
    
    print("\n" + "=" * 60)
    print("OpenF1 data extraction completed!")
    print(f"All files saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)

if __name__ == "__main__":
    extract_2025_data()
