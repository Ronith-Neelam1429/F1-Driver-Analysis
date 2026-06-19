#!/usr/bin/env python3
"""
Extract detailed telemetry data for Australia 2025 race.
"""

import fastf1
import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')

OUTPUT_DIR = "data/raw_data/fastf1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Enable caching
cache_dir = os.path.expanduser('~/.cache/fastf1')
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

def extract_race_telemetry():
    """Extract telemetry for Australia 2025 race."""
    
    print("Loading Australia 2025 Race session...")
    
    try:
        # Load race session
        race = fastf1.get_session(2025, 1, 'R')
        race.load(drivers=['1', '4', '81', '63', '22', '23', '16', '44', '10', '55', '6', '14', '18', '7', '5', '12', '27', '30', '31', '87'])
        
        print(f"✓ Loaded race with {len(race.drivers)} drivers")
        print(f"  Drivers: {list(race.drivers)}")
        
        all_telemetry = []
        
        for driver_num in race.drivers:
            print(f"\n  Processing driver {driver_num}...")
            
            try:
                # Get all laps for this driver
                laps = race.laps.pick_driver(driver_num)
                
                if laps is None or laps.empty:
                    print(f"    No laps for driver {driver_num}")
                    continue
                
                driver_name = laps.iloc[0]['Driver'] if not laps.empty else f"Driver {driver_num}"
                print(f"    Found {len(laps)} laps for {driver_name}")
                
                for lap_idx, (lap_num, lap) in enumerate(laps.iterrows()):
                    try:
                        telemetry = lap.get_telemetry()
                        
                        if telemetry is None or telemetry.empty:
                            continue
                        
                        # Add lap and session info
                        telemetry['DriverNumber'] = driver_num
                        telemetry['Driver'] = lap['Driver']
                        telemetry['LapNumber'] = lap['LapNumber']
                        telemetry['LapTime'] = lap['LapTime']
                        telemetry['Sector1Time'] = lap['Sector1Time']
                        telemetry['Sector2Time'] = lap['Sector2Time']
                        telemetry['Sector3Time'] = lap['Sector3Time']
                        telemetry['Compound'] = lap['Compound']
                        telemetry['FreshTyre'] = lap['FreshTyre']
                        telemetry['SessionType'] = 'Race'
                        telemetry['Round'] = 1
                        telemetry['Year'] = 2025
                        
                        # Calculate time in lap
                        telemetry['TimeInLap'] = (telemetry['Time'] - telemetry['Time'].iloc[0]).dt.total_seconds()
                        
                        all_telemetry.append(telemetry)
                        
                        if (lap_idx + 1) % 5 == 0:
                            print(f"      ✓ Processed {lap_idx + 1}/{len(laps)} laps")
                    
                    except Exception as e:
                        print(f"      ✗ Error processing lap {lap_num}: {str(e)[:60]}")
                        continue
            
            except Exception as e:
                print(f"    ✗ Error with driver {driver_num}: {str(e)[:60]}")
                continue
        
        if all_telemetry:
            print(f"\nCombining {len(all_telemetry)} telemetry dataframes...")
            telemetry_df = pd.concat(all_telemetry, ignore_index=True)
            
            # Reorder columns
            cols = ['Year', 'Round', 'SessionType', 'DriverNumber', 'Driver', 
                   'LapNumber', 'TimeInLap', 'Time',
                   'Position', 'Throttle', 'Brake', 'DRS', 'RPM', 'Speed',
                   'X', 'Y', 'Z',
                   'Steering', 'Gear',
                   'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time',
                   'Compound', 'FreshTyre']
            
            existing_cols = [col for col in cols if col in telemetry_df.columns]
            telemetry_df = telemetry_df[existing_cols]
            
            # Save to CSV
            output_file = os.path.join(OUTPUT_DIR, 'australia_2025_telemetry_race.csv')
            telemetry_df.to_csv(output_file, index=False)
            
            print(f"\n✓ Saved {len(telemetry_df)} telemetry records to australia_2025_telemetry_race.csv")
            print(f"  File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
            print(f"\nTelemetry columns:")
            for col in existing_cols:
                print(f"  - {col}")
        else:
            print("✗ No telemetry data collected")
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    extract_race_telemetry()
