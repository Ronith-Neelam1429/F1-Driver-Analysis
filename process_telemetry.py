#!/usr/bin/env python3
"""
Process qualifying telemetry with acceleration, deceleration, and corner detection.
"""

import pandas as pd
import numpy as np

# Albert Park corner coordinates (approximate center points)
ALBERT_PARK_CORNERS = {
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
}

def identify_corner(x, y):
    """Identify which corner a position is closest to."""
    if pd.isna(x) or pd.isna(y):
        return None
    
    min_distance = float('inf')
    closest_corner = None
    
    for corner_name, bounds in ALBERT_PARK_CORNERS.items():
        # Calculate distance to corner center
        corner_x = (bounds['x'][0] + bounds['x'][1]) / 2
        corner_y = (bounds['y'][0] + bounds['y'][1]) / 2
        distance = np.sqrt((x - corner_x)**2 + (y - corner_y)**2)
        
        if distance < min_distance:
            min_distance = distance
            closest_corner = corner_name
    
    return closest_corner if min_distance < 500 else None

def process_qualifying_telemetry():
    """Process full qualifying telemetry with acceleration and corner data."""
    
    print("Loading full qualifying telemetry...")
    quali_df = pd.read_csv('data/raw_data/fastf1/australia_2025_telemetry_qualifying.csv')
    
    print(f"Processing {len(quali_df):,} data points...")
    
    # Add corner identification
    print("Identifying corners...")
    quali_df['Corner'] = quali_df.apply(
        lambda row: identify_corner(row['X'], row['Y']), 
        axis=1
    )
    
    # Calculate acceleration/deceleration
    print("Calculating acceleration and deceleration...")
    quali_df = quali_df.sort_values(['DriverNumber', 'LapNumber', 'TimeInLap']).reset_index(drop=True)
    
    # Calculate speed change per second
    quali_df['Speed_ms'] = quali_df['Speed'] / 3.6  # Convert km/h to m/s
    
    # Group by driver and lap to calculate acceleration within each lap
    accel_values = []
    for (driver, lap), group_indices in quali_df.groupby(['DriverNumber', 'LapNumber']).groups.items():
        indices = sorted(group_indices)
        acceleration = np.full(len(indices), np.nan)
        
        for i in range(1, len(indices)):
            curr_idx = indices[i]
            prev_idx = indices[i-1]
            
            time_diff = quali_df.loc[curr_idx, 'TimeInLap'] - quali_df.loc[prev_idx, 'TimeInLap']
            speed_diff = quali_df.loc[curr_idx, 'Speed_ms'] - quali_df.loc[prev_idx, 'Speed_ms']
            
            if time_diff > 0:
                accel = speed_diff / time_diff
                # Clip to reasonable bounds
                acceleration[i] = np.clip(accel, -15, 15)
        
        accel_values.extend(acceleration)
    
    quali_df['Acceleration'] = accel_values
    
    # Classify as acceleration or deceleration
    quali_df['Accel_Type'] = quali_df['Acceleration'].apply(
        lambda x: 'Acceleration' if x > 0.5 else ('Deceleration' if x < -0.5 else 'Neutral') if pd.notna(x) else None
    )
    
    # Save full processed data
    print("Saving processed telemetry...")
    import os
    os.makedirs('data/processed', exist_ok=True)
    
    quali_df.to_csv('data/processed/australia_2025_quali_telemetry_processed.csv', index=False)
    print(f"✓ Saved to data/processed/australia_2025_quali_telemetry_processed.csv")
    
    # Create corner-focused summary
    print("\nGenerating corner analysis...")
    
    corner_summary = []
    for driver in quali_df['Driver'].unique():
        driver_data = quali_df[quali_df['Driver'] == driver]
        
        for corner in driver_data['Corner'].dropna().unique():
            corner_data = driver_data[driver_data['Corner'] == corner]
            
            if len(corner_data) > 0:
                decel_data = corner_data[corner_data['Accel_Type'] == 'Deceleration']
                accel_data = corner_data[corner_data['Accel_Type'] == 'Acceleration']
                
                corner_summary.append({
                    'Driver': driver,
                    'Corner': corner,
                    'Min_Speed': corner_data['Speed'].min(),
                    'Max_Speed_Before': decel_data['Speed'].max() if len(decel_data) > 0 else np.nan,
                    'Max_Deceleration': decel_data['Acceleration'].min() if len(decel_data) > 0 else np.nan,
                    'Max_Acceleration_Exit': accel_data['Acceleration'].max() if len(accel_data) > 0 else np.nan,
                    'Data_Points': len(corner_data)
                })
    
    corner_summary_df = pd.DataFrame(corner_summary).sort_values(['Corner', 'Driver'])
    corner_summary_df.to_csv('data/processed/australia_2025_quali_corner_analysis.csv', index=False)
    print(f"✓ Saved corner analysis to data/processed/australia_2025_quali_corner_analysis.csv")
    
    # Print summary
    print("\n" + "="*70)
    print("QUALIFYING TELEMETRY PROCESSING COMPLETE")
    print("="*70)
    print(f"\nTotal records: {len(quali_df):,}")
    print(f"Drivers: {quali_df['Driver'].nunique()}")
    print(f"Corners identified: {quali_df['Corner'].nunique()}")
    
    print("\nFiles created:")
    print("  1. australia_2025_quali_telemetry_processed.csv - Full data with acceleration & corners")
    print("  2. australia_2025_quali_corner_analysis.csv - Corner statistics by driver")
    
    return quali_df, corner_summary_df

if __name__ == "__main__":
    quali_df, corner_df = process_qualifying_telemetry()
    
    print("\n" + "="*70)
    print("CORNER ANALYSIS - TOP ENTRIES BY DECELERATION")
    print("="*70)
    top_decel = corner_df.nlargest(20, 'Max_Deceleration')[
        ['Driver', 'Corner', 'Min_Speed', 'Max_Speed_Before', 'Max_Deceleration', 'Max_Acceleration_Exit']
    ].copy()
    print(top_decel.to_string())
