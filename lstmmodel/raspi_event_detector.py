#!/usr/bin/env python3
"""
RASPBERRY PI EVENT DETECTION SYSTEM
Single file - no external dependencies except pandas and numpy
Usage: python3 raspi_event_detector.py sensor_data.csv
"""

import sys
import os
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("ERROR: Required packages not found!")
    print("Install with: pip3 install pandas numpy")
    sys.exit(1)

# ============================================================================
# CONFIGURATION - Adjust these thresholds as needed
# ============================================================================
THRESHOLDS = {
    'sudden_acceleration': 2.0,      # g force
    'sudden_braking': -2.0,          # g force
    'turn_threshold': 0.8,           # g lateral
    'sharp_turn': 1.5,               # g lateral
    'high_speed': 30.0,              # knots
    'vertical_bump': 2.5,            # g vertical
    'harsh_bump': 3.5,               # g vertical
    'gyro_turn': 20.0,               # degrees/sec
    'aggressive_rotation': 50.0,     # degrees/sec
    'straight_tolerance': 0.3,       # g lateral
}

# ============================================================================
# EVENT DETECTION FUNCTIONS
# ============================================================================

def get_sensor_value(row, primary, *alternatives):
    """Get sensor value trying multiple column name variations"""
    for col in [primary] + list(alternatives):
        if col in row and pd.notna(row[col]):
            try:
                return float(row[col])
            except:
                pass
    return 0.0

def detect_event(row):
    """Detect driving events from sensor data"""
    events = []
    
    # Get sensor values with flexible column names
    ax = get_sensor_value(row, 'ax_g', 'Ax', 'ax', 'accel_x')
    ay = get_sensor_value(row, 'ay_g', 'Ay', 'ay', 'accel_y')
    az = get_sensor_value(row, 'az_g', 'Az', 'az', 'accel_z')
    gx = get_sensor_value(row, 'gx_dps', 'Gx', 'gx', 'gyro_x')
    gy = get_sensor_value(row, 'gy_dps', 'Gy', 'gy', 'gyro_y')
    gz = get_sensor_value(row, 'gz_dps', 'Gz', 'gz', 'gyro_z')
    
    # Get speed
    speed = get_sensor_value(row, 'gps_speed_kn', 'Speed', 'speed', 'velocity')
    
    # Calculate movement characteristics
    lateral_accel = ay
    longitudinal_accel = ax
    vertical_accel = az - 9.8
    yaw_rate = gz
    rotation_magnitude = np.sqrt(gx**2 + gy**2 + gz**2)
    
    is_moving = speed > 2 or abs(longitudinal_accel) > 0.5
    
    # Detect stationary
    if not is_moving:
        events.append('STATIONARY')
    
    # Detect acceleration/braking
    if longitudinal_accel > THRESHOLDS['sudden_acceleration']:
        events.append('SUDDEN_ACCELERATION')
    if longitudinal_accel < THRESHOLDS['sudden_braking']:
        events.append('SUDDEN_BRAKING')
    
    # Detect turns
    abs_lateral = abs(lateral_accel)
    abs_yaw = abs(yaw_rate)
    
    if abs_lateral > THRESHOLDS['turn_threshold'] or abs_yaw > THRESHOLDS['gyro_turn']:
        turn_direction = 'RIGHT' if (lateral_accel > 0 or yaw_rate > 0) else 'LEFT'
        if abs_lateral > THRESHOLDS['sharp_turn'] or abs_yaw > THRESHOLDS['aggressive_rotation']:
            events.append(f'SHARP_{turn_direction}_TURN')
        else:
            events.append(f'{turn_direction}_TURN')
    elif is_moving and abs_lateral < THRESHOLDS['straight_tolerance']:
        events.append('STRAIGHT_DRIVING')
    
    # Detect bumps
    if vertical_accel > THRESHOLDS['vertical_bump']:
        if vertical_accel > THRESHOLDS['harsh_bump']:
            events.append('HARSH_BUMP')
        else:
            events.append('BUMP_UP')
    elif vertical_accel < -THRESHOLDS['vertical_bump']:
        events.append('BUMP_DOWN')
    
    # Detect high speed
    if speed > THRESHOLDS['high_speed']:
        events.append('HIGH_SPEED')
    
    # Detect aggressive maneuvers
    if rotation_magnitude > THRESHOLDS['aggressive_rotation']:
        events.append('AGGRESSIVE_MANEUVER')
    
    return ' | '.join(events) if events else 'NORMAL'

def calculate_severity(row, event):
    """Calculate severity score 0-10"""
    if event in ['NORMAL', 'STRAIGHT_DRIVING', 'STATIONARY']:
        return 0.0
    
    ax = get_sensor_value(row, 'ax_g', 'Ax', 'ax')
    ay = get_sensor_value(row, 'ay_g', 'Ay', 'ay')
    az = abs(get_sensor_value(row, 'az_g', 'Az', 'az') - 9.8)
    gx = get_sensor_value(row, 'gx_dps', 'Gx', 'gx')
    gy = get_sensor_value(row, 'gy_dps', 'Gy', 'gy')
    gz = get_sensor_value(row, 'gz_dps', 'Gz', 'gz')
    
    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    
    severity = min(10.0, (accel_mag + gyro_mag / 10) * 1.5)
    return round(severity, 1)

# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_sensor_data(input_file):
    """Process sensor data and detect events"""
    
    print("="*70)
    print("  RASPBERRY PI EVENT DETECTION SYSTEM")
    print("="*70)
    print(f"\n📂 Input: {input_file}")
    
    # Check file exists
    if not Path(input_file).exists():
        print(f"❌ ERROR: File not found: {input_file}")
        return False
    
    try:
        # Read CSV
        df = pd.read_csv(input_file)
        print(f"✓ Loaded {len(df)} records")
        print(f"✓ Columns: {', '.join(df.columns.tolist()[:5])}...")
        
        # Detect events
        print("\n🔍 Detecting events...")
        df['EVENT'] = df.apply(detect_event, axis=1)
        df['SEVERITY'] = df.apply(lambda row: calculate_severity(row, row['EVENT']), axis=1)
        
        # Add timestamp if not exists
        if 'timestamp' not in df.columns and 'Timestamp' not in df.columns:
            from datetime import datetime, timedelta
            print("⚠️  No timestamp column, generating timestamps...")
            start = datetime.now()
            df['TIMESTAMP'] = [(start + timedelta(seconds=i*0.1)).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] 
                               for i in range(len(df))]
        else:
            timestamp_col = 'timestamp' if 'timestamp' in df.columns else 'Timestamp'
            df['TIMESTAMP'] = df[timestamp_col]
        
        # Reorder columns
        other_cols = [col for col in df.columns if col not in ['TIMESTAMP', 'EVENT', 'SEVERITY']]
        df = df[['TIMESTAMP', 'EVENT', 'SEVERITY'] + other_cols]
        
        # Save output
        output_file = input_file.replace('.csv', '_with_events.csv')
        df.to_csv(output_file, index=False)
        
        print(f"\n✅ SUCCESS! Saved to: {output_file}")
        
        # Print statistics
        print("\n📊 EVENT STATISTICS:")
        print("-"*70)
        
        event_counts = {}
        for events_str in df['EVENT']:
            for event in str(events_str).split(' | '):
                event_counts[event] = event_counts.get(event, 0) + 1
        
        total = len(df)
        for event in sorted(event_counts.keys()):
            count = event_counts[event]
            percentage = (count / total) * 100
            print(f"  {event:25s} : {count:5d} ({percentage:5.1f}%)")
        
        # Show dangerous events
        dangerous = df[df['SEVERITY'] >= 7]
        if len(dangerous) > 0:
            print(f"\n⚠️  DANGEROUS EVENTS (Severity ≥ 7): {len(dangerous)}")
            print("-"*70)
            print(dangerous[['TIMESTAMP', 'EVENT', 'SEVERITY']].head(5).to_string(index=False))
        
        # Show timeline
        print(f"\n📅 EVENT TIMELINE (First 5):")
        print("-"*70)
        print(df[['TIMESTAMP', 'EVENT', 'SEVERITY']].head(5).to_string(index=False))
        
        print("\n" + "="*70)
        print(f"✓ Complete! Output file: {output_file}")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("\n" + "="*70)
        print("  RASPBERRY PI EVENT DETECTION SYSTEM")
        print("="*70)
        print("\nUsage:")
        print("  python3 raspi_event_detector.py <sensor_data.csv>")
        print("\nExample:")
        print("  python3 raspi_event_detector.py data/rider01/sensor_data.csv")
        print("\nOutput:")
        print("  Creates: sensor_data_with_events.csv")
        print("\nEvents Detected:")
        print("  ✓ LEFT_TURN, RIGHT_TURN, SHARP_LEFT_TURN, SHARP_RIGHT_TURN")
        print("  ✓ STRAIGHT_DRIVING, STATIONARY")
        print("  ✓ SUDDEN_ACCELERATION, SUDDEN_BRAKING")
        print("  ✓ BUMP_UP, BUMP_DOWN, HARSH_BUMP")
        print("  ✓ HIGH_SPEED, AGGRESSIVE_MANEUVER")
        print("\n" + "="*70)
        sys.exit(1)
    
    input_file = sys.argv[1]
    success = process_sensor_data(input_file)
    sys.exit(0 if success else 1)
