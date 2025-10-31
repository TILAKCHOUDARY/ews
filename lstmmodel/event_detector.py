#!/usr/bin/env python3
"""
Event Detection Script for Raspberry Pi Sensor Data
Processes rider data folders and detects events from sensor readings
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Event detection thresholds
THRESHOLDS = {
    'sudden_acceleration': 2.0,     # g force forward
    'sudden_braking': -2.0,         # g force backward
    'turn_threshold': 0.8,          # g force lateral for turn detection
    'sharp_turn': 1.5,              # g force lateral for sharp turn
    'high_speed': 30.0,             # knots
    'vertical_bump': 2.5,           # g force vertical (upward bump)
    'harsh_bump': 3.5,              # g force vertical (harsh bump)
    'gyro_turn': 20.0,              # deg/s for turn detection
    'aggressive_rotation': 50.0,    # deg/s for aggressive maneuver
    'straight_tolerance': 0.3,      # g lateral tolerance for straight
}

def detect_events(row, prev_row=None):
    """
    Detect various driving events based on sensor data
    
    Events detected:
    - sudden_acceleration: High positive forward acceleration
    - sudden_braking: High negative forward acceleration
    - left_turn: Turning left (gentle)
    - right_turn: Turning right (gentle)
    - sharp_left_turn: Aggressive left turn
    - sharp_right_turn: Aggressive right turn
    - straight: Moving straight ahead
    - high_speed: Speed exceeds threshold
    - bump_up: Hitting upward bump
    - bump_down: Hitting downward depression
    - harsh_bump: Severe vertical impact
    - pothole: Sudden downward impact
    - stationary: Not moving
    - aggressive_maneuver: Complex aggressive driving
    """
    events = []
    
    # Extract sensor values
    ax = row['ax_g']  # Forward/backward acceleration
    ay = row['ay_g']  # Left/right (lateral) acceleration
    az = row['az_g']  # Up/down (vertical) acceleration
    gx = row['gx_dps']  # Roll (tilt left/right)
    gy = row['gy_dps']  # Pitch (tilt forward/back)
    gz = row['gz_dps']  # Yaw (rotation/turning)
    
    # Handle GPS speed (might be N/A)
    try:
        speed = float(row['gps_speed_kn']) if row['gps_speed_kn'] != 'N/A' else 0
    except:
        speed = 0
    
    # Calculate magnitudes
    lateral_accel = ay  # Positive = right, Negative = left
    longitudinal_accel = ax  # Positive = forward, Negative = backward
    vertical_accel = az - 9.8  # Remove gravity, Positive = up, Negative = down
    
    # Rotation magnitudes
    yaw_rate = abs(gz)  # Turning rate
    total_rotation = np.sqrt(gx**2 + gy**2 + gz**2)
    
    # ==== MOVEMENT DIRECTION ====
    
    # Detect stationary
    if speed < 1.0 and abs(longitudinal_accel) < 0.3 and abs(lateral_accel) < 0.3:
        events.append('stationary')
        return ','.join(events) if events else 'stationary'
    
    # Detect straight movement (low lateral acceleration and low yaw)
    if abs(lateral_accel) < THRESHOLDS['straight_tolerance'] and yaw_rate < THRESHOLDS['gyro_turn']:
        if speed > 2.0:  # Only if actually moving
            events.append('straight')
    
    # ==== LONGITUDINAL EVENTS ====
    
    # Detect sudden acceleration
    if longitudinal_accel > THRESHOLDS['sudden_acceleration']:
        events.append('sudden_acceleration')
    
    # Detect sudden braking
    if longitudinal_accel < THRESHOLDS['sudden_braking']:
        events.append('sudden_braking')
    
    # ==== TURNING EVENTS ====
    
    # Sharp turns (high lateral acceleration)
    if abs(lateral_accel) > THRESHOLDS['sharp_turn']:
        if lateral_accel < 0:  # Negative = left
            events.append('sharp_left_turn')
        else:  # Positive = right
            events.append('sharp_right_turn')
    
    # Gentle turns (moderate lateral acceleration)
    elif abs(lateral_accel) > THRESHOLDS['turn_threshold']:
        if lateral_accel < 0:
            events.append('left_turn')
        else:
            events.append('right_turn')
    
    # Alternative turn detection using gyroscope yaw
    if yaw_rate > THRESHOLDS['gyro_turn'] and 'turn' not in ','.join(events):
        if gz < 0:  # Negative yaw = left turn
            events.append('left_turn')
        else:  # Positive yaw = right turn
            events.append('right_turn')
    
    # ==== VERTICAL EVENTS ====
    
    # Harsh bump (severe vertical impact)
    if abs(vertical_accel) > THRESHOLDS['harsh_bump']:
        if vertical_accel > 0:
            events.append('harsh_bump_up')
        else:
            events.append('harsh_bump_down')
    
    # Regular bump
    elif abs(vertical_accel) > THRESHOLDS['vertical_bump']:
        if vertical_accel > 0:
            events.append('bump_up')
        else:
            events.append('bump_down')
    
    # Pothole detection (sudden downward followed by upward - requires previous reading)
    if prev_row is not None:
        prev_vertical = prev_row['az_g'] - 9.8
        if prev_vertical < -THRESHOLDS['vertical_bump'] and vertical_accel > THRESHOLDS['vertical_bump']:
            events.append('pothole')
    
    # ==== SPEED EVENTS ====
    
    # Detect high speed
    if speed > THRESHOLDS['high_speed']:
        events.append('high_speed')
    
    # ==== COMPLEX MANEUVERS ====
    
    # Detect aggressive maneuver (high rotation + high lateral/longitudinal)
    if total_rotation > THRESHOLDS['aggressive_rotation']:
        events.append('aggressive_maneuver')
    
    # Swerving (rapid left-right changes)
    if prev_row is not None:
        prev_lateral = prev_row['ay_g']
        # If lateral acceleration changed sign significantly
        if abs(lateral_accel - prev_lateral) > 1.5 and abs(lateral_accel) > 0.8:
            events.append('swerving')
    
    # Return comma-separated events or 'normal'
    return ','.join(events) if events else 'normal'


def calculate_severity(row, event):
    """Calculate severity score (0-10) for detected events"""
    if event == 'normal' or event == 'stationary' or event == 'straight':
        return 0
    
    ax = row['ax_g']
    ay = row['ay_g']
    az = abs(row['az_g'] - 9.8)
    gx = row['gx_dps']
    gy = row['gy_dps']
    gz = row['gz_dps']
    
    # Calculate based on magnitude of forces and type of event
    accel_magnitude = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_magnitude = np.sqrt(gx**2 + gy**2 + gz**2)
    
    # Different severity calculations based on event type
    if 'harsh' in event or 'sharp' in event or 'sudden' in event:
        # High severity events
        severity = min(10, (accel_magnitude * 2 + gyro_magnitude / 5))
    elif 'bump' in event or 'pothole' in event:
        # Vertical events - focus on vertical acceleration
        severity = min(10, abs(az) * 2.5)
    elif 'turn' in event:
        # Turning events - focus on lateral and yaw
        severity = min(10, (abs(ay) + abs(gz) / 10) * 3)
    elif 'swerving' in event or 'aggressive' in event:
        # Complex maneuvers
        severity = min(10, (accel_magnitude + gyro_magnitude / 8) * 2.5)
    else:
        # General severity
        severity = min(10, (accel_magnitude + gyro_magnitude / 10) * 2)
    
    return round(severity, 2)


def process_rider_folder(rider_path):
    """
    Process a single rider folder and create event detection CSV
    
    Args:
        rider_path: Path to rider folder (e.g., data/rider01)
    """
    rider_path = Path(rider_path)
    
    if not rider_path.exists():
        print(f"❌ Rider folder not found: {rider_path}")
        return
    
    # Find CSV file in the rider folder
    csv_files = list(rider_path.glob('*.csv'))
    
    if not csv_files:
        print(f"❌ No CSV files found in {rider_path}")
        return
    
    # Process the first CSV file found
    csv_file = csv_files[0]
    print(f"\n📂 Processing: {csv_file}")
    
    try:
        # Read sensor data
        df = pd.read_csv(csv_file)
        print(f"   Found {len(df)} sensor records")
        
        # Check required columns
        required_cols = ['timestamp', 'ax_g', 'ay_g', 'az_g', 'gx_dps', 'gy_dps', 'gz_dps']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return
        
        # Detect events for each timestamp
        events_data = []
        
        for idx, row in df.iterrows():
            prev_row = df.iloc[idx - 1] if idx > 0 else None
            
            event_type = detect_events(row, prev_row)
            severity = calculate_severity(row, event_type)
            
            # Create event record
            event_record = {
                'timestamp': row['timestamp'],
                'epoch_time': row.get('epoch_time', ''),
                'event_type': event_type,
                'severity': severity,
                'ax_g': round(row['ax_g'], 4),
                'ay_g': round(row['ay_g'], 4),
                'az_g': round(row['az_g'], 4),
                'gx_dps': round(row['gx_dps'], 3),
                'gy_dps': round(row['gy_dps'], 3),
                'gz_dps': round(row['gz_dps'], 3),
                'gps_speed_kn': row.get('gps_speed_kn', 'N/A'),
                'gps_lat': row.get('gps_lat', 'N/A'),
                'gps_lon': row.get('gps_lon', 'N/A'),
                'image_filename': row.get('image_filename', ''),
            }
            
            events_data.append(event_record)
        
        # Create events DataFrame
        df_events = pd.DataFrame(events_data)
        
        # Save to new CSV file
        output_file = rider_path / 'events_detected.csv'
        df_events.to_csv(output_file, index=False)
        
        # Print summary
        event_counts = df_events['event_type'].value_counts()
        print(f"\n✅ Events detected and saved to: {output_file}")
        print(f"\n📊 Event Summary:")
        for event, count in event_counts.items():
            print(f"   {event}: {count}")
        
        # Show some example events (non-normal)
        significant_events = df_events[df_events['event_type'] != 'normal']
        if len(significant_events) > 0:
            print(f"\n⚠️  Significant Events (first 5):")
            print(significant_events[['timestamp', 'event_type', 'severity']].head().to_string(index=False))
        
    except Exception as e:
        print(f"❌ Error processing {csv_file}: {e}")
        import traceback
        traceback.print_exc()


def process_all_riders(data_folder='data'):
    """
    Process all rider folders in the data directory
    
    Args:
        data_folder: Path to data folder containing rider folders
    """
    data_path = Path(data_folder)
    
    if not data_path.exists():
        print(f"❌ Data folder not found: {data_path}")
        print(f"   Creating example structure...")
        data_path.mkdir(exist_ok=True)
        return
    
    # Find all rider folders
    rider_folders = sorted([f for f in data_path.iterdir() if f.is_dir() and f.name.startswith('rider')])
    
    if not rider_folders:
        print(f"❌ No rider folders found in {data_path}")
        print(f"   Expected folders like: rider01, rider02, etc.")
        return
    
    print(f"🔍 Found {len(rider_folders)} rider folder(s)")
    
    # Process each rider folder
    for rider_folder in rider_folders:
        process_rider_folder(rider_folder)
    
    print(f"\n✅ Processing complete!")


def create_sample_data():
    """Create sample data structure for testing"""
    print("📝 Creating sample data structure...")
    
    # Create data folder and rider01
    data_path = Path('data')
    rider_path = data_path / 'rider01'
    images_path = rider_path / 'images'
    
    rider_path.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(exist_ok=True)
    
    # Generate sample sensor data
    np.random.seed(42)
    n_samples = 100
    
    data = []
    for i in range(n_samples):
        timestamp = f"2025-09-13_22-34-{47+i//10:02d}-{(i*100)%1000:03d}"
        epoch_time = 1757783087.0 + i * 0.1
        
        # Simulate different driving patterns
        if i < 20:  # Normal driving
            ax = np.random.normal(0.6, 0.1)
            ay = np.random.normal(0.25, 0.05)
            az = np.random.normal(0.75, 0.05)
            gx = np.random.normal(-3.3, 0.2)
            gy = np.random.normal(0.9, 0.3)
            gz = np.random.normal(0.3, 0.2)
        elif i < 25:  # Sudden braking
            ax = np.random.normal(-2.5, 0.3)
            ay = np.random.normal(0.3, 0.1)
            az = np.random.normal(0.8, 0.1)
            gx = np.random.normal(-4.0, 0.5)
            gy = np.random.normal(1.5, 0.4)
            gz = np.random.normal(0.5, 0.2)
        elif i < 35:  # Sharp turn
            ax = np.random.normal(0.8, 0.2)
            ay = np.random.normal(2.0, 0.3)
            az = np.random.normal(0.7, 0.1)
            gx = np.random.normal(-5.0, 0.8)
            gy = np.random.normal(3.0, 0.5)
            gz = np.random.normal(2.5, 0.4)
        elif i < 40:  # Harsh bump
            ax = np.random.normal(0.6, 0.2)
            ay = np.random.normal(0.3, 0.1)
            az = np.random.normal(13.0, 0.5)
            gx = np.random.normal(-3.5, 0.3)
            gy = np.random.normal(1.0, 0.3)
            gz = np.random.normal(0.4, 0.2)
        else:  # Normal driving
            ax = np.random.normal(0.6, 0.1)
            ay = np.random.normal(0.25, 0.05)
            az = np.random.normal(0.75, 0.05)
            gx = np.random.normal(-3.3, 0.2)
            gy = np.random.normal(0.9, 0.3)
            gz = np.random.normal(0.3, 0.2)
        
        image_file = f"images/image_{timestamp}_{i+1:04d}.jpg"
        
        data.append({
            'timestamp': timestamp,
            'epoch_time': epoch_time,
            'image_filename': image_file,
            'ax_g': round(ax, 4),
            'ay_g': round(ay, 4),
            'az_g': round(az, 4),
            'gx_dps': round(gx, 3),
            'gy_dps': round(gy, 3),
            'gz_dps': round(gz, 3),
            'gps_utc': 'N/A',
            'gps_lat': 'N/A',
            'gps_ns': 'N/A',
            'gps_lon': 'N/A',
            'gps_ew': 'N/A',
            'gps_speed_kn': 'N/A',
            'gps_course_deg': 'N/A',
            'gps_valid': 'V'
        })
    
    df = pd.DataFrame(data)
    csv_file = rider_path / 'sensor_data.csv'
    df.to_csv(csv_file, index=False)
    
    print(f"✅ Created sample data: {csv_file}")
    print(f"   {len(df)} sensor records")
    return str(rider_path)


if __name__ == '__main__':
    import sys
    
    print("=" * 60)
    print("  Event Detection System for Raspberry Pi Sensor Data")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        # Process specific rider folder
        rider_folder = sys.argv[1]
        process_rider_folder(rider_folder)
    else:
        # Check if data folder exists
        if not Path('data').exists() or not list(Path('data').glob('rider*')):
            print("\n⚠️  No data folder found. Creating sample data for testing...")
            rider_path = create_sample_data()
            print("\n" + "=" * 60)
            input("Press Enter to process the sample data...")
        
        # Process all rider folders
        process_all_riders('data')
    
    print("\n" + "=" * 60)
    print("Done!")
