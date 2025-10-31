"""
Test script for Driving Score Analyzer
Creates synthetic test data to validate all functionality
"""

import os
import pandas as pd
import numpy as np
import sys

def create_test_data(folder_name="test_ride"):
    """Create synthetic test data with all event types"""
    
    # Create test folder
    test_folder = os.path.join("..", "data", folder_name)
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(os.path.join(test_folder, "images"), exist_ok=True)
    
    print(f"📁 Creating test data in: {test_folder}")
    
    # Generate combined_data.csv
    n_samples = 200
    timestamps = [f"2025-01-01_12-00-{i:02d}-000" for i in range(n_samples)]
    
    # Simulate driving data
    base_ax = 0.6 + np.random.normal(0, 0.05, n_samples)
    base_ay = 0.25 + np.random.normal(0, 0.05, n_samples)
    base_az = 0.75 + np.random.normal(0, 0.05, n_samples)
    
    # Add bumps at specific locations (sudden Z spike)
    bump_indices = [50, 120]
    for idx in bump_indices:
        base_az[idx] = 1.5  # Bump spike
    
    # Add pothole at specific location (sudden Z drop)
    pothole_indices = [80]
    for idx in pothole_indices:
        base_az[idx] = 0.4  # Pothole drop
    
    # Gyroscope data
    base_gx = -3.2 + np.random.normal(0, 0.5, n_samples)
    base_gy = 1.0 + np.random.normal(0, 0.3, n_samples)
    base_gz = 0.5 + np.random.normal(0, 0.2, n_samples)
    
    # Add turns (high gz values)
    turn_indices = [30, 90, 150]
    for idx in turn_indices:
        base_gz[idx:idx+5] = 25.0 + np.random.normal(0, 2, 5)  # Right turn
    
    left_turn_indices = [60, 130]
    for idx in left_turn_indices:
        base_gz[idx:idx+5] = -25.0 + np.random.normal(0, 2, 5)  # Left turn
    
    # Speed data (in knots, will be converted to km/h)
    base_speed = np.linspace(15, 35, n_samples)  # Gradual speed increase
    base_speed += np.random.normal(0, 2, n_samples)  # Add noise
    
    # Reduce speed before turns and bumps
    for idx in turn_indices + left_turn_indices + bump_indices + pothole_indices:
        if idx > 3:
            base_speed[idx-3:idx] *= 0.7  # Slow down before event
    
    # Add stop event (speed drops to 0)
    stop_index = 170
    base_speed[stop_index:stop_index+5] = np.linspace(20, 0, 5)
    
    # Create DataFrame
    imu_data = pd.DataFrame({
        'timestamp': timestamps,
        'epoch_time': np.arange(n_samples) * 0.2,
        'image_filename': [f'images/img_{i:04d}.jpg' for i in range(n_samples)],
        'ax_g': base_ax,
        'ay_g': base_ay,
        'az_g': base_az,
        'gx_dps': base_gx,
        'gy_dps': base_gy,
        'gz_dps': base_gz,
        'gps_utc': ['N/A'] * n_samples,
        'gps_lat': ['N/A'] * n_samples,
        'gps_ns': ['N'] * n_samples,
        'gps_lon': ['N/A'] * n_samples,
        'gps_ew': ['E'] * n_samples,
        'gps_speed_kn': base_speed,
        'gps_course_deg': ['N/A'] * n_samples,
        'gps_valid': ['A'] * n_samples
    })
    
    csv_path = os.path.join(test_folder, "combined_data.csv")
    imu_data.to_csv(csv_path, index=False)
    print(f"✅ Created combined_data.csv ({len(imu_data)} samples)")
    
    # Generate events.csv
    events = []
    
    # Straight events
    events.append({'timestamp': timestamps[10], 'event': 'straight'})
    events.append({'timestamp': timestamps[100], 'event': 'straight'})
    
    # Turn events
    for idx in turn_indices:
        events.append({'timestamp': timestamps[idx], 'event': 'right'})
    
    for idx in left_turn_indices:
        events.append({'timestamp': timestamps[idx], 'event': 'left'})
    
    # Bump events
    for idx in bump_indices:
        events.append({'timestamp': timestamps[idx], 'event': 'bump'})
    
    # Pothole events
    for idx in pothole_indices:
        events.append({'timestamp': timestamps[idx], 'event': 'pothole'})
    
    # Stop event
    events.append({'timestamp': timestamps[stop_index], 'event': 'stop'})
    
    events_df = pd.DataFrame(events)
    events_path = os.path.join(test_folder, "events.csv")
    events_df.to_csv(events_path, index=False)
    print(f"✅ Created events.csv ({len(events_df)} events)")
    
    # Print event summary
    print("\n📊 Event Summary:")
    for event_type in events_df['event'].unique():
        count = len(events_df[events_df['event'] == event_type])
        print(f"   {event_type:12s}: {count}")
    
    return test_folder


def test_analyzer(test_folder):
    """Test the analyzer on synthetic data"""
    print(f"\n{'='*60}")
    print("   🧪 TESTING DRIVING SCORE ANALYZER")
    print(f"{'='*60}\n")
    
    try:
        from driving_score_analyzer import DrivingScoreAnalyzer
        
        # Run analysis
        analyzer = DrivingScoreAnalyzer(test_folder)
        final_score, metrics = analyzer.analyze_all()
        
        # Verify all metrics exist
        print("\n✅ TEST RESULTS:")
        print("-" * 60)
        
        required_metrics = [
            'speed_management',
            'acceleration_smoothness',
            'turning_quality',
            'event_responses'
        ]
        
        all_passed = True
        for metric in required_metrics:
            if metric in metrics:
                score = metrics[metric].get('score', 0)
                print(f"✓ {metric:30s}: {score:.2f}/100")
            else:
                print(f"✗ {metric:30s}: MISSING")
                all_passed = False
        
        print("-" * 60)
        print(f"{'FINAL SCORE':30s}: {final_score:.2f}/100")
        print(f"{'RATING':30s}: {metrics['final']['rating']}")
        print("=" * 60)
        
        # Check event responses details
        if 'event_responses' in metrics:
            er = metrics['event_responses']
            print(f"\n📊 Event Analysis Details:")
            print(f"   Bumps analyzed:    {er.get('bumps', 0)}")
            print(f"   Potholes analyzed: {er.get('potholes', 0)}")
            print(f"   Stops analyzed:    {er.get('stops', 0)}")
            print(f"   Total events:      {er.get('events_analyzed', 0)}")
        
        if all_passed:
            print(f"\n{'✅ ALL TESTS PASSED!':^60}")
        else:
            print(f"\n{'❌ SOME TESTS FAILED':^60}")
        
        return all_passed
        
    except Exception as e:
        print(f"\n❌ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases():
    """Test edge cases"""
    print(f"\n{'='*60}")
    print("   🔍 TESTING EDGE CASES")
    print(f"{'='*60}\n")
    
    # Test 1: No events
    print("Test 1: Ride with no events...")
    folder = create_minimal_test_data("test_no_events", has_events=False)
    test_analyzer(folder)
    
    # Test 2: No speed data
    print("\nTest 2: Ride with no speed data...")
    folder = create_minimal_test_data("test_no_speed", has_speed=False)
    test_analyzer(folder)
    
    print("\n✅ Edge case testing complete")


def create_minimal_test_data(folder_name, has_events=True, has_speed=True):
    """Create minimal test data for edge cases"""
    test_folder = os.path.join("..", "data", folder_name)
    os.makedirs(test_folder, exist_ok=True)
    
    n_samples = 50
    timestamps = [f"2025-01-01_12-00-{i:02d}-000" for i in range(n_samples)]
    
    imu_data = pd.DataFrame({
        'timestamp': timestamps,
        'epoch_time': np.arange(n_samples) * 0.2,
        'image_filename': [f'images/img_{i:04d}.jpg' for i in range(n_samples)],
        'ax_g': np.random.normal(0.6, 0.05, n_samples),
        'ay_g': np.random.normal(0.25, 0.05, n_samples),
        'az_g': np.random.normal(0.75, 0.05, n_samples),
        'gx_dps': np.random.normal(-3.2, 0.5, n_samples),
        'gy_dps': np.random.normal(1.0, 0.3, n_samples),
        'gz_dps': np.random.normal(0.5, 0.2, n_samples),
        'gps_utc': ['N/A'] * n_samples,
        'gps_lat': ['N/A'] * n_samples,
        'gps_ns': ['N'] * n_samples,
        'gps_lon': ['N/A'] * n_samples,
        'gps_ew': ['E'] * n_samples,
        'gps_speed_kn': np.random.normal(20, 2, n_samples) if has_speed else [0] * n_samples,
        'gps_course_deg': ['N/A'] * n_samples,
        'gps_valid': ['A'] * n_samples
    })
    
    imu_data.to_csv(os.path.join(test_folder, "combined_data.csv"), index=False)
    
    if has_events:
        events = pd.DataFrame([
            {'timestamp': timestamps[10], 'event': 'straight'}
        ])
        events.to_csv(os.path.join(test_folder, "events.csv"), index=False)
    else:
        # Create empty events file
        pd.DataFrame(columns=['timestamp', 'event']).to_csv(
            os.path.join(test_folder, "events.csv"), index=False
        )
    
    return test_folder


def main():
    """Main test function"""
    print("╔" + "="*58 + "╗")
    print("║" + " "*15 + "ANALYZER TEST SUITE" + " "*24 + "║")
    print("╚" + "="*58 + "╝\n")
    
    # Create test data
    print("Step 1: Creating test data...")
    test_folder = create_test_data()
    
    # Run basic tests
    print("\nStep 2: Testing analyzer...")
    basic_passed = test_analyzer(test_folder)
    
    # Run edge case tests
    print("\nStep 3: Testing edge cases...")
    test_edge_cases()
    
    print("\n" + "="*60)
    if basic_passed:
        print("   ✅ ANALYZER IS WORKING CORRECTLY!")
    else:
        print("   ⚠️  ANALYZER HAS ISSUES - CHECK OUTPUT ABOVE")
    print("="*60 + "\n")
    
    print("💡 To test on real data:")
    print("   python driving_score_analyzer.py ../data/ride01\n")


if __name__ == "__main__":
    main()
