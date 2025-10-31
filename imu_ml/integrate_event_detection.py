"""
Event Detection Integration for Raspberry Pi IMU Data Collection

This script integrates real-time event detection with your existing new.py system.
Simply run this script after starting your main data collection system.

Instructions:
1. Copy event_detector.py and this file to your Raspberry Pi
2. Install required dependencies: pip install numpy
3. Start your main data collection system (new.py) 
4. In a separate terminal, run: python integrate_event_detection.py

The system will automatically detect when data collection starts/stops and 
create events.csv files in each data folder with detected driving events.
"""

import os
import time
import sys
from pathlib import Path
from event_detector import start_event_detection, stop_event_detection

class AutoEventDetection:
    """
    Automatically monitors for new data collection sessions and starts event detection
    """
    
    def __init__(self, base_data_dir="data"):
        self.base_data_dir = base_data_dir
        self.current_session = None
        self.monitoring = False
        
    def start_monitoring(self):
        """Start monitoring for new data collection sessions"""
        self.monitoring = True
        print("=" * 60)
        print("   🎯 Event Detection Integration Started")
        print("=" * 60)
        print("📂 Monitoring data directory:", os.path.abspath(self.base_data_dir))
        print("🔍 Waiting for data collection to start...")
        print("-" * 60)
        
        try:
            while self.monitoring:
                self._check_for_active_session()
                time.sleep(2)  # Check every 2 seconds
        except KeyboardInterrupt:
            print("\n⛔ Received stop signal...")
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop monitoring and cleanup"""
        self.monitoring = False
        if self.current_session:
            stop_event_detection()
            print(f"🛑 Stopped event detection for session: {self.current_session}")
        print("👋 Event detection integration ended.")
    
    def _check_for_active_session(self):
        """Check for active data collection session"""
        if not os.path.exists(self.base_data_dir):
            return
        
        # Find the most recent ride folder
        try:
            ride_folders = [d for d in os.listdir(self.base_data_dir) 
                          if d.startswith("ride") and os.path.isdir(os.path.join(self.base_data_dir, d))]
            
            if not ride_folders:
                return
            
            # Sort to get the latest ride folder
            ride_folders.sort()
            latest_ride = ride_folders[-1]
            latest_path = os.path.join(self.base_data_dir, latest_ride)
            
            # Check if this is a new session or if data is being actively written
            csv_file = os.path.join(latest_path, "combined_data.csv")
            
            if os.path.exists(csv_file):
                # Check if file is being actively written (modified recently)
                file_mod_time = os.path.getmtime(csv_file)
                time_diff = time.time() - file_mod_time
                
                if time_diff < 10:  # File modified within last 10 seconds
                    if self.current_session != latest_ride:
                        # New active session detected
                        if self.current_session:
                            stop_event_detection()
                            print(f"🛑 Stopped event detection for previous session")
                        
                        self.current_session = latest_ride
                        start_event_detection(latest_path)
                        print(f"🚀 Started event detection for session: {latest_ride}")
                        print(f"📁 Monitoring: {latest_path}")
                        print(f"📝 Events will be logged to: {os.path.join(latest_path, 'events.csv')}")
                        print("-" * 60)
                
                elif self.current_session == latest_ride and time_diff > 30:
                    # Session appears to have ended (no writes for 30+ seconds)
                    stop_event_detection()
                    print(f"✅ Session completed: {self.current_session}")
                    print(f"📊 Events logged to: {os.path.join(latest_path, 'events.csv')}")
                    print("🔍 Waiting for next data collection session...")
                    print("-" * 60)
                    self.current_session = None
                    
        except Exception as e:
            print(f"❌ Error monitoring sessions: {e}")


def main():
    """Main function"""
    print("🔧 Checking dependencies...")
    
    # Check if numpy is available
    try:
        import numpy as np
        print("✅ NumPy found")
    except ImportError:
        print("❌ NumPy not found. Please install with: pip install numpy")
        sys.exit(1)
    
    # Check if event_detector module is available
    try:
        from event_detector import EventDetector
        print("✅ Event detector module found")
    except ImportError:
        print("❌ event_detector.py not found. Please ensure it's in the same directory.")
        sys.exit(1)
    
    # Start auto event detection
    detector = AutoEventDetection()
    
    try:
        detector.start_monitoring()
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        detector.stop_monitoring()


if __name__ == "__main__":
    main()