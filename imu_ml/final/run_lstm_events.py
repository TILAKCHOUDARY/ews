"""
Auto-Integration for LSTM Event Detection
Automatically monitors for data collection and runs real-time LSTM predictions
"""

import os
import time
import sys
from pathlib import Path
from realtime_lstm_detector import start_lstm_detection, stop_lstm_detection

class AutoLSTMDetection:
    """
    Automatically monitors for new data collection sessions and starts LSTM detection
    """
    
    def __init__(self, base_data_dir="data", model_weights='lstm_model_weights.weights.h5'):
        self.base_data_dir = base_data_dir
        self.model_weights = model_weights
        self.current_session = None
        self.monitoring = False
        
    def start_monitoring(self):
        """Start monitoring for new data collection sessions"""
        self.monitoring = True
        print("=" * 60)
        print("   🎯 LSTM Event Detection - AUTO MODE")
        print("=" * 60)
        print(f"📂 Monitoring: {os.path.abspath(self.base_data_dir)}")
        print(f"🧠 Model: {self.model_weights}")
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
            stop_lstm_detection()
            print(f"🛑 Stopped LSTM detection for: {self.current_session}")
        print("👋 LSTM event detection ended.")
    
    def _check_for_active_session(self):
        """Check for active data collection session"""
        if not os.path.exists(self.base_data_dir):
            return
        
        try:
            # Find the most recent ride folder
            ride_folders = [d for d in os.listdir(self.base_data_dir) 
                          if d.startswith("ride") and os.path.isdir(os.path.join(self.base_data_dir, d))]
            
            if not ride_folders:
                return
            
            # Sort to get the latest ride folder
            ride_folders.sort()
            latest_ride = ride_folders[-1]
            latest_path = os.path.join(self.base_data_dir, latest_ride)
            
            # Check if combined_data.csv exists and is being written
            csv_file = os.path.join(latest_path, "combined_data.csv")
            
            if os.path.exists(csv_file):
                # Check if file is being actively written (modified recently)
                file_mod_time = os.path.getmtime(csv_file)
                time_diff = time.time() - file_mod_time
                
                if time_diff < 10:  # File modified within last 10 seconds
                    if self.current_session != latest_ride:
                        # New active session detected
                        if self.current_session:
                            stop_lstm_detection()
                            print(f"🛑 Stopped previous session")
                        
                        self.current_session = latest_ride
                        start_lstm_detection(latest_path, self.model_weights)
                        print(f"🚀 Started LSTM detection: {latest_ride}")
                        print(f"📁 Folder: {latest_path}")
                        print(f"📝 Events → {os.path.join(latest_path, 'events.csv')}")
                        print("-" * 60)
                
                elif self.current_session == latest_ride and time_diff > 30:
                    # Session appears to have ended (no writes for 30+ seconds)
                    stop_lstm_detection()
                    print(f"✅ Session completed: {self.current_session}")
                    print(f"📊 Events logged to: {os.path.join(latest_path, 'events.csv')}")
                    print("🔍 Waiting for next session...")
                    print("-" * 60)
                    self.current_session = None
                    
        except Exception as e:
            print(f"❌ Error monitoring sessions: {e}")


def main():
    """Main function"""
    print("🔧 Checking dependencies...")
    
    # Check if TensorFlow/Keras is available
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
    except ImportError:
        print("❌ TensorFlow not found.")
        print("   Install with: pip install tensorflow")
        print("\n   For Raspberry Pi, use:")
        print("   pip install tensorflow-aarch64")
        print("   OR")
        print("   pip install tflite-runtime")
        sys.exit(1)
    
    # Check if numpy is available
    try:
        import numpy as np
        print(f"✅ NumPy version: {np.__version__}")
    except ImportError:
        print("❌ NumPy not found. Install with: pip install numpy")
        sys.exit(1)
    
    # Check if model weights exist
    model_file = 'lstm_model_weights.weights.h5'
    if not os.path.exists(model_file):
        print(f"❌ Model weights not found: {model_file}")
        print("   Please ensure the model file is in the same directory")
        sys.exit(1)
    else:
        print(f"✅ Model weights found: {model_file}")
    
    # Check if realtime_lstm_detector module is available
    try:
        from realtime_lstm_detector import LSTMEventDetector
        print("✅ LSTM detector module found")
    except ImportError:
        print("❌ realtime_lstm_detector.py not found")
        print("   Please ensure it's in the same directory")
        sys.exit(1)
    
    print("\n✅ All dependencies OK!")
    print("-" * 60)
    
    # Start auto detection
    detector = AutoLSTMDetection()
    
    try:
        detector.start_monitoring()
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        detector.stop_monitoring()


if __name__ == "__main__":
    main()
