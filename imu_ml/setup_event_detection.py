#!/usr/bin/env python3
"""
Event Detection Setup Script for Raspberry Pi

This script helps set up the event detection system on your Raspberry Pi.
Run this script once to check dependencies and set up the environment.
"""

import os
import sys
import subprocess
import importlib.util

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 6):
        print("❌ Python 3.6 or higher required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_and_install_numpy():
    """Check if NumPy is installed, install if not"""
    try:
        import numpy as np
        print(f"✅ NumPy version: {np.__version__}")
        return True
    except ImportError:
        print("⚠️  NumPy not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
            import numpy as np
            print(f"✅ NumPy installed successfully: {np.__version__}")
            return True
        except Exception as e:
            print(f"❌ Failed to install NumPy: {e}")
            print("   Please install manually: pip install numpy")
            return False

def check_files():
    """Check if required files are present"""
    required_files = [
        "event_detector.py",
        "integrate_event_detection.py"
    ]
    
    all_present = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")
            all_present = False
    
    return all_present

def check_data_directory():
    """Check if data directory exists or create it"""
    data_dir = "data"
    if os.path.exists(data_dir):
        print(f"✅ Data directory exists: {os.path.abspath(data_dir)}")
    else:
        try:
            os.makedirs(data_dir)
            print(f"✅ Created data directory: {os.path.abspath(data_dir)}")
        except Exception as e:
            print(f"⚠️  Could not create data directory: {e}")
            print("   This is okay - it will be created when you start data collection")

def create_start_script():
    """Create a convenient start script"""
    script_content = """#!/bin/bash
# Event Detection Starter Script
echo "🎯 Starting Event Detection System"
echo "Make sure your main data collection (new.py) is running first!"
echo "Press Ctrl+C to stop"
echo ""
python3 integrate_event_detection.py
"""
    
    try:
        with open("start_events.sh", "w") as f:
            f.write(script_content)
        os.chmod("start_events.sh", 0o755)  # Make executable
        print("✅ Created start_events.sh script")
        print("   You can run: ./start_events.sh")
    except Exception as e:
        print(f"⚠️  Could not create start script: {e}")

def test_event_detector():
    """Test the event detector with sample data"""
    try:
        from event_detector import EventDetector
        detector = EventDetector()
        
        # Add a few test samples
        test_samples = [
            (0.63, 0.25, 0.75, -3.2, 1.0, 0.5),
            (0.64, 0.25, 0.75, -3.1, 0.8, 0.3),
            (0.63, 0.24, 0.76, -3.3, 0.9, 0.2),
        ]
        
        for i, (ax, ay, az, gx, gy, gz) in enumerate(test_samples):
            detector.add_sample(f"test_{i}", ax, ay, az, gx, gy, gz)
        
        print("✅ Event detector test passed")
        return True
    except Exception as e:
        print(f"❌ Event detector test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("=" * 60)
    print("   🎯 Event Detection System Setup")
    print("=" * 60)
    
    # Check system requirements
    if not check_python_version():
        sys.exit(1)
    
    if not check_and_install_numpy():
        sys.exit(1)
    
    if not check_files():
        print("\n❌ Missing required files!")
        print("   Please ensure event_detector.py and integrate_event_detection.py")
        print("   are in the same directory as this script.")
        sys.exit(1)
    
    # Optional setup
    check_data_directory()
    create_start_script()
    
    # Test the system
    print("\n🧪 Testing event detector...")
    if test_event_detector():
        print("\n" + "=" * 60)
        print("   ✅ Setup Complete!")
        print("=" * 60)
        print("📋 Next steps:")
        print("1. Start your main data collection: python new.py")
        print("2. In another terminal, start event detection:")
        print("   python integrate_event_detection.py")
        print("   OR")
        print("   ./start_events.sh")
        print("\n📊 Events will be saved to events.csv in each ride folder")
        print("=" * 60)
    else:
        print("\n❌ Setup incomplete - please check errors above")
        sys.exit(1)

if __name__ == "__main__":
    main()