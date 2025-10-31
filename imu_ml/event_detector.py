import numpy as np
import csv
import os
import time
from datetime import datetime
from threading import Thread, Lock
from collections import deque
import math

class EventDetector:
    """
    Real-time event detection from IMU data for driving events:
    - Straight driving
    - Left turn
    - Right turn  
    - Bump
    - Pothole
    """
    
    def __init__(self, window_size=10, sample_rate=5):
        """
        Initialize event detector
        
        Args:
            window_size: Number of samples to use for detection (default: 10 samples = 2 seconds at 5Hz)
            sample_rate: Expected sample rate in Hz (default: 5Hz matching your FPS)
        """
        self.window_size = window_size
        self.sample_rate = sample_rate
        
        # Rolling windows for sensor data
        self.accel_x = deque(maxlen=window_size)
        self.accel_y = deque(maxlen=window_size)
        self.accel_z = deque(maxlen=window_size)
        self.gyro_x = deque(maxlen=window_size)
        self.gyro_y = deque(maxlen=window_size)
        self.gyro_z = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
        # Event detection thresholds (tuned for typical driving scenarios)
        self.TURN_GYRO_Z_THRESHOLD = 15.0      # deg/s for turn detection
        self.STRAIGHT_GYRO_Z_THRESHOLD = 5.0   # deg/s for straight driving
        self.BUMP_ACCEL_Z_THRESHOLD = 1.2      # g for bump detection (upward acceleration)
        self.POTHOLE_ACCEL_Z_THRESHOLD = 0.6   # g for pothole detection (downward acceleration)
        self.MIN_EVENT_DURATION = 0.5          # seconds - minimum event duration
        
        # State tracking
        self.last_event = None
        self.last_event_time = 0
        self.event_cooldown = 1.0  # seconds between same event type
        
    def add_sample(self, timestamp, ax, ay, az, gx, gy, gz):
        """
        Add new IMU sample to the detection window
        
        Args:
            timestamp: Sample timestamp string
            ax, ay, az: Accelerometer readings in g
            gx, gy, gz: Gyroscope readings in deg/s
        """
        self.timestamps.append(timestamp)
        self.accel_x.append(ax)
        self.accel_y.append(ay)
        self.accel_z.append(az)
        self.gyro_x.append(gx)
        self.gyro_y.append(gy)
        self.gyro_z.append(gz)
        
    def detect_events(self):
        """
        Analyze current window and detect driving events
        
        Returns:
            list: Detected events as [{'event': 'event_name', 'timestamp': 'timestamp', 'confidence': float}]
        """
        if len(self.timestamps) < self.window_size:
            return []
            
        events = []
        current_time = time.time()
        
        # Calculate statistical measures for detection
        gyro_z_mean = np.mean(self.gyro_z)
        gyro_z_std = np.std(self.gyro_z)
        accel_z_mean = np.mean(self.accel_z)
        accel_z_std = np.std(self.accel_z)
        accel_z_max = np.max(self.accel_z)
        accel_z_min = np.min(self.accel_z)
        
        # Get latest timestamp
        latest_timestamp = self.timestamps[-1]
        
        # 1. TURN DETECTION (based on gyroscope Z-axis)
        if abs(gyro_z_mean) > self.TURN_GYRO_Z_THRESHOLD and gyro_z_std > 5.0:
            if gyro_z_mean > 0:
                event_type = "right_turn"
            else:
                event_type = "left_turn"
            
            confidence = min(1.0, abs(gyro_z_mean) / 50.0)  # Scale confidence
            
            if self._should_report_event(event_type, current_time):
                events.append({
                    'event': event_type,
                    'timestamp': latest_timestamp,
                    'confidence': round(confidence, 3)
                })
        
        # 2. STRAIGHT DRIVING (low gyroscope activity, stable acceleration)
        elif abs(gyro_z_mean) < self.STRAIGHT_GYRO_Z_THRESHOLD and gyro_z_std < 3.0 and accel_z_std < 0.1:
            event_type = "straight"
            confidence = 1.0 - (abs(gyro_z_mean) / self.STRAIGHT_GYRO_Z_THRESHOLD)
            
            if self._should_report_event(event_type, current_time):
                events.append({
                    'event': event_type,
                    'timestamp': latest_timestamp,
                    'confidence': round(confidence, 3)
                })
        
        # 3. BUMP DETECTION (sudden upward acceleration spike)
        if accel_z_max > self.BUMP_ACCEL_Z_THRESHOLD and accel_z_std > 0.2:
            event_type = "bump"
            confidence = min(1.0, (accel_z_max - 1.0) / 0.5)  # Scale based on magnitude
            
            if self._should_report_event(event_type, current_time):
                events.append({
                    'event': event_type,
                    'timestamp': latest_timestamp,
                    'confidence': round(confidence, 3)
                })
        
        # 4. POTHOLE DETECTION (sudden downward acceleration drop)
        if accel_z_min < self.POTHOLE_ACCEL_Z_THRESHOLD and accel_z_std > 0.2:
            event_type = "pothole"
            confidence = min(1.0, (1.0 - accel_z_min) / 0.4)  # Scale based on magnitude
            
            if self._should_report_event(event_type, current_time):
                events.append({
                    'event': event_type,
                    'timestamp': latest_timestamp,
                    'confidence': round(confidence, 3)
                })
        
        return events
    
    def _should_report_event(self, event_type, current_time):
        """
        Check if event should be reported based on cooldown and repetition rules
        """
        if self.last_event == event_type and (current_time - self.last_event_time) < self.event_cooldown:
            return False
        
        self.last_event = event_type
        self.last_event_time = current_time
        return True


class RealTimeEventProcessor:
    """
    Real-time processor that monitors CSV file and detects events
    """
    
    def __init__(self, data_folder):
        """
        Initialize processor for a specific data folder
        
        Args:
            data_folder: Path to folder containing combined_data.csv
        """
        self.data_folder = data_folder
        self.csv_file_path = os.path.join(data_folder, "combined_data.csv")
        self.event_file_path = os.path.join(data_folder, "events.csv")
        
        self.detector = EventDetector(window_size=10, sample_rate=5)
        self.last_processed_line = 0
        self.processing = False
        self.process_thread = None
        self.lock = Lock()
        
        # Initialize events CSV file
        self._initialize_events_file()
    
    def _initialize_events_file(self):
        """Initialize the events.csv file with headers"""
        try:
            with open(self.event_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'event'])
            print(f"📝 Events file initialized: {self.event_file_path}")
        except Exception as e:
            print(f"❌ Error initializing events file: {e}")
    
    def start_processing(self):
        """Start real-time event processing"""
        if self.processing:
            return
        
        self.processing = True
        self.process_thread = Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        print("🎯 Event detection started")
    
    def stop_processing(self):
        """Stop real-time event processing"""
        self.processing = False
        if self.process_thread:
            self.process_thread.join(timeout=2)
        print("🛑 Event detection stopped")
    
    def _process_loop(self):
        """Main processing loop that monitors CSV file for new data"""
        while self.processing:
            try:
                self._process_new_data()
                time.sleep(0.2)  # Check every 200ms for new data
            except Exception as e:
                if self.processing:  # Only log if we're still supposed to be running
                    print(f"❌ Event processing error: {e}")
                time.sleep(1)
    
    def _process_new_data(self):
        """Process any new lines in the CSV file"""
        if not os.path.exists(self.csv_file_path):
            return
        
        try:
            with open(self.csv_file_path, 'r', newline='') as f:
                reader = csv.reader(f)
                lines = list(reader)
            
            # Skip header and already processed lines
            new_lines = lines[max(1, self.last_processed_line + 1):]
            
            for line in new_lines:
                if len(line) >= 9:  # Ensure we have all IMU data
                    self._process_line(line)
                    self.last_processed_line += 1
                    
        except Exception as e:
            if self.processing:
                print(f"❌ Error reading CSV file: {e}")
    
    def _process_line(self, line):
        """Process a single CSV line for event detection"""
        try:
            # Parse CSV line: timestamp,epoch_time,image_filename,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,...
            timestamp = line[0]
            ax = float(line[3])
            ay = float(line[4]) 
            az = float(line[5])
            gx = float(line[6])
            gy = float(line[7])
            gz = float(line[8])
            
            # Add sample to detector
            with self.lock:
                self.detector.add_sample(timestamp, ax, ay, az, gx, gy, gz)
                
                # Detect events
                events = self.detector.detect_events()
                
                # Log any detected events
                for event in events:
                    self._log_event(event)
                    
        except Exception as e:
            print(f"❌ Error processing line: {e}")
    
    def _log_event(self, event):
        """Log detected event to events.csv file"""
        try:
            with open(self.event_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    event['timestamp'],
                    event['event']
                ])
            
            print(f"🎯 EVENT DETECTED: {event['event'].upper()} at {event['timestamp']}")
            
        except Exception as e:
            print(f"❌ Error logging event: {e}")


# Global event processor instance
current_event_processor = None
processor_lock = Lock()

def start_event_detection(data_folder):
    """
    Start event detection for a specific data folder
    
    Args:
        data_folder: Path to data folder containing combined_data.csv
    """
    global current_event_processor
    
    with processor_lock:
        # Stop any existing processor
        if current_event_processor:
            current_event_processor.stop_processing()
        
        # Start new processor
        current_event_processor = RealTimeEventProcessor(data_folder)
        current_event_processor.start_processing()

def stop_event_detection():
    """Stop current event detection"""
    global current_event_processor
    
    with processor_lock:
        if current_event_processor:
            current_event_processor.stop_processing()
            current_event_processor = None


if __name__ == "__main__":
    # Test the event detector with sample data
    print("🧪 Testing Event Detector with sample data...")
    
    detector = EventDetector()
    
    # Test with some sample IMU data (simulating different events)
    test_samples = [
        # Straight driving
        (0.63, 0.25, 0.75, -3.2, 1.0, 0.5),
        (0.64, 0.25, 0.75, -3.1, 0.8, 0.3),
        (0.63, 0.24, 0.76, -3.3, 0.9, 0.2),
        
        # Right turn
        (0.65, 0.30, 0.70, -2.8, 1.2, 25.0),
        (0.68, 0.35, 0.68, -2.5, 1.5, 28.0),
        (0.70, 0.40, 0.65, -2.2, 1.8, 30.0),
        
        # Bump
        (0.63, 0.25, 1.5, -3.2, 1.0, 0.5),
        (0.64, 0.26, 1.8, -3.1, 0.8, 0.3),
        (0.63, 0.24, 1.6, -3.3, 0.9, 0.2),
    ]
    
    for i, (ax, ay, az, gx, gy, gz) in enumerate(test_samples):
        timestamp = f"2025-01-01_12-00-{i:02d}-000"
        detector.add_sample(timestamp, ax, ay, az, gx, gy, gz)
        
        if len(detector.timestamps) >= detector.window_size:
            events = detector.detect_events()
            if events:
                for event in events:
                    print(f"✅ Detected: {event['event']} (confidence: {event['confidence']})")
    
    print("✅ Event detector test completed!")