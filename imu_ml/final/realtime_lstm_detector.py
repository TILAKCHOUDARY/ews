"""
Real-Time LSTM Event Detection for Raspberry Pi
Uses your trained LSTM model to detect driving events in real-time
"""

import numpy as np
import pandas as pd
import csv
import os
import time
from datetime import datetime
from threading import Thread, Lock
from collections import deque

# Try to import Keras/TensorFlow (will work on Raspberry Pi after installation)
try:
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    KERAS_AVAILABLE = True
except ImportError:
    print("⚠️  Keras not available. Please install: pip install tensorflow")
    KERAS_AVAILABLE = False


class LSTMEventDetector:
    """
    Real-time LSTM-based event detection from IMU sensor data
    """
    
    def __init__(self, model_weights_path='lstm_model_weights.weights.h5'):
        """
        Initialize LSTM detector with pre-trained weights
        
        Args:
            model_weights_path: Path to trained model weights file
        """
        self.model = None
        self.model_loaded = False
        self.class_names = ['BUMP', 'LEFT', 'RIGHT', 'STOP', 'STRAIGHT']
        
        # Buffer for collecting 104 timesteps (required for LSTM input)
        self.n_time_steps = 104
        self.n_features = 7  # Ax, Ay, Az, Gx, Gy, Gz, Speed
        
        # Rolling buffer for sensor data
        self.sensor_buffer = deque(maxlen=self.n_time_steps)
        self.timestamps_buffer = deque(maxlen=self.n_time_steps)
        
        # Load model
        if KERAS_AVAILABLE:
            self._load_model(model_weights_path)
        else:
            print("❌ Keras not available - running in demo mode")
    
    def _create_model(self):
        """Create LSTM model with same architecture as training"""
        model = Sequential()
        model.add(LSTM(units=100, input_shape=(self.n_time_steps, self.n_features)))
        model.add(Dropout(0.5))
        model.add(Dense(units=10, activation='relu'))
        model.add(Dense(5, activation='softmax'))
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def _load_model(self, weights_path):
        """Load pre-trained model weights"""
        try:
            print(f"🔧 Loading LSTM model from: {weights_path}")
            self.model = self._create_model()
            self.model.load_weights(weights_path)
            self.model_loaded = True
            print(f"✅ LSTM model loaded successfully!")
            print(f"📊 Model parameters: {self.model.count_params():,}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("   Make sure lstm_model_weights.weights.h5 is in the same directory")
            self.model_loaded = False
    
    def add_sample(self, timestamp, ax, ay, az, gx, gy, gz, speed=0.0):
        """
        Add new IMU sample to the buffer
        
        Args:
            timestamp: Sample timestamp string
            ax, ay, az: Accelerometer readings in g
            gx, gy, gz: Gyroscope readings in deg/s
            speed: Vehicle speed (default 0 if not available)
        """
        # Store timestamp
        self.timestamps_buffer.append(timestamp)
        
        # Create feature vector: [Ax, Ay, Az, Gx, Gy, Gz, Speed]
        sample = [ax, ay, az, gx, gy, gz, speed]
        self.sensor_buffer.append(sample)
    
    def can_predict(self):
        """Check if we have enough data to make a prediction"""
        return len(self.sensor_buffer) >= self.n_time_steps
    
    def predict(self):
        """
        Make prediction using current buffer
        
        Returns:
            dict: {'event': str, 'timestamp': str, 'confidence': float} or None if not ready
        """
        if not self.can_predict():
            return None
        
        if not self.model_loaded:
            return None
        
        try:
            # Convert buffer to numpy array
            data = np.array(list(self.sensor_buffer), dtype=np.float32)
            
            # Reshape for model input: (1, 104, 7)
            data = data.reshape(1, self.n_time_steps, self.n_features)
            
            # Make prediction
            prediction = self.model.predict(data, verbose=0)[0]
            predicted_idx = np.argmax(prediction)
            confidence = float(prediction[predicted_idx])
            
            # Get latest timestamp
            latest_timestamp = self.timestamps_buffer[-1]
            
            return {
                'event': self.class_names[predicted_idx].lower(),
                'timestamp': latest_timestamp,
                'confidence': confidence
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None


class RealTimeLSTMProcessor:
    """
    Real-time processor that monitors CSV file and runs LSTM predictions
    """
    
    def __init__(self, data_folder, model_weights_path='lstm_model_weights.weights.h5'):
        """
        Initialize processor for a specific data folder
        
        Args:
            data_folder: Path to folder containing combined_data.csv
            model_weights_path: Path to LSTM model weights
        """
        self.data_folder = data_folder
        self.csv_file_path = os.path.join(data_folder, "combined_data.csv")
        self.event_file_path = os.path.join(data_folder, "events.csv")
        
        self.detector = LSTMEventDetector(model_weights_path)
        self.last_processed_line = 0
        self.processing = False
        self.process_thread = None
        self.lock = Lock()
        
        # Initialize events CSV file
        self._initialize_events_file()
        
        # Track last predicted event to avoid duplicates
        self.last_event = None
        self.last_event_time = 0
        self.event_cooldown = 2.0  # seconds between predictions
    
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
        print("🎯 LSTM event detection started")
    
    def stop_processing(self):
        """Stop real-time event processing"""
        self.processing = False
        if self.process_thread:
            self.process_thread.join(timeout=2)
        print("🛑 LSTM event detection stopped")
    
    def _process_loop(self):
        """Main processing loop that monitors CSV file for new data"""
        while self.processing:
            try:
                self._process_new_data()
                time.sleep(0.2)  # Check every 200ms for new data
            except Exception as e:
                if self.processing:
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
            
            # Speed from GPS if available (column 14), otherwise 0
            try:
                speed_str = line[14]
                speed = float(speed_str) if speed_str != 'N/A' else 0.0
            except (IndexError, ValueError):
                speed = 0.0
            
            # Add sample to detector
            with self.lock:
                self.detector.add_sample(timestamp, ax, ay, az, gx, gy, gz, speed)
                
                # Try to make prediction if we have enough data
                if self.detector.can_predict():
                    result = self.detector.predict()
                    
                    if result and self._should_log_event(result):
                        self._log_event(result)
                    
        except Exception as e:
            print(f"❌ Error processing line: {e}")
    
    def _should_log_event(self, result):
        """
        Check if event should be logged based on cooldown
        Also filter out continuous STRAIGHT events
        """
        current_time = time.time()
        event = result['event']
        
        # Always log non-STRAIGHT events after cooldown
        if event != 'straight':
            if (current_time - self.last_event_time) >= self.event_cooldown:
                self.last_event = event
                self.last_event_time = current_time
                return True
        else:
            # Log STRAIGHT less frequently (every 10 seconds)
            if self.last_event != 'straight' or (current_time - self.last_event_time) >= 10.0:
                self.last_event = event
                self.last_event_time = current_time
                return True
        
        return False
    
    def _log_event(self, result):
        """Log detected event to events.csv file"""
        try:
            with open(self.event_file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    result['timestamp'],
                    result['event']
                ])
            
            # Console output with confidence
            confidence_pct = result['confidence'] * 100
            print(f"🎯 EVENT: {result['event'].upper()} ({confidence_pct:.1f}%) at {result['timestamp']}")
            
        except Exception as e:
            print(f"❌ Error logging event: {e}")


# Global processor instance
current_lstm_processor = None
processor_lock = Lock()


def start_lstm_detection(data_folder, model_weights_path='lstm_model_weights.weights.h5'):
    """
    Start LSTM-based event detection for a specific data folder
    
    Args:
        data_folder: Path to data folder containing combined_data.csv
        model_weights_path: Path to LSTM model weights file
    """
    global current_lstm_processor
    
    with processor_lock:
        # Stop any existing processor
        if current_lstm_processor:
            current_lstm_processor.stop_processing()
        
        # Start new processor
        current_lstm_processor = RealTimeLSTMProcessor(data_folder, model_weights_path)
        current_lstm_processor.start_processing()


def stop_lstm_detection():
    """Stop current LSTM event detection"""
    global current_lstm_processor
    
    with processor_lock:
        if current_lstm_processor:
            current_lstm_processor.stop_processing()
            current_lstm_processor = None


if __name__ == "__main__":
    print("=" * 60)
    print("   🧪 Testing LSTM Event Detector")
    print("=" * 60)
    
    # Test with sample data file if it exists
    sample_file = "sample_imudata.txt"
    
    if os.path.exists(sample_file):
        print(f"\n📂 Found sample data: {sample_file}")
        print("Testing event detection...")
        
        # Create a test detector
        detector = LSTMEventDetector('lstm_model_weights.weights.h5')
        
        if detector.model_loaded:
            # Read sample data
            df = pd.read_csv(sample_file)
            print(f"✅ Loaded {len(df)} samples")
            
            # Process samples
            for i, row in df.iterrows():
                detector.add_sample(
                    row['timestamp'],
                    row['ax_g'], row['ay_g'], row['az_g'],
                    row['gx_dps'], row['gy_dps'], row['gz_dps'],
                    0.0  # speed
                )
                
                if detector.can_predict() and i % 10 == 0:  # Predict every 10 samples
                    result = detector.predict()
                    if result:
                        print(f"  Sample {i}: {result['event'].upper()} ({result['confidence']*100:.1f}%)")
            
            print("\n✅ Test completed!")
        else:
            print("❌ Could not load model for testing")
    else:
        print(f"\n⚠️  Sample data file not found: {sample_file}")
        print("Place the script in the same directory as your data to test.")
    
    print("\n💡 To use in production:")
    print("   from realtime_lstm_detector import start_lstm_detection")
    print("   start_lstm_detection('data/ride01')")
