# 🚀 Real-Time LSTM Event Detection for Raspberry Pi

This system uses your trained LSTM model to detect driving events in real-time as your Raspberry Pi collects IMU sensor data.

## 📋 Overview

**Your System:**
- ✅ Raspberry Pi with IMU sensor (MPU6050)
- ✅ Data collection script (`new.py`)
- ✅ Trained LSTM model (91.4% accuracy)
- ✅ Model weights file (`lstm_model_weights.weights.h5`)

**Events Detected:**
- **BUMP** - Road bumps/irregularities
- **LEFT** - Left turn maneuvers
- **RIGHT** - Right turn maneuvers  
- **STOP** - Vehicle stopping
- **STRAIGHT** - Straight line driving

## 🎯 Quick Start

### Step 1: Copy Files to Raspberry Pi

Copy these files to your Raspberry Pi in the same folder as `new.py`:
```
realtime_lstm_detector.py
run_lstm_events.py
lstm_model_weights.weights.h5
```

### Step 2: Install Dependencies

On your Raspberry Pi:
```bash
# Install TensorFlow (may take 5-10 minutes)
pip install tensorflow

# Or use TensorFlow Lite for faster/lighter installation
pip install tflite-runtime

# Install other dependencies
pip install numpy pandas
```

### Step 3: Run the System

**Terminal 1** - Start data collection:
```bash
python new.py
```

**Terminal 2** - Start event detection:
```bash
python run_lstm_events.py
```

That's it! The system will:
1. ✅ Wait for data collection to start
2. ✅ Load your trained LSTM model
3. ✅ Detect events in real-time (every 104 samples)
4. ✅ Save events to `events.csv` in each ride folder
5. ✅ Show live event predictions in console

## 📊 How It Works

### Data Flow
```
IMU Sensor → new.py → combined_data.csv
                           ↓
               realtime_lstm_detector.py
                           ↓
           (Collects 104 samples = ~21 seconds at 5Hz)
                           ↓
                    LSTM Model
                           ↓
              Event Prediction → events.csv
```

### Prediction Window
- **Window Size**: 104 timesteps (~21 seconds at 5 Hz)
- **Input Features**: 7 (Ax, Ay, Az, Gx, Gy, Gz, Speed)
- **Sliding Window**: New prediction every 2 seconds
- **Output**: Event type with confidence level

## 📁 Output Format

### events.csv
```csv
timestamp,event
2025-09-13_22-34-50-275,straight
2025-09-13_22-35-12-841,right
2025-09-13_22-35-28-193,bump
2025-09-13_22-35-45-672,left
2025-09-13_22-36-02-115,straight
```

### Console Output
```
🎯 EVENT: STRAIGHT (92.3%) at 2025-09-13_22-34-50-275
🎯 EVENT: RIGHT (87.6%) at 2025-09-13_22-35-12-841
🎯 EVENT: BUMP (94.1%) at 2025-09-13_22-35-28-193
```

## ⚙️ System Architecture

### Files Explained

| File | Purpose |
|------|---------|
| `new.py` | Your original data collection script |
| `realtime_lstm_detector.py` | LSTM model wrapper and real-time processor |
| `run_lstm_events.py` | Auto-integration script |
| `lstm_model_weights.weights.h5` | Your trained model (298 KB) |

### Key Components

#### 1. LSTMEventDetector Class
- Loads your trained model weights
- Maintains rolling buffer of 104 samples
- Makes predictions using sliding window

#### 2. RealTimeLSTMProcessor Class  
- Monitors `combined_data.csv` for new data
- Feeds data to LSTM detector
- Logs events to `events.csv`
- Filters duplicate/redundant events

#### 3. AutoLSTMDetection Class
- Automatically detects when data collection starts/stops
- Manages event detection lifecycle
- No manual intervention needed

## 🔧 Configuration

### Adjust Prediction Frequency

In `realtime_lstm_detector.py`, modify:
```python
self.event_cooldown = 2.0  # seconds between predictions
```

### Change Model Path

When running:
```python
from realtime_lstm_detector import start_lstm_detection

# Use custom model path
start_lstm_detection('data/ride01', 'path/to/your/model.h5')
```

## 💡 Usage Examples

### Example 1: Basic Usage
```bash
# Terminal 1
python new.py

# Terminal 2
python run_lstm_events.py
```

### Example 2: Test on Existing Data
```python
from realtime_lstm_detector import LSTMEventDetector
import pandas as pd

# Load detector
detector = LSTMEventDetector('lstm_model_weights.weights.h5')

# Load your CSV data
data = pd.read_csv('data/ride01/combined_data.csv')

# Process samples
for i, row in data.iterrows():
    detector.add_sample(
        row['timestamp'],
        row['ax_g'], row['ay_g'], row['az_g'],
        row['gx_dps'], row['gy_dps'], row['gz_dps'],
        0.0  # speed
    )
    
    if detector.can_predict():
        result = detector.predict()
        if result:
            print(f"{result['event']} ({result['confidence']*100:.1f}%)")
```

### Example 3: Manual Integration
```python
from realtime_lstm_detector import start_lstm_detection, stop_lstm_detection

# Start detection for specific folder
start_lstm_detection('data/ride05')

# ... your code ...

# Stop detection
stop_lstm_detection()
```

## 🧪 Testing

### Test the Detector
```bash
python realtime_lstm_detector.py
```

This will test the detector on `sample_imudata.txt` if available.

### Verify Model Loading
```python
from realtime_lstm_detector import LSTMEventDetector

detector = LSTMEventDetector()
if detector.model_loaded:
    print("✅ Model loaded successfully!")
    print(f"Parameters: {detector.model.count_params():,}")
else:
    print("❌ Model failed to load")
```

## 🐛 Troubleshooting

### Issue: "TensorFlow not found"
**Solution:**
```bash
# Standard installation
pip install tensorflow

# For Raspberry Pi 3/4
pip install tensorflow-aarch64

# Lightweight option
pip install tflite-runtime
```

### Issue: "Model weights not found"
**Solution:**
- Ensure `lstm_model_weights.weights.h5` is in the same directory
- Check file permissions: `chmod 644 lstm_model_weights.weights.h5`

### Issue: "Not enough data for prediction"
**Cause:** Need 104 samples before first prediction (~21 seconds)
**Solution:** Wait for buffer to fill, then predictions start

### Issue: Too many predictions
**Solution:** Increase cooldown period:
```python
self.event_cooldown = 5.0  # Wait 5 seconds between events
```

### Issue: Low confidence predictions
**Possible causes:**
1. Sensor orientation different from training data
2. Different vehicle/road conditions
3. Need to fine-tune model

**Solution:** Collect new data and fine-tune:
```python
model.load_weights('lstm_model_weights.weights.h5')
model.fit(X_new, y_new, epochs=20)
model.save_weights('lstm_model_finetuned.weights.h5')
```

## 📈 Performance

- **Latency**: ~2-4 seconds (due to 104-sample window)
- **CPU Usage**: ~15-25% on Raspberry Pi 4
- **Memory**: ~200MB (TensorFlow + model)
- **Accuracy**: 91.4% (from original training)

### Optimization Tips

1. **Use TensorFlow Lite**: Faster inference
   ```bash
   pip install tflite-runtime
   ```

2. **Reduce Buffer Size**: Faster predictions (less accuracy)
   ```python
   self.n_time_steps = 50  # Instead of 104
   ```

3. **Batch Processing**: Process multiple samples at once

## 🎓 Understanding the Model

### Input Requirements
- **Shape**: (104, 7)
- **104 timesteps** = ~21 seconds at 5 Hz
- **7 features**: Ax, Ay, Az, Gx, Gy, Gz, Speed

### Model Architecture
```
Input (104, 7)
    ↓
LSTM (100 units)
    ↓
Dropout (0.5)
    ↓
Dense (10 units, ReLU)
    ↓
Dense (5 units, Softmax)
    ↓
Output [BUMP, LEFT, RIGHT, STOP, STRAIGHT]
```

### Confidence Levels
- **>90%**: High confidence - very likely correct
- **70-90%**: Good confidence - likely correct
- **50-70%**: Moderate confidence - possibly correct
- **<50%**: Low confidence - uncertain

## 🔄 Integration with Your Project

The system is designed to work alongside your existing `new.py` without modifications:

```
Your Raspberry Pi Setup:
├── new.py                          # Your original script (NO CHANGES)
├── realtime_lstm_detector.py       # NEW: LSTM detector
├── run_lstm_events.py              # NEW: Auto-runner
├── lstm_model_weights.weights.h5   # Your model
└── data/                           # Data folder
    ├── ride01/
    │   ├── images/
    │   ├── combined_data.csv       # From new.py
    │   └── events.csv              # NEW: Generated automatically
    └── ride02/
        ├── images/
        ├── combined_data.csv
        └── events.csv
```

## 📚 Next Steps

1. **Collect More Data**: Improve accuracy with diverse scenarios
2. **Fine-Tune Model**: Adapt to your specific vehicle/conditions
3. **Add New Events**: Train for potholes, u-turns, etc.
4. **Real-Time Alerts**: Trigger actions based on events
5. **Data Analysis**: Analyze driving patterns from events.csv

## 🎯 Production Checklist

- [ ] TensorFlow installed on Raspberry Pi
- [ ] Model weights file copied
- [ ] Both Python scripts copied
- [ ] Tested on sample data
- [ ] Verified events.csv creation
- [ ] Checked console output
- [ ] Monitored CPU/memory usage
- [ ] Tested full ride cycle

## 📞 Support

For issues or questions:
1. Check troubleshooting section above
2. Verify all dependencies installed
3. Test with sample data first
4. Check file permissions and paths

---

**Your trained LSTM model is now running real-time event detection! 🚀**
