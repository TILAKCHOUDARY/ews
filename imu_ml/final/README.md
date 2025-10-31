# Real-Time Event Detection for Raspberry Pi

Detects driving events (BUMP, LEFT, RIGHT, STOP, STRAIGHT) using your trained LSTM model.

---

## 📦 Files in This Folder

```
final/
├── README.md                      ← This file
├── realtime_lstm_detector.py      ← Event detector engine
├── run_lstm_events.py             ← Auto-runner script
├── requirements.txt               ← Python dependencies
└── lstm_model_weights.weights.h5  ← Your trained model (291 KB)
```

---

## 🚀 Setup (3 Steps)

### 1. Copy to Raspberry Pi
Copy this entire `final` folder to your Raspberry Pi, in the **same directory** where your `new.py` is located.

### 2. Install Dependencies (One Time)
```bash
cd ~/final
pip install -r requirements.txt
```
Wait 5-10 minutes for TensorFlow to install.

### 3. Run It
Open two terminals:

**Terminal 1** - Start data collection:
```bash
python new.py
```

**Terminal 2** - Start event detection:
```bash
cd final
python run_lstm_events.py
```

---

## 📊 What You'll See

### Console Output:
```
🎯 EVENT: STRAIGHT (92.3%) at 2025-09-13_22-34-50-275
🎯 EVENT: RIGHT (87.6%) at 2025-09-13_22-35-12-841
🎯 EVENT: BUMP (94.1%) at 2025-09-13_22-35-28-193
```

### Output File: `data/rideXX/events.csv`
```csv
timestamp,event
2025-09-13_22-34-50-275,straight
2025-09-13_22-35-12-841,right
2025-09-13_22-35-28-193,bump
```

---

## ⏱️ Timing

- First prediction: After 21 seconds (needs 104 samples)
- New predictions: Every 2 seconds
- No changes to your `new.py` needed!

---

## 🐛 Troubleshooting

**"TensorFlow not found"**  
→ `pip install tensorflow`

**"Model weights not found"**  
→ Make sure `lstm_model_weights.weights.h5` is in the `final` folder

**"No predictions"**  
→ Wait 21 seconds for buffer to fill  
→ Check if `new.py` is running and collecting data

**"Memory error"**  
→ Close other apps  
→ Try: `pip install tflite-runtime` (lighter version)

---

## ✅ How It Works

1. Your `new.py` collects IMU data → `combined_data.csv`
2. `run_lstm_events.py` monitors for new data
3. Collects 104 samples (21 seconds)
4. LSTM model predicts event type
5. Logs to `events.csv`

**Events Detected:**
- BUMP (road bumps)
- LEFT (left turns)
- RIGHT (right turns)
- STOP (vehicle stopping)
- STRAIGHT (straight driving)

---

## 📁 Output Structure

```
data/
└── ride01/
    ├── images/
    ├── combined_data.csv    ← From new.py
    └── events.csv           ← From detector ✨
```

---

That's it! Your system will detect events in real-time! 🎯
