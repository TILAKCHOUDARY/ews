# 🚀 START HERE - Real-Time Event Detection

## What This Does
Your trained LSTM model will detect driving events (BUMP, LEFT, RIGHT, STOP, STRAIGHT) in **real-time** as your Raspberry Pi collects IMU data!

## 📦 Files You Need on Raspberry Pi

Copy these 3 files to your Raspberry Pi (same folder as `new.py`):

1. ✅ `realtime_lstm_detector.py` - The LSTM detector
2. ✅ `run_lstm_events.py` - Auto-runner
3. ✅ `lstm_model_weights.weights.h5` - Your trained model

## ⚡ Quick Setup (3 Steps)

### 1. Install Dependencies (One Time Only)
```bash
pip install tensorflow numpy pandas
```
*(Takes 5-10 minutes on Raspberry Pi)*

### 2. Start Data Collection
```bash
python new.py
```

### 3. Start Event Detection (New Terminal)
```bash
python run_lstm_events.py
```

## 🎯 What Happens

```
Terminal 1: new.py running, collecting IMU data...
Terminal 2: run_lstm_events.py detects events...

🎯 EVENT: STRAIGHT (92.3%) at 2025-09-13_22-34-50-275
🎯 EVENT: RIGHT (87.6%) at 2025-09-13_22-35-12-841
🎯 EVENT: BUMP (94.1%) at 2025-09-13_22-35-28-193
```

## 📁 Output Location

Events saved to: `data/ride##/events.csv`

Example:
```csv
timestamp,event
2025-09-13_22-34-50-275,straight
2025-09-13_22-35-12-841,right
2025-09-13_22-35-28-193,bump
```

## ⏱️ Timing

- **First prediction**: After ~21 seconds (needs 104 samples)
- **Then**: New prediction every 2 seconds
- **No changes to your existing code needed!**

## 📚 Full Documentation

- `LSTM_EVENT_DETECTION_README.md` - Complete guide
- `realtime_lstm_detector.py` - Core detector code
- `run_lstm_events.py` - Auto-runner code

## ❓ Quick Troubleshooting

**"TensorFlow not found"**
→ `pip install tensorflow`

**"Model weights not found"**  
→ Copy `lstm_model_weights.weights.h5` to same folder

**"Not enough data"**
→ Wait 21 seconds for buffer to fill

## ✅ That's It!

Your real-time event detection is now running! 🎉

---

**Need more info?** Read `LSTM_EVENT_DETECTION_README.md`
