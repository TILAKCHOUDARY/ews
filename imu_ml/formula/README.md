# Driving Performance Score Analyzer

Analyzes how well a driver performed during a ride based on IMU data, GPS speed, and detected events.

**Based on research papers on driver behavior analysis**

---

## 🎯 What It Does

Calculates a **comprehensive driving score (0-100)** based on:

1. **Speed Management (25%)** - Speed consistency, no sudden changes
2. **Acceleration Smoothness (30%)** - Smooth acceleration/braking (low jerk)
3. **Turning Quality (25%)** - Slowing down before turns, smooth rotation
4. **Event Responses (20%)** - Proper handling of bumps, potholes, and stops

---

## 🚀 How to Use

### Command Format:
```bash
python driving_score_analyzer.py <path/to/rideXX>
```

### Examples:

**From formula folder:**
```bash
cd formula
python driving_score_analyzer.py ../data/ride01
python driving_score_analyzer.py ../data/ride05
```

**From anywhere (absolute path):**
```bash
python driving_score_analyzer.py /home/pi/data/ride01
python driving_score_analyzer.py C:\data\ride03
```

**Analyze multiple rides:**
```bash
# Linux/Mac
for ride in ../data/ride*; do
    python driving_score_analyzer.py "$ride"
done

# Windows
foreach ($ride in Get-ChildItem ../data/ride*) {
    python driving_score_analyzer.py $ride.FullName
}
```

---

## 📊 Output

### Console Report:
```
============================================================
   🌟 DRIVING PERFORMANCE REPORT
============================================================

🚗 Ride: ride01
📊 Samples: 245
🎯 Events: 12

📈 DETAILED SCORES:
------------------------------------------------------------
  Speed Management              :  85.50/100
  Acceleration Smoothness       :  78.30/100
  Turning Quality               :  92.10/100
  Event Responses               :  88.00/100
------------------------------------------------------------
  FINAL SCORE                   :  85.48/100
  RATING                        : VERY GOOD
============================================================
```

### JSON Report: `data/rideXX/driving_score_report.json`
```json
{
  "ride": "ride01",
  "final_score": 85.48,
  "metrics": {
    "speed_management": {
      "score": 85.5,
      "avg_speed": 35.2,
      "speed_variation": 15.3,
      "consistency": 82.0
    },
    ...
  }
}
```

---

## 📈 Scoring System

### Rating Scale:
- **90-100**: EXCELLENT 🌟
- **80-89**: VERY GOOD ⭐
- **70-79**: GOOD ✅
- **60-69**: AVERAGE 👍
- **<60**: NEEDS IMPROVEMENT ⚠️

### What's Measured:

#### 1. Speed Management
- ✅ Consistent speed (low variation)
- ✅ Minimal sudden speed changes
- ❌ Penalty for erratic speed

#### 2. Acceleration Smoothness
- ✅ Low "jerk" (smooth acceleration changes)
- ✅ No harsh braking/acceleration
- ❌ Penalty for harsh events

#### 3. Turning Quality
- ✅ Slowing down before turns
- ✅ Smooth rotation (gyroscope)
- ✅ Controlled exit
- ❌ Penalty for high-speed turns

#### 4. Event Responses
- ✅ Slowing down before bumps
- ✅ Slowing down before potholes (stricter than bumps)
- ✅ Smooth stops
- ❌ Penalty for hitting bumps at high speed
- ❌ Penalty for hitting potholes at high speed (higher penalty)

---

## 🔬 Formula Details

### Speed Management Score
```
Speed Variation Index = (std_dev / mean) * 100
Score = 100 - penalties for high variation
```

### Acceleration Smoothness Score
```
Jerk = rate of change of acceleration
Smoothness = 100 - (avg_jerk * 100) - harsh_events_penalty
```

### Turning Quality Score
```
For each turn:
  - Speed reduction before turn: +30 points
  - Gyroscope smoothness: +20 points
Average all turns
```

### Event Response Score
```
For bumps: 
  Speed < 20 km/h = 100 points
  Speed 20-40 km/h = 90-70 points
  Speed > 40 km/h = < 70 points

For potholes (stricter):
  Speed < 15 km/h = 100 points
  Speed 15-30 km/h = 95-65 points  
  Speed > 30 km/h = < 65 points

For stops: 
  Smooth deceleration = higher score
```

### Final Score
```
Final = (Speed * 0.25) + (Acceleration * 0.30) + 
        (Turning * 0.25) + (Events * 0.20)
```

---

## 📁 Required Files

The analyzer needs these files in the ride folder:

### Must Have:
- **`combined_data.csv`** - IMU sensor data and GPS speed
  - Columns: timestamp, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps, gps_speed_kn, etc.

### Optional (but recommended):
- **`events.csv`** - Detected events from LSTM detector
  - Format: timestamp, event
  - Events: straight, left, right, bump, pothole, stop
  - If missing: Event analysis will show neutral scores

---

## 💡 Complete Workflow

### Step 1: Collect Data
```bash
# Run your data collection
python new.py
# Creates: data/ride01/combined_data.csv
```

### Step 2: Detect Events (from final folder)
```bash
cd final
python run_lstm_events.py
# Creates: data/ride01/events.csv
```

### Step 3: Analyze Performance (from formula folder)
```bash
cd formula
python driving_score_analyzer.py ../data/ride01
# Creates: data/ride01/driving_score_report.json
```

---

## 📊 Understanding Your Score

### High Score (85+)
- Smooth, consistent driving
- Proper speed management
- Good anticipation of events

### Medium Score (70-84)
- Generally good driving
- Some rough patches
- Room for improvement

### Low Score (<70)
- Aggressive driving patterns
- Harsh acceleration/braking
- Poor event handling
- Consider defensive driving training

---

---

## 📋 All Detected Events

The analyzer handles all these event types:

| Event | What It Measures | Scoring Criteria |
|-------|------------------|------------------|
| **straight** | Baseline driving | Speed consistency |
| **left** | Left turns | Speed reduction + smoothness |
| **right** | Right turns | Speed reduction + smoothness |
| **bump** | Road bumps | Speed before bump |
| **pothole** | Potholes | Speed before pothole (strict) |
| **stop** | Vehicle stops | Deceleration smoothness |

---

## 🔍 What Gets Analyzed

### From `combined_data.csv`:
- ✅ Accelerometer data (ax, ay, az) → Jerk calculation
- ✅ Gyroscope data (gx, gy, gz) → Turn smoothness
- ✅ GPS speed → Speed management, event response
- ✅ Timestamps → Event correlation

### From `events.csv`:
- ✅ Event types → Turn quality, event responses
- ✅ Event timestamps → Speed before/during/after events
- ✅ Event frequency → Overall behavior patterns

---

## ⚙️ How It Works

```
┌─────────────────────────────────────────────┐
│  Input: data/rideXX/                        │
│  - combined_data.csv (IMU + GPS)            │
│  - events.csv (detected events)             │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│  Analysis:                                   │
│  1. Speed Management (25%)                   │
│     - Calculate speed variation              │
│     - Count sudden changes                   │
│                                              │
│  2. Acceleration Smoothness (30%)            │
│     - Calculate jerk (d/dt acceleration)     │
│     - Count harsh events                     │
│                                              │
│  3. Turning Quality (25%)                    │
│     - Analyze each turn event                │
│     - Check speed reduction                  │
│     - Measure gyroscope smoothness           │
│                                              │
│  4. Event Responses (20%)                    │
│     - Analyze bumps: speed before            │
│     - Analyze potholes: speed before (strict)│
│     - Analyze stops: deceleration smoothness │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│  Output:                                     │
│  - Console report with scores                │
│  - JSON file: driving_score_report.json      │
│  - Final score (0-100) with rating           │
└─────────────────────────────────────────────┘
```

---

## 🎯 Scoring Details

### Speed Management (25 points)
```python
speed_variation = (std_dev / mean) * 100

if variation < 30%: score = 100
elif variation < 50%: score = 100 - (variation - 30) * 2
else: score = max(20, 100 - variation)

# Penalty for sudden changes (>5 km/h)
consistency_penalty = sudden_changes * 2

final = variation_score * 0.6 + consistency_score * 0.4
```

### Acceleration Smoothness (30 points)
```python
jerk = abs(diff(acceleration_magnitude))
avg_jerk = mean(jerk)

if avg_jerk < 0.1: score = 100
elif avg_jerk < 0.2: score = 90 - (avg_jerk - 0.1) * 100  
elif avg_jerk < 0.3: score = 70 - (avg_jerk - 0.2) * 200
else: score = max(20, 50 - avg_jerk * 100)

# Penalty for harsh events (jerk > 0.5)
harsh_penalty = min(30, harsh_count * 2)
final = max(0, score - harsh_penalty)
```

### Turning Quality (25 points)
```python
For each turn:
  speed_reduction = speed_before - speed_during
  
  if reduction > 5 km/h: speed_score = 100
  elif reduction > 0: speed_score = 70 + reduction * 6
  else: speed_score = max(30, 70 + reduction * 10)
  
  smoothness = 100 - min(50, std(gyro_z) * 2)
  
  turn_score = speed_score * 0.6 + smoothness * 0.4

average_all_turns()
```

### Event Responses (20 points)
```python
Bumps:
  if speed < 20: score = 100
  elif speed < 40: score = 90 - (speed - 20)
  else: score = max(30, 70 - (speed - 40))

Potholes (stricter):
  if speed < 15: score = 100
  elif speed < 30: score = 95 - (speed - 15) * 2
  else: score = max(20, 65 - (speed - 30))

Stops:
  smoothness = 100 - min(50, std(deceleration) * 50)

average_all_events()
```

---

## 🚨 Common Issues

### Error: "Folder not found"
**Cause:** Wrong path to ride folder  
**Solution:** Use correct path, e.g., `../data/ride01` from formula folder

### Error: "No such file: combined_data.csv"
**Cause:** Ride folder doesn't have data  
**Solution:** Make sure data collection completed successfully

### Warning: "No events.csv found"
**Cause:** Event detection not run  
**Solution:** This is OK - analyzer will work with IMU data only (events score = neutral)

### Warning: "No speed data available"
**Cause:** GPS speed not captured  
**Solution:** Speed metrics will show neutral scores (50/100)

---

## 📊 Output Files

### Console Report
Shows detailed breakdown of all metrics with scores

### JSON Report: `driving_score_report.json`
```json
{
  "ride": "ride01",
  "timestamp": "2025-01-01T12:00:00",
  "samples": 245,
  "events": 12,
  "metrics": {
    "speed_management": {
      "score": 85.5,
      "avg_speed": 35.2,
      "speed_variation": 15.3,
      "consistency": 82.0
    },
    "acceleration_smoothness": {
      "score": 78.3,
      "avg_jerk": 0.15,
      "harsh_events": 5,
      "moderate_events": 12
    },
    "turning_quality": {
      "score": 92.1,
      "turns_analyzed": 5,
      "left_turns": 2,
      "right_turns": 3
    },
    "event_responses": {
      "score": 88.0,
      "events_analyzed": 7,
      "bumps": 2,
      "potholes": 1,
      "stops": 4
    },
    "final": {
      "score": 85.48,
      "rating": "VERY GOOD",
      "emoji": "⭐"
    }
  },
  "final_score": 85.48
}
```

---

## ✅ Testing

A test script is included: `test_analyzer.py`

Run on Raspberry Pi to validate:
```bash
cd formula
python test_analyzer.py
```

This creates synthetic test data with all event types and validates the analyzer.

---

That's it! Run it after each ride to get your driving performance score! 🚗
