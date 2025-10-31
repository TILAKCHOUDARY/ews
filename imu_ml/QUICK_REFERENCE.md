# Quick Reference Cheat Sheet

## 🚀 Quick Start Commands

### Run the Test Script
```bash
python load_and_test_model.py
```

### Load Model in Your Code
```python
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Define model
model = Sequential()
model.add(LSTM(units=100, input_shape=(104, 7)))
model.add(Dropout(0.5))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load weights
model.load_weights('lstm_model_weights.weights.h5')
```

### Make a Prediction
```python
import numpy as np

# Your sensor data: 104 timesteps × 7 features
sensor_data = np.array([...])  # Shape: (104, 7)

# Reshape and predict
data = sensor_data.reshape(1, 104, 7)
prediction = model.predict(data)
predicted_class = np.argmax(prediction)

class_names = ['BUMP', 'LEFT', 'RIGHT', 'STOP', 'STRAIGHT']
print(f"Predicted: {class_names[predicted_class]}")
```

---

## 📁 Files in Your Project

| File | Purpose | Size |
|------|---------|------|
| `LSTM-ESW.ipynb` | Original training notebook | 154 KB |
| `lstm_model_weights.weights.h5` | **Pre-trained model weights** | 298 KB |
| `MODEL_USAGE_GUIDE.md` | Complete tutorial (this guide) | - |
| `load_and_test_model.py` | Ready-to-run test script | - |
| `QUICK_REFERENCE.md` | This cheat sheet | - |

---

## 🎯 Your Model Specs

- **Architecture**: LSTM → Dropout → Dense → Softmax
- **Input**: `(104, 7)` = 104 timesteps of 7 sensor readings
- **Output**: 5 classes (BUMP, LEFT, RIGHT, STOP, STRAIGHT)
- **Parameters**: 44,265 trainable weights
- **Performance**: 91.4% test accuracy

### Input Features (7 total)
1. `Ax` - Accelerometer X-axis
2. `Ay` - Accelerometer Y-axis
3. `Az` - Accelerometer Z-axis
4. `Gx` - Gyroscope X-axis
5. `Gy` - Gyroscope Y-axis
6. `Gz` - Gyroscope Z-axis
7. `Speed` - Vehicle speed

---

## 🔄 Common Operations

### 1️⃣ Load Complete Model (Easier)
```python
from keras.models import load_model

# If you saved full model (recommended for future)
model = load_model('lstm_model_complete.h5')
```

### 2️⃣ Save Complete Model
```python
# After training or fine-tuning
model.save('lstm_model_complete.h5')
```

### 3️⃣ Fine-Tune with New Data
```python
# Load existing weights
model.load_weights('lstm_model_weights.weights.h5')

# Train on new data (fewer epochs)
model.fit(X_new, y_new, epochs=20, batch_size=16)

# Save updated weights
model.save_weights('lstm_model_finetuned.weights.h5')
```

### 4️⃣ Freeze Layers During Fine-Tuning
```python
# Freeze all layers except last 2
for layer in model.layers[:-2]:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### 5️⃣ Evaluate Model
```python
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy*100:.2f}%")
```

### 6️⃣ Get Predictions with Probabilities
```python
predictions = model.predict(X_test)
# predictions[i] = [prob_BUMP, prob_LEFT, prob_RIGHT, prob_STOP, prob_STRAIGHT]

# Get class with highest probability
predicted_classes = np.argmax(predictions, axis=1)
```

---

## 🛠️ Troubleshooting

### ❌ "ValueError: Cannot create a tensor proto whose content is larger than 2GB"
**Solution**: Use `.weights.h5` format instead of `.h5`
```python
model.save_weights('model_weights.weights.h5')  # ✅ Works
```

### ❌ "Cannot load model: incompatible architecture"
**Solution**: Define exact same architecture before loading weights
```python
model = create_model()  # Same architecture as training
model.load_weights('lstm_model_weights.weights.h5')
```

### ❌ "Shape mismatch error"
**Problem**: Input data shape doesn't match expected (104, 7)
```python
# ❌ Wrong
data = np.array([1,2,3,4,5,6,7])

# ✅ Correct
data = np.zeros((104, 7))  # 104 timesteps, 7 features
```

---

## 📊 Expected Results

Based on your original training:

```
Test Accuracy: 91.35%
Test Loss: 0.4684

Class Distribution:
- BUMP: 31 samples (7.2%)
- LEFT: 18 samples (4.2%)
- RIGHT: 23 samples (5.4%)
- STOP: 14 samples (3.3%)
- STRAIGHT: 342 samples (79.9%)  ← Majority class
```

**Note**: Model performs best on STRAIGHT (most common), may need fine-tuning for rare classes.

---

## 💡 Pro Tips

1. **Always save complete model** after training:
   ```python
   model.save('my_model_v1.h5')  # Saves everything
   ```

2. **Use lower learning rate** for fine-tuning:
   ```python
   from keras.optimizers import Adam
   model.compile(optimizer=Adam(0.0001), ...)  # 10x lower
   ```

3. **Monitor validation loss** to prevent overfitting:
   ```python
   from keras.callbacks import EarlyStopping
   early_stop = EarlyStopping(monitor='val_loss', patience=5)
   model.fit(..., callbacks=[early_stop])
   ```

4. **Test on real sensor data** before deploying:
   ```python
   # Collect 104 samples → predict → verify result
   ```

---

## 🎓 How This Helps Your ESW Project

### Use Cases:
- ✅ **Vehicle Behavior Classification**: Detect turns, bumps, stops in real-time
- ✅ **Driver Assistance System**: Alert driver based on predicted action
- ✅ **Autonomous Navigation**: Input to control system (e.g., slow down before bump)
- ✅ **Data Logging**: Record driving patterns with classifications

### Integration Example:
```python
# Pseudocode for embedded system
while True:
    sensor_reading = read_sensors()  # Get [Ax,Ay,Az,Gx,Gy,Gz,Speed]
    buffer.append(sensor_reading)
    
    if len(buffer) == 104:
        action = model.predict(buffer)
        
        if action == 'BUMP':
            reduce_speed()
        elif action == 'LEFT' or action == 'RIGHT':
            activate_turn_signal()
        
        buffer = buffer[1:]  # Sliding window
```

---

## 📚 Additional Resources

- **TensorFlow Lite**: Convert model for microcontrollers (ESP32)
  ```python
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()
  ```

- **Edge Deployment**: Consider Edge Impulse for embedded ML deployment

- **Data Collection**: Use MPU6050 sensor with Arduino/ESP32 for live data

---

## 📞 Need More Help?

Check out:
1. `MODEL_USAGE_GUIDE.md` - Full detailed tutorial
2. `load_and_test_model.py` - Working example script
3. Original notebook: `LSTM-ESW.ipynb`

**Key Concept**: Your model is a **pattern recognizer** trained on sensor data. It learned what sensor patterns look like for each action (BUMP, LEFT, etc.). Loading weights = transferring that learned knowledge without retraining! 🧠
