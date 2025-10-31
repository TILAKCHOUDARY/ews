# Complete Guide: Loading Model Weights & Fine-Tuning Your LSTM Model

## Table of Contents
1. [Understanding Your Current Model](#understanding-your-current-model)
2. [How to Load Model Weights](#how-to-load-model-weights)
3. [Fine-Tuning the Model with New Data](#fine-tuning-the-model)
4. [How This Helps Your Project](#how-this-helps-your-project)
5. [Step-by-Step Implementation](#step-by-step-implementation)

---

## 1. Understanding Your Current Model

### What You Have
You're working with an **LSTM (Long Short-Term Memory)** neural network for **time-series sensor data classification**. Based on your notebook:

- **Input Data**: Sensor readings (Accelerometer: Ax, Ay, Az; Gyroscope: Gx, Gy, Gz; Speed)
- **Labels**: 5 classes (BUMP, LEFT, RIGHT, STOP, STRAIGHT)
- **Model Architecture**:
  - LSTM layer (100 units)
  - Dropout (0.5)
  - Dense layer (10 units, ReLU)
  - Output Dense layer (5 units, Softmax)
- **Current Performance**: 91.4% test accuracy

### Saved Model Weight File
You have: `lstm_model_weights.weights.h5` (298KB)

---

## 2. How to Load Model Weights

### Method 1: Load into Same Architecture (Recommended)

```python
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Define the exact same architecture
def create_model():
    model = Sequential()
    model.add(LSTM(units=100, input_shape=(104, 7)))
    model.add(Dropout(0.5))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Create model and load weights
model = create_model()
model.load_weights('lstm_model_weights.weights.h5')

print("✅ Model weights loaded successfully!")
```

### Method 2: Save and Load Complete Model (Better Practice)

```python
# SAVE complete model (do this after training)
model.save('lstm_model_complete.h5')  # Saves architecture + weights + optimizer

# LOAD complete model (easier)
from keras.models import load_model
model = load_model('lstm_model_complete.h5')
```

### Method 3: Use SavedModel Format (TensorFlow Standard)

```python
# Save
model.save('saved_model/')  # Creates a directory

# Load
import tensorflow as tf
model = tf.keras.models.load_model('saved_model/')
```

---

## 3. Fine-Tuning the Model with New Data

### What is Fine-Tuning?
**Fine-tuning** means taking your pre-trained model and continuing to train it on **new data** without starting from scratch. This is useful when:
- You collect more data
- You want to adapt the model to slightly different conditions
- You want to improve performance on specific cases

### Step-by-Step Fine-Tuning Process

```python
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.model_selection import train_test_split

# ===========================
# STEP 1: Load Pre-trained Model
# ===========================
model = create_model()
model.load_weights('lstm_model_weights.weights.h5')
print("✅ Loaded pre-trained weights")

# ===========================
# STEP 2: Prepare New Data
# ===========================
# Load your new data
new_data = pd.read_csv('new_data_collected.csv')

# Preprocess exactly like your training data
new_data = new_data.drop(['TIME IN GMT','TIME IN IST','Lat ','Long', 'Time','T','Date'], axis=1)
new_data.dropna(axis=0, how='any', inplace=True)

# Create segments (same as original)
n_time_steps = 104
n_features = 7
step = 104

segments = []
labels = []

for i in range(0, new_data.shape[0] - n_time_steps, step):
    Ax_tr = new_data['Ax'].values[i: i + n_time_steps]
    Ay_tr = new_data['Ay'].values[i: i + n_time_steps]
    Az_tr = new_data['Az'].values[i: i + n_time_steps]
    Gx_tr = new_data['Gx'].values[i: i + n_time_steps]
    Gy_tr = new_data['Gy'].values[i: i + n_time_steps]
    Gz_tr = new_data['Gz'].values[i: i + n_time_steps]
    Speed_tr = new_data['Speed'].values[i: i + n_time_steps]
    
    # Get label
    from collections import Counter
    label_slice = new_data['Label'][i: i + n_time_steps]
    label_tr = Counter(label_slice).most_common(1)[0][0]
    
    segments.append([Ax_tr, Ay_tr, Az_tr, Gx_tr, Gy_tr, Gz_tr, Speed_tr])
    labels.append(label_tr)

# Convert to numpy arrays
X_new = np.asarray(segments, dtype=np.float32).reshape(-1, n_time_steps, n_features)
y_new = pd.get_dummies(labels).astype(np.float32)

print(f"New data shape: {X_new.shape}")

# ===========================
# STEP 3: Fine-Tune the Model
# ===========================
# Option A: Fine-tune entire model (all layers trainable)
history = model.fit(
    X_new, y_new,
    epochs=20,  # Fewer epochs than initial training
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# Option B: Freeze early layers, only train final layers
# (Useful when new data is similar but you want faster training)
for layer in model.layers[:-2]:  # Freeze all except last 2 layers
    layer.trainable = False

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_new, y_new,
    epochs=20,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# ===========================
# STEP 4: Save Fine-Tuned Model
# ===========================
model.save_weights('lstm_model_finetuned.weights.h5')
model.save('lstm_model_finetuned_complete.h5')

print("✅ Fine-tuning complete! Model saved.")
```

### Important Fine-Tuning Tips

1. **Use Lower Learning Rate**: Prevents destroying pre-trained weights
```python
from keras.optimizers import Adam
optimizer = Adam(learning_rate=0.0001)  # Lower than original 0.001
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
```

2. **Fewer Epochs**: You don't need to train as long
```python
epochs = 10-30  # Instead of 100
```

3. **Monitor Validation Loss**: Stop if model starts overfitting
```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_new, y_new,
    epochs=30,
    callbacks=[early_stop],
    validation_split=0.2
)
```

---

## 4. How This Helps Your Project

### Benefits of Loading Pre-trained Weights

✅ **Save Training Time**
- No need to retrain from scratch (100 epochs = ~5 minutes saved)
- Immediately start testing/deploying

✅ **Transfer Learning**
- Model already learned patterns from 178k training samples
- New data builds on existing knowledge

✅ **Consistent Results**
- Always start from same baseline
- Reproducible experiments

✅ **Deployment Ready**
- Load weights on edge device (ESP32, Raspberry Pi, etc.)
- Make real-time predictions

### Benefits of Fine-Tuning

✅ **Adapt to New Conditions**
- Different vehicle/driver behavior
- New road conditions
- Updated sensor calibration

✅ **Improve Accuracy**
- Fix misclassifications on specific events
- Handle edge cases better

✅ **Incremental Learning**
- Add new classes (e.g., POTHOLE, U-TURN)
- Update with fresh data regularly

---

## 5. Step-by-Step Implementation

### Complete Working Example

```python
# ====================================================
# COMPLETE WORKFLOW: Load, Test, Fine-Tune, Deploy
# ====================================================

import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------
# 1. MODEL DEFINITION
# ----------------------
def create_model():
    model = Sequential()
    model.add(LSTM(units=100, input_shape=(104, 7)))
    model.add(Dropout(0.5))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ----------------------
# 2. LOAD PRE-TRAINED WEIGHTS
# ----------------------
model = create_model()
model.load_weights('lstm_model_weights.weights.h5')
print("✅ Model loaded")

# ----------------------
# 3. TEST ON EXISTING TEST DATA
# ----------------------
# Assuming you have X_test, y_test from your notebook
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"📊 Test Accuracy: {test_acc*100:.2f}%")

# Get predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Classification report
labels = ['BUMP', 'LEFT', 'RIGHT', 'STOP', 'STRAIGHT']
print("\n" + classification_report(y_test_classes, y_pred_classes, target_names=labels))

# ----------------------
# 4. FINE-TUNE WITH NEW DATA
# ----------------------
# Load new data you collected
new_data = pd.read_csv('new_data.csv')

# [Process new_data exactly like training data - see Step 3 above]
# ... (data preprocessing code)

# Fine-tune
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_new, y_new,
    epochs=20,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# ----------------------
# 5. SAVE FINE-TUNED MODEL
# ----------------------
model.save('lstm_model_finetuned.h5')
print("✅ Fine-tuned model saved!")

# ----------------------
# 6. MAKE PREDICTIONS (DEPLOYMENT)
# ----------------------
def predict_action(sensor_data):
    """
    sensor_data: shape (104, 7) - one time window
    """
    # Reshape for model input
    data = sensor_data.reshape(1, 104, 7)
    
    # Predict
    prediction = model.predict(data, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    
    class_names = ['BUMP', 'LEFT', 'RIGHT', 'STOP', 'STRAIGHT']
    
    return class_names[predicted_class], confidence

# Example usage
# sample_data = ... # Get 104 timesteps of sensor data
# action, confidence = predict_action(sample_data)
# print(f"Predicted Action: {action} (Confidence: {confidence*100:.2f}%)")
```

---

## Bonus: Visualize Training Progress

```python
import matplotlib.pyplot as plt

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    
    # Loss
    ax2.plot(history.history['loss'], label='Training')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('fine_tuning_progress.png')
    plt.show()

# After fine-tuning
plot_training_history(history)
```

---

## Summary

### What You Learned
1. ✅ How to **load pre-trained weights** into your LSTM model
2. ✅ How to **fine-tune** the model with new data
3. ✅ Why this approach **saves time** and **improves performance**
4. ✅ How to **deploy** the model for real-time predictions

### Next Steps for Your Project
1. Collect more sensor data in different scenarios
2. Fine-tune model with new data
3. Test on edge device (ESP32/Raspberry Pi)
4. Integrate with your ESW project (e.g., autonomous vehicle control)

### Key Takeaways
- **Never retrain from scratch** if you have working weights!
- **Fine-tuning is faster** than full training (10-20 epochs vs 100)
- **Test thoroughly** after loading weights
- **Save different versions** of your model as you improve it

---

## Questions?

If you need help with:
- Loading weights on embedded devices
- Converting model to TensorFlow Lite (for ESP32)
- Handling real-time sensor streams
- Adding new classes

Let me know! 🚀
