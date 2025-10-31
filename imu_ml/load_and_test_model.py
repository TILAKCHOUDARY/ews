"""
Script to Load Pre-trained LSTM Model and Make Predictions
Usage: python load_and_test_model.py
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# STEP 1: Define Model Architecture (Same as Training)
# =====================================================
def create_model():
    """
    Creates LSTM model with same architecture used during training
    """
    model = Sequential()
    
    # LSTM layer
    model.add(LSTM(units=100, input_shape=(104, 7)))
    
    # Dropout layer
    model.add(Dropout(0.5))
    
    # Dense layer with ReLU
    model.add(Dense(units=10, activation='relu'))
    
    # Output layer (5 classes)
    model.add(Dense(5, activation='softmax'))
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# =====================================================
# STEP 2: Load Pre-trained Weights
# =====================================================
print("🔧 Loading model...")
model = create_model()

try:
    model.load_weights('lstm_model_weights.weights.h5')
    print("✅ Model weights loaded successfully!")
    print(f"📊 Model has {model.count_params():,} trainable parameters")
    
    # Display model architecture
    print("\n📋 Model Architecture:")
    model.summary()
    
except Exception as e:
    print(f"❌ Error loading weights: {e}")
    print("Make sure 'lstm_model_weights.weights.h5' exists in this directory")
    exit(1)


# =====================================================
# STEP 3: Load Test Data (from your original notebook)
# =====================================================
print("\n🔍 Loading test data...")

try:
    # Load test data
    test = pd.read_csv('../data_test_24-29.csv')
    
    # Preprocess (same as training)
    test_data = test.drop(['TIME IN GMT','TIME IN IST','Lat ','Long', 'Time','T','Date'], axis=1)
    test_data.dropna(axis=0, how='any', inplace=True)
    
    print(f"✅ Test data loaded: {test_data.shape}")
    
except FileNotFoundError:
    print("⚠️  Test data file not found")
    print("Using demo mode (random data for testing)")
    # Create dummy data for demonstration
    X_test = np.random.rand(100, 104, 7).astype(np.float32)
    y_test = np.eye(5)[np.random.randint(0, 5, 100)].astype(np.float32)
    test_data_loaded = False
else:
    # Create segments for test data
    n_time_steps = 104
    n_features = 7
    step = 104
    
    segments_test = []
    labels_test = []
    
    print("📦 Creating test segments...")
    for i in range(0, test_data.shape[0] - n_time_steps, step):
        # Extract sensor data
        Ax_tr = test_data['Ax'].values[i: i + n_time_steps]
        Ay_tr = test_data['Ay'].values[i: i + n_time_steps]
        Az_tr = test_data['Az'].values[i: i + n_time_steps]
        Gx_tr = test_data['Gx'].values[i: i + n_time_steps]
        Gy_tr = test_data['Gy'].values[i: i + n_time_steps]
        Gz_tr = test_data['Gz'].values[i: i + n_time_steps]
        Speed_tr = test_data['Speed'].values[i: i + n_time_steps]
        
        # Get label
        from collections import Counter
        label_slice = test_data['Label'][i: i + n_time_steps]
        label_tr = Counter(label_slice).most_common(1)[0][0]
        
        segments_test.append([Ax_tr, Ay_tr, Az_tr, Gx_tr, Gy_tr, Gz_tr, Speed_tr])
        labels_test.append(label_tr)
    
    # Convert to numpy arrays
    X_test = np.asarray(segments_test, dtype=np.float32).reshape(-1, n_time_steps, n_features)
    y_test = np.asarray(pd.get_dummies(labels_test), dtype=np.float32)
    
    print(f"✅ Created {len(segments_test)} test segments")
    test_data_loaded = True


# =====================================================
# STEP 4: Evaluate Model on Test Data
# =====================================================
print("\n🧪 Evaluating model on test data...")

# Get predictions
loss, accuracy = model.evaluate(X_test, y_test, batch_size=16, verbose=0)

print(f"\n📊 Test Results:")
print(f"  Loss: {loss:.4f}")
print(f"  Accuracy: {accuracy*100:.2f}%")


# =====================================================
# STEP 5: Detailed Classification Report
# =====================================================
if test_data_loaded:
    print("\n📈 Generating detailed classification report...")
    
    # Get predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Class names
    class_names = ['BUMP', 'LEFT', 'RIGHT', 'STOP', 'STRAIGHT']
    
    # Classification report
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_test_classes, y_pred_classes, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    print("\n💾 Confusion matrix saved as 'confusion_matrix.png'")
    
    # Class distribution
    print("\n📊 Test Data Distribution:")
    unique, counts = np.unique(y_test_classes, return_counts=True)
    for label_idx, count in zip(unique, counts):
        print(f"  {class_names[label_idx]}: {count} samples ({count/len(y_test_classes)*100:.1f}%)")


# =====================================================
# STEP 6: Example Prediction Function
# =====================================================
def predict_single_sample(sensor_data):
    """
    Make prediction on a single sensor data window
    
    Args:
        sensor_data: numpy array of shape (104, 7)
        
    Returns:
        predicted_class: string (class name)
        confidence: float (0-1)
        all_probabilities: dict of all class probabilities
    """
    # Ensure correct shape
    if sensor_data.shape != (104, 7):
        raise ValueError(f"Expected shape (104, 7), got {sensor_data.shape}")
    
    # Reshape for model input
    data = sensor_data.reshape(1, 104, 7)
    
    # Make prediction
    prediction = model.predict(data, verbose=0)[0]
    predicted_idx = np.argmax(prediction)
    confidence = prediction[predicted_idx]
    
    class_names = ['BUMP', 'LEFT', 'RIGHT', 'STOP', 'STRAIGHT']
    
    # Create probability dict
    probabilities = {class_names[i]: float(prediction[i]) for i in range(5)}
    
    return class_names[predicted_idx], confidence, probabilities


# Demo prediction
print("\n🎯 Testing prediction function...")
sample_data = X_test[0]  # Get first test sample
predicted_class, confidence, probs = predict_single_sample(sample_data)

print(f"\nPrediction: {predicted_class} (Confidence: {confidence*100:.2f}%)")
print("\nAll class probabilities:")
for class_name, prob in probs.items():
    print(f"  {class_name}: {prob*100:.2f}%")


# =====================================================
# STEP 7: Real-time Prediction Template
# =====================================================
print("\n" + "="*50)
print("REAL-TIME PREDICTION EXAMPLE")
print("="*50)
print("""
To use this model in real-time (e.g., with live sensor data):

# Collect 104 timesteps of sensor data
sensor_buffer = []  # List to store sensor readings

while True:
    # Read sensor data (Ax, Ay, Az, Gx, Gy, Gz, Speed)
    sensor_reading = [ax, ay, az, gx, gy, gz, speed]
    sensor_buffer.append(sensor_reading)
    
    # When buffer reaches 104 samples
    if len(sensor_buffer) == 104:
        # Convert to numpy array
        data = np.array(sensor_buffer)
        
        # Make prediction
        action, confidence, probs = predict_single_sample(data)
        
        print(f"Predicted Action: {action} ({confidence*100:.1f}%)")
        
        # Slide window (remove oldest, keep last 103)
        sensor_buffer = sensor_buffer[1:]
""")

print("\n✅ Script completed successfully!")
print("\n💡 Next steps:")
print("  1. Review the confusion matrix: confusion_matrix.png")
print("  2. Use predict_single_sample() for new predictions")
print("  3. See MODEL_USAGE_GUIDE.md for fine-tuning instructions")
