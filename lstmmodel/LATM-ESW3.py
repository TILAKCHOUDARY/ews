# Converted from LSTM-ESW(3).ipynb
# Paste this file into VS Code and run: python LSTM-ESW3.py
# Notes:
# - Update the CSV paths if needed.
# - Requires: numpy, pandas, matplotlib, seaborn, keras, tensorflow, scikit-learn

import os
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mode
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, accuracy_score

warnings.filterwarnings("ignore")

# If you want to see TF logs you can comment out the next two lines.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ---------
# Config
# ---------
n_time_steps = 104
n_features = 7
step = 104
n_classes = 5
n_epochs = 100
batch_size = 16
learning_rate = 0.0025
l2_loss = 0.0015
PATIENCE = 10

# ---------
# Load data (update paths if necessary)
# ---------
# The notebook used relative paths ../data_train_24-29.csv and ../data_test_24-29.csv
# If those files are next to this script, change paths accordingly.
train_csv = "../data_train_24-29.csv"
test_csv = "../data_test_24-29.csv"

if not os.path.exists(train_csv) or not os.path.exists(test_csv):
    # try current directory - helpful if you place CSVs next to script
    train_csv = "data_train_24-29.csv"
    test_csv = "data_test_24-29.csv"

if not os.path.exists(train_csv) or not os.path.exists(test_csv):
    raise FileNotFoundError(
        f"Train or test CSV not found. Tried: ../data_train_24-29.csv, ../data_test_24-29.csv, "
        f"data_train_24-29.csv, data_test_24-29.csv. Please adjust file paths in the script."
    )

print("Loading:", train_csv, test_csv)
test = pd.read_csv(test_csv)
train = pd.read_csv(train_csv)

# ---------
# Preprocess: drop unused columns and NaNs
# ---------
drop_cols = ["TIME IN GMT", "TIME IN IST", "Lat ", "Long", "Time", "T", "Date"]
# Some CSVs might not have all of these columns; only drop those present
drop_cols = [c for c in drop_cols if c in train.columns]
train_data = train.drop(drop_cols, axis=1)
drop_cols_test = [c for c in drop_cols if c in test.columns]
test_data = test.drop(drop_cols_test, axis=1)

train_data.dropna(axis=0, how="any", inplace=True)
test_data.dropna(axis=0, how="any", inplace=True)

print("Train shape after dropna:", train_data.shape)
print("Test shape after dropna :", test_data.shape)

# ---------
# Segment creation
# ---------
segments = []
labels = []
segments_test = []
labels_test = []

# Training segments
for i in range(0, train_data.shape[0] - n_time_steps, step):
    Ax_tr = train_data["Ax"].values[i : i + n_time_steps]
    Ay_tr = train_data["Ay"].values[i : i + n_time_steps]
    Az_tr = train_data["Az"].values[i : i + n_time_steps]
    Gx_tr = train_data["Gx"].values[i : i + n_time_steps]
    Gy_tr = train_data["Gy"].values[i : i + n_time_steps]
    Gz_tr = train_data["Gz"].values[i : i + n_time_steps]
    Speed_tr = train_data["Speed"].values[i : i + n_time_steps]

    label_slice = train_data["Label"][i : i + n_time_steps]
    try:
        label_tr = Counter(label_slice).most_common(1)[0][0]
    except Exception as e:
        print("Error at iteration:", i, "error:", e)
        continue

    segments.append([Ax_tr, Ay_tr, Az_tr, Gx_tr, Gy_tr, Gz_tr, Speed_tr])
    labels.append(label_tr)

unique_labels, label_counts = np.unique(labels, return_counts=True)
for label, count in zip(unique_labels, label_counts):
    print("Train label {}: {} cases".format(label, count))

# Test segments
for i in range(0, test_data.shape[0] - n_time_steps, step):
    Ax_tr = test_data["Ax"].values[i : i + n_time_steps]
    Ay_tr = test_data["Ay"].values[i : i + n_time_steps]
    Az_tr = test_data["Az"].values[i : i + n_time_steps]
    Gx_tr = test_data["Gx"].values[i : i + n_time_steps]
    Gy_tr = test_data["Gy"].values[i : i + n_time_steps]
    Gz_tr = test_data["Gz"].values[i : i + n_time_steps]
    Speed_tr = test_data["Speed"].values[i : i + n_time_steps]

    label_slice = test_data["Label"][i : i + n_time_steps]
    try:
        label_tr = Counter(label_slice).most_common(1)[0][0]
    except Exception as e:
        print("Error at iteration (test):", i, "error:", e)
        continue

    segments_test.append([Ax_tr, Ay_tr, Az_tr, Gx_tr, Gy_tr, Gz_tr, Speed_tr])
    labels_test.append(label_tr)

unique_test_labels, label_test_counts = np.unique(labels_test, return_counts=True)
for label, count in zip(unique_test_labels, label_test_counts):
    print("Test label {}: {} cases".format(label, count))

# ---------
# Convert to numpy arrays and one-hot encode labels
# ---------
X_train = np.asarray(segments, dtype=np.float32).reshape(-1, n_time_steps, n_features)
y_train = np.asarray(pd.get_dummies(labels), dtype=np.float32)
X_test = np.asarray(segments_test, dtype=np.float32).reshape(-1, n_time_steps, n_features)
y_test = np.asarray(pd.get_dummies(labels_test), dtype=np.float32)

print("X_train shape:", X_train.shape)
print("X_test  shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test  shape:", y_test.shape)

# ---------
# Build model
# ---------
model = Sequential()
model.add(LSTM(units=100, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.5))
model.add(Dense(units=10, activation="relu"))
model.add(Dense(y_train.shape[1], activation="softmax"))

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ---------
# Train
# ---------
history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, verbose=1)

# ---------
# Evaluate
# ---------
loss, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
print("Test Accuracy :", accuracy)
print("Test Loss :", loss)

# ---------
# Plot training history
# ---------
plt.figure()
plt.plot(np.array(history.history["loss"]), "r--", label="Train loss")
plt.plot(np.array(history.history["accuracy"]), "g--", label="Train accuracy")
plt.title("Training session's progress over iterations")
plt.legend(loc="lower left")
plt.ylabel("Training Progress (Loss/Accuracy)")
plt.xlabel("Training Epoch")
plt.ylim(0)
plt.show()

# ---------
# Predictions + Confusion matrix
# ---------
classes = pd.get_dummies(labels).columns.tolist()

y_pred = model.predict(X_test, batch_size=batch_size)
print("Shape of y_pred is", y_pred.shape)
y_pred_classes = np.argmax(y_pred, axis=1)
print("Shape of y_pred_classes is", y_pred_classes.shape)
y_test_categorical = np.argmax(y_test, axis=1)
print("Shape of y_test_encoded is", y_test.shape)
print("Shape of y_test_categorical is", y_test_categorical.shape)

conf_matrix = confusion_matrix(y_test_categorical, y_pred_classes)
# avoid division by zero
with np.errstate(all="ignore"):
    conf_matrix_normalized = conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
    conf_matrix_normalized = np.nan_to_num(conf_matrix_normalized)

plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix_normalized,
    annot=True,
    fmt=".3f",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes,
)
plt.title("Normalized Confusion Matrix for Test Dataset\nAccuracy: {:.2%}".format(accuracy))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

print(f"Accuracy:{accuracy*100:0.8f}%")

# ---------
# Save weights
# ---------
model.save_weights("lstm_model.weights.h5")
print("Saved weights to lstm_model.weights.h5")