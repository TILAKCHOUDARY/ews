import os, csv, time, serial, uuid
from datetime import datetime
from threading import Thread, Lock, Event
from queue import Queue
from smbus2 import SMBus
from supabase import create_client
from collections import deque

# Event detection imports
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Pothole detection imports
from torchvision import models, transforms
from PIL import Image
import cv2
from picamera2 import Picamera2

# ========= Supabase Config =========
SUPABASE_URL = "https://ghtqafnlnijxvsmzdnmh.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdodHFhZm5sbmlqeHZzbXpkbm1oIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTk3NDQyNjcsImV4cCI6MjA3NTMyMDI2N30.Q1LGQP8JQdWn6rJJ1XRYT8rfo9b2Q5YfWUytrzQEsa0"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ========= Hardware Config =========
BUS_NUM = 1
ADDR = 0x68
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43
GPS_PORT = "/dev/serial0"
GPS_BAUD = 9600
IMU_SAMPLE_RATE = 104

# ========= Event Detection Config =========
EVENT_MODEL_PATH = "imu_classifier_model.pth"
EVENT_CLASSES = ['STRAIGHT', 'BUMP', 'LEFT', 'RIGHT', 'STOP']
WINDOW_SIZE = 50  # Collect 50 samples before prediction
MIN_EVENT_DURATION = 0.5  # Minimum 0.5 seconds for event to be logged
CONFIDENCE_THRESHOLD = 0.6  # 60% confidence threshold

# ========= Neural Network Definition (matching training architecture) =========
class IMUClassifier(nn.Module):
    def __init__(self, input_size=7, hidden_sizes=[512, 512, 256, 256, 128], num_classes=5, dropout=0.4):
        super(IMUClassifier, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 64))
        layers.append(nn.BatchNorm1d(64))
        layers.append(nn.LeakyReLU(0.1))
        layers.append(nn.Dropout(dropout * 0.5))
        layers.append(nn.Linear(64, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# ========= Globals =========
running = False
rider_id = None
current_file_id = None
current_folder = None
pothole_images_folder = None
pothole_log_file = None
pothole_log_writer = None
events_log_file = None
events_log_writer = None

gps_lock = Lock()
latest_gps = {
    "utc": "N/A", "lat": "N/A", "ns": "N/A",
    "lon": "N/A", "ew": "N/A", "speed": "N/A",
    "course": "N/A", "date": "N/A", "valid": "N/A"
}

supabase_queue = Queue(maxsize=1000)
imu_data_queue = Queue(maxsize=500)  # For event detection
gps_thread_obj = None
imu_thread_obj = None
supabase_thread_obj = None
pothole_thread_obj = None
event_thread_obj = None
stop_event = Event()

# Event tracking
event_lock = Lock()
current_event = None
current_event_start = None
current_event_start_str = None
current_event_confidences = []

# Event detection model globals
event_model = None
event_scaler = None
label_encoder = None

# Pothole model config
POTHOLE_MODEL_PATH = "pothole_model.pth"
classes_pothole = ['Plain', 'Pothole']
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
show_preview = True

# ---------- Helper: IMU read ----------
def read_word(bus, addr, reg):
    try:
        high = bus.read_byte_data(addr, reg)
        low  = bus.read_byte_data(addr, reg+1)
        val = (high << 8) | low
        if val & 0x8000:
            val -= 0x10000
        return val
    except Exception as e:
        print(f"IMU read error: {e}")
        return 0

# ---------- Helper: setup folder ----------
def setup_data_folder():
    global pothole_images_folder, pothole_log_file, pothole_log_writer
    global events_log_file, events_log_writer
    
    main_data_dir = "data"
    os.makedirs(main_data_dir, exist_ok=True)
    n = 1
    while os.path.exists(os.path.join(main_data_dir, f"ride{n:02d}")):
        n += 1
    session_folder = os.path.join(main_data_dir, f"ride{n:02d}")
    os.makedirs(session_folder, exist_ok=True)
    
    # Create pothole_image subfolder
    pothole_images_folder = os.path.join(session_folder, "pothole_image")
    os.makedirs(pothole_images_folder, exist_ok=True)
    
    # Create pothole_log.csv
    pothole_log_path = os.path.join(session_folder, "pothole_log.csv")
    pothole_log_file = open(pothole_log_path, "w", newline="")
    pothole_log_writer = csv.writer(pothole_log_file)
    pothole_log_writer.writerow(["timestamp", "epoch_time", "image_filename", "confidence_percent"])
    print(f"📋 Pothole log created: {pothole_log_path}")
    
    # Create events.csv with new format
    events_log_path = os.path.join(session_folder, "events.csv")
    events_log_file = open(events_log_path, "w", newline="")
    events_log_writer = csv.writer(events_log_file)
    events_log_writer.writerow(["start_timestamp", "end_timestamp", "start_epoch", "end_epoch", "duration_sec", "event_type", "avg_confidence_percent"])
    events_log_file.flush()
    print(f"📋 Events log created: {events_log_path}")
    
    return session_folder

# ---------- Helper: parse GPS ----------
def parse_gprmc(line):
    try:
        parts = line.split(",")
        if len(parts) >= 10:
            return {
                "utc": parts[1] or "N/A",
                "valid": parts[2] or "N/A",
                "lat": parts[3] or "N/A",
                "ns": parts[4] or "N/A",
                "lon": parts[5] or "N/A",
                "ew": parts[6] or "N/A",
                "speed": parts[7] or "N/A",
                "course": parts[8] or "N/A",
                "date": parts[9] or "N/A"
            }
    except:
        pass
    return None

# ---------- Load Event Detection Model ----------
def load_event_model():
    global event_model, event_scaler, label_encoder
    
    try:
        print("🧠 Loading event detection model...")
        checkpoint = torch.load(EVENT_MODEL_PATH, map_location='cpu')
        
        # Load scaler and label encoder
        event_scaler = checkpoint['scaler']
        label_encoder = checkpoint['label_encoder']
        
        # Create model architecture
        num_classes = len(label_encoder.classes_)
        event_model = IMUClassifier(input_size=7, num_classes=num_classes)
        event_model.load_state_dict(checkpoint['model_state_dict'])
        event_model.eval()
        
        print(f"✅ Event model loaded successfully!")
        print(f"   Classes: {list(label_encoder.classes_)}")
        print(f"   Test accuracy: {checkpoint.get('test_accuracy', 'N/A')}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load event model: {e}")
        print(f"   Make sure '{EVENT_MODEL_PATH}' exists in the same directory")
        return False

# ========= Supabase Upload Thread =========
def supabase_upload_thread():
    print("☁  Supabase upload thread started...")
    while not stop_event.is_set() or not supabase_queue.empty():
        try:
            data = supabase_queue.get(timeout=1)
            try:
                supabase.table("riderdata").insert(data).execute()
            except Exception as e:
                pass
            supabase_queue.task_done()
        except Exception:
            pass
    print("☁  Supabase upload thread exiting...")

# ========= GPS Thread =========
def gps_thread():
    print("📡 GPS collection starting...")
    try:
        with serial.Serial(GPS_PORT, GPS_BAUD, timeout=1) as ser:
            while not stop_event.is_set():
                try:
                    line = ser.readline().decode("ascii", errors="ignore").strip()
                    if line.startswith("$GPRMC"):
                        gps_data = parse_gprmc(line)
                        if gps_data:
                            with gps_lock:
                                latest_gps.update(gps_data)
                except Exception as e:
                    if not stop_event.is_set():
                        print(f"GPS error: {e}")
                    time.sleep(1)
    except Exception as e:
        print(f"GPS serial error: {e}")
    print("📡 GPS thread exiting...")

# ========= IMU Thread =========
def imu_thread():
    print("📊 IMU initialization starting...")
    bus = None
    try:
        bus = SMBus(BUS_NUM)
        bus.write_byte_data(ADDR, PWR_MGMT_1, 0)
        time.sleep(0.1)
        print("✅ IMU initialized successfully (104 Hz sampling)")
    except Exception as e:
        print(f"❌ IMU initialization error: {e}")
        return

    csv_file = None
    csv_writer = None
    sample_count = 0

    try:
        csv_path = os.path.join(current_folder, "sensor_data.csv")
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "timestamp", "epoch_time",
            "ax_g", "ay_g", "az_g",
            "gx_dps", "gy_dps", "gz_dps",
            "gps_utc", "gps_lat", "gps_ns", "gps_lon", "gps_ew",
            "gps_speed_kn", "gps_course_deg", "gps_valid"
        ])
        print(f"📝 CSV file created: {csv_path}")

        sample_interval = 1.0 / IMU_SAMPLE_RATE
        next_sample_time = time.time()

        while not stop_event.is_set():
            current_time = time.time()
            if current_time >= next_sample_time:
                now = datetime.now()
                timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
                epoch_time = current_time

                ax = read_word(bus, ADDR, ACCEL_XOUT_H)
                ay = read_word(bus, ADDR, ACCEL_XOUT_H+2)
                az = read_word(bus, ADDR, ACCEL_XOUT_H+4)
                gx = read_word(bus, ADDR, GYRO_XOUT_H)
                gy = read_word(bus, ADDR, GYRO_XOUT_H+2)
                gz = read_word(bus, ADDR, GYRO_XOUT_H+4)

                ax_g = ax / 16384.0
                ay_g = ay / 16384.0
                az_g = az / 16384.0
                gx_dps = gx / 131.0
                gy_dps = gy / 131.0
                gz_dps = gz / 131.0

                with gps_lock:
                    gps_copy = latest_gps.copy()

                csv_writer.writerow([
                    timestamp_str, epoch_time,
                    round(ax_g, 4), round(ay_g, 4), round(az_g, 4),
                    round(gx_dps, 3), round(gy_dps, 3), round(gz_dps, 3),
                    gps_copy["utc"], gps_copy["lat"], gps_copy["ns"],
                    gps_copy["lon"], gps_copy["ew"],
                    gps_copy["speed"], gps_copy["course"], gps_copy["valid"]
                ])

                sample_count += 1
                if sample_count % 100 == 0:
                    csv_file.flush()

                # Get speed from GPS
                speed_val = 0.0
                if gps_copy["speed"] != "N/A":
                    try:
                        speed_val = float(gps_copy["speed"])
                    except:
                        pass

                # Put IMU data into queue for event detection
                try:
                    imu_data_queue.put_nowait({
                        'timestamp': timestamp_str,
                        'epoch': epoch_time,
                        'speed': speed_val,
                        'ax': ax_g,
                        'ay': ay_g,
                        'az': az_g,
                        'gx': gx_dps,
                        'gy': gy_dps,
                        'gz': gz_dps
                    })
                except:
                    pass

                # Supabase upload
                course_val = None
                if gps_copy["course"] != "N/A":
                    try:
                        course_val = float(gps_copy["course"])
                    except:
                        pass

                try:
                    supabase_queue.put_nowait({
                        "rider_id": rider_id,
                        "file_id": current_file_id,
                        "timestamp": now.isoformat(),
                        "ax": round(ax_g, 4),
                        "ay": round(ay_g, 4),
                        "az": round(az_g, 4),
                        "gx": round(gx_dps, 3),
                        "gy": round(gy_dps, 3),
                        "gz": round(gz_dps, 3),
                        "gps_utc": gps_copy["utc"],
                        "gps_lat": gps_copy["lat"],
                        "gps_ns": gps_copy["ns"],
                        "gps_lon": gps_copy["lon"],
                        "gps_ew": gps_copy["ew"],
                        "gps_speed_kn": speed_val,
                        "gps_course_deg": course_val,
                        "gps_valid": gps_copy["valid"]
                    })
                except:
                    pass

                if sample_count % 10 == 0:
                    print(f"✅ [{timestamp_str}] IMU sample logged")
                next_sample_time += sample_interval
            else:
                time.sleep(0.0001)

    except Exception as e:
        if not stop_event.is_set():
            print(f"❌ IMU main loop error: {e}")
    finally:
        try:
            if csv_file:
                csv_file.close()
                print("💾 CSV file closed")
        except Exception as e:
            print(f"CSV cleanup error: {e}")
        try:
            if bus:
                bus.close()
                print("🔌 IMU bus closed")
        except Exception as e:
            print(f"IMU cleanup error: {e}")
    print("📊 IMU thread exiting...")

# ========= Event Detection Thread =========
def event_detection_thread():
    global current_event, current_event_start, current_event_start_str, current_event_confidences
    global events_log_writer, events_log_file
    
    print("🧠 Event detection thread starting...")
    
    if event_model is None or event_scaler is None:
        print("❌ Event model not loaded. Cannot start event detection.")
        return
    
    window_buffer = deque(maxlen=WINDOW_SIZE)
    event_count = 0
    
    try:
        while not stop_event.is_set():
            try:
                data = imu_data_queue.get(timeout=0.5)
                
                # Add to window buffer
                window_buffer.append(data)
                
                # Wait until we have enough samples
                if len(window_buffer) < WINDOW_SIZE:
                    continue
                
                # Extract features from window (using latest sample as representative)
                latest = window_buffer[-1]
                features = np.array([[
                    latest['speed'],
                    latest['ax'],
                    latest['ay'],
                    latest['az'],
                    latest['gx'],
                    latest['gy'],
                    latest['gz']
                ]])
                
                # Scale and predict
                features_scaled = event_scaler.transform(features)
                features_tensor = torch.FloatTensor(features_scaled)
                
                with torch.no_grad():
                    output = event_model(features_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    confidence, predicted_class = torch.max(probabilities, 1)
                    
                    predicted_label = label_encoder.inverse_transform([predicted_class.item()])[0]
                    conf_percent = confidence.item() * 100
                
                # Event tracking logic
                with event_lock:
                    if conf_percent >= CONFIDENCE_THRESHOLD * 100:
                        # High confidence prediction
                        if current_event is None:
                            # Start new event
                            current_event = predicted_label
                            current_event_start = latest['epoch']
                            current_event_start_str = latest['timestamp']
                            current_event_confidences = [conf_percent]
                            print(f"🎯 Event started: {predicted_label} ({conf_percent:.1f}%)")
                            
                        elif current_event == predicted_label:
                            # Continue current event
                            current_event_confidences.append(conf_percent)
                            
                        else:
                            # Different event detected - save previous and start new
                            end_epoch = latest['epoch']
                            end_timestamp = latest['timestamp']
                            duration = end_epoch - current_event_start
                            
                            if duration >= MIN_EVENT_DURATION:
                                avg_conf = np.mean(current_event_confidences)
                                event_count += 1
                                
                                if events_log_writer:
                                    try:
                                        events_log_writer.writerow([
                                            current_event_start_str,
                                            end_timestamp,
                                            round(current_event_start, 3),
                                            round(end_epoch, 3),
                                            round(duration, 2),
                                            current_event,
                                            round(avg_conf, 2)
                                        ])
                                        events_log_file.flush()
                                        print(f"📝 Event #{event_count} logged: {current_event} ({duration:.2f}s, {avg_conf:.1f}%)")
                                    except Exception as e:
                                        print(f"❌ Error writing event: {e}")
                            
                            # Start new event
                            current_event = predicted_label
                            current_event_start = latest['epoch']
                            current_event_start_str = latest['timestamp']
                            current_event_confidences = [conf_percent]
                            print(f"🎯 Event started: {predicted_label} ({conf_percent:.1f}%)")
                    
                    else:
                        # Low confidence - end current event if exists
                        if current_event is not None:
                            end_epoch = latest['epoch']
                            end_timestamp = latest['timestamp']
                            duration = end_epoch - current_event_start
                            
                            if duration >= MIN_EVENT_DURATION:
                                avg_conf = np.mean(current_event_confidences)
                                event_count += 1
                                
                                if events_log_writer:
                                    try:
                                        events_log_writer.writerow([
                                            current_event_start_str,
                                            end_timestamp,
                                            round(current_event_start, 3),
                                            round(end_epoch, 3),
                                            round(duration, 2),
                                            current_event,
                                            round(avg_conf, 2)
                                        ])
                                        events_log_file.flush()
                                        print(f"📝 Event #{event_count} logged: {current_event} ({duration:.2f}s, {avg_conf:.1f}%)")
                                    except Exception as e:
                                        print(f"❌ Error writing event: {e}")
                            
                            current_event = None
                            current_event_start = None
                            current_event_start_str = None
                            current_event_confidences = []
                
            except Exception as e:
                if not stop_event.is_set():
                    time.sleep(0.01)
    
    except Exception as e:
        if not stop_event.is_set():
            print(f"❌ Event detection error: {e}")
    
    finally:
        # Save any ongoing event when thread stops
        with event_lock:
            if current_event is not None and events_log_writer:
                try:
                    end_epoch = time.time()
                    end_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
                    duration = end_epoch - current_event_start
                    
                    if duration >= MIN_EVENT_DURATION:
                        avg_conf = np.mean(current_event_confidences)
                        event_count += 1
                        
                        events_log_writer.writerow([
                            current_event_start_str,
                            end_timestamp,
                            round(current_event_start, 3),
                            round(end_epoch, 3),
                            round(duration, 2),
                            current_event,
                            round(avg_conf, 2)
                        ])
                        events_log_file.flush()
                        print(f"📝 Final event #{event_count} logged: {current_event} ({duration:.2f}s, {avg_conf:.1f}%)")
                except Exception as e:
                    print(f"❌ Error writing final event: {e}")
        
        print(f"🧠 Event detection thread exiting. Total events logged: {event_count}")

# ========= Pothole Detection Thread =========
def pothole_thread():
    global pothole_images_folder, rider_id, current_file_id, pothole_log_writer, pothole_log_file, current_folder

    print("📷 Pothole thread initializing...")

    # Load pothole model
    try:
        model_pothole = models.resnet18()
        num_features = model_pothole.fc.in_features
        model_pothole.fc = torch.nn.Linear(num_features, 2)
        model_pothole.load_state_dict(torch.load(POTHOLE_MODEL_PATH, map_location='cpu'))
        model_pothole.eval()
        print("✅ Pothole model loaded")
    except Exception as e:
        print(f"❌ Failed to load pothole model: {e}")
        return

    # Start camera
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(1)
        print("✅ Picamera2 started")
    except Exception as e:
        print(f"❌ Picamera2 start error: {e}")
        return

    # Setup video writer
    video_writer = None
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"ride_video_{timestamp}.mp4"
        video_path = os.path.join(current_folder, video_filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30.0
        frame_size = (640, 480)
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
        
        if video_writer.isOpened():
            print(f"🎥 Video recording started: {video_path}")
        else:
            print("❌ Failed to open video writer")
            video_writer = None
    except Exception as e:
        print(f"❌ Video writer setup error: {e}")
        video_writer = None

    frame_count = 0
    pothole_count = 0

    try:
        while not stop_event.is_set():
            frame = picam2.capture_array()
            if frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Write frame to video
            if video_writer and video_writer.isOpened():
                video_writer.write(frame_bgr)

            # Model inference
            img = Image.fromarray(frame)
            img_t = transform(img).unsqueeze(0)

            with torch.no_grad():
                output = model_pothole(img_t)
                probs = torch.nn.functional.softmax(output, dim=1)
                confidence, pred = torch.max(probs, 1)
                label = classes_pothole[pred.item()]
                conf_percent = confidence.item() * 100

            # Draw debug text
            color = (0, 255, 0) if label == "Plain" else (0, 0, 255)
            text = f"{label}: {conf_percent:.1f}%"
            cv2.putText(frame_bgr, text, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(frame_bgr, f"Frame: {frame_count}", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Pothole detected
            if label == "Pothole" and conf_percent > 70:
                pothole_count += 1

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                epoch_time = time.time()
                filename = f"pothole_{timestamp}.jpg"
                filepath = os.path.join(pothole_images_folder, filename)
                cv2.imwrite(filepath, frame_bgr)

                print(f"🕳 Pothole #{pothole_count} detected ({conf_percent:.1f}%)")

                if pothole_log_writer:
                    try:
                        pothole_log_writer.writerow([timestamp, epoch_time, filename, round(conf_percent, 2)])
                        pothole_log_file.flush()
                    except Exception as e:
                        print(f"❌ Error writing to pothole_log.csv: {e}")

                if rider_id and current_file_id:
                    try:
                        supabase.table("pothole_events").insert({
                            "rider_id": rider_id,
                            "file_id": current_file_id,
                            "detected_at": datetime.now().isoformat()
                        }).execute()
                    except Exception as e:
                        print(f"❌ Pothole upload error: {e}")

            # Show preview
            if show_preview:
                cv2.imshow("Pothole Detection (Press 'q' to stop)", frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("⌨ 'q' pressed → stopping")
                    stop_event.set()
                    break

            time.sleep(0.001)

    except Exception as e:
        if not stop_event.is_set():
            print("❌ Pothole thread error:", e)

    finally:
        try:
            picam2.stop()
        except:
            pass

        if video_writer:
            try:
                video_writer.release()
                print("🎥 Video recording saved")
            except:
                pass

        if show_preview:
            try:
                cv2.destroyAllWindows()
            except:
                pass

        print(f"📷 Pothole thread exiting. Frames={frame_count}, potholes={pothole_count}")

# ========= Command Listener Thread =========
def command_listener():
    global running, current_file_id, rider_id, current_folder
    global gps_thread_obj, imu_thread_obj, supabase_thread_obj, pothole_thread_obj, event_thread_obj, stop_event
    global pothole_log_file, pothole_log_writer, events_log_file, events_log_writer

    last_command_id = None
    print("👂 Command listener started - waiting for commands...")

    while True:
        try:
            result = supabase.table("rider_commands")\
                .select("*")\
                .eq("status", "pending")\
                .order("timestamp", desc=True)\
                .limit(1)\
                .execute()

            if result.data:
                latest = result.data[0]
                if latest["id"] != last_command_id:
                    last_command_id = latest["id"]
                    command = latest["command"].lower()
                    rider_id = latest["rider_id"]

                    if command == "start" and not running:
                        running = True
                        stop_event.clear()
                        while not supabase_queue.empty():
                            try:
                                supabase_queue.get_nowait()
                            except:
                                break
                        while not imu_data_queue.empty():
                            try:
                                imu_data_queue.get_nowait()
                            except:
                                break

                        current_folder = setup_data_folder()

                        file_res = supabase.table("riderfiles").insert({
                            "rider_id": rider_id,
                            "filename": f"ride_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "filepath": current_folder
                        }).execute()
                        try:
                            current_file_id = file_res.data[0]["id"]
                        except:
                            current_file_id = None

                        print(f"\n{'='*60}")
                        print(f"🚀 START command received!")
                        print(f"👤 Rider ID: {rider_id}")
                        print(f"📁 Data folder: {current_folder}")
                        print(f"{'='*60}\n")

                        # Start threads
                        gps_thread_obj = Thread(target=gps_thread, daemon=False)
                        gps_thread_obj.start()
                        print("📡 GPS thread STARTED")

                        supabase_thread_obj = Thread(target=supabase_upload_thread, daemon=False)
                        supabase_thread_obj.start()
                        print("☁  Supabase upload thread STARTED")

                        imu_thread_obj = Thread(target=imu_thread, daemon=False)
                        imu_thread_obj.start()
                        print("📊 IMU thread STARTED (104 Hz)")

                        # Start Event detection thread
                        event_thread_obj = Thread(target=event_detection_thread, daemon=False)
                        event_thread_obj.start()
                        print("🧠 Event detection thread STARTED")

                        # Start Pothole detection thread
                        pothole_thread_obj = Thread(target=pothole_thread, daemon=False)
                        pothole_thread_obj.start()
                        print("📷 Pothole detection thread STARTED")

                        # Mark command as executed
                        supabase.table("rider_commands")\
                            .update({"status": "executed"})\
                            .eq("id", latest["id"])\
                            .execute()

                    elif command == "stop" and running:
                        running = False
                        stop_event.set()

                        print(f"\n{'='*60}")
                        print(f"🛑 STOP command received!")
                        print(f"👤 Rider ID: {rider_id}")
                        print(f"ℹ  Stopping threads...")
                        print(f"{'='*60}\n")

                        # Wait for threads to finish
                        if gps_thread_obj and gps_thread_obj.is_alive():
                            gps_thread_obj.join(timeout=5)
                            print("✅ GPS thread STOPPED")
                        if imu_thread_obj and imu_thread_obj.is_alive():
                            imu_thread_obj.join(timeout=5)
                            print("✅ IMU thread STOPPED")
                        if event_thread_obj and event_thread_obj.is_alive():
                            event_thread_obj.join(timeout=5)
                            print("✅ Event detection thread STOPPED")
                        if supabase_thread_obj and supabase_thread_obj.is_alive():
                            supabase_thread_obj.join(timeout=10)
                            print("✅ Supabase upload thread STOPPED")
                        if pothole_thread_obj and pothole_thread_obj.is_alive():
                            pothole_thread_obj.join(timeout=10)
                            print("✅ Pothole detection thread STOPPED")

                        # Close CSV files
                        try:
                            if pothole_log_file:
                                pothole_log_file.close()
                                print("💾 Pothole log closed")
                        except:
                            pass
                        
                        try:
                            if events_log_file:
                                events_log_file.close()
                                print("💾 Events log closed")
                        except:
                            pass

                        print(f"💾 Data saved in: {current_folder}\n")

                        supabase.table("rider_commands")\
                            .update({"status": "executed"})\
                            .eq("id", latest["id"])\
                            .execute()

                        # Reset session variables
                        current_file_id = None
                        current_folder = None
                        gps_thread_obj = None
                        imu_thread_obj = None
                        supabase_thread_obj = None
                        pothole_thread_obj = None
                        event_thread_obj = None
                        pothole_log_file = None
                        pothole_log_writer = None
                        events_log_file = None
                        events_log_writer = None

                        print("🔄 System ready for next ride\n")

        except Exception as e:
            print(f"❌ Command listener error: {e}")

        time.sleep(1)

# ========= Main =========
if __name__ == "__main__":
    print("=" * 80)
    print("   IMU Event Detection + GPS + Pothole Detection System")
    print("=" * 80)
    print(f"☁  Supabase: Real-time updates enabled")
    print(f"📊 IMU Sampling Rate: {IMU_SAMPLE_RATE} Hz")
    print(f"🧠 Event Classes: {EVENT_CLASSES}")
    print(f"📏 Window Size: {WINDOW_SIZE} samples")
    print(f"⏱  Min Event Duration: {MIN_EVENT_DURATION} seconds")
    print(f"🎯 Confidence Threshold: {CONFIDENCE_THRESHOLD * 100}%")
    print("-" * 80)

    # Load event detection model
    if not load_event_model():
        print("\n❌ CRITICAL: Event model failed to load!")
        print("   Please ensure 'imu_classifier_model.pth' exists in the same directory.")
        print("   The system will continue but event detection will NOT work.")
        print("\n   To train the model, run the training notebook first.")
    
    print("\n" + "=" * 80)

    try:
        t_cmd = Thread(target=command_listener, daemon=True)
        print("🚀 Starting command listener...")
        t_cmd.start()

        print("\n⏳ WAITING FOR START COMMAND FROM WEBSITE...")
        print("   All features will start when START is received:")
        print("   • GPS tracking")
        print("   • IMU data logging (104 Hz)")
        print("   • Event detection (STRAIGHT, BUMP, LEFT, RIGHT, STOP)")
        print("   • Pothole detection with camera")
        print("=" * 80)

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n⛔ Received stop signal...")
        stop_event.set()
        print("👋 Exiting program...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        print("✅ Program ended.")