import os, csv, time, serial, uuid
from datetime import datetime
from threading import Thread, Lock, Event
from queue import Queue
from smbus2 import SMBus
from supabase import create_client
import cv2
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput

# ========= Supabase Config =========
SUPABASE_URL = "https://ghtqafnlnijxvsmzdnmh.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdodHFhZm5sbmlqeHZzbXpkbm1oIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTk3NDQyNjcsImV4cCI6MjA3NTMyMDI2N30.Q1LGQP8JQdWn6rJJ1XRYT8rfo9b2Q5YfWUytrzQEsa0"

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ========= Hardware Config =========
BUS_NUM = 1
ADDR = 0x68
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43
GPS_PORT = "/dev/serial0"
GPS_BAUD = 9600
IMU_SAMPLE_RATE = 100  # 100 samples per second for IMU

# ========= Video Recording Config =========
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 30
VIDEO_BITRATE = 2000000  # 2 Mbps

# ========= Globals =========
running = False
rider_id = None
current_file_id = None
current_folder = None
video_filename = None
gps_lock = Lock()
latest_gps = {
    "utc": "N/A", "lat": "N/A", "ns": "N/A",
    "lon": "N/A", "ew": "N/A", "speed": "N/A",
    "course": "N/A", "date": "N/A", "valid": "N/A"
}

# Queue for Supabase uploads
supabase_queue = Queue(maxsize=1000)

# Thread references
gps_thread_obj = None
imu_thread_obj = None
supabase_thread_obj = None
video_thread_obj = None
stop_event = Event()  # Signal to stop threads

# ========= Helpers =========
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

def setup_data_folder():
    main_data_dir = "data"
    os.makedirs(main_data_dir, exist_ok=True)
    
    n = 1
    while os.path.exists(os.path.join(main_data_dir, f"ride{n:02d}")):
        n += 1
    
    session_folder = os.path.join(main_data_dir, f"ride{n:02d}")
    os.makedirs(session_folder, exist_ok=True)
    
    return session_folder

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

# ========= Video Recording Thread =========
def video_recording_thread():
    """Video recording thread using Picamera2"""
    global video_filename
    
    print("🎥 Video recording starting...")
    
    picam2 = None
    
    try:
        # Generate video filename with start timestamp
        start_time = datetime.now()
        video_filename = f"{start_time.strftime('%Y%m%d_%H%M%S')}.mp4"
        video_path = os.path.join(current_folder, video_filename)
        
        print(f"📹 Video will be saved as: {video_filename}")
        
        # Initialize camera
        picam2 = Picamera2()
        
        # Configure camera for video recording
        video_config = picam2.create_video_configuration(
            main={"size": (VIDEO_WIDTH, VIDEO_HEIGHT), "format": "RGB888"},
            controls={"FrameRate": VIDEO_FPS}
        )
        picam2.configure(video_config)
        
        # Create encoder and output
        encoder = H264Encoder(bitrate=VIDEO_BITRATE)
        output = FfmpegOutput(video_path)
        
        print("✅ Camera initialized successfully")
        print(f"📊 Recording at {VIDEO_WIDTH}x{VIDEO_HEIGHT} @ {VIDEO_FPS}fps")
        
        # Start recording
        picam2.start_recording(encoder, output)
        print("🔴 RECORDING STARTED")
        
        # Keep recording until stop signal
        while not stop_event.is_set():
            time.sleep(0.1)
        
        # Stop recording
        print("⏹️  Stopping video recording...")
        picam2.stop_recording()
        
        # Get final video file size
        if os.path.exists(video_path):
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            duration = (datetime.now() - start_time).total_seconds()
            print(f"✅ Video saved: {video_filename}")
            print(f"   Size: {file_size_mb:.2f} MB")
            print(f"   Duration: {duration:.1f} seconds")
            
            # Update Supabase with video info
            try:
                supabase.table("riderfiles").update({
                    "video_filename": video_filename,
                    "video_duration_sec": round(duration, 1),
                    "video_size_mb": round(file_size_mb, 2)
                }).eq("id", current_file_id).execute()
                print("☁️  Video info uploaded to Supabase")
            except Exception as e:
                print(f"Video info upload error: {e}")
        
    except Exception as e:
        print(f"❌ Video recording error: {e}")
    
    finally:
        # Cleanup
        if picam2:
            try:
                picam2.close()
                print("📷 Camera closed")
            except:
                pass
    
    print("🎥 Video recording thread exiting...")

# ========= Supabase Upload Thread =========
def supabase_upload_thread():
    """Dedicated thread for uploading data to Supabase"""
    print("☁️  Supabase upload thread started...")
    
    while not stop_event.is_set() or not supabase_queue.empty():
        try:
            # Get data from queue with timeout
            data = supabase_queue.get(timeout=1)
            
            # Upload to Supabase
            supabase.table("riderdata").insert(data).execute()
            
            supabase_queue.task_done()
            
        except Exception as e:
            if not stop_event.is_set():
                pass  # Silently handle errors to not slow down
    
    print("☁️  Supabase upload thread exiting...")

# ========= Command Listener Thread =========
def command_listener():
    """Listen for START/STOP commands from Supabase"""
    global running, current_file_id, rider_id, current_folder, video_filename
    global gps_thread_obj, imu_thread_obj, supabase_thread_obj, video_thread_obj, stop_event

    last_command_id = None
    print("👂 Command listener started - waiting for commands...")

    while True:
        try:
            # Fetch latest pending command (from any rider)
            result = supabase.table("rider_commands")\
                .select("*")\
                .eq("status", "pending")\
                .order("timestamp", desc=True)\
                .limit(1)\
                .execute()

            if result.data:
                latest = result.data[0]
                
                # Check if this is a new command
                if latest["id"] != last_command_id:
                    last_command_id = latest["id"]
                    command = latest["command"].lower()
                    
                    # Get rider_id from the command itself
                    rider_id = latest["rider_id"]

                    if command == "start" and not running:
                        running = True
                        stop_event.clear()  # Reset stop signal
                        
                        # Clear queue
                        while not supabase_queue.empty():
                            try:
                                supabase_queue.get_nowait()
                            except:
                                break
                        
                        # Create new data folder
                        current_folder = setup_data_folder()
                        
                        # Create riderfiles entry
                        file_res = supabase.table("riderfiles").insert({
                            "rider_id": rider_id,
                            "filename": f"ride_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "filepath": current_folder
                        }).execute()
                        current_file_id = file_res.data[0]["id"]

                        print(f"\n{'='*60}")
                        print(f"🚀 START command received!")
                        print(f"👤 Rider ID: {rider_id}")
                        print(f"📄 File ID: {current_file_id}")
                        print(f"📁 Data folder: {current_folder}")
                        print(f"{'='*60}\n")

                        # START GPS THREAD
                        gps_thread_obj = Thread(target=gps_thread, daemon=False)
                        gps_thread_obj.start()
                        print("📡 GPS thread STARTED")

                        # START SUPABASE UPLOAD THREAD
                        supabase_thread_obj = Thread(target=supabase_upload_thread, daemon=False)
                        supabase_thread_obj.start()
                        print("☁️  Supabase upload thread STARTED")

                        # START VIDEO RECORDING THREAD
                        video_thread_obj = Thread(target=video_recording_thread, daemon=False)
                        video_thread_obj.start()
                        print("🎥 Video recording thread STARTED")

                        # START IMU THREAD (100 Hz)
                        imu_thread_obj = Thread(target=imu_thread, daemon=False)
                        imu_thread_obj.start()
                        print("📊 IMU thread STARTED (100 Hz)")

                        # Mark command as executed
                        supabase.table("rider_commands")\
                            .update({"status": "executed"})\
                            .eq("id", latest["id"])\
                            .execute()

                    elif command == "stop" and running:
                        running = False
                        stop_event.set()  # Signal threads to stop
                        
                        print(f"\n{'='*60}")
                        print(f"🛑 STOP command received!")
                        print(f"👤 Rider ID: {rider_id}")
                        print(f"⏹️  Stopping threads...")
                        print(f"{'='*60}\n")
                        
                        # Wait for threads to finish
                        if gps_thread_obj and gps_thread_obj.is_alive():
                            gps_thread_obj.join(timeout=5)
                            print("✅ GPS thread STOPPED")
                        
                        if imu_thread_obj and imu_thread_obj.is_alive():
                            imu_thread_obj.join(timeout=5)
                            print("✅ IMU thread STOPPED")
                        
                        if video_thread_obj and video_thread_obj.is_alive():
                            video_thread_obj.join(timeout=10)
                            print("✅ Video recording thread STOPPED")
                        
                        if supabase_thread_obj and supabase_thread_obj.is_alive():
                            supabase_thread_obj.join(timeout=10)
                            print("✅ Supabase upload thread STOPPED")
                        
                        print(f"💾 Data saved in: {current_folder}")
                        print(f"🎬 Video saved as: {video_filename}\n")
                        
                        # Mark command as executed
                        supabase.table("rider_commands")\
                            .update({"status": "executed"})\
                            .eq("id", latest["id"])\
                            .execute()
                        
                        current_file_id = None
                        current_folder = None
                        video_filename = None
                        gps_thread_obj = None
                        imu_thread_obj = None
                        supabase_thread_obj = None
                        video_thread_obj = None
                        
                        print("🔄 System ready for next ride\n")

        except Exception as e:
            print(f"❌ Command listener error: {e}")
        
        time.sleep(1)  # Check for commands every second

# ========= GPS Thread =========
def gps_thread():
    """GPS data collection thread - stops when stop_event is set"""
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
    """IMU data collection thread - 100 Hz sampling - stops when stop_event is set"""
    print("📊 IMU initialization starting...")

    # Setup IMU
    bus = None
    try:
        bus = SMBus(BUS_NUM)
        bus.write_byte_data(ADDR, PWR_MGMT_1, 0)
        time.sleep(0.1)
        print("✅ IMU initialized successfully (100 Hz sampling)")
    except Exception as e:
        print(f"❌ IMU initialization error: {e}")
        return

    csv_file = None
    csv_writer = None
    sample_count = 0

    try:
        # Open CSV file
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

        sample_interval = 1.0 / IMU_SAMPLE_RATE  # 0.01 seconds = 10ms
        next_sample_time = time.time()
        
        while not stop_event.is_set():
            current_time = time.time()
            
            # Only sample if it's time
            if current_time >= next_sample_time:
                # Get synchronized timestamp
                now = datetime.now()
                timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
                epoch_time = current_time

                # Read IMU data
                ax = read_word(bus, ADDR, ACCEL_XOUT_H)
                ay = read_word(bus, ADDR, ACCEL_XOUT_H+2)
                az = read_word(bus, ADDR, ACCEL_XOUT_H+4)
                gx = read_word(bus, ADDR, GYRO_XOUT_H)
                gy = read_word(bus, ADDR, GYRO_XOUT_H+2)
                gz = read_word(bus, ADDR, GYRO_XOUT_H+4)

                # Convert to physical units
                ax_g = ax / 16384.0
                ay_g = ay / 16384.0
                az_g = az / 16384.0
                gx_dps = gx / 131.0
                gy_dps = gy / 131.0
                gz_dps = gz / 131.0

                # Get latest GPS data safely
                with gps_lock:
                    gps_copy = latest_gps.copy()

                # Write to CSV backup
                csv_writer.writerow([
                    timestamp_str, epoch_time,
                    round(ax_g, 4), round(ay_g, 4), round(az_g, 4),
                    round(gx_dps, 3), round(gy_dps, 3), round(gz_dps, 3),
                    gps_copy["utc"], gps_copy["lat"], gps_copy["ns"],
                    gps_copy["lon"], gps_copy["ew"],
                    gps_copy["speed"], gps_copy["course"], gps_copy["valid"]
                ])
                
                # Flush every 100 samples
                sample_count += 1
                if sample_count % 100 == 0:
                    csv_file.flush()

                # Prepare Supabase data
                speed_val = None
                course_val = None
                
                if gps_copy["speed"] != "N/A":
                    try:
                        speed_val = float(gps_copy["speed"])
                    except:
                        pass
                
                if gps_copy["course"] != "N/A":
                    try:
                        course_val = float(gps_copy["course"])
                    except:
                        pass

                # Add to upload queue (non-blocking)
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
                    pass  # Queue full, skip this sample

                # Console output (every 10th sample to avoid flooding)
                if sample_count % 10 == 0:
                    print(f"✅ [{timestamp_str}] Data pushed to Supabase")
                    print(f"📊 IMU -> ACC: {ax_g:.3f}, {ay_g:.3f}, {az_g:.3f} | "
                          f"GYRO: {gx_dps:.3f}, {gy_dps:.3f}, {gz_dps:.3f}")
                    print(f"   GPS -> Lat:{gps_copy['lat']}{gps_copy['ns']} | "
                          f"Lon:{gps_copy['lon']}{gps_copy['ew']} | Speed:{gps_copy['speed']}kn | Valid:{gps_copy['valid']}")
                    print("-" * 100)

                # Calculate next sample time
                next_sample_time += sample_interval
            else:
                # Sleep for a very short time to avoid busy waiting
                time.sleep(0.0001)

    except Exception as e:
        if not stop_event.is_set():
            print(f"❌ Main loop error: {e}")
    finally:
        # Cleanup
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

# ========= Main =========
if __name__ == "__main__":
    print("=" * 60)
    print("   IMU (100 Hz) + GPS + VIDEO Recording Logger")
    print("=" * 60)
    print(f"☁️  Supabase: Real-time updates enabled")
    print(f"📊 IMU Sampling Rate: {IMU_SAMPLE_RATE} Hz")
    print(f"🎥 Video Recording: {VIDEO_WIDTH}x{VIDEO_HEIGHT} @ {VIDEO_FPS}fps")
    print(f"🎧 Listening for commands from ANY rider")
    print("-" * 60)
    
    try:
        # Only start command listener thread
        t_cmd = Thread(target=command_listener, daemon=True)
        print("🚀 Starting command listener...")
        t_cmd.start()
        
        print("\n⏳ WAITING FOR START COMMAND FROM WEBSITE...")
        print("   GPS, IMU, and Video Recording will start when START is received")
        print("=" * 60)
        
        # Keep main thread alive
        while True:
            time.sleep(1)
        
    except KeyboardInterrupt:
        print("\n⛔ Received stop signal...")
        stop_event.set()  # Signal all threads to stop
        print("👋 Exiting program...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        print("✅ Program ended.")