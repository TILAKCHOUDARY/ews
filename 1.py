import os, csv, time, serial, uuid, subprocess
from datetime import datetime
from threading import Thread, Lock, Event
from queue import Queue
from smbus2 import SMBus
from picamera2 import Picamera2
from supabase import create_client

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

# ========= Sampling Rates =========
IMU_RATE = 100      # 100 Hz - IMU samples per second
CAMERA_RATE = 10    # 10 Hz - Camera captures per second
GPS_RATE = 1        # 1 Hz - GPS update rate (typical)

# ========= Globals =========
running = False
rider_id = None
current_file_id = None
current_folder = None

# Locks for thread-safe access
gps_lock = Lock()
camera_lock = Lock()
latest_image_lock = Lock()

# Latest GPS data (updated ~1 Hz, used by all IMU samples)
latest_gps = {
    "utc": "N/A", "lat": "N/A", "ns": "N/A",
    "lon": "N/A", "ew": "N/A", "speed": "N/A",
    "course": "N/A", "date": "N/A", "valid": "N/A"
}

# Latest camera image filename (updated 10 Hz, used by all IMU samples)
latest_image_filename = "N/A"

# Upload queue for non-blocking Supabase uploads
upload_queue = Queue()

# Thread references
gps_thread_obj = None
imu_thread_obj = None
camera_thread_obj = None
upload_thread_obj = None
stop_event = Event()

# Camera object
picam2_global = None

# ========= Camera Reset Function =========
def reset_camera():
    """Reset camera by killing any existing processes"""
    try:
        print("🔄 Resetting camera processes...")
        subprocess.run(['sudo', 'killall', '-9', 'libcamera-hello'], 
                      capture_output=True, stderr=subprocess.DEVNULL)
        subprocess.run(['sudo', 'killall', '-9', 'libcamera-still'], 
                      capture_output=True, stderr=subprocess.DEVNULL)
        subprocess.run(['sudo', 'killall', '-9', 'libcamera-vid'], 
                      capture_output=True, stderr=subprocess.DEVNULL)
        subprocess.run(['sudo', 'pkill', '-9', '-f', 'libcamera'], 
                      capture_output=True, stderr=subprocess.DEVNULL)
        time.sleep(3)
        print("✅ Camera processes reset complete")
    except Exception as e:
        print(f"⚠️ Camera reset error (non-critical): {e}")

def cleanup_camera():
    """Properly cleanup camera object"""
    global picam2_global
    with camera_lock:
        if picam2_global is not None:
            try:
                print("🛑 Stopping camera...")
                picam2_global.stop()
                picam2_global.close()
                time.sleep(1)
                print("✅ Camera stopped and closed")
            except Exception as e:
                print(f"⚠️ Camera cleanup error: {e}")
            finally:
                picam2_global = None

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
    os.makedirs(os.path.join(session_folder, "images"), exist_ok=True)
    
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

# ========= Upload Thread (Non-blocking Supabase uploads) =========
def upload_thread():
    """Dedicated thread for Supabase uploads - doesn't block IMU sampling"""
    print("📤 Upload thread started...")
    
    while not stop_event.is_set():
        try:
            # Get data from queue (with timeout to check stop_event)
            try:
                data = upload_queue.get(timeout=1)
            except:
                continue
            
            # Upload to Supabase
            try:
                supabase.table("riderdata").insert(data).execute()
            except Exception as e:
                if not stop_event.is_set():
                    print(f"❌ Supabase upload error: {e}")
            finally:
                upload_queue.task_done()
                
        except Exception as e:
            if not stop_event.is_set():
                print(f"❌ Upload thread error: {e}")
    
    print("📤 Upload thread exiting...")

# ========= Command Listener Thread =========
def command_listener():
    """Listen for START/STOP commands from Supabase"""
    global running, current_file_id, rider_id, current_folder
    global gps_thread_obj, imu_thread_obj, camera_thread_obj, upload_thread_obj, stop_event
    global latest_image_filename

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
                        
                        # Reset latest image filename
                        with latest_image_lock:
                            latest_image_filename = "N/A"
                        
                        current_folder = setup_data_folder()
                        
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

                        # START UPLOAD THREAD (handles Supabase uploads without blocking)
                        upload_thread_obj = Thread(target=upload_thread, daemon=False)
                        upload_thread_obj.start()
                        print("📤 Upload thread STARTED (non-blocking uploads)")

                        # START GPS THREAD (1 Hz - updates latest_gps)
                        gps_thread_obj = Thread(target=gps_thread, daemon=False)
                        gps_thread_obj.start()
                        print("📡 GPS thread STARTED (1 Hz - background updates)")

                        # START CAMERA THREAD (10 Hz - updates latest_image_filename)
                        camera_thread_obj = Thread(target=camera_thread, daemon=False)
                        camera_thread_obj.start()
                        print("📷 Camera thread STARTED (10 Hz - background updates)")

                        # START IMU THREAD (100 Hz - uses latest GPS + latest image)
                        imu_thread_obj = Thread(target=imu_thread, daemon=False)
                        imu_thread_obj.start()
                        print("📊 IMU thread STARTED (100 Hz - uses latest GPS & image)")

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
                        print(f"ℹ️ Stopping threads...")
                        print(f"{'='*60}\n")
                        
                        # Wait for threads
                        if gps_thread_obj and gps_thread_obj.is_alive():
                            gps_thread_obj.join(timeout=5)
                            print("✅ GPS thread STOPPED")
                        
                        if camera_thread_obj and camera_thread_obj.is_alive():
                            camera_thread_obj.join(timeout=10)
                            print("✅ Camera thread STOPPED")
                        
                        if imu_thread_obj and imu_thread_obj.is_alive():
                            imu_thread_obj.join(timeout=5)
                            print("✅ IMU thread STOPPED")
                        
                        if upload_thread_obj and upload_thread_obj.is_alive():
                            # Wait for queue to be empty
                            upload_queue.join()
                            upload_thread_obj.join(timeout=10)
                            print("✅ Upload thread STOPPED")
                        
                        cleanup_camera()
                        reset_camera()
                        
                        print(f"💾 Data saved in: {current_folder}\n")
                        
                        supabase.table("rider_commands")\
                            .update({"status": "executed"})\
                            .eq("id", latest["id"])\
                            .execute()
                        
                        current_file_id = None
                        current_folder = None
                        gps_thread_obj = None
                        imu_thread_obj = None
                        camera_thread_obj = None
                        upload_thread_obj = None
                        
                        print("🔄 System ready for next ride\n")

        except Exception as e:
            print(f"❌ Command listener error: {e}")
        
        time.sleep(1)

# ========= GPS Thread (1 Hz - Background Updates) =========
def gps_thread():
    """GPS data collection - updates latest_gps in background"""
    print("📡 GPS collection starting (background updates)...")
    
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
                            print(f"🛰️ GPS updated: {gps_data['lat']}{gps_data['ns']} {gps_data['lon']}{gps_data['ew']} (used by all IMU samples)")
                except Exception as e:
                    if not stop_event.is_set():
                        print(f"GPS error: {e}")
                    time.sleep(1)
    except Exception as e:
        print(f"GPS serial error: {e}")
    
    print("📡 GPS thread exiting...")

# ========= Camera Thread (10 Hz - Background Updates) =========
def camera_thread():
    """Camera capture at 10 Hz - updates latest_image_filename in background"""
    global picam2_global, latest_image_filename
    
    print("📷 Camera initialization starting...")
    reset_camera()

    # Setup camera
    try:
        with camera_lock:
            print("📷 Creating Picamera2 instance...")
            picam2_global = Picamera2()
            
            print("📷 Configuring camera...")
            config = picam2_global.create_video_configuration(
                main={"size": (1920, 1080), "format": "RGB888"}
            )
            picam2_global.configure(config)
            
            print("📷 Starting camera...")
            picam2_global.start()
            
        time.sleep(3)
        print("✅ Camera initialized successfully")
        
    except Exception as e:
        print(f"❌ Camera initialization error: {e}")
        return

    img_count = 0
    interval = 1.0 / CAMERA_RATE  # 0.1 seconds (100ms)

    try:
        while not stop_event.is_set():
            loop_start = time.time()
            
            # Get timestamp
            now = datetime.now()
            timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]

            # Capture image
            img_count += 1
            filename = f"image_{timestamp_str}_{img_count:04d}.jpg"
            filepath = os.path.join(current_folder, "images", filename)
            
            try:
                with camera_lock:
                    if picam2_global is not None:
                        picam2_global.capture_file(filepath)
                
                # Update latest image filename - all IMU samples will use this
                with latest_image_lock:
                    latest_image_filename = f"images/{filename}"
                
                print(f"📷 Image captured: {filename} (now used by all IMU samples)")
            except Exception as e:
                if not stop_event.is_set():
                    print(f"⚠️ Camera capture error: {e}")

            # Maintain 10 Hz timing
            elapsed = time.time() - loop_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        if not stop_event.is_set():
            print(f"❌ Camera loop error: {e}")
    finally:
        with camera_lock:
            if picam2_global is not None:
                picam2_global.stop()
                print("📷 Camera stopped in thread")
    
    print("📷 Camera thread exiting...")

# ========= IMU Thread (100 Hz - Uses Latest GPS + Latest Image) =========
def imu_thread():
    """High-speed IMU data collection at 100 Hz - uses latest GPS and image"""
    print("📊 IMU collection starting at 100 Hz...")
    print("   🛰️ Using latest GPS data (updates ~1 Hz)")
    print("   📷 Using latest image filename (updates 10 Hz)")
    
    # Setup IMU
    bus = None
    try:
        bus = SMBus(BUS_NUM)
        bus.write_byte_data(ADDR, PWR_MGMT_1, 0)
        time.sleep(0.1)
        print("✅ IMU initialized successfully")
    except Exception as e:
        print(f"❌ IMU initialization error: {e}")
        return

    csv_file = None
    csv_writer = None
    sample_count = 0
    last_print_time = time.time()

    try:
        # Open CSV file for IMU data
        csv_path = os.path.join(current_folder, "combined_data.csv")
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "timestamp", "epoch_time", "sample_num", "image_filename",
            "ax_g", "ay_g", "az_g",
            "gx_dps", "gy_dps", "gz_dps",
            "gps_utc", "gps_lat", "gps_ns", "gps_lon", "gps_ew",
            "gps_speed_kn", "gps_course_deg", "gps_valid"
        ])
        print(f"📝 CSV file created: {csv_path}")

        interval = 1.0 / IMU_RATE  # 0.01 seconds (10ms)
        
        while not stop_event.is_set():
            loop_start = time.time()
            
            # Get timestamp
            now = datetime.now()
            timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
            epoch_time = time.time()
            sample_count += 1

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

            # Get LATEST GPS data (same value until GPS updates)
            with gps_lock:
                gps_copy = latest_gps.copy()

            # Get LATEST image filename (same value until camera captures new image)
            with latest_image_lock:
                current_image = latest_image_filename

            # Write to CSV - EVERY sample gets latest GPS + latest image
            csv_writer.writerow([
                timestamp_str, epoch_time, sample_count, current_image,
                round(ax_g, 4), round(ay_g, 4), round(az_g, 4),
                round(gx_dps, 3), round(gy_dps, 3), round(gz_dps, 3),
                gps_copy["utc"], gps_copy["lat"], gps_copy["ns"],
                gps_copy["lon"], gps_copy["ew"],
                gps_copy["speed"], gps_copy["course"], gps_copy["valid"]
            ])
            
            # Flush every 100 samples (1 second)
            if sample_count % 100 == 0:
                csv_file.flush()
                current_time = time.time()
                actual_rate = 100 / (current_time - last_print_time)
                print(f"📊 IMU: {sample_count} samples | Actual rate: {actual_rate:.1f} Hz | GPS: {gps_copy['lat']}{gps_copy['ns']} | Image: {current_image}")
                last_print_time = current_time

            # Queue data for Supabase upload (non-blocking)
            try:
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

                # Put data in queue - this is non-blocking
                upload_queue.put({
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
                    "gps_valid": gps_copy["valid"],
                    "image_filename": current_image
                })

            except Exception as e:
                if not stop_event.is_set():
                    print(f"❌ Queue error: {e}")

            # Maintain 100 Hz timing
            elapsed = time.time() - loop_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        if not stop_event.is_set():
            print(f"❌ IMU loop error: {e}")
    finally:
        if csv_file:
            csv_file.close()
            print("💾 IMU CSV file closed")
        if bus:
            bus.close()
            print("🔌 IMU bus closed")
    
    print("📊 IMU thread exiting...")

# ========= Main =========
if __name__ == "__main__":
    print("=" * 60)
    print("  SYNCHRONIZED: IMU (100Hz) + Camera (10Hz) + GPS (1Hz)")
    print("=" * 60)
    print(f"📊 IMU: 100 samples/sec → uses LATEST GPS + LATEST image")
    print(f"📷 Camera: 10 images/sec → updates image reference")
    print(f"🛰️ GPS: ~1 update/sec → updates position data")
    print(f"📤 Upload: Non-blocking queue → doesn't slow down IMU")
    print(f"")
    print(f"HOW IT WORKS:")
    print(f"  • GPS updates ~every 1 second (background)")
    print(f"  • Camera captures every 0.1 seconds (background)")
    print(f"  • IMU samples EVERY 0.01 seconds (100 Hz) using:")
    print(f"    - Latest available GPS position (same until GPS updates)")
    print(f"    - Latest captured image filename (same until camera captures)")
    print(f"  • Uploads happen in separate thread (no blocking)")
    print(f"")
    print(f"RESULT: 100 IMU samples per second, each with most recent")
    print(f"        GPS and image data available at that instant")
    print("-" * 60)
    
    reset_camera()
    
    try:
        t_cmd = Thread(target=command_listener, daemon=True)
        print("🚀 Starting command listener...")
        t_cmd.start()
        
        print("\n⏳ WAITING FOR START COMMAND FROM WEBSITE...")
        print("=" * 60)
        
        while True:
            time.sleep(1)
        
    except KeyboardInterrupt:
        print("\n⛔ Received stop signal...")
        stop_event.set()
        cleanup_camera()
        reset_camera()
        print("👋 Exiting program...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        cleanup_camera()
        print("✅ Program ended.")