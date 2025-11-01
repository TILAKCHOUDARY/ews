import os, csv, time, serial, subprocess
from datetime import datetime
from threading import Thread, Lock, Event
from smbus2 import SMBus
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
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
VIDEO_FPS = 30  # Increased for smoother video
IMU_RATE = 10

# ========= Globals =========
running = False
rider_id = None
current_file_id = None
current_folder = None
gps_lock = Lock()
latest_gps = {
    "utc": "N/A", "lat": "N/A", "ns": "N/A",
    "lon": "N/A", "ew": "N/A", "speed": "N/A",
    "course": "N/A", "date": "N/A", "valid": "N/A"
}

gps_thread_obj = None
cam_imu_thread_obj = None
stop_event = Event()

picam2_global = None
camera_lock = Lock()

# ========= Camera Management =========
def reset_camera():
    """Kill all camera processes"""
    try:
        print("🔄 Resetting camera...")
        subprocess.run(['sudo', 'killall', '-9', 'libcamera-hello'], stderr=subprocess.DEVNULL)
        subprocess.run(['sudo', 'killall', '-9', 'libcamera-still'], stderr=subprocess.DEVNULL)
        subprocess.run(['sudo', 'killall', '-9', 'libcamera-vid'], stderr=subprocess.DEVNULL)
        subprocess.run(['sudo', 'pkill', '-9', '-f', 'libcamera'], stderr=subprocess.DEVNULL)
        time.sleep(2)
        print("✅ Camera reset complete")
    except Exception as e:
        print(f"⚠️ Camera reset warning: {e}")

def cleanup_camera():
    """Properly cleanup camera"""
    global picam2_global
    with camera_lock:
        if picam2_global is not None:
            try:
                print("🛑 Stopping camera...")
                picam2_global.stop_recording()
                time.sleep(0.5)
                picam2_global.stop()
                picam2_global.close()
                time.sleep(0.5)
                print("✅ Camera closed")
            except Exception as e:
                print(f"⚠️ Camera cleanup: {e}")
            finally:
                picam2_global = None

# ========= Helpers =========
def read_word(bus, addr, reg):
    try:
        high = bus.read_byte_data(addr, reg)
        low = bus.read_byte_data(addr, reg+1)
        val = (high << 8) | low
        if val & 0x8000:
            val -= 0x10000
        return val
    except:
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

# ========= Command Listener =========
def command_listener():
    global running, current_file_id, rider_id, current_folder
    global gps_thread_obj, cam_imu_thread_obj, stop_event

    last_command_id = None
    print("👂 Command listener started...")

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
                        
                        current_folder = setup_data_folder()
                        
                        file_res = supabase.table("riderfiles").insert({
                            "rider_id": rider_id,
                            "filename": f"ride_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "filepath": current_folder
                        }).execute()
                        current_file_id = file_res.data[0]["id"]

                        print(f"\n{'='*60}")
                        print(f"🚀 START - Rider ID: {rider_id}")
                        print(f"📁 Folder: {current_folder}")
                        print(f"{'='*60}\n")

                        gps_thread_obj = Thread(target=gps_thread, daemon=False)
                        gps_thread_obj.start()

                        cam_imu_thread_obj = Thread(target=cam_imu_thread, daemon=False)
                        cam_imu_thread_obj.start()

                        supabase.table("rider_commands")\
                            .update({"status": "executed"})\
                            .eq("id", latest["id"])\
                            .execute()

                    elif command == "stop" and running:
                        running = False
                        stop_event.set()
                        
                        print(f"\n{'='*60}")
                        print(f"🛑 STOP - Waiting for threads...")
                        print(f"{'='*60}\n")
                        
                        if gps_thread_obj:
                            gps_thread_obj.join(timeout=5)
                        
                        if cam_imu_thread_obj:
                            cam_imu_thread_obj.join(timeout=15)
                        
                        cleanup_camera()
                        time.sleep(1)
                        reset_camera()
                        
                        print(f"\n✅ Data saved: {current_folder}\n")
                        
                        supabase.table("rider_commands")\
                            .update({"status": "executed"})\
                            .eq("id", latest["id"])\
                            .execute()
                        
                        current_file_id = None
                        current_folder = None
                        gps_thread_obj = None
                        cam_imu_thread_obj = None

        except Exception as e:
            print(f"❌ Command error: {e}")
        
        time.sleep(1)

# ========= GPS Thread =========
def gps_thread():
    print("📡 GPS starting...")
    
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
                except:
                    if not stop_event.is_set():
                        time.sleep(0.5)
    except Exception as e:
        print(f"❌ GPS error: {e}")
    
    print("📡 GPS stopped")

# ========= Video + IMU Thread =========
def cam_imu_thread():
    global picam2_global
    
    print("🎥 Initializing camera and IMU...")
    reset_camera()

    # IMU Setup
    bus = None
    try:
        bus = SMBus(BUS_NUM)
        bus.write_byte_data(ADDR, PWR_MGMT_1, 0)
        time.sleep(0.1)
        print("✅ IMU ready")
    except Exception as e:
        print(f"❌ IMU error: {e}")
        return

    # Camera Setup
    try:
        with camera_lock:
            picam2_global = Picamera2()
            video_config = picam2_global.create_video_configuration(
                main={"size": (1920, 1080), "format": "RGB888"}
            )
            picam2_global.configure(video_config)
            picam2_global.start()
            
        time.sleep(2)
        print("✅ Camera ready")
        
    except Exception as e:
        print(f"❌ Camera error: {e}")
        if bus:
            bus.close()
        return

    csv_file = None
    csv_writer = None
    video_path = None
    encoder = None
    recording_start_time = None

    try:
        # CSV File
        csv_path = os.path.join(current_folder, "combined_data.csv")
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "timestamp", "epoch_time",
            "ax_g", "ay_g", "az_g",
            "gx_dps", "gy_dps", "gz_dps",
            "gps_utc", "gps_lat", "gps_ns", "gps_lon", "gps_ew",
            "gps_speed_kn", "gps_course_deg", "gps_valid"
        ])
        print(f"📝 CSV: {csv_path}")

        # Video Recording
        video_path = os.path.join(current_folder, "video.h264")
        encoder = H264Encoder(bitrate=10000000)
        
        with camera_lock:
            if picam2_global is not None:
                picam2_global.start_recording(encoder, video_path)
                recording_start_time = datetime.now()
                print(f"🎬 Recording: {video_path}\n")

        imu_count = 0
        last_print = time.time()

        while not stop_event.is_set():
            loop_start = time.time()
            
            now = datetime.now()
            timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
            epoch_time = time.time()

            # IMU Read
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

            # Write CSV
            csv_writer.writerow([
                timestamp_str, epoch_time,
                round(ax_g, 4), round(ay_g, 4), round(az_g, 4),
                round(gx_dps, 3), round(gy_dps, 3), round(gz_dps, 3),
                gps_copy["utc"], gps_copy["lat"], gps_copy["ns"],
                gps_copy["lon"], gps_copy["ew"],
                gps_copy["speed"], gps_copy["course"], gps_copy["valid"]
            ])
            csv_file.flush()

            # Supabase Push
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

                supabase.table("riderdata").insert({
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
                    "image_filename": "video.mp4"
                }).execute()

            except Exception as e:
                if not stop_event.is_set():
                    print(f"⚠️ Supabase: {e}")

            # Status Print
            imu_count += 1
            if time.time() - last_print >= 1.0:
                print(f"🎥 {now.strftime('%I:%M:%S %p')} | IMU: {imu_count}Hz | GPS: {gps_copy['lat']}{gps_copy['ns']}")
                imu_count = 0
                last_print = time.time()

            # Rate Control
            elapsed = time.time() - loop_start
            sleep_time = (1.0 / IMU_RATE) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        if not stop_event.is_set():
            print(f"❌ Loop error: {e}")
    finally:
        # Cleanup
        if csv_file:
            csv_file.close()
            print("\n💾 CSV saved")
        
        with camera_lock:
            if picam2_global is not None:
                try:
                    picam2_global.stop_recording()
                    time.sleep(0.5)
                    picam2_global.stop()
                    print("🎥 Recording stopped")
                except:
                    pass
        
        # Convert to MP4 with timestamp
        if video_path and os.path.exists(video_path):
            try:
                print("\n🔄 Converting to MP4 with timestamp...")
                mp4_path = os.path.join(current_folder, "video.mp4")
                
                # Generate timestamp file for ffmpeg
                timestamp_file = os.path.join(current_folder, "timestamps.txt")
                with open(timestamp_file, 'w') as tf:
                    # Calculate duration
                    duration = (datetime.now() - recording_start_time).total_seconds()
                    current_time = recording_start_time
                    
                    # Write timestamp every 0.1 seconds
                    for i in range(int(duration * 10)):
                        timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                        tf.write(f"{i/10:.1f} {timestamp}\n")
                        current_time += timedelta(milliseconds=100)
                
                # Use drawtext with file for precise timestamps
                subprocess.run([
                    'ffmpeg', '-y', 
                    '-i', video_path,
                    '-vf', f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:textfile={timestamp_file}:reload=1:fontcolor=white:fontsize=40:box=1:boxcolor=black@0.8:boxborderw=5:x=w-tw-20:y=h-th-20",
                    '-c:v', 'libx264', 
                    '-preset', 'fast', 
                    '-crf', 23,
                    mp4_path
                ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, timeout=120)
                
                os.remove(video_path)
                os.remove(timestamp_file)
                print(f"✅ Video saved: video.mp4")
            except subprocess.TimeoutExpired:
                print("⚠️ Video conversion timeout - using fallback method")
                try:
                    # Fallback: simple timestamp overlay
                    subprocess.run([
                        'ffmpeg', '-y', '-i', video_path,
                        '-vf', "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='%{localtime}':fontcolor=white:fontsize=40:box=1:boxcolor=black@0.8:boxborderw=5:x=w-tw-20:y=h-th-20",
                        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', 23,
                        mp4_path
                    ], check=True, timeout=60)
                    os.remove(video_path)
                    print(f"✅ Video saved (fallback): video.mp4")
                except:
                    print(f"⚠️ H264 file kept: {video_path}")
            except Exception as e:
                print(f"⚠️ Conversion error: {e}")
                print(f"   H264 available: {video_path}")
        
        if bus:
            bus.close()
    
    print("🎥 Thread stopped")

# ========= Main =========
if __name__ == "__main__":
    print("=" * 60)
    print("   IMU + Video + GPS Logger")
    print("=" * 60)
    print(f"📊 IMU: {IMU_RATE}Hz | Video: {VIDEO_FPS}FPS")
    print("-" * 60)
    
    from datetime import timedelta  # Import for video timestamp generation
    
    reset_camera()
    
    try:
        t_cmd = Thread(target=command_listener, daemon=True)
        t_cmd.start()
        
        print("\n⏳ WAITING FOR START COMMAND...")
        print("=" * 60)
        
        while True:
            time.sleep(1)
        
    except KeyboardInterrupt:
        print("\n⛔ Stopping...")
        stop_event.set()
        cleanup_camera()
        reset_camera()
        print("👋 Exit")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        cleanup_camera()
        print("✅ Done")