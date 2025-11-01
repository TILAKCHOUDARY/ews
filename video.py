import os, csv, time, serial, subprocess, sys
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
VIDEO_FPS = 5
IMU_RATE = 10  # 10 readings per second

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

# ========= Camera Reset Function =========
def reset_camera():
    """Reset camera by killing any existing processes"""
    try:
        sys.stdout.write("🔄 Resetting camera processes...\n")
        sys.stdout.flush()
        subprocess.run(['sudo', 'killall', '-9', 'libcamera-hello'], 
                      stderr=subprocess.DEVNULL)
        subprocess.run(['sudo', 'killall', '-9', 'libcamera-still'], 
                      stderr=subprocess.DEVNULL)
        subprocess.run(['sudo', 'killall', '-9', 'libcamera-vid'], 
                      stderr=subprocess.DEVNULL)
        subprocess.run(['sudo', 'pkill', '-9', '-f', 'libcamera'], 
                      stderr=subprocess.DEVNULL)
        time.sleep(3)
        sys.stdout.write("✅ Camera processes reset complete\n")
        sys.stdout.flush()
    except Exception as e:
        sys.stdout.write(f"⚠️  Camera reset error (non-critical): {e}\n")
        sys.stdout.flush()

def cleanup_camera():
    """Properly cleanup camera object"""
    global picam2_global
    with camera_lock:
        if picam2_global is not None:
            try:
                sys.stdout.write("🛑 Stopping camera...\n")
                sys.stdout.flush()
                picam2_global.stop_recording()
                time.sleep(0.5)
                picam2_global.stop()
                picam2_global.close()
                time.sleep(1)
                sys.stdout.write("✅ Camera stopped and closed\n")
                sys.stdout.flush()
            except Exception as e:
                sys.stdout.write(f"⚠️  Camera cleanup error: {e}\n")
                sys.stdout.flush()
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
        sys.stdout.write(f"IMU read error: {e}\n")
        sys.stdout.flush()
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

def format_timestamp_for_video():
    """Format timestamp as HH:MM:SS AM/PM"""
    now = datetime.now()
    return now.strftime("%I:%M:%S %p")

# ========= Command Listener Thread =========
def command_listener():
    """Listen for START/STOP commands from Supabase"""
    global running, current_file_id, rider_id, current_folder
    global gps_thread_obj, cam_imu_thread_obj, stop_event

    last_command_id = None
    sys.stdout.write("👂 Command listener started - waiting for commands...\n")
    sys.stdout.flush()

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

                        sys.stdout.write(f"\n{'='*60}\n")
                        sys.stdout.write(f"🚀 START command received!\n")
                        sys.stdout.write(f"👤 Rider ID: {rider_id}\n")
                        sys.stdout.write(f"📄 File ID: {current_file_id}\n")
                        sys.stdout.write(f"📁 Data folder: {current_folder}\n")
                        sys.stdout.write(f"{'='*60}\n\n")
                        sys.stdout.flush()

                        gps_thread_obj = Thread(target=gps_thread, daemon=False)
                        gps_thread_obj.start()
                        sys.stdout.write("📡  GPS thread STARTED\n")
                        sys.stdout.flush()

                        cam_imu_thread_obj = Thread(target=cam_imu_thread, daemon=False)
                        cam_imu_thread_obj.start()
                        sys.stdout.write("🎥 Video/IMU thread STARTED\n\n")
                        sys.stdout.flush()

                        supabase.table("rider_commands")\
                            .update({"status": "executed"})\
                            .eq("id", latest["id"])\
                            .execute()

                    elif command == "stop" and running:
                        running = False
                        stop_event.set()
                        
                        sys.stdout.write(f"\n{'='*60}\n")
                        sys.stdout.write(f"🛑 STOP command received!\n")
                        sys.stdout.write(f"👤 Rider ID: {rider_id}\n")
                        sys.stdout.write(f"ℹ️ Stopping threads...\n")
                        sys.stdout.write(f"{'='*60}\n\n")
                        sys.stdout.flush()
                        
                        if gps_thread_obj and gps_thread_obj.is_alive():
                            gps_thread_obj.join(timeout=5)
                            sys.stdout.write("✅ GPS thread STOPPED\n")
                            sys.stdout.flush()
                        
                        if cam_imu_thread_obj and cam_imu_thread_obj.is_alive():
                            cam_imu_thread_obj.join(timeout=10)
                            sys.stdout.write("✅ Video/IMU thread STOPPED\n")
                            sys.stdout.flush()
                        
                        cleanup_camera()
                        reset_camera()
                        
                        sys.stdout.write(f"💾 Data saved in: {current_folder}\n\n")
                        sys.stdout.flush()
                        
                        supabase.table("rider_commands")\
                            .update({"status": "executed"})\
                            .eq("id", latest["id"])\
                            .execute()
                        
                        current_file_id = None
                        current_folder = None
                        gps_thread_obj = None
                        cam_imu_thread_obj = None
                        
                        sys.stdout.write("🔄 System ready for next ride\n\n")
                        sys.stdout.flush()

        except Exception as e:
            sys.stdout.write(f"❌  Command listener error: {e}\n")
            sys.stdout.flush()
        
        time.sleep(1)

# ========= GPS Thread =========
def gps_thread():
    """GPS data collection thread"""
    sys.stdout.write("📡  GPS collection starting...\n")
    sys.stdout.flush()
    
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
                        sys.stdout.write(f"GPS error: {e}\n")
                        sys.stdout.flush()
                    time.sleep(1)
    except Exception as e:
        sys.stdout.write(f"GPS serial error: {e}\n")
        sys.stdout.flush()
    
    sys.stdout.write("📡  GPS thread exiting...\n")
    sys.stdout.flush()

# ========= Video + IMU Thread =========
def cam_imu_thread():
    """Combined video recording and IMU thread - IMU at 10Hz"""
    global picam2_global
    
    sys.stdout.write("🎥 Video/IMU initialization starting...\n")
    sys.stdout.flush()
    reset_camera()

    # Setup IMU
    bus = None
    try:
        bus = SMBus(BUS_NUM)
        bus.write_byte_data(ADDR, PWR_MGMT_1, 0)
        time.sleep(0.1)
        sys.stdout.write("✅ IMU initialized successfully\n")
        sys.stdout.flush()
    except Exception as e:
        sys.stdout.write(f"❌ IMU initialization error: {e}\n")
        sys.stdout.flush()
        return

    # Setup camera for video recording
    try:
        with camera_lock:
            sys.stdout.write("🎥 Creating Picamera2 instance...\n")
            sys.stdout.flush()
            picam2_global = Picamera2()
            
            sys.stdout.write("🎥 Configuring camera for video...\n")
            sys.stdout.flush()
            video_config = picam2_global.create_video_configuration(
                main={"size": (1920, 1080), "format": "RGB888"}
            )
            picam2_global.configure(video_config)
            
            sys.stdout.write("🎥 Starting camera...\n")
            sys.stdout.flush()
            picam2_global.start()
            
        time.sleep(2)
        sys.stdout.write("✅ Camera initialized successfully\n")
        sys.stdout.flush()
        
    except Exception as e:
        sys.stdout.write(f"❌ Camera initialization error: {e}\n")
        sys.stdout.flush()
        if bus:
            bus.close()
        return

    csv_file = None
    csv_writer = None
    video_path = None
    encoder = None

    try:
        # Open CSV file
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
        sys.stdout.write(f"📝 CSV file created: {csv_path}\n")
        sys.stdout.flush()

        # Start video recording
        video_path = os.path.join(current_folder, "video.h264")
        encoder = H264Encoder(bitrate=10000000)
        
        with camera_lock:
            if picam2_global is not None:
                picam2_global.start_recording(encoder, video_path)
                sys.stdout.write(f"🎬 Video recording started: {video_path}\n\n")
                sys.stdout.flush()

        imu_sample_count = 0
        last_print_time = time.time()

        while not stop_event.is_set():
            loop_start = time.time()
            
            # Get synchronized timestamp
            now = datetime.now()
            timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
            epoch_time = time.time()

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
            csv_file.flush()

            # ========= PUSH TO SUPABASE IN REAL-TIME =========
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
                    sys.stdout.write(f"❌  Supabase push error: {e}\n")
                    sys.stdout.flush()

            # Console output every second (after 10 samples)
            imu_sample_count += 1
            if time.time() - last_print_time >= 1.0:
                video_timestamp = format_timestamp_for_video()
                sys.stdout.write(f"🎥 Recording: {video_timestamp}\n")
                sys.stdout.write(f"   IMU ({imu_sample_count} samples/s) -> ACC: {ax_g:.3f}, {ay_g:.3f}, {az_g:.3f} | GYRO: {gx_dps:.3f}, {gy_dps:.3f}, {gz_dps:.3f}\n")
                sys.stdout.write(f"   GPS -> Lat:{gps_copy['lat']}{gps_copy['ns']} | Lon:{gps_copy['lon']}{gps_copy['ew']} | Speed:{gps_copy['speed']}kn | Valid:{gps_copy['valid']}\n")
                sys.stdout.write("-" * 100 + "\n")
                sys.stdout.flush()
                imu_sample_count = 0
                last_print_time = time.time()

            # Control IMU sampling rate (10 Hz = 0.1 seconds)
            elapsed = time.time() - loop_start
            sleep_time = (1.0 / IMU_RATE) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        if not stop_event.is_set():
            sys.stdout.write(f"❌ Main loop error: {e}\n")
            sys.stdout.flush()
    finally:
        # Cleanup
        try:
            if csv_file:
                csv_file.close()
                sys.stdout.write("💾 CSV file closed\n")
                sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(f"CSV cleanup error: {e}\n")
            sys.stdout.flush()
        
        try:
            with camera_lock:
                if picam2_global is not None:
                    picam2_global.stop_recording()
                    time.sleep(0.5)
                    picam2_global.stop()
                    sys.stdout.write("🎥 Video recording stopped\n")
                    sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(f"Camera stop error: {e}\n")
            sys.stdout.flush()
        
        # Convert H264 to MP4 with timestamp overlay
        if video_path and os.path.exists(video_path):
            try:
                sys.stdout.write("\n🔄 Converting video to MP4 with timestamp overlay...\n")
                sys.stdout.flush()
                mp4_path = os.path.join(current_folder, "video.mp4")
                
                # Add timestamp overlay at bottom-right corner using ffmpeg
                result = subprocess.run([
                    'ffmpeg', '-y', '-i', video_path,
                    '-vf', "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='%{localtime\\:%I\\\\:%M\\\\:%S %p}':fontcolor=white:fontsize=48:box=1:boxcolor=black@0.7:boxborderw=8:x=w-tw-30:y=h-th-30",
                    '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
                    mp4_path
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    os.remove(video_path)
                    sys.stdout.write(f"✅ Video saved as: video.mp4\n\n")
                    sys.stdout.flush()
                else:
                    sys.stdout.write(f"⚠️  FFmpeg error: {result.stderr}\n\n")
                    sys.stdout.flush()
            except Exception as e:
                sys.stdout.write(f"⚠️  Video conversion error: {e}\n\n")
                sys.stdout.flush()
        
        try:
            if bus:
                bus.close()
                sys.stdout.write("🔌 IMU bus closed\n")
                sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(f"IMU cleanup error: {e}\n")
            sys.stdout.flush()
    
    sys.stdout.write("🎥 Video/IMU thread exiting...\n")
    sys.stdout.flush()

# ========= Main =========
if __name__ == "__main__":
    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.write("   IMU + Video + GPS Data Logger with Supabase\n")
    sys.stdout.write("=" * 60 + "\n")
    sys.stdout.write(f"☁️  Supabase: Real-time updates enabled\n")
    sys.stdout.write(f"🎧 Listening for commands from ANY rider\n")
    sys.stdout.write(f"📊 IMU Rate: {IMU_RATE} Hz | Video: {VIDEO_FPS} FPS\n")
    sys.stdout.write("-" * 60 + "\n")
    sys.stdout.flush()
    
    reset_camera()
    
    try:
        t_cmd = Thread(target=command_listener, daemon=True)
        sys.stdout.write("🚀 Starting command listener...\n")
        sys.stdout.flush()
        t_cmd.start()
        
        sys.stdout.write("\n⏳  WAITING FOR START COMMAND FROM WEBSITE...\n")
        sys.stdout.write("   GPS and Video threads will start when START is received\n")
        sys.stdout.write("=" * 60 + "\n")
        sys.stdout.flush()
        
        while True:
            time.sleep(1)
        
    except KeyboardInterrupt:
        sys.stdout.write("\n⛔ Received stop signal...\n")
        sys.stdout.flush()
        stop_event.set()
        cleanup_camera()
        reset_camera()
        sys.stdout.write("👋 Exiting program...\n")
        sys.stdout.flush()
    except Exception as e:
        sys.stdout.write(f"❌ Error: {e}\n")
        sys.stdout.flush()
    finally:
        cleanup_camera()
        sys.stdout.write("✅ Program ended.\n")
        sys.stdout.flush()