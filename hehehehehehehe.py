import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'

import cv2
import time
import threading
import platform
import random
import numpy as np
import serial
import serial.tools.list_ports
import sqlite3
import math
import psutil
import secrets
import signal
import sys
import atexit
from datetime import date
from flask import Flask, render_template_string, Response, jsonify, request, session
from contextlib import contextmanager

# ==========================================
# AUDIO & AI INITIALIZATION
# ==========================================
try:
    import sounddevice as sd
    AUDIO_ENABLED = True
except ImportError:
    AUDIO_ENABLED = False
    print("[-] sounddevice not installed. Audio features disabled.")

try:
    import mediapipe as mp
    MP_AVAILABLE = True
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils
except Exception as e:
    MP_AVAILABLE = False
    hands_detector = None
    mp_draw = None
    print(f"[-] MediaPipe initialization notice: {e}")

# ==========================================
# 1. FLASK APP & PERSISTENT SECURITY
# ==========================================
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'pitpaws_data.db')
SECRET_KEY_FILE = os.path.join(BASE_DIR, '.secret_key')

def get_or_create_secret_key():
    try:
        if os.path.exists(SECRET_KEY_FILE):
            with open(SECRET_KEY_FILE, 'r') as f:
                return f.read().strip()
        key = secrets.token_hex(32)
        with open(SECRET_KEY_FILE, 'w') as f:
            f.write(key)
        try:
            os.chmod(SECRET_KEY_FILE, 0o600)
        except Exception:
            pass
        return key
    except Exception:
        return secrets.token_hex(32)

app.secret_key = os.environ.get("PITPAWS_SECRET_KEY") or get_or_create_secret_key()
AUTH_USER = os.environ.get("PITPAWS_USER", "aatriv")
AUTH_PASS = os.environ.get("PITPAWS_PASS", "carbon")

# Synchronization Locks
state_lock = threading.RLock()
pet_profiles_lock = threading.RLock()
frame_lock = threading.Lock()
db_lock = threading.Lock()

MAX_TREAT_CAPACITY = 25
CALIBRATED_METRIC_SCALE = 0.0046
running = True

telemetry_state = {
    "active_baseline_mps": 3.5, 
    "gait_asymmetry": 8.2,
    "prev_day_asymmetry": 18.5,
    "limp_delta": 0.0,                
    "limp_trend": 0.0,                
    "current_velocity_mps": 0.0,
    "current_stride_length_m": 0.0,  
    "top_sprint_mps": 0.0,
    "avg_sprint_mps": 0.0,
    "min_sprint_mps": 99.9,
    "sprint_sum": 0.0,
    "sprint_count": 0,
    "turret_angle_deg": 90,
    "launch_distance_ft": 6,
    "treats_remaining": 25,
    "treats_dispensed_today": 0,
    "treat_level_pct": 100,
    "fall_count": 0,
    "sleep_duration_sec": 0.0,
    "is_sleeping": 0,          
    "turret_mode": "MANUAL",
    "camera_mode": "LAPTOP",   
    "fetch_active": 0,         
    "fetch_start_time": 0.0,
    "last_lap_sec": 0.0,
    "fastest_lap_sec": 99.9,
    "night_mode": 0,            
    "bark_detected": 0,        
    "cpu_usage": 0.0,        
    "ram_usage": 0.0,
    "is_pre_launch": 0,        
    "pre_launch_timer": 0.0,         
    "pet_detected_ai": 0,      
    "pet_detected_type": "None",
    "security_mode": False,
    "security_alert_triggered": False,
    "recall_alert_triggered": False,
    "cognitive_reaction_time_sec": 0.0,
    "expected_reaction_sec": 0.5, 
    "waiting_for_reaction": False,
    "buzzer_timestamp": 0.0,
    
    # Hydration & Respiration FFT
    "continuous_run_time_sec": 0.0,
    "run_session_start": None,
    "panting_level": 0.0,
    "panting_detected": False,
    "hydration_alert": False,

    # Lead-Predictive Aiming
    "target_centroid_x": 320,
    "target_centroid_y": 240,
    "target_vx_pps": 0.0,
    "predicted_lead_angle": 90,

    # Adaptive HIIT Engine
    "hiit_tier": "BALANCED",
    "hiit_consecutive_fast_laps": 0,
    "hiit_consecutive_slow_laps": 0,
    "hiit_next_action_time": 0.0,
    "hiit_adaptive_rest_sec": 8.0,

    # Master Gesture Toggle & Telemetry
    "gesture_detection_enabled": True,
    "active_gesture": "NONE",
    "gesture_finger_count": 0
}

pet_profiles = {
    "Dog": {"name": "Carbon", "birthday": "", "breed": "", "weight": 0.0},
    "Cat": {"name": "Kitty", "birthday": "", "breed": "", "weight": 0.0},
    "Human": {"name": "Owner", "birthday": "", "breed": "", "weight": 0.0}
}

output_frame = None
latest_inference_frame = None
cached_hand_landmarks = None
last_security_alert_time = 0.0
last_water_alert_sent = 0.0
db_dirty = False

arduino = None
camera_cap = None

# ==========================================
# 2. DATABASE CONTEXT MANAGER & SCHEMAS
# ==========================================
@contextmanager
def get_db_connection():
    conn = None
    try:
        with db_lock:
            conn = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
        yield conn
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass

def init_db():
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS daily_stats 
                        (date TEXT PRIMARY KEY, top_speed REAL, fastest_lap REAL, 
                         falls INTEGER, treats INTEGER DEFAULT 0, limp_delta_val REAL DEFAULT 0.0)''')
            c.execute('''CREATE TABLE IF NOT EXISTS telemetry_logs 
                        (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, 
                         event_type TEXT, details TEXT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS profiles 
                        (species TEXT PRIMARY KEY, name TEXT, birthday TEXT, 
                         breed TEXT DEFAULT '', weight REAL DEFAULT 0.0)''')
            
            c.execute("INSERT OR IGNORE INTO profiles (species, name, birthday, breed, weight) VALUES ('Dog', 'Carbon', '', '', 0.0)")
            c.execute("INSERT OR IGNORE INTO profiles (species, name, birthday, breed, weight) VALUES ('Cat', 'Kitty', '', '', 0.0)")
            c.execute("INSERT OR IGNORE INTO profiles (species, name, birthday, breed, weight) VALUES ('Human', 'Owner', '', '', 0.0)")
            conn.commit()
    except Exception as e:
        print(f"[-] Database initialization error: {e}")

def update_dynamic_baseline():
    try:
        with pet_profiles_lock:
            dog = pet_profiles.get("Dog", {})
            breed = str(dog.get("breed", "")).lower()
            bday = dog.get("birthday", "")
            weight = float(dog.get("weight", 0.0) or 0.0)

        breed_speeds = {
            "greyhound": 8.0, "border collie": 6.5, "labrador": 5.0, 
            "retriever": 4.5, "bulldog": 2.5, "pug": 2.0
        }
        base_spd = 3.5 
        base_rx = 0.5  
        
        for k, v in breed_speeds.items():
            if k in breed:
                base_spd = v
                break

        try:
            if bday:
                b_year = int(bday.split("-")[0])
                age = date.today().year - b_year
                if age >= 8: 
                    base_spd *= 0.85
                    base_rx += 0.4 
                elif age >= 5:
                    base_rx += 0.2
        except Exception:
            pass

        if weight > 35.0:
            base_spd *= 0.90

        with state_lock:
            telemetry_state["active_baseline_mps"] = round(base_spd, 2)
            telemetry_state["expected_reaction_sec"] = round(base_rx, 2)
    except Exception as e:
        print(f"[-] Baseline update error: {e}")

def load_db_stats():
    today = str(date.today())
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT top_speed, fastest_lap, falls, treats, limp_delta_val FROM daily_stats WHERE date=?", (today,))
            row = c.fetchone()
            if row:
                with state_lock:
                    telemetry_state["top_sprint_mps"] = row[0] or 0.0
                    telemetry_state["fastest_lap_sec"] = row[1] or 99.9
                    telemetry_state["fall_count"] = row[2] or 0
                    telemetry_state["treats_dispensed_today"] = row[3] or 0
                    telemetry_state["limp_delta"] = row[4] or 0.0
                    
            c.execute("SELECT limp_delta_val FROM daily_stats WHERE date < ? ORDER BY date DESC LIMIT 1", (today,))
            prev_row = c.fetchone()
            if prev_row and prev_row[0] is not None:
                with state_lock:
                    telemetry_state["prev_day_asymmetry"] = prev_row[0]
            
            c.execute("SELECT species, name, birthday, breed, weight FROM profiles")
            profiles_data = c.fetchall()
            
        with pet_profiles_lock:
            for row in profiles_data:
                pet_profiles[row[0]] = {
                    "name": row[1], 
                    "birthday": row[2], 
                    "breed": row[3], 
                    "weight": row[4] or 0.0
                }
    except Exception as e:
        print(f"[-] Database load error: {e}")
    
    update_dynamic_baseline()

def db_writer_thread():
    global db_dirty
    while running:
        time.sleep(2.0)
        if db_dirty:
            today = str(date.today())
            with state_lock:
                top = telemetry_state["top_sprint_mps"]
                lap = telemetry_state["fastest_lap_sec"]
                falls = telemetry_state["fall_count"]
                treats = telemetry_state["treats_dispensed_today"]
                limp_val = telemetry_state["limp_delta"]
                db_dirty = False
            
            try:
                with get_db_connection() as conn:
                    c = conn.cursor()
                    c.execute("""INSERT OR REPLACE INTO daily_stats 
                               (date, top_speed, fastest_lap, falls, treats, limp_delta_val) 
                               VALUES (?, ?, ?, ?, ?, ?)""", 
                             (today, top, lap, falls, treats, limp_val))
                    conn.commit()
            except Exception as e:
                print(f"[-] Database write error: {e}")

# ==========================================
# 3. NORMALIZED EUCLIDEAN GESTURE CLASSIFIER
# ==========================================
def classify_landmarks(landmarks):
    try:
        w = landmarks[0]
        palm_len = math.hypot(landmarks[9].x - w.x, landmarks[9].y - w.y) + 1e-5

        def is_open(tip_idx, pip_idx):
            d_tip = math.hypot(landmarks[tip_idx].x - w.x, landmarks[tip_idx].y - w.y)
            d_pip = math.hypot(landmarks[pip_idx].x - w.x, landmarks[pip_idx].y - w.y)
            return d_tip > (d_pip * 1.05)

        index_open = is_open(8, 6)
        middle_open = is_open(12, 10)
        ring_open = is_open(16, 14)
        pinky_open = is_open(20, 18)

        thumb_span = math.hypot(landmarks[4].x - landmarks[17].x, landmarks[4].y - landmarks[17].y) / palm_len
        thumb_open = thumb_span > 0.75

        # 🤙 SHAKA: Thumb + Pinky open, Middle 3 curled
        if thumb_open and pinky_open and (not index_open) and (not middle_open) and (not ring_open):
            return "SHAKA_SECURITY", 2

        # ☝️ POINTING: Index open only
        if index_open and (not middle_open) and (not ring_open) and (not pinky_open):
            return "POINT_LAUNCH", 1

        # ✌️ PEACE: Index & Middle open, Others curled
        if index_open and middle_open and (not ring_open) and (not pinky_open):
            return "PEACE_HIIT", 2

        # 🖐️ OPEN PALM: 4 or 5 fingers open
        extended_count = sum([thumb_open, index_open, middle_open, ring_open, pinky_open])
        if extended_count >= 4:
            return "OPEN_PALM_RECALL", extended_count

        # ✊ FIST: All fingers curled
        if not any([index_open, middle_open, ring_open, pinky_open, thumb_open]):
            return "FIST_STOP", 0

        return "HOLD", extended_count
    except Exception as e:
        print(f"[-] Gesture classification error: {e}")
        return "NONE", 0

# ==========================================
# 4. ASYNC MEDIAPIPE INFERENCE WORKER
# ==========================================
def mediapipe_worker_thread():
    global latest_inference_frame, cached_hand_landmarks
    last_action_time = time.time()
    
    while running:
        try:
            with state_lock:
                gestures_enabled = telemetry_state.get("gesture_detection_enabled", True)

            if not gestures_enabled or not MP_AVAILABLE or latest_inference_frame is None:
                with frame_lock:
                    cached_hand_landmarks = None
                with state_lock:
                    telemetry_state["active_gesture"] = "DISABLED" if not gestures_enabled else "NONE"
                    telemetry_state["gesture_finger_count"] = 0
                time.sleep(0.08)
                continue
                
            with frame_lock:
                frame_rgb = latest_inference_frame.copy()

            results = hands_detector.process(frame_rgb)
            gesture = "NONE"
            fingers = 0
            
            if results.multi_hand_landmarks:
                with frame_lock:
                    cached_hand_landmarks = list(results.multi_hand_landmarks)
                for hand_lms in results.multi_hand_landmarks:
                    gesture, fingers = classify_landmarks(hand_lms.landmark)
                    break
            else:
                with frame_lock:
                    cached_hand_landmarks = None

            now = time.time()
            action_to_fire = None
            
            with state_lock:
                telemetry_state["active_gesture"] = gesture
                telemetry_state["gesture_finger_count"] = fingers

                if gesture not in ["NONE", "HOLD", "DISABLED"] and (now - last_action_time > 1.5):
                    if gesture == "SHAKA_SECURITY":
                        telemetry_state["security_mode"] = not telemetry_state["security_mode"]
                        last_action_time = now

                    elif gesture == "OPEN_PALM_RECALL":
                        telemetry_state["recall_alert_triggered"] = True
                        action_to_fire = "WATER_ALERT"
                        last_action_time = now

                    elif gesture == "POINT_LAUNCH":
                        angle = telemetry_state["turret_angle_deg"]
                        dist = telemetry_state["launch_distance_ft"]
                        action_to_fire = ("POINT_LAUNCH", angle, dist)
                        last_action_time = now

                    elif gesture == "PEACE_HIIT":
                        curr_m = telemetry_state["turret_mode"]
                        telemetry_state["turret_mode"] = "MANUAL" if curr_m == "HIIT_ROUTINE" else "HIIT_ROUTINE"
                        last_action_time = now

                    elif gesture == "FIST_STOP":
                        telemetry_state["turret_mode"] = "MANUAL"
                        telemetry_state["is_pre_launch"] = 0
                        telemetry_state["security_mode"] = False
                        last_action_time = now

            if action_to_fire == "WATER_ALERT":
                if arduino and arduino.is_open:
                    try:
                        arduino.write(b"ALERT:WATER\n")
                    except Exception as e:
                        print(f"[-] Serial write error: {e}")
            elif isinstance(action_to_fire, tuple) and action_to_fire[0] == "POINT_LAUNCH":
                trigger_launch_sequence(action_to_fire[1], action_to_fire[2], is_auto=False)

        except Exception as e:
            print(f"[-] MediaPipe worker error: {e}")
        
        time.sleep(0.035)

# ==========================================
# 5. AUDIO & SYSTEM TELEMETRY
# ==========================================
def audio_monitor_thread():
    if not AUDIO_ENABLED:
        return
    
    def audio_callback(indata, frames, time_info, status):
        try:
            if status:
                pass
            
            if indata.shape[1] > 1:
                audio_data = np.mean(indata, axis=1)
            else:
                audio_data = indata[:, 0]
            
            if len(audio_data) == 0:
                return
            
            fft_data = np.abs(np.fft.rfft(audio_data))
            freqs = np.fft.rfftfreq(len(audio_data), 1.0/44100)
            volume_norm = np.linalg.norm(audio_data) * 10
            
            if volume_norm > 45.0 and len(freqs) > 0:
                dominant_freq = freqs[np.argmax(fft_data)]
                if 500 < dominant_freq < 3000:
                    with state_lock: 
                        telemetry_state["bark_detected"] = 1

            pant_mask = (freqs >= 150) & (freqs <= 600)
            if np.any(pant_mask):
                pant_energy = float(np.mean(fft_data[pant_mask]))
                with state_lock:
                    telemetry_state["panting_level"] = round(pant_energy, 2)
                    if pant_energy > 450.0:
                        telemetry_state["panting_detected"] = True
                    elif pant_energy < 200.0:
                        telemetry_state["panting_detected"] = False
        except Exception as e:
            print(f"[-] Audio callback error: {e}")

    try:
        with sd.InputStream(callback=audio_callback, samplerate=44100, channels=1, blocksize=2048):
            while running:
                time.sleep(0.1) 
                with state_lock:
                    is_bark = telemetry_state["bark_detected"]
                if is_bark:
                    time.sleep(1.0) 
                    with state_lock: 
                        telemetry_state["bark_detected"] = 0
    except Exception as e:
        print(f"[-] Audio Stream Warning: {e}")

def system_monitor_thread():
    while running:
        try:
            cpu_usage = psutil.cpu_percent(interval=1)
            ram_usage = psutil.virtual_memory().percent
            with state_lock:
                telemetry_state["cpu_usage"] = cpu_usage
                telemetry_state["ram_usage"] = ram_usage
        except Exception as e:
            print(f"[-] System monitor error: {e}")
        time.sleep(2)

# ==========================================
# 6. HARDWARE CONTROLLER & LAUNCH ENGINE
# ==========================================
def get_universal_arduino_port():
    try:
        for port in serial.tools.list_ports.comports():
            desc = port.description.lower()
            device = port.device.lower()
            if any(k in desc or k in device for k in ["arduino", "ch340", "cp210", "usbmodem", "ttyacm", "ttyusb"]):
                return port.device
    except Exception as e:
        print(f"[-] Port scanning error: {e}")
    return None

def init_arduino():
    global arduino
    arduino_port = get_universal_arduino_port()
    if not arduino_port:
        print("[-] No Arduino found on serial bus")
        return None
    try:
        arduino = serial.Serial(arduino_port, 9600, timeout=1, write_timeout=0.5)
        print(f"[+] Arduino connected on {arduino_port}")
        return arduino
    except Exception as e:
        print(f"[-] Arduino connection failed: {e}")
        return None

def get_camera_for_mode(cam_mode):
    global camera_cap
    if camera_cap is not None:
        try:
            camera_cap.release()
        except Exception:
            pass
        camera_cap = None
    
    os_name = platform.system()
    target_indices = [1, 0] if cam_mode in ["EXTENDED", "360_PANORAMIC"] else [0, 1]
    
    for idx in target_indices:
        try:
            backend = cv2.CAP_AVFOUNDATION if os_name == "Darwin" else cv2.CAP_ANY
            cap = cv2.VideoCapture(idx, backend)
            if cap is not None and cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    camera_cap = cap
                    return cap
                cap.release()
        except Exception as e:
            print(f"[-] Camera index {idx} error: {e}")
            
    try:
        cap = cv2.VideoCapture(0)
        if cap is not None and cap.isOpened():
            camera_cap = cap
            return cap
    except Exception as e:
        print(f"[-] Fallback camera error: {e}")
    return None

def fire_hardware_turret(angle, dist):
    if arduino and arduino.is_open:
        cmd = f"FIRE:{angle}:{dist}\n"
        try: 
            arduino.write(cmd.encode())
        except Exception as e:
            print(f"[-] Serial write error: {e}")

def trigger_launch_sequence(angle, distance, is_auto=False):
    global db_dirty
    should_fire_now = False
    
    with state_lock:
        if telemetry_state["treats_remaining"] > 0:
            telemetry_state["turret_angle_deg"] = angle
            telemetry_state["launch_distance_ft"] = distance
            
            if is_auto:
                if not telemetry_state["is_pre_launch"]:
                    telemetry_state["is_pre_launch"] = 1
                    telemetry_state["pre_launch_timer"] = time.time()
                    telemetry_state["buzzer_timestamp"] = time.time()
                    telemetry_state["waiting_for_reaction"] = True
            else:
                telemetry_state["treats_remaining"] -= 1
                telemetry_state["treats_dispensed_today"] += 1
                telemetry_state["fetch_active"] = 1
                telemetry_state["fetch_start_time"] = time.time()
                db_dirty = True
                should_fire_now = True

    if should_fire_now:
        fire_hardware_turret(angle, distance)

def turret_logic_thread():
    global db_dirty, last_water_alert_sent
    auto_fire_timer = time.time()
    
    while running:
        now = time.time()
        should_fire_cmd = None
        should_trigger_auto = None
        
        with state_lock:
            if telemetry_state["run_session_start"] is not None:
                telemetry_state["continuous_run_time_sec"] = round(now - telemetry_state["run_session_start"], 1)
                if (telemetry_state["continuous_run_time_sec"] >= 300.0) and telemetry_state["panting_detected"]:
                    if not telemetry_state["hydration_alert"]:
                        telemetry_state["hydration_alert"] = True
                        if arduino and arduino.is_open and (now - last_water_alert_sent > 15.0):
                            try:
                                arduino.write(b"ALERT:WATER\n")
                                last_water_alert_sent = now
                            except Exception:
                                pass
            else:
                telemetry_state["continuous_run_time_sec"] = 0.0

            if telemetry_state["is_pre_launch"]:
                if now - telemetry_state["pre_launch_timer"] > 3.0:
                    telemetry_state["is_pre_launch"] = 0
                    telemetry_state["waiting_for_reaction"] = False
                    if telemetry_state["treats_remaining"] > 0:
                        telemetry_state["treats_remaining"] -= 1
                        telemetry_state["treats_dispensed_today"] += 1
                        telemetry_state["fetch_active"] = 1
                        telemetry_state["fetch_start_time"] = now
                        db_dirty = True
                        should_fire_cmd = (telemetry_state['turret_angle_deg'], telemetry_state['launch_distance_ft'])
                    auto_fire_timer = now

            if not telemetry_state["is_pre_launch"]:
                mode = telemetry_state["turret_mode"]
                lead_angle = telemetry_state["predicted_lead_angle"]
                
                if mode == "AUTO_LURE":
                    if telemetry_state["bark_detected"] and (now - auto_fire_timer > 3.0): 
                        should_trigger_auto = (lead_angle, random.randint(5, 8))
                    elif not telemetry_state["is_sleeping"] and (now - auto_fire_timer > 8.0): 
                        should_trigger_auto = (lead_angle, random.randint(5, 8))

                elif mode == "HIIT_ROUTINE":
                    if now > telemetry_state["hiit_next_action_time"]:
                        tier = telemetry_state["hiit_tier"]
                        if tier == "ENDURANCE_BOOST":
                            dist = random.choice([8, 9, 10])
                            rest_delay = 5.0
                        elif tier == "RECOVERY":
                            dist = random.choice([3, 4, 5])
                            rest_delay = 14.0
                        else:
                            dist = random.choice([5, 6, 7])
                            rest_delay = 8.0
                        
                        telemetry_state["hiit_next_action_time"] = now + rest_delay
                        telemetry_state["hiit_adaptive_rest_sec"] = rest_delay
                        should_trigger_auto = (lead_angle, dist)

            if telemetry_state["fetch_active"] and (now - telemetry_state["fetch_start_time"] > 25.0):
                telemetry_state["fetch_active"] = 0

            if telemetry_state["sleep_duration_sec"] > 300 and not telemetry_state["is_sleeping"]:
                telemetry_state["is_sleeping"] = 1
            elif telemetry_state["sleep_duration_sec"] < 60 and telemetry_state["is_sleeping"]:
                telemetry_state["is_sleeping"] = 0

        if should_fire_cmd:
            fire_hardware_turret(should_fire_cmd[0], should_fire_cmd[1])
        elif should_trigger_auto:
            trigger_launch_sequence(should_trigger_auto[0], should_trigger_auto[1], is_auto=True)

        time.sleep(0.1)

# ==========================================
# 7. MAIN CAPTURE & OPENCV TRACKING (30 FPS)
# ==========================================
def vision_thread_loop():
    global output_frame, latest_inference_frame, camera_cap, last_security_alert_time, db_dirty
    current_cam_mode = "LAPTOP"
    camera_cap = get_camera_for_mode(current_cam_mode)
    
    prev_gray_small = None
    inactivity_start_time = None
    last_tick_time = time.time()
    last_fall_time = time.time()
    last_kinematic_time = time.time()
    last_motion_time = time.time()
    
    prev_cx, prev_cy = 320, 240
    prev_centroid_time = time.time()
    consecutive_read_failures = 0

    while running:
        try:
            with state_lock:
                target_cam_mode = telemetry_state.get("camera_mode", "LAPTOP")
                active_base = telemetry_state["active_baseline_mps"]
                gestures_enabled = telemetry_state.get("gesture_detection_enabled", True)

            if target_cam_mode != current_cam_mode or camera_cap is None or not camera_cap.isOpened():
                if camera_cap is not None:
                    try:
                        camera_cap.release()
                    except Exception:
                        pass
                current_cam_mode = target_cam_mode
                camera_cap = get_camera_for_mode(current_cam_mode)
                prev_gray_small = None
                consecutive_read_failures = 0
                
                if camera_cap is None:
                    time.sleep(1.0)
                    continue

            ret, frame = camera_cap.read()
            if not ret or frame is None:
                consecutive_read_failures += 1
                if consecutive_read_failures > 15:
                    print("[-] Camera feed lost (sleep wake detected). Re-initializing hardware...")
                    try:
                        camera_cap.release()
                    except Exception:
                        pass
                    camera_cap = get_camera_for_mode(current_cam_mode)
                    consecutive_read_failures = 0
                time.sleep(0.05)
                continue
            
            consecutive_read_failures = 0
                
            try:
                frame = cv2.resize(frame, (640, 480))
                frame = cv2.flip(frame, 1)
            except Exception:
                continue

            now = time.time()
            dt = max(0.001, now - last_tick_time)
            last_tick_time = now

            if gestures_enabled:
                with frame_lock:
                    latest_inference_frame = cv2.cvtColor(cv2.resize(frame, (240, 180)), cv2.COLOR_BGR2RGB)
                
                with frame_lock:
                    landmarks_to_draw = list(cached_hand_landmarks) if cached_hand_landmarks else None
                if landmarks_to_draw and MP_AVAILABLE and mp_draw:
                    for hand_lms in landmarks_to_draw:
                        mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            # Fixed-Interval Velocity Differencing (20Hz)
            if (now - last_kinematic_time) > 0.05:
                last_kinematic_time = now
                small_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (240, 180))
                small_blurred = cv2.GaussianBlur(small_gray, (9, 9), 0)

                cx, cy = prev_cx, prev_cy
                vx_pps = 0.0

                if prev_gray_small is not None:
                    frame_diff = cv2.absdiff(prev_gray_small, small_blurred)
                    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
                    motion_ratio = (float(np.count_nonzero(thresh)) / (240 * 180)) * 100.0

                    is_locked = False
                    if motion_ratio > 1.5:
                        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            largest = max(contours, key=cv2.contourArea)
                            if cv2.contourArea(largest) > 250:
                                x, y, w, h = cv2.boundingRect(largest)
                                cx, cy = int((x + w // 2) * 2.66), int((y + h // 2) * 2.66)
                                is_locked = True
                                cv2.rectangle(frame, (int(x * 2.66), int(y * 2.66)), (int((x + w) * 2.66), int((y + h) * 2.66)), (0, 255, 136), 2)

                    dt_pos = max(0.01, now - prev_centroid_time)
                    vx_pps = (cx - prev_cx) / dt_pos
                    prev_cx, prev_cy = cx, cy
                    prev_centroid_time = now

                    predicted_cx = np.clip(cx + (vx_pps * 0.4), 40, 600)
                    lead_angle = int(np.clip(160.0 - (predicted_cx / 640.0) * 140.0, 20, 160))

                    raw_velocity = (motion_ratio * CALIBRATED_METRIC_SCALE * 7.5)
                    raw_velocity = min(12.0, round(raw_velocity, 2))
                    stride = round(raw_velocity / (2.0 + raw_velocity * 0.2), 2) if raw_velocity > 0.2 else 0.0

                    with state_lock:
                        telemetry_state["target_centroid_x"] = cx
                        telemetry_state["target_centroid_y"] = cy
                        telemetry_state["target_vx_pps"] = round(vx_pps, 1)
                        telemetry_state["predicted_lead_angle"] = lead_angle
                        telemetry_state["current_velocity_mps"] = raw_velocity
                        telemetry_state["current_stride_length_m"] = stride

                        if motion_ratio < 0.8:
                            if inactivity_start_time is None: 
                                inactivity_start_time = now
                            elif (now - inactivity_start_time) > 3.0:
                                telemetry_state["is_sleeping"] = 1
                                telemetry_state["sleep_duration_sec"] += dt
                        else:
                            inactivity_start_time = None
                            telemetry_state["is_sleeping"] = 0

                        if motion_ratio > 35.0 and (now - last_fall_time > 4.0):
                            telemetry_state["fall_count"] += 1
                            last_fall_time = now
                            db_dirty = True

                        if telemetry_state.get("security_mode", False) and motion_ratio > 4.0 and (now - last_security_alert_time > 3.0):
                            telemetry_state["security_alert_triggered"] = True
                            last_security_alert_time = now

                        # Run session detection
                        if raw_velocity > 0.3:
                            last_motion_time = now
                            if raw_velocity > 1.2 and telemetry_state["run_session_start"] is None:
                                telemetry_state["run_session_start"] = now
                        else:
                            if telemetry_state["run_session_start"] is not None and (now - last_motion_time > 20.0):
                                telemetry_state["run_session_start"] = None

                        if raw_velocity > telemetry_state["top_sprint_mps"]:
                            telemetry_state["top_sprint_mps"] = raw_velocity
                            db_dirty = True

                        if raw_velocity > 1.5:
                            if telemetry_state["min_sprint_mps"] == 99.9 or raw_velocity < telemetry_state["min_sprint_mps"]:
                                telemetry_state["min_sprint_mps"] = raw_velocity

                            telemetry_state["sprint_sum"] += raw_velocity
                            telemetry_state["sprint_count"] += 1
                            if telemetry_state["sprint_count"] > 0:
                                telemetry_state["avg_sprint_mps"] = round(telemetry_state["sprint_sum"] / telemetry_state["sprint_count"], 2)
                                delta = ((telemetry_state["avg_sprint_mps"] - active_base) / active_base) * 100
                                telemetry_state["limp_delta"] = round(delta, 1)
                                telemetry_state["limp_trend"] = round(telemetry_state["limp_delta"] - telemetry_state["prev_day_asymmetry"], 1)

                prev_gray_small = small_blurred

            # HUD Drawing
            with state_lock:
                velocity_val = telemetry_state['current_velocity_mps']
                mode_val = telemetry_state['turret_mode']
                angle_val = telemetry_state['turret_angle_deg']
                dist_val = telemetry_state['launch_distance_ft']
                stride_val = telemetry_state['current_stride_length_m']
                falls_val = telemetry_state['fall_count']
                sec_mode_active = telemetry_state['security_mode']
                water_alert_active = telemetry_state['hydration_alert']
                hiit_tier_val = telemetry_state['hiit_tier']
                gesture_val = telemetry_state['active_gesture']
                g_enabled = telemetry_state.get('gesture_detection_enabled', True)

            cam_bottom_x, cam_bottom_y = 320, 470
            math_angle = math.radians(180 - angle_val)
            target_x = int(cam_bottom_x + math.cos(math_angle) * (dist_val * 25))
            target_y = int(cam_bottom_y - math.sin(math_angle) * (dist_val * 25))
            
            cv2.line(frame, (cam_bottom_x, cam_bottom_y), (target_x, target_y), (0, 255, 255), 2)
            cv2.circle(frame, (target_x, target_y), 10, (0, 165, 255), 2)
            
            cv2.rectangle(frame, (10, 10), (450, 95), (0, 0, 0), -1)
            mode_color = (0, 200, 255) if water_alert_active else ((0, 0, 255) if sec_mode_active else ((0, 255, 136) if mode_val != 'MANUAL' else (255, 200, 0)))
            status_text = "HYDRATION NEEDED 💧" if water_alert_active else ("SECURITY ARMED 🚨" if sec_mode_active else f"{mode_val} [{hiit_tier_val}]")
            
            cv2.putText(frame, f"MODE: {status_text}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 2)
            cv2.putText(frame, f"VEL: {velocity_val} m/s | STRIDE: {stride_val} m", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 242, 254), 1)
            
            if not g_enabled:
                cv2.putText(frame, "AI GESTURE: DISABLED (OFF)", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 116, 139), 1)
            else:
                g_col = (0, 255, 136) if gesture_val not in ["NONE", "HOLD", "DISABLED"] else (160, 160, 160)
                cv2.putText(frame, f"AI GESTURE: {gesture_val}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, g_col, 2)

            with frame_lock:
                output_frame = frame.copy()

        except Exception as e:
            print(f"[-] Vision thread iteration error: {e}")

        time.sleep(0.001)

# ==========================================
# 8. FLASK STREAMING & REST ENDPOINTS
# ==========================================
def generate_mjpeg_stream():
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 45]
    while running:
        with frame_lock:
            frame_to_send = output_frame.copy() if output_frame is not None else None
        
        if frame_to_send is None:
            time.sleep(0.01)
            continue

        ret, encoded_image = cv2.imencode(".jpg", frame_to_send, encode_param)
        if ret:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')
        
        time.sleep(0.02)

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

def require_auth():
    if not session.get("authenticated"):
        return jsonify({"error": "Unauthorized"}), 401
    return None

@app.route('/api/telemetry')
def get_telemetry(): 
    auth_err = require_auth()
    if auth_err: return auth_err
    with state_lock: return jsonify(telemetry_state)

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json(silent=True) or {}
    if str(data.get("username", "")) == AUTH_USER and str(data.get("password", "")) == AUTH_PASS:
        session["authenticated"] = True
        return jsonify({"success": True})
    return jsonify({"error": "Invalid credentials."}), 401

@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.pop("authenticated", None)
    return jsonify({"success": True})

@app.route('/api/auth-status')
def api_auth_status():
    return jsonify({"authenticated": session.get("authenticated", False)})

@app.route('/api/gesture/toggle', methods=['POST'])
def toggle_gesture_detection():
    auth_err = require_auth()
    if auth_err: return auth_err
    with state_lock:
        telemetry_state["gesture_detection_enabled"] = not telemetry_state["gesture_detection_enabled"]
        current = telemetry_state["gesture_detection_enabled"]
    return jsonify({"status": "success", "gesture_detection_enabled": current})

@app.route('/api/profiles', methods=['GET', 'POST'])
def manage_profiles():
    auth_err = require_auth()
    if auth_err: return auth_err
    if request.method == 'POST':
        data = request.get_json() or {}
        dog = data.get("Dog", {})
        cat = data.get("Cat", {})
        human = data.get("Human", {})
        with pet_profiles_lock:
            if dog: pet_profiles["Dog"] = {"name": dog.get("name", "Carbon"), "birthday": dog.get("birthday", ""), "breed": dog.get("breed", ""), "weight": float(dog.get("weight", 0) or 0)}
            if cat: pet_profiles["Cat"] = {"name": cat.get("name", "Kitty"), "birthday": cat.get("birthday", ""), "breed": "", "weight": 0.0}
            if human: pet_profiles["Human"] = {"name": human.get("name", "Owner"), "birthday": human.get("birthday", ""), "breed": "", "weight": 0.0}
        
        try:
            with get_db_connection() as conn:
                c = conn.cursor()
                for species, pdata in [("Dog", dog), ("Cat", cat), ("Human", human)]:
                    if pdata:
                        c.execute("""INSERT OR REPLACE INTO profiles (species, name, birthday, breed, weight)
                                     VALUES (?, ?, ?, ?, ?)""",
                                  (species, pdata.get("name", ""), pdata.get("birthday", ""), 
                                   pdata.get("breed", ""), float(pdata.get("weight", 0) or 0)))
                conn.commit()
        except Exception as e:
            print(f"[-] Profile db sync error: {e}")

        update_dynamic_baseline()
        return jsonify({"status": "updated"})
    
    with pet_profiles_lock:
        return jsonify(pet_profiles)

@app.route('/api/turret/mode', methods=['POST'])
def toggle_mode():
    auth_err = require_auth()
    if auth_err: return auth_err
    data = request.get_json() or {}
    mode = data.get("mode", "MANUAL")
    with state_lock: 
        telemetry_state["turret_mode"] = mode
        if mode == "HIIT_ROUTINE": 
            telemetry_state["hiit_next_action_time"] = time.time() + 2.0
    return jsonify({"mode": mode})

@app.route('/api/camera/mode', methods=['POST'])
def toggle_camera_mode():
    auth_err = require_auth()
    if auth_err: return auth_err
    data = request.get_json() or {}
    cam_mode = data.get("camera_mode", "LAPTOP")
    with state_lock:
        telemetry_state["camera_mode"] = cam_mode
    return jsonify({"camera_mode": cam_mode})

@app.route('/api/turret/aim', methods=['POST'])
def aim_turret():
    auth_err = require_auth()
    if auth_err: return auth_err
    data = request.get_json() or {}
    with state_lock:
        telemetry_state["turret_mode"] = "MANUAL" 
        telemetry_state["turret_angle_deg"] = int(data.get("angle", 90))
        telemetry_state["launch_distance_ft"] = int(data.get("distance", 6))
    return jsonify({"status": "AIMED"})

@app.route('/api/turret/fire', methods=['POST'])
def fire_turret():
    auth_err = require_auth()
    if auth_err: return auth_err
    with state_lock:
        telemetry_state["turret_mode"] = "MANUAL"
        angle = telemetry_state["turret_angle_deg"]
        dist = telemetry_state["launch_distance_ft"]
    trigger_launch_sequence(angle, dist, is_auto=False)
    return jsonify({"status": "INSTANT_LAUNCH"})

@app.route('/api/turret/refill', methods=['POST'])
def refill_turret():
    auth_err = require_auth()
    if auth_err: return auth_err
    with state_lock: 
        telemetry_state["treats_remaining"] = MAX_TREAT_CAPACITY
    return jsonify({"status": "REFILLED"})

@app.route('/api/security/toggle', methods=['POST'])
def toggle_security_mode():
    auth_err = require_auth()
    if auth_err: return auth_err
    with state_lock:
        telemetry_state["security_mode"] = not telemetry_state["security_mode"]
        current = telemetry_state["security_mode"]
    return jsonify({"status": "success", "security_mode": current})

@app.route('/api/security/disarm', methods=['POST'])
def force_disarm_security():
    auth_err = require_auth()
    if auth_err: return auth_err
    with state_lock:
        telemetry_state["security_mode"] = False
        telemetry_state["security_alert_triggered"] = False
    return jsonify({"status": "DISARMED"})

@app.route('/api/security/status', methods=['GET'])
def check_security_status():
    auth_err = require_auth()
    if auth_err: return auth_err
    with state_lock:
        alert = telemetry_state["security_alert_triggered"]
        telemetry_state["security_alert_triggered"] = False  
    return jsonify({"alert": alert})

@app.route('/api/recall/status', methods=['GET'])
def check_recall_status():
    auth_err = require_auth()
    if auth_err: return auth_err
    with state_lock:
        alert = telemetry_state.get("recall_alert_triggered", False)
        telemetry_state["recall_alert_triggered"] = False
    return jsonify({"alert": alert})

@app.route('/api/hydration/dismiss', methods=['POST'])
def dismiss_hydration():
    auth_err = require_auth()
    if auth_err: return auth_err
    with state_lock:
        telemetry_state["hydration_alert"] = False
        telemetry_state["run_session_start"] = None
        telemetry_state["continuous_run_time_sec"] = 0.0
    return jsonify({"status": "DISMISSED"})

@app.route('/api/reset_stats', methods=['POST'])
def reset_stats():
    global db_dirty
    auth_err = require_auth()
    if auth_err: return auth_err
    with state_lock:
        telemetry_state["fall_count"] = 0
        telemetry_state["treats_dispensed_today"] = 0
        telemetry_state["hydration_alert"] = False
        telemetry_state["continuous_run_time_sec"] = 0.0
        telemetry_state["sprint_sum"] = 0.0
        telemetry_state["sprint_count"] = 0
        telemetry_state["min_sprint_mps"] = 99.9
        telemetry_state["top_sprint_mps"] = 0.0
        telemetry_state["avg_sprint_mps"] = 0.0
        db_dirty = True
    return jsonify({"status": "STATS_RESET"})

@app.route('/api/export_vet_report')
def export_vet_report():
    auth_err = require_auth()
    if auth_err: return auth_err
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT date, top_speed, fastest_lap, falls, treats, limp_delta_val FROM daily_stats ORDER BY date DESC")
            rows = c.fetchall()
        csv_data = "Date,Top_Speed_mps,Fastest_Lap_sec,Falls_Detected,Treats_Dispensed,Gait_Asymmetry_Delta\n"
        for r in rows:
            csv_data += f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]}%\n"
        return Response(csv_data, mimetype="text/csv", headers={"Content-disposition": "attachment; filename=Carbon_Vet_Report.csv"})
    except Exception as e: 
        return f"Error: {e}"

# ==========================================
# 9. TITAN DASHBOARD UI TEMPLATE
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>PitPaws // Titan Station</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;800&display=swap');
        body { font-family: 'Inter', sans-serif; -webkit-tap-highlight-color: transparent; }
        .mono { font-family: 'JetBrains Mono', monospace; }
        .glass { background: rgba(18, 24, 38, 0.75); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); transition: all 0.5s ease; }
        .glow-cyan { box-shadow: 0 0 20px rgba(6, 182, 212, 0.15); }
        .glow-red { box-shadow: 0 0 25px rgba(239, 68, 68, 0.4); border-color: rgba(239, 68, 68, 0.6); }
        .glow-orange { box-shadow: 0 0 30px rgba(249, 115, 22, 0.6); border-color: rgba(249, 115, 22, 0.8); }
        .glow-water { box-shadow: 0 0 35px rgba(14, 165, 233, 0.7); border-color: rgba(14, 165, 233, 0.9); }
    </style>
</head>
<body id="app-body" onclick="forceUnlockAudio()" class="bg-[#080b11] text-slate-200 min-h-screen p-4 lg:p-6 pb-20 transition-all duration-700">

    <div id="audio-unlock-banner" class="fixed top-2 left-1/2 -translate-x-1/2 z-50 bg-amber-500 text-black px-4 py-1.5 rounded-full text-xs font-black shadow-lg cursor-pointer animate-pulse flex items-center gap-2">
        <i data-lucide="volume-2" class="w-4 h-4"></i> CLICK ONCE TO ENABLE BROWSER AUDIO & ALARMS
    </div>

    <div id="water-alert-banner" class="hidden fixed top-5 left-1/2 -translate-x-1/2 z-50 w-11/12 max-w-2xl bg-cyan-950/95 border-2 border-cyan-400 text-cyan-200 p-4 rounded-2xl shadow-2xl backdrop-blur-md flex items-center justify-between glow-water animate-bounce">
        <div class="flex items-center gap-3">
            <span class="text-3xl">💧</span>
            <div>
                <h3 class="font-extrabold text-white text-sm uppercase tracking-wider">Hydration Alert // Give Water</h3>
                <p class="text-xs text-cyan-200 mt-0.5">Continuous running has exceeded 5 minutes with elevated panting audio energy.</p>
            </div>
        </div>
        <button onclick="dismissWaterAlert()" class="bg-cyan-400 hover:bg-cyan-300 text-slate-950 font-black text-xs px-4 py-2 rounded-xl transition active:scale-95">
            DISMISS
        </button>
    </div>

    <div id="modal-login" class="fixed inset-0 z-50 bg-black/90 flex items-center justify-center backdrop-blur-md">
        <div class="glass p-8 rounded-2xl w-full max-w-sm border border-slate-700 m-4 shadow-2xl">
            <div class="text-center mb-6">
                <div class="inline-block p-3 bg-cyan-500/10 border border-cyan-500/30 rounded-2xl text-cyan-400 mb-3">
                    <i data-lucide="paw-print" class="w-8 h-8"></i>
                </div>
                <h2 class="text-xl font-extrabold text-white tracking-wider">PITPAWS ACCESS</h2>
                <p class="text-xs text-slate-400 mt-1">Please log in to view the station.</p>
            </div>
            <div class="space-y-3 mb-5">
                <input id="input-username" type="text" placeholder="Username" class="w-full bg-slate-900 border border-slate-700 rounded-xl p-3 text-white text-sm focus:outline-none focus:border-cyan-500">
                <input id="input-password" type="password" placeholder="Password" class="w-full bg-slate-900 border border-slate-700 rounded-xl p-3 text-white text-sm focus:outline-none focus:border-cyan-500">
            </div>
            <button onclick="submitLogin()" class="w-full py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-bold text-sm shadow-lg shadow-cyan-500/20 active:scale-95 transition">Authorize Station</button>
        </div>
    </div>

    <div id="pet-modal" class="hidden fixed inset-0 z-50 bg-black/80 flex items-center justify-center backdrop-blur-sm">
        <div class="glass p-6 rounded-2xl w-full max-w-md border border-slate-700 m-4 max-h-[90vh] overflow-y-auto">
            <h2 class="text-xl font-bold text-white mb-5"><i data-lucide="scan-face" class="inline w-5 h-5 mr-2"></i>Entity Profiles</h2>
            <div class="mb-4 p-4 rounded-xl bg-cyan-900/10 border border-cyan-500/20 space-y-3">
                <h3 class="text-cyan-400 text-sm font-semibold flex items-center gap-2"><i data-lucide="dog" class="w-4 h-4"></i> PRIMARY DOG</h3>
                <input id="dog-name" type="text" placeholder="Dog's Name" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-white text-sm focus:outline-none focus:border-cyan-500">
                <input id="dog-breed" type="text" placeholder="Breed (e.g., Labrador)" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-white text-sm focus:outline-none focus:border-cyan-500">
                <div class="flex gap-2">
                    <input id="dog-bday" type="date" title="Birthday" class="w-1/2 bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-400 text-sm focus:outline-none focus:border-cyan-500">
                    <input id="dog-weight" type="number" placeholder="Weight (kg)" class="w-1/2 bg-slate-900 border border-slate-700 rounded-lg p-2 text-white text-sm focus:outline-none focus:border-cyan-500">
                </div>
            </div>
            <div class="mb-4 p-4 rounded-xl bg-orange-900/10 border border-orange-500/20">
                <h3 class="text-orange-400 text-sm font-semibold mb-3 flex items-center gap-2"><i data-lucide="cat" class="w-4 h-4"></i> PRIMARY CAT</h3>
                <input id="cat-name" type="text" placeholder="Cat's Name" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-white text-sm focus:outline-none border-orange-500">
            </div>
            <div class="mb-6 p-4 rounded-xl bg-emerald-900/10 border border-emerald-500/20">
                <h3 class="text-emerald-400 text-sm font-semibold mb-3 flex items-center gap-2"><i data-lucide="user" class="w-4 h-4"></i> PRIMARY HUMAN</h3>
                <input id="human-name" type="text" placeholder="Owner's Name" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-white text-sm focus:outline-none focus:border-emerald-500">
            </div>
            <div class="flex justify-end gap-3 sticky bottom-0">
                <button onclick="closePetModal()" class="px-5 py-2.5 rounded-lg bg-slate-800 text-slate-300 font-semibold text-sm">Cancel</button>
                <button onclick="savePetProfiles()" class="px-5 py-2.5 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-bold text-sm">Save Profiles</button>
            </div>
        </div>
    </div>

    <header class="flex justify-between items-center pb-6 mb-6 border-b border-slate-800/80">
        <div class="flex items-center gap-3">
            <div class="p-2.5 bg-cyan-500/10 border border-cyan-500/30 rounded-xl text-cyan-400">
                <i data-lucide="crosshair" class="w-6 h-6"></i>
            </div>
            <div>
                <h1 class="font-extrabold text-xl tracking-wider text-white">PITPAWS <span class="text-cyan-400 text-xs px-2 py-0.5 rounded bg-cyan-950 border border-cyan-800 mono hidden sm:inline-block">TITAN DIAGNOSTICS</span></h1>
            </div>
        </div>
        <div class="flex items-center gap-3">
            <button onclick="handleLogout()" class="px-3 py-1.5 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-xs font-semibold hover:bg-red-500/20 transition">LOGOUT</button>
            <div class="hidden md:flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-slate-800 border border-slate-700 text-slate-300 text-[10px] mono">
                <i data-lucide="clock" class="w-3 h-3"></i> <span id="clock-display">--:--:--</span>
            </div>
            <div id="gesture-badge" class="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-amber-500/10 border border-amber-500/20 text-amber-400 text-[10px] mono">
                <i data-lucide="hand" class="w-3 h-3"></i> GESTURE: <span id="gesture-display" class="font-bold">NONE</span>
            </div>
            <div id="mode-badge" class="flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-[10px] sm:text-xs font-semibold">
                MANUAL MODE
            </div>
        </div>
    </header>

    <main class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div class="lg:col-span-2 space-y-6">
            <div id="video-container" class="glass rounded-2xl overflow-hidden relative border border-slate-800 glow-cyan transition-all duration-300">
                <div class="absolute top-4 left-4 z-10 flex items-center gap-2 px-3 py-1 bg-black/70 backdrop-blur rounded-md text-[10px] sm:text-xs mono text-cyan-400 border border-cyan-500/30">
                    <i data-lucide="camera" class="w-3.5 h-3.5"></i> <span id="camera-label">FIXED KINEMATICS</span>
                </div>
                <div class="aspect-video bg-black flex items-center justify-center relative">
                    <img src="/video_feed" class="w-full h-full object-contain" alt="Kinematic Vision Feed">
                </div>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div id="limp-alert-box" class="glass rounded-2xl p-5 border border-slate-800 text-center transition-all duration-500 flex flex-col justify-center items-center">
                    <h3 class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 flex items-center justify-center gap-2">
                        <i data-lucide="activity" class="w-4 h-4"></i> Orthopedic Status & Falls
                    </h3>
                    <div class="flex items-center gap-3 mt-1 flex-wrap justify-center">
                        <div class="text-xs text-slate-400 mono">Trend: <span id="limp-trend-val" class="font-bold text-base text-cyan-400">0.0%</span></div>
                        <div id="ortho-status-badge" class="px-3 py-1 rounded-full text-xs font-extrabold uppercase tracking-wider bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">Normal</div>
                    </div>
                    <div class="text-xs text-slate-400 mt-3 mono">Falls Registered Today: <span id="fall-count-val" class="font-bold text-red-400">0</span></div>
                </div>

                <div class="glass rounded-2xl p-5 border border-slate-800 text-center flex flex-col justify-center items-center">
                    <h3 class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 flex items-center justify-center gap-2">
                        <i data-lucide="zap" class="w-4 h-4 text-purple-400"></i> Adaptive HIIT & Vitals
                    </h3>
                    <div class="grid grid-cols-2 gap-3 w-full mt-1">
                        <div class="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800">
                            <span class="text-[10px] text-slate-400 uppercase mono">HIIT Difficulty</span>
                            <div class="text-xs font-bold text-purple-400 uppercase tracking-wider mt-1.5" id="hiit-tier-display">BALANCED</div>
                        </div>
                        <div class="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800">
                            <span class="text-[10px] text-slate-400 uppercase mono">Adaptive Rest</span>
                            <div class="text-xs font-bold text-white mono mt-1.5"><span id="hiit-rest-display">8.0</span>s</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="glass rounded-2xl p-5 border border-slate-800 hidden md:block">
                <div class="flex justify-between items-center mb-2">
                    <h3 class="text-xs font-semibold text-slate-400 uppercase tracking-wider mono flex items-center gap-2">
                        <i data-lucide="gauge" class="w-4 h-4 text-cyan-400"></i> Sprint Velocity (m/s)
                    </h3>
                </div>
                <div class="h-20 w-full relative">
                    <canvas id="velocityChart"></canvas>
                    <div class="absolute bottom-0 right-0 text-[9px] text-slate-500 mono" id="active-baseline-display">Base: 3.5 m/s</div>
                </div>
            </div>
        </div>

        <div class="space-y-4">
            <div class="glass rounded-2xl p-4 border border-slate-800 text-center">
                <span class="text-xs text-slate-400 uppercase tracking-wider mono flex items-center justify-center gap-1.5 mb-2.5">
                    <i data-lucide="hand-metal" class="w-4 h-4 text-amber-400"></i> AI Gesture Engine Power
                </span>
                <button id="gesture-toggle-btn" onclick="toggleGestureSystem()" class="w-full py-2.5 px-3 rounded-xl border border-emerald-500/40 bg-emerald-500/20 text-emerald-300 font-extrabold text-xs transition active:scale-95 flex items-center justify-center gap-1.5">
                    <i data-lucide="sparkles" class="w-4 h-4"></i> GESTURE DETECTION: ACTIVE (ON)
                </button>
            </div>

            <div class="glass rounded-2xl p-4 border border-slate-800 text-center">
                <span class="text-xs text-slate-400 uppercase tracking-wider mono flex items-center justify-center gap-1.5 mb-3">
                    <i data-lucide="shield-alert" class="w-4 h-4 text-red-400"></i> Surveillance Security Mode (🤙)
                </span>
                <div class="grid grid-cols-2 gap-2">
                    <button id="sec-btn" onclick="toggleSecurityMode()" class="w-full py-2.5 px-3 rounded-xl border border-slate-700 bg-slate-800 text-slate-300 font-bold text-xs transition active:scale-95 flex items-center justify-center gap-1.5">
                        <i data-lucide="shield" class="w-4 h-4"></i> ARM (🤙)
                    </button>
                    <button onclick="forceDisarmSecurity()" class="w-full py-2.5 px-3 rounded-xl border border-red-500/30 bg-red-500/10 text-red-400 font-bold text-xs hover:bg-red-500/20 transition active:scale-95 flex items-center justify-center gap-1.5">
                        <i data-lucide="shield-off" class="w-4 h-4"></i> DISARM
                    </button>
                </div>
            </div>

            <div class="glass rounded-2xl p-4 border border-slate-800 text-center">
                <span class="text-xs text-slate-400 uppercase tracking-wider mono flex items-center justify-center gap-1.5 mb-2">
                    <i data-lucide="video" class="w-4 h-4 text-cyan-400"></i> Camera Hardware Option
                </span>
                <div class="grid grid-cols-3 gap-2">
                    <button id="cam-btn-laptop" onclick="setCameraMode('LAPTOP')" class="py-2 px-1 rounded-xl border border-cyan-500/40 bg-cyan-500/20 text-cyan-300 font-semibold text-[10px] transition active:scale-95">LAPTOP CAM</button>
                    <button id="cam-btn-extended" onclick="setCameraMode('EXTENDED')" class="py-2 px-1 rounded-xl border border-slate-700 bg-slate-800 text-slate-400 font-semibold text-[10px] hover:border-slate-600 transition active:scale-95">EXTENDED CAM</button>
                    <button id="cam-btn-360" onclick="setCameraMode('360_PANORAMIC')" class="py-2 px-1 rounded-xl border border-slate-700 bg-slate-800 text-slate-400 font-semibold text-[10px] hover:border-slate-600 transition active:scale-95">360° PANORAMIC</button>
                </div>
            </div>

            <div class="glass rounded-2xl p-5 border border-slate-800 text-center">
                <span class="text-xs text-slate-400 uppercase tracking-wider mono flex items-center justify-center gap-1.5 mb-3">
                    <i data-lucide="bot" class="w-4 h-4 text-cyan-400"></i> Turret Mode Selector
                </span>
                <div class="grid grid-cols-3 gap-2">
                    <button id="btn-manual" onclick="setTurretMode('MANUAL')" class="py-2 px-2 rounded-xl border border-cyan-500/40 bg-cyan-500/20 text-cyan-300 font-semibold text-[10px] transition active:scale-95">MANUAL</button>
                    <button id="btn-auto" onclick="setTurretMode('AUTO_LURE')" class="py-2 px-2 rounded-xl border border-emerald-500/40 bg-emerald-500/20 text-emerald-300 font-semibold text-[10px] transition active:scale-95">AUTO LEAD</button>
                    <button id="btn-hiit" onclick="setTurretMode('HIIT_ROUTINE')" class="py-2 px-2 rounded-xl border border-purple-500/40 bg-purple-500/20 text-purple-300 font-semibold text-[10px] transition active:scale-95">ADAPTIVE HIIT</button>
                </div>
            </div>

            <div class="glass rounded-2xl p-5 border border-amber-500/30 bg-amber-500/5">
                <div class="space-y-3 mb-4 text-left">
                    <div>
                        <div class="flex justify-between text-[10px] sm:text-xs mono text-slate-400 mb-1">
                            <span>PAN:</span> <span id="angle-val" class="text-cyan-400 font-bold">90°</span>
                        </div>
                        <input type="range" id="angle-slider" min="0" max="180" value="90" oninput="updateAimText()" onchange="updateAim()" class="w-full accent-cyan-400 bg-slate-800 h-2 rounded-lg cursor-pointer">
                    </div>
                    <div>
                        <div class="flex justify-between text-[10px] sm:text-xs mono text-slate-400 mb-1">
                            <span>DISTANCE:</span> <span id="dist-val" class="text-amber-400 font-bold">6 FT</span>
                        </div>
                        <input type="range" id="dist-slider" min="3" max="10" value="6" oninput="updateAimText()" onchange="updateAim()" class="w-full accent-amber-400 bg-slate-800 h-2 rounded-lg cursor-pointer">
                    </div>
                </div>
                <button id="btn-fire" onclick="fireTurret()" class="w-full py-4 px-4 bg-gradient-to-r from-amber-500 to-orange-500 text-black font-extrabold rounded-xl transition shadow-lg shadow-amber-500/20 flex items-center justify-center gap-2 active:scale-95">
                    <i data-lucide="zap" class="w-5 h-5 fill-black"></i> INITIATE LAUNCH
                </button>
            </div>

            <div class="grid grid-cols-3 gap-3 mt-1">
                <div class="glass rounded-2xl p-4 border border-slate-800 bg-gradient-to-br from-indigo-500/5 to-transparent">
                    <span class="text-[10px] text-slate-400 uppercase tracking-wider mono flex items-center gap-1">
                        <i data-lucide="flag" class="w-3 h-3 text-indigo-400"></i> Last Lap
                    </span>
                    <div class="text-xl font-extrabold text-indigo-400 mt-1 mono"><span id="last-lap">0.0</span><span class="text-[10px] text-slate-500">s</span></div>
                </div>
                
                <div class="glass rounded-2xl p-4 border border-slate-800 bg-gradient-to-br from-yellow-500/5 to-transparent">
                    <span class="text-[10px] text-slate-400 uppercase tracking-wider mono flex items-center gap-1">
                        <i data-lucide="database" class="w-3 h-3 text-yellow-400"></i> Best (DB)
                    </span>
                    <div class="text-xl font-extrabold text-yellow-400 mt-1 mono"><span id="fastest-lap">--</span><span class="text-[10px] text-slate-500 hidden" id="fastest-lap-unit">s</span></div>
                </div>

                <div id="reaction-box" class="glass rounded-2xl p-4 border border-slate-800 bg-gradient-to-br from-purple-500/5 to-transparent transition-all">
                    <span class="text-[10px] text-slate-400 uppercase tracking-wider mono flex items-center gap-1">
                        <i data-lucide="brain" class="w-3 h-3 text-purple-400" id="reaction-icon"></i> Reaction
                    </span>
                    <div id="reaction-text" class="text-xl font-extrabold text-purple-400 mt-1 mono"><span id="reaction-time">0.0</span><span class="text-[10px] text-slate-500">s</span></div>
                    <div class="text-[9px] text-slate-500 mono mt-1" id="reaction-baseline-display">Expected: 0.5s</div>
                </div>
            </div>

            <div class="grid grid-cols-2 gap-3">
                <div class="glass rounded-2xl p-3 border border-slate-800">
                    <span class="text-[10px] text-slate-400 uppercase tracking-wider mono">Peak Spd</span>
                    <div class="text-lg font-extrabold text-emerald-400 mt-1 mono"><span id="metric-top-speed">0.0</span></div>
                </div>
                <div class="glass rounded-2xl p-3 border border-slate-800" id="avg-speed-box">
                    <span class="text-[10px] text-slate-400 uppercase tracking-wider mono">Avg Spd</span>
                    <div class="text-lg font-extrabold text-blue-400 mt-1 mono"><span id="metric-avg-speed">0.0</span></div>
                    <div class="text-[9px] text-slate-500 mono mt-1" id="avg-speed-expected">Expected: 3.5m/s</div>
                </div>
                <div class="glass rounded-2xl p-3 border border-slate-800">
                    <span class="text-[10px] text-slate-400 uppercase tracking-wider mono">Min Spd</span>
                    <div class="text-lg font-extrabold text-purple-400 mt-1 mono"><span id="metric-min-speed">0.0</span></div>
                </div>
                <div class="glass rounded-2xl p-3 border border-pink-500/30 bg-pink-500/5">
                    <span class="text-[10px] text-pink-400 uppercase tracking-wider mono flex items-center gap-1">
                        <i data-lucide="ruler" class="w-3 h-3"></i> Stride Len
                    </span>
                    <div class="text-lg font-extrabold text-pink-400 mt-1 mono"><span id="metric-stride">0.00</span><span class="text-[10px] text-slate-500 ml-1">m</span></div>
                </div>
            </div>

            <div class="glass rounded-2xl p-4 border border-slate-800 flex justify-between items-center">
                <div>
                    <span class="text-xs text-slate-400 uppercase tracking-wider mono">Treat Reservoir</span>
                    <div class="text-2xl font-extrabold text-white mt-0.5 mono"><span id="treat-remaining">25</span><span class="text-slate-500 text-xs"> / 25</span></div>
                    <div class="text-[10px] text-emerald-400 mt-1 mono font-semibold tracking-wider">GIVEN TODAY: <span id="treats-dispensed">0</span></div>
                </div>
                <button onclick="refillTurret()" class="py-2 px-4 bg-slate-800 text-slate-300 text-xs font-semibold rounded-lg border border-slate-700 active:scale-95">REFILL</button>
            </div>

            <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
                <button onclick="resetStats()" class="w-full py-3 rounded-xl border border-red-900/30 bg-red-900/10 text-red-400 font-semibold text-[10px] transition hover:bg-red-900/20 active:scale-95 flex items-center justify-center gap-1.5">
                    <i data-lucide="rotate-ccw" class="w-3.5 h-3.5"></i> RESET
                </button>
                <button onclick="openPetModal()" class="w-full py-3 rounded-xl border border-slate-700 bg-slate-800 text-slate-300 font-semibold text-[10px] transition hover:bg-slate-700 active:scale-95 flex items-center justify-center gap-1.5">
                    <i data-lucide="users" class="w-3.5 h-3.5"></i> PROFILES
                </button>
                <a href="/api/export_vet_report" class="w-full py-3 rounded-xl border border-emerald-500/40 bg-emerald-500/20 text-emerald-300 font-semibold text-[10px] transition hover:bg-emerald-500/30 active:scale-95 flex items-center justify-center gap-1.5">
                    <i data-lucide="download" class="w-3.5 h-3.5"></i> VET REPORT
                </a>
                <button id="btn-buzzer" onclick="toggleManualBuzzer()" class="w-full py-3 rounded-xl border border-slate-700 bg-slate-800 text-slate-300 font-semibold text-[10px] transition hover:bg-slate-700 active:scale-95 flex items-center justify-center gap-1.5">
                    <i data-lucide="volume-x" class="w-3.5 h-3.5"></i> TEST BUZZER
                </button>
            </div>
        </div>
    </main>

    <script>
        try { if (typeof lucide !== 'undefined') lucide.createIcons(); } catch(e) {}
        
        let audioCtx = null;
        function forceUnlockAudio() {
            if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            if (audioCtx.state === 'suspended') audioCtx.resume();
            const banner = document.getElementById('audio-unlock-banner');
            if (banner) banner.classList.add('hidden');
        }
        window.addEventListener('click', forceUnlockAudio, { once: true });
        window.addEventListener('touchstart', forceUnlockAudio, { once: true });

        let manualBuzzerActive = false;

        function playBuzzerTone(freq = 880, durationSec = 0.25, type = 'square') {
            try {
                let ctx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
                if (!audioCtx) audioCtx = ctx;
                if (ctx.state === 'suspended') ctx.resume();

                let osc = ctx.createOscillator();
                let gain = ctx.createGain();
                osc.type = type;
                osc.frequency.setValueAtTime(freq, ctx.currentTime);
                gain.gain.setValueAtTime(0.6, ctx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + durationSec);
                
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.start(ctx.currentTime);
                osc.stop(ctx.currentTime + durationSec);
            } catch(e) {}
        }

        function toggleManualBuzzer() {
            manualBuzzerActive = !manualBuzzerActive;
            const btn = document.getElementById('btn-buzzer');
            forceUnlockAudio();

            if (manualBuzzerActive) {
                btn.classList.add('bg-red-600', 'text-white', 'border-red-500', 'animate-pulse');
                btn.innerHTML = '<i data-lucide="volume-2" class="w-3.5 h-3.5"></i> STOP BUZZER';
                playBuzzerTone(900, 0.5, 'square');
            } else {
                btn.classList.remove('bg-red-600', 'text-white', 'border-red-500', 'animate-pulse');
                btn.innerHTML = '<i data-lucide="volume-x" class="w-3.5 h-3.5"></i> TEST BUZZER';
            }
            try { if (typeof lucide !== 'undefined') lucide.createIcons(); } catch(e){}
        }

        function playRecallWhistle() {
            try {
                let ctx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
                if (!audioCtx) audioCtx = ctx;
                if (ctx.state === 'suspended') ctx.resume();

                const tones = [850, 1250, 1900];
                tones.forEach((freq, idx) => {
                    let osc = ctx.createOscillator();
                    let gain = ctx.createGain();
                    osc.type = 'sine';
                    osc.frequency.setValueAtTime(freq, ctx.currentTime + (idx * 0.18));
                    gain.gain.setValueAtTime(0.7, ctx.currentTime + (idx * 0.18));
                    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + (idx * 0.18) + 0.16);
                    osc.connect(gain);
                    gain.connect(ctx.destination);
                    osc.start(ctx.currentTime + (idx * 0.18));
                    osc.stop(ctx.currentTime + (idx * 0.18) + 0.17);
                });
            } catch(e) {}
        }

        async function checkAuthGate() {
            try {
                const res = await fetch('/api/auth-status');
                const data = await res.json();
                if (data.authenticated) document.getElementById('modal-login').classList.add('hidden');
                else document.getElementById('modal-login').classList.remove('hidden');
            } catch(e) {}
        }
        setInterval(checkAuthGate, 4000);

        async function submitLogin() {
            const username = document.getElementById('input-username').value;
            const password = document.getElementById('input-password').value;
            try {
                const res = await fetch('/api/login', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ username, password })
                });
                const data = await res.json();
                if (res.ok && data.success) {
                    document.getElementById('modal-login').classList.add('hidden');
                    forceUnlockAudio();
                } else alert(data.error || "Login failed.");
            } catch(e) { alert("Network error."); }
        }

        async function handleLogout() {
            await fetch('/api/logout', { method: 'POST' });
            checkAuthGate();
        }

        setInterval(() => {
            const timeStr = new Date().toLocaleTimeString('en-US', { hour12: false });
            const clockEl = document.getElementById('clock-display');
            if (clockEl) clockEl.innerText = timeStr;
        }, 1000);

        let currentProfiles = { Dog: {name: 'Carbon', breed: '', birthday: '', weight: 0.0}, Cat: {name: 'Kitty'}, Human: {name: 'Owner'} };
        async function loadProfiles() {
            try {
                const res = await fetch('/api/profiles');
                if (res.ok) currentProfiles = await res.json();
            } catch(e) {}
        }
        loadProfiles();

        function openPetModal() {
            document.getElementById('dog-name').value = currentProfiles.Dog.name || '';
            document.getElementById('dog-breed').value = currentProfiles.Dog.breed || '';
            document.getElementById('dog-bday').value = currentProfiles.Dog.birthday || '';
            document.getElementById('dog-weight').value = currentProfiles.Dog.weight || '';
            document.getElementById('cat-name').value = currentProfiles.Cat.name || '';
            document.getElementById('human-name').value = currentProfiles.Human.name || '';
            document.getElementById('pet-modal').classList.remove('hidden');
        }
        function closePetModal() { document.getElementById('pet-modal').classList.add('hidden'); }
        async function savePetProfiles() {
            const payload = {
                Dog: { name: document.getElementById('dog-name').value, breed: document.getElementById('dog-breed').value, birthday: document.getElementById('dog-bday').value, weight: parseFloat(document.getElementById('dog-weight').value) || 0.0 },
                Cat: { name: document.getElementById('cat-name').value, birthday: '' },
                Human: { name: document.getElementById('human-name').value, birthday: '' }
            };
            await fetch('/api/profiles', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload) });
            currentProfiles = payload;
            closePetModal();
        }

        let velocityChart = null;
        try {
            const ctx = document.getElementById('velocityChart').getContext('2d');
            velocityChart = new Chart(ctx, {
                type: 'line',
                data: { labels: Array(25).fill(''), datasets: [{ data: Array(25).fill(0), borderColor: '#06b6d4', borderWidth: 2, tension: 0.3, fill: true, backgroundColor: 'rgba(6, 182, 212, 0.08)', pointRadius: 0 }] },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { display: false }, y: { min: 0, max: 5 } } }
            });
        } catch(e) {}

        function setTurretMode(mode) {
            updateModeButtonsUI(mode);
            fetch('/api/turret/mode', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({mode}) });
        }

        function updateModeButtonsUI(mode) {
            const mBtn = document.getElementById('btn-manual');
            const aBtn = document.getElementById('btn-auto');
            const hBtn = document.getElementById('btn-hiit');
            const badge = document.getElementById('mode-badge');
            
            const activeClass = 'py-2 px-2 rounded-xl border border-cyan-500/40 bg-cyan-500/20 text-cyan-300 font-semibold text-[10px] transition active:scale-95';
            const inactiveClass = 'py-2 px-2 rounded-xl border border-slate-700 bg-slate-800 text-slate-400 font-semibold text-[10px] hover:border-slate-600 transition active:scale-95';

            mBtn.className = (mode === 'MANUAL') ? activeClass : inactiveClass;
            aBtn.className = (mode === 'AUTO_LURE') ? 'py-2 px-2 rounded-xl border border-emerald-500/40 bg-emerald-500/20 text-emerald-300 font-semibold text-[10px] transition active:scale-95' : inactiveClass;
            hBtn.className = (mode === 'HIIT_ROUTINE') ? 'py-2 px-2 rounded-xl border border-purple-500/40 bg-purple-500/20 text-purple-300 font-semibold text-[10px] transition active:scale-95' : inactiveClass;
            badge.innerText = mode === 'AUTO_LURE' ? 'AUTO LEAD' : (mode === 'HIIT_ROUTINE' ? 'ADAPTIVE HIIT' : 'MANUAL MODE');
        }

        function setCameraMode(camMode) {
            fetch('/api/camera/mode', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({camera_mode: camMode}) });
        }

        async function toggleSecurityMode() {
            const res = await fetch('/api/security/toggle', { method: 'POST' });
            const data = await res.json();
            updateSecurityUI(data.security_mode);
        }

        async function forceDisarmSecurity() {
            await fetch('/api/security/disarm', { method: 'POST' });
            updateSecurityUI(false);
        }

        async function toggleGestureSystem() {
            const res = await fetch('/api/gesture/toggle', { method: 'POST' });
            const data = await res.json();
            updateGestureButtonUI(data.gesture_detection_enabled);
        }

        function updateGestureButtonUI(isEnabled) {
            const btn = document.getElementById('gesture-toggle-btn');
            if (isEnabled) {
                btn.innerHTML = '<i data-lucide="sparkles" class="w-4 h-4"></i> GESTURE DETECTION: ACTIVE (ON)';
                btn.className = 'w-full py-2.5 px-3 rounded-xl border border-emerald-500/40 bg-emerald-500/20 text-emerald-300 font-extrabold text-xs transition active:scale-95 flex items-center justify-center gap-1.5';
            } else {
                btn.innerHTML = '<i data-lucide="eye-off" class="w-4 h-4"></i> GESTURE DETECTION: DISABLED (OFF)';
                btn.className = 'w-full py-2.5 px-3 rounded-xl border border-slate-700 bg-slate-800/80 text-slate-400 font-bold text-xs transition active:scale-95 flex items-center justify-center gap-1.5';
            }
            try { if (typeof lucide !== 'undefined') lucide.createIcons(); } catch(e){}
        }

        function updateSecurityUI(isActive) {
            const secBtn = document.getElementById('sec-btn');
            if (isActive) {
                secBtn.innerHTML = '<i data-lucide="shield-alert" class="w-4 h-4 text-black"></i> ARMED 🚨';
                secBtn.className = 'w-full py-2.5 px-3 rounded-xl border border-red-500 bg-red-600 text-black font-extrabold text-xs transition active:scale-95 flex items-center justify-center gap-1.5 animate-pulse';
            } else {
                secBtn.innerHTML = '<i data-lucide="shield" class="w-4 h-4"></i> ARM (🤙)';
                secBtn.className = 'w-full py-2.5 px-3 rounded-xl border border-slate-700 bg-slate-800 text-slate-300 font-bold text-xs transition active:scale-95 flex items-center justify-center gap-1.5';
            }
            try { if (typeof lucide !== 'undefined') lucide.createIcons(); } catch(e){}
        }

        function updateAimText() { 
            document.getElementById('angle-val').innerText = `${document.getElementById('angle-slider').value}°`; 
            document.getElementById('dist-val').innerText = `${document.getElementById('dist-slider').value} FT`; 
        }
        async function updateAim() { 
            await fetch('/api/turret/aim', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({angle: document.getElementById('angle-slider').value, distance: document.getElementById('dist-slider').value}) }); 
        }
        async function fireTurret() { await fetch('/api/turret/fire', { method: 'POST' }); }
        async function refillTurret() { await fetch('/api/turret/refill', { method: 'POST' }); }
        async function resetStats() { if(confirm("Reset metrics?")) await fetch('/api/reset_stats', { method: 'POST' }); }

        async function dismissWaterAlert() {
            await fetch('/api/hydration/dismiss', { method: 'POST' });
            document.getElementById('water-alert-banner').classList.add('hidden');
        }

        let wasPreLaunch = false;
        let securityActiveState = false;
        let lastGestureState = true;

        async function syncTelemetry() {
            try {
                const res = await fetch('/api/telemetry');
                if (!res.ok) return;
                const data = await res.json();
                
                const secRes = await fetch('/api/security/status');
                if (secRes.ok) {
                    const secData = await secRes.json();
                    if (secData.alert) playBuzzerTone(1100, 0.4, 'sawtooth');
                }

                const recRes = await fetch('/api/recall/status');
                if (recRes.ok) {
                    const recData = await recRes.json();
                    if (recData.alert) playRecallWhistle();
                }

                if (data.security_mode !== securityActiveState) {
                    securityActiveState = data.security_mode;
                    updateSecurityUI(securityActiveState);
                }

                if (data.gesture_detection_enabled !== lastGestureState) {
                    lastGestureState = data.gesture_detection_enabled;
                    updateGestureButtonUI(lastGestureState);
                }
                
                const waterModal = document.getElementById('water-alert-banner');
                if (data.hydration_alert) waterModal.classList.remove('hidden');

                document.getElementById('hiit-tier-display').innerText = data.hiit_tier;
                document.getElementById('hiit-rest-display').innerText = data.hiit_adaptive_rest_sec.toFixed(1);

                const gestureEl = document.getElementById('gesture-display');
                if (gestureEl) {
                    if (!data.gesture_detection_enabled) gestureEl.innerText = 'OFF';
                    else if (data.active_gesture === 'SHAKA_SECURITY') gestureEl.innerText = '🤙 SHAKA SEC';
                    else gestureEl.innerText = data.active_gesture;
                }

                const vidContainer = document.getElementById('video-container');
                const camLabel = document.getElementById('camera-label');
                const fireBtn = document.getElementById('btn-fire');
                
                const camLaptopBtn = document.getElementById('cam-btn-laptop');
                const camExtendedBtn = document.getElementById('cam-btn-extended');
                const cam360Btn = document.getElementById('cam-btn-360');
                
                const activeCamClass = 'py-2 px-1 rounded-xl border border-cyan-500/40 bg-cyan-500/20 text-cyan-300 font-semibold text-[10px] transition active:scale-95';
                const inactiveCamClass = 'py-2 px-1 rounded-xl border border-slate-700 bg-slate-800 text-slate-400 font-semibold text-[10px] hover:border-slate-600 transition active:scale-95';

                camLaptopBtn.className = inactiveCamClass;
                camExtendedBtn.className = inactiveCamClass;
                cam360Btn.className = inactiveCamClass;

                if (data.camera_mode === '360_PANORAMIC') cam360Btn.className = activeCamClass;
                else if (data.camera_mode === 'EXTENDED') camExtendedBtn.className = activeCamClass;
                else camLaptopBtn.className = activeCamClass;

                if (data.treats_remaining <= 0) {
                    fireBtn.innerHTML = '<i data-lucide="x-circle" class="w-5 h-5 fill-slate-800"></i> HOPPER EMPTY';
                    fireBtn.className = 'w-full py-4 px-4 bg-slate-800 text-slate-500 font-extrabold rounded-xl transition flex items-center justify-center gap-2 cursor-not-allowed';
                    fireBtn.disabled = true;
                    try { if (typeof lucide !== 'undefined') lucide.createIcons(); } catch(e){}
                } else if (data.is_pre_launch && !wasPreLaunch) {
                    playBuzzerTone(440, 0.4, 'square');
                    fireBtn.innerHTML = '<i data-lucide="alert-triangle" class="w-5 h-5 fill-black"></i> STAND CLEAR - LAUNCHING';
                    fireBtn.className = 'w-full py-4 px-4 bg-red-500 text-black font-extrabold rounded-xl transition shadow-lg shadow-red-500/50 flex items-center justify-center gap-2 animate-pulse';
                    fireBtn.disabled = true;
                    try { if (typeof lucide !== 'undefined') lucide.createIcons(); } catch(e){}
                } else if (!data.is_pre_launch && (wasPreLaunch || fireBtn.disabled)) {
                    fireBtn.innerHTML = '<i data-lucide="zap" class="w-5 h-5 fill-black"></i> INITIATE LAUNCH';
                    fireBtn.className = 'w-full py-4 px-4 bg-gradient-to-r from-amber-500 to-orange-500 text-black font-extrabold rounded-xl transition shadow-lg shadow-amber-500/20 flex items-center justify-center gap-2 active:scale-95';
                    fireBtn.disabled = false;
                    try { if (typeof lucide !== 'undefined') lucide.createIcons(); } catch(e){}
                }
                wasPreLaunch = data.is_pre_launch;

                if (data.hydration_alert) {
                    vidContainer.className = 'glass rounded-2xl overflow-hidden relative border border-cyan-400 glow-water transition-all duration-300';
                    camLabel.innerHTML = '<span class="text-cyan-400 font-bold">HYDRATION REMINDER ACTIVE 💧</span>';
                } else if (data.is_pre_launch) {
                    vidContainer.className = 'glass rounded-2xl overflow-hidden relative border border-orange-500 glow-orange transition-all duration-300';
                    camLabel.innerHTML = '<span class="text-orange-400 font-bold">PAVLOVIAN PROTOCOL ACTIVE</span>';
                } else if (data.security_mode) {
                    vidContainer.className = 'glass rounded-2xl overflow-hidden relative border border-red-500 glow-red transition-all duration-300';
                    camLabel.innerHTML = '<span class="text-red-400 font-bold">SURVEILLANCE SECURITY ARMED</span>';
                } else {
                    vidContainer.className = 'glass rounded-2xl overflow-hidden relative border border-slate-800 glow-cyan transition-all duration-300';
                    if (data.camera_mode === '360_PANORAMIC') camLabel.innerText = '360° PANORAMIC FEED';
                    else if (data.camera_mode === 'EXTENDED') camLabel.innerText = 'EXTENDED USB CAMERA';
                    else camLabel.innerText = 'LAPTOP WEBCAM FEED';
                }

                document.getElementById('metric-top-speed').innerText = data.top_sprint_mps;
                document.getElementById('metric-avg-speed').innerText = data.avg_sprint_mps;
                document.getElementById('metric-min-speed').innerText = data.min_sprint_mps === 99.9 ? '0.0' : data.min_sprint_mps;
                document.getElementById('metric-stride').innerText = data.current_stride_length_m.toFixed(2);
                
                document.getElementById('last-lap').innerText = data.last_lap_sec;
                if (data.fastest_lap_sec !== 99.9) {
                    document.getElementById('fastest-lap').innerText = data.fastest_lap_sec;
                    const unitEl = document.getElementById('fastest-lap-unit');
                    if (unitEl && unitEl.classList.contains('hidden')) unitEl.classList.remove('hidden');
                }

                document.getElementById('reaction-time').innerText = data.cognitive_reaction_time_sec;
                const rxBaseDisplay = document.getElementById('reaction-baseline-display');
                if (rxBaseDisplay) rxBaseDisplay.innerText = `Expected: ${data.expected_reaction_sec}s`;

                document.getElementById('treat-remaining').innerText = data.treats_remaining;
                const treatsDispensedEl = document.getElementById('treats-dispensed');
                if (treatsDispensedEl) treatsDispensedEl.innerText = data.treats_dispensed_today;
                
                const trendVal = data.limp_trend;
                const trendEl = document.getElementById('limp-trend-val');
                if (trendEl) trendEl.innerText = (trendVal > 0 ? "+" : "") + trendVal + "%";

                const fallCountEl = document.getElementById('fall-count-val');
                if (fallCountEl) fallCountEl.innerText = data.fall_count;

                const limpBox = document.getElementById('limp-alert-box');
                const statusBadge = document.getElementById('ortho-status-badge');
                
                if (data.limp_delta <= -12.0) {
                    limpBox.className = 'glass rounded-2xl p-5 border border-red-500 bg-red-500/10 text-center glow-red transition-all duration-500 flex flex-col justify-center items-center';
                    statusBadge.className = 'px-3 py-1 rounded-full text-xs font-extrabold uppercase tracking-wider bg-red-500/20 text-red-400 border border-red-500/40 animate-pulse';
                    statusBadge.innerText = 'Critical // Take to Doctor';
                } else if (data.limp_delta < -5.0) {
                    limpBox.className = 'glass rounded-2xl p-5 border border-amber-500/30 bg-amber-500/10 text-center transition-all duration-500 flex flex-col justify-center items-center';
                    statusBadge.className = 'px-3 py-1 rounded-full text-xs font-extrabold uppercase tracking-wider bg-amber-500/20 text-amber-400 border border-amber-500/40';
                    statusBadge.innerText = 'Sub-Clinical';
                } else {
                    limpBox.className = 'glass rounded-2xl p-5 border border-slate-800 text-center transition-all duration-500 flex flex-col justify-center items-center';
                    statusBadge.className = 'px-3 py-1 rounded-full text-xs font-extrabold uppercase tracking-wider bg-emerald-500/20 text-emerald-400 border border-emerald-500/30';
                    statusBadge.innerText = 'Normal';
                }
                
                const activeBaseDisplay = document.getElementById('active-baseline-display');
                if (activeBaseDisplay) activeBaseDisplay.innerText = `Base: ${data.active_baseline_mps} m/s`;

                const avgSpeedExpected = document.getElementById('avg-speed-expected');
                if (avgSpeedExpected) {
                    avgSpeedExpected.innerText = `Expected: ${data.active_baseline_mps} m/s`;
                    if (data.avg_sprint_mps > 0 && data.avg_sprint_mps < (data.active_baseline_mps - 0.5)) {
                        avgSpeedExpected.className = 'text-[9px] text-red-400 mono mt-1 font-bold';
                    } else {
                        avgSpeedExpected.className = 'text-[9px] text-slate-500 mono mt-1';
                    }
                }

                if (velocityChart) {
                    velocityChart.data.datasets[0].data.push(data.current_velocity_mps);
                    velocityChart.data.datasets[0].shift();
                    velocityChart.update('none');
                }
            } catch (e) {}
        }
        setInterval(syncTelemetry, 250);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

# ==========================================
# 10. SYSTEM SHUTDOWN & ENTRY POINT
# ==========================================
def cleanup_resources():
    global running, camera_cap, arduino
    print("\n[+] Initiating graceful hardware shutdown...")
    running = False
    
    if camera_cap is not None:
        try:
            camera_cap.release()
            print("[+] Camera hardware released.")
        except Exception:
            pass
            
    if arduino and arduino.is_open:
        try:
            arduino.close()
            print("[+] Arduino serial port closed.")
        except Exception:
            pass

atexit.register(cleanup_resources)

def handle_sigint(sig, frame):
    cleanup_resources()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)
signal.signal(signal.SIGTERM, handle_sigint)

if __name__ == '__main__':
    init_db()
    load_db_stats()
    init_arduino()
    
    # Start Vision & AI Threads
    threading.Thread(target=vision_thread_loop, daemon=True).start()
    threading.Thread(target=mediapipe_worker_thread, daemon=True).start()
    
    # Start Background Subsystems
    threading.Thread(target=db_writer_thread, daemon=True).start()
    threading.Thread(target=system_monitor_thread, daemon=True).start()
    threading.Thread(target=turret_logic_thread, daemon=True).start()
    
    if AUDIO_ENABLED:
        threading.Thread(target=audio_monitor_thread, daemon=True).start()

    print("[+] PitPaws Titan Station active on http://0.0.0.0:5001")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
