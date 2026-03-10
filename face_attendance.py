import cv2
import face_recognition
import os
import numpy as np
import requests
import json
import time
from datetime import datetime
from flask import Flask, Response, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configuration (Use Environment Variables for Cloud Deployment)
BASE_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1/MP/backend/")
ATTENDANCE_API = BASE_URL + "mark_attendance.php"
PRISONER_LIST_API = BASE_URL + "get_prisoners_list.php"
PRISONER_UPLOADS = os.environ.get("UPLOADS_PATH", "../uploads/prisoners/")
IS_RENDER = os.environ.get("RENDER", "false").lower() == "true"

# Global state
current_shift = "General"
current_type = "Entry"
last_recognition_time = {}
known_encodings = []
known_ids = []
known_names = {}

def normalize_id(pid):
    """Normalize ID by removing common prefixes like 'P-2026-' to match DB keys"""
    if pid.startswith("P-"):
        # e.g., "P-2026-3248" -> "2026-3248" or "3248" depending on DB
        # If DB has "26-3248", we might need more aggressive stripping
        parts = pid.split('-')
        if len(parts) >= 2:
            return "-".join(parts[1:]) # "2026-3248"
    return pid

def get_best_name(pid):
    """Try to find the name using full ID, normalized ID, or partial ID"""
    if pid in known_names:
        return known_names[pid]
    
    norm = normalize_id(pid)
    if norm in known_names:
        return known_names[norm]
        
    # Check if DB has "26-3248" but folder is "P-2026-3248"
    short_norm = norm[2:] if len(norm) > 2 else norm
    if short_norm in known_names:
        return known_names[short_norm]
        
    return "Unknown"

def load_face_data():
    global known_encodings, known_ids, known_names
    print("\n--- RELOADING FACE DATABASE ---")
    known_encodings = []
    known_ids = []
    known_names = {}
    
    # Fetch names with retries
    for attempt in range(3):
        try:
            res = requests.get(PRISONER_LIST_API, timeout=5)
            if res.status_code == 200:
                known_names = res.json()
                print(f"Successfully fetched {len(known_names)} prisoner names.")
                break
        except Exception as e:
            print(f"Attempt {attempt+1}: Could not fetch prisoner names ({e})")
            time.sleep(1)

    # Sync faces from cloud if deployed
    print("Checking for missing face images from cloud...")
    try:
        faces_res = requests.get(BASE_URL + "get_prisoner_faces.php", timeout=5)
        if faces_res.status_code == 200:
            faces_data = faces_res.json()
            for pid, photo_path in faces_data.items():
                local_dir = os.path.join(PRISONER_UPLOADS, pid)
                os.makedirs(local_dir, exist_ok=True)
                local_path = os.path.join(local_dir, "front.jpg")
                if not os.path.exists(local_path):
                    print(f"Downloading missing face for {pid}...")
                    clean_path = photo_path.replace("../", "")
                    img_url = BASE_URL.replace("backend/", "") + clean_path
                    try:
                        img_res = requests.get(img_url, timeout=10)
                        if img_res.status_code == 200:
                            with open(local_path, "wb") as f:
                                f.write(img_res.content)
                        else:
                            print(f"Failed to fetch {img_url} - Status: {img_res.status_code}")
                    except Exception as e:
                        print(f"Error downloading {img_url}: {e}")
    except Exception as e:
        print(f"Could not sync face list from cloud: {e}")

    if not os.path.exists(PRISONER_UPLOADS):
        print(f"CRITICAL: Uploads directory not found at {PRISONER_UPLOADS}")
        return

    for prisoner_id in os.listdir(PRISONER_UPLOADS):
        prisoner_dir = os.path.join(PRISONER_UPLOADS, prisoner_id)
        if os.path.isdir(prisoner_dir):
            for img_file in os.listdir(prisoner_dir):
                if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(prisoner_dir, img_file)
                    try:
                        image = face_recognition.load_image_file(img_path)
                        encs = face_recognition.face_encodings(image, num_jitters=1)
                        if len(encs) > 0:
                            known_encodings.append(encs[0])
                            known_ids.append(prisoner_id) # Store original folder name as ID
                    except Exception as e:
                        print(f"Error encoding {img_path}: {e}")
    print(f"Face database updated. Encoded {len(known_ids)} samples.\n")

load_face_data()

def mark_attendance(prisoner_id, recognition_date=None, recognition_time=None):
    try:
        # Before sending, we should really send the ID that the DB expects
        # If DB expects normalized ID, we send that.
        
        target_id = prisoner_id
        norm = normalize_id(prisoner_id)
        if norm in known_names:
            target_id = norm
        else:
            short_norm = norm[2:] if len(norm) > 2 else norm
            if short_norm in known_names:
                target_id = short_norm

        data = {
            "prisoner_id": target_id,
            "shift_name": current_shift,
            "movement_type": current_type,
            "attendance_date": recognition_date,
            "time_in": recognition_time
        }
        print(f"Sending Attendance: {target_id} | {current_shift} | {current_type} | {recognition_time}")
        response = requests.post(ATTENDANCE_API, json=data, timeout=5)
        result = response.json()
        print(f"API Response: {result.get('status')} - {result.get('message')}")
        return result.get('status') in ['success', 'info']
    except Exception as e:
        print(f"Error calling API for {prisoner_id}: {e}")
        return False

def gen_frames():
    """Generator for video frames with robust camera handling"""
    # Try different backends and indices
    camera = None
    if IS_RENDER:
        print("INFO: Running on Render (Headless). Skipping camera initialization.")
        while True:
            err_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(err_frame, "RENDER CLOUD: NO CAMERA ACCESS", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(err_frame, "This node acts as a Recognition API.", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            ret, buffer = cv2.imencode('.jpg', err_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(5)

    for attempt in [ (0, cv2.CAP_DSHOW), (0, None), (1, cv2.CAP_DSHOW), (1, None) ]:
        idx, backend = attempt
        print(f"DEBUG: Attempting to open camera {idx} with backend {backend}...")
        if backend:
            camera = cv2.VideoCapture(idx, backend)
        else:
            camera = cv2.VideoCapture(idx)
            
        if camera.isOpened():
            print(f"DEBUG: Camera {idx} opened successfully!")
            break
        else:
            camera.release()
            camera = None

    if camera is None or not camera.isOpened():
        print("CRITICAL: All camera attempts failed.")
        # Return a "No Camera" placeholder frame if possible, or just exit
        while True:
            err_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(err_frame, "CAMERA ERROR: CANNOT OPEN DEVICE", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(err_frame, "Ensure camera is connected and not in use.", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            ret, buffer = cv2.imencode('.jpg', err_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1)

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("ERROR: Lost camera frame. Attempting to recover...")
                time.sleep(0.5)
                continue
            
            # Resize for speed/accuracy balance
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
                prisoner_id = "Unknown"
                prisoner_name = "Unknown"
                min_dist = 1.0

                if len(known_encodings) > 0:
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    min_dist = face_distances[best_match_index]
                    
                    if matches[best_match_index]:
                        raw_id = known_ids[best_match_index]
                        prisoner_name = get_best_name(raw_id)
                        prisoner_id = raw_id
                        print(f"DEBUG: Recognized {prisoner_name} ({prisoner_id}) dist: {min_dist:.2f}")

                # Mark Attendance
                if prisoner_name != "Unknown":
                    now = datetime.now()
                    current_time_val = time.time()
                    throttle_key = f"{prisoner_id}_{current_shift}_{current_type}"
                    if throttle_key not in last_recognition_time or (current_time_val - last_recognition_time[throttle_key] > 10):
                        rec_date = now.strftime("%Y-%m-%d")
                        rec_time = now.strftime("%H:%M:%S")
                        if mark_attendance(prisoner_id, rec_date, rec_time):
                            last_recognition_time[throttle_key] = current_time_val

                # Draw Overlay
                top, right, bottom, left = [v * 2 for v in face_location]
                off = 50
                t, r, b, l = top-off, right+off, bottom+off, left-off
                color = (0, 255, 0) if prisoner_name != "Unknown" else (0, 0, 255)
                
                # Brackets
                ln, th = 50, 4
                cv2.line(frame, (l, t), (l+ln, t), color, th)
                cv2.line(frame, (l, t), (l, t+ln), color, th)
                cv2.line(frame, (r, t), (r-ln, t), color, th)
                cv2.line(frame, (r, t), (r, t+ln), color, th)
                cv2.line(frame, (l, b), (l+ln, b), color, th)
                cv2.line(frame, (l, b), (l, b-ln), color, th)
                cv2.line(frame, (r, b), (r-ln, b), color, th)
                cv2.line(frame, (r, b), (r, b-ln), color, th)

                # HUD Text
                if prisoner_name != "Unknown":
                    cv2.putText(frame, prisoner_name.upper(), (l+10, b-25), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1)
                    cv2.putText(frame, f"ID: {prisoner_id} ({min_dist:.2f})", (l+10, b-10), cv2.FONT_HERSHEY_DUPLEX, 0.35, color, 1)
                else:
                    cv2.putText(frame, f"SCANNING... ({min_dist:.2f})", (l+10, t+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        if camera:
            camera.release()
            print("DEBUG: Camera released.")

@app.route('/ping', methods=['GET', 'POST'])
def ping():
    return json.dumps({"status": "online", "current_shift": current_shift, "current_type": current_type})

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_context', methods=['POST'])
def set_context():
    global current_shift, current_type
    data = request.json or {}
    print(f"DEBUG: Setting Context -> {data}")
    if 'shift' in data: current_shift = data.get('shift')
    if 'type' in data: current_type = data.get('type')
    return json.dumps({"status": "success"})

@app.route('/recognize_image', methods=['POST'])
def recognize_image():
    """API endpoint to process a single image from a remote client"""
    if 'image' not in request.files:
        return json.dumps({"status": "error", "message": "No image provided"}), 400
    
    file = request.files['image']
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    results = []
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
        prisoner_id = "Unknown"
        prisoner_name = "Unknown"
        
        if len(known_encodings) > 0:
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                raw_id = known_ids[best_match_index]
                prisoner_id = raw_id
                prisoner_name = get_best_name(raw_id)
                # Mark attendance if found
                now = datetime.now()
                mark_attendance(prisoner_id, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"))
        
        results.append({
            "prisoner_id": prisoner_id,
            "prisoner_name": prisoner_name,
            "location": face_location
        })
    
    return json.dumps({"status": "success", "results": results})

@app.route('/reload_data', methods=['POST', 'GET'])
def reload_data():
    load_face_data()
    return json.dumps({"status": "success", "count": len(known_ids)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
