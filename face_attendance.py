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

# Configuration
BASE_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1/MP/backend/")
ATTENDANCE_API = BASE_URL + "mark_attendance.php"
PRISONER_LIST_API = BASE_URL + "get_prisoners_list.php"
PRISONER_UPLOADS = os.environ.get("UPLOADS_PATH", "../uploads/prisoners/")
IS_RENDER = os.environ.get("RENDER", "false").lower() == "true"

# Global state
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
current_shift = "General"
current_type = "Entry"
last_recognition_time = {}
known_encodings = []
known_ids = []
known_names = {}
load_status = {"names": 0, "images": 0, "errors": []}

def normalize_id(pid):
    if pid.startswith("P-"):
        parts = pid.split('-')
        if len(parts) >= 2:
            return "-".join(parts[1:]) 
    return pid

def get_best_name(pid):
    if pid in known_names: return known_names[pid]
    norm = normalize_id(pid)
    if norm in known_names: return known_names[norm]
    return "Unknown"

def load_face_data():
    global known_encodings, known_ids, known_names, load_status
    print("\n--- RELOADING FACE DATABASE ---")
    known_encodings = []
    known_ids = []
    known_names = {}
    load_status = {"names": 0, "images": 0, "errors": []}
    
    # 1. Fetch names
    try:
        res = requests.get(PRISONER_LIST_API, headers=HEADERS, timeout=10)
        if res.status_code == 200:
            known_names = res.json()
            load_status["names"] = len(known_names)
            print(f"Fetched {len(known_names)} names.")
        else:
            load_status["errors"].append(f"Names API error: {res.status_code}")
    except Exception as e:
        load_status["errors"].append(f"Names Fetch Error: {str(e)}")
        print(f"Error fetching names: {e}")

    # 2. Prepare Uploads Dir
    if not os.path.exists(PRISONER_UPLOADS):
        os.makedirs(PRISONER_UPLOADS, exist_ok=True)

    # 3. Sync images from InfinityFree
    print("Syncing images from Cloud Backend...")
    try:
        faces_api = BASE_URL + "get_prisoner_faces.php"
        faces_res = requests.get(faces_api, headers=HEADERS, timeout=10)
        if faces_res.status_code == 200:
            try:
                faces_map = faces_res.json()
                for pid, photo_path in faces_map.items():
                    clean_path = photo_path.replace("../", "")
                    img_url = BASE_URL.replace("backend/", "") + clean_path
                    
                    pid_dir = os.path.join(PRISONER_UPLOADS, pid)
                    os.makedirs(pid_dir, exist_ok=True)
                    local_img_path = os.path.join(pid_dir, "front.jpg")
                    
                    if not os.path.exists(local_img_path):
                        print(f"Downloading {pid}...")
                        img_data = requests.get(img_url, headers=HEADERS, timeout=10).content
                        with open(local_img_path, "wb") as f:
                            f.write(img_data)
                    load_status["images"] += 1
            except Exception as json_err:
                load_status["errors"].append(f"Invalid JSON from Faces API: {str(json_err)}")
        else:
            load_status["errors"].append(f"Faces API error: {faces_res.status_code}")
    except Exception as e:
        load_status["errors"].append(f"Sync Error: {str(e)}")
        print(f"Sync failed: {e}")

    # 4. Load into Memory
    for prisoner_id in os.listdir(PRISONER_UPLOADS):
        prisoner_dir = os.path.join(PRISONER_UPLOADS, prisoner_id)
        if os.path.isdir(prisoner_dir):
            for img_file in os.listdir(prisoner_dir):
                if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(prisoner_dir, img_file)
                    try:
                        image = face_recognition.load_image_file(img_path)
                        encs = face_recognition.face_encodings(image)
                        for enc in encs:
                            known_encodings.append(enc)
                            known_ids.append(prisoner_id)
                    except: pass
    print(f"Face database ready: {len(known_ids)} encodings loaded.")

load_face_data()

def mark_attendance(prisoner_id, rec_date, rec_time):
    try:
        data = {
            "prisoner_id": normalize_id(prisoner_id),
            "shift_name": current_shift,
            "movement_type": current_type,
            "attendance_date": rec_date,
            "time_in": rec_time
        }
        requests.post(ATTENDANCE_API, json=data, headers=HEADERS, timeout=5)
        return True
    except Exception as e:
        print(f"Failed to mark attendance: {e}")
        return False

@app.route('/recognize_image', methods=['POST'])
def recognize_image():
    if 'image' not in request.files: return json.dumps({"status": "error", "message": "No image"}), 400
    file = request.files['image']
    img = face_recognition.load_image_file(file)
    
    # 1. Detect faces in current frame
    face_locations = face_recognition.face_locations(img)
    encodings = face_recognition.face_encodings(img, face_locations)
    
    print(f"Server: Analyzing frame. Found {len(encodings)} faces.")
    
    results_list = []
    now = datetime.now()
    now_str_date = now.strftime("%Y-%m-%d")
    now_str_time = now.strftime("%H:%M:%S")
    timestamp = time.time()

    for encoding in encodings:
        if not known_encodings: break
        
        # Calculate distances to all known faces
        face_distances = face_recognition.face_distance(known_encodings, encoding)
        best_match_index = np.argmin(face_distances)
        
        # Stricter tolerance (0.5 instead of 0.6)
        if face_distances[best_match_index] < 0.5:
            p_id = known_ids[best_match_index]
            
            # 30-second Cooldown to prevent duplicate logs in same session
            last_time = last_recognition_time.get(p_id, 0)
            if (timestamp - last_time) > 30:
                print(f"Server: BEST MATCH FOUND -> {p_id} (Dist: {face_distances[best_match_index]:.2f})")
                mark_attendance(p_id, now_str_date, now_str_time)
                last_recognition_time[p_id] = timestamp
                results_list.append({"prisoner_id": p_id, "prisoner_name": get_best_name(p_id)})
            else:
                print(f"Server: Match skipped (cooldown active) -> {p_id}")
        else:
            print(f"Server: Face detected but distance too high ({np.min(face_distances):.2f})")
            
    return json.dumps({"status": "success", "results": results_list, "faces_count": len(encodings)})

@app.route('/sync_prisoner', methods=['POST'])
def sync_prisoner():
    pid = request.form.get('id')
    name = request.form.get('name')
    file = request.files.get('image')
    
    if not pid or not file: 
        return json.dumps({"status": "error", "message": "Missing ID or Image"}), 400
    
    try:
        # 1. Save locally for persistence
        pid_dir = os.path.join(PRISONER_UPLOADS, pid)
        os.makedirs(pid_dir, exist_ok=True)
        img_path = os.path.join(pid_dir, "front.jpg")
        file.save(img_path)
        
        # 2. Update Names Mapping
        known_names[pid] = name
        
        # 3. Process Encoding
        image = face_recognition.load_image_file(img_path)
        encs = face_recognition.face_encodings(image)
        
        if len(encs) > 0:
            # Remove old encodings for this specific ID to prevent duplicates
            indices = [i for i, x in enumerate(known_ids) if x == pid]
            for i in reversed(indices):
                known_encodings.pop(i)
                known_ids.pop(i)
                
            for enc in encs:
                known_encodings.append(enc)
                known_ids.append(pid)
            
            return json.dumps({"status": "success", "samples": len(encs)})
        else:
            return json.dumps({"status": "error", "message": "No face found in synced image"}), 422
            
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}), 500

@app.route('/set_context', methods=['POST'])
def set_context():
    global current_shift, current_type
    try:
        data = request.get_json()
        current_shift = data.get('shift', 'General')
        current_type = data.get('type', 'Entry')
        print(f"Context updated: {current_shift} - {current_type}")
        return json.dumps({"status": "success"})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}), 400

@app.route('/reload_data', methods=['POST', 'GET'])
def reload_data():
    load_face_data()
    return json.dumps({
        "status": "success", 
        "count": len(known_ids),
        "names_found": len(known_names),
        "images_local": load_status["images"],
        "errors": load_status["errors"]
    })

@app.route('/video_feed')
def video_feed():
    # Cloud version doesn't server camera feed, returning empty to avoid 404
    return "Cloud server cannot serve camera feed directly. Use the Web Scanner client.", 200

@app.route('/ping', methods=['GET', 'POST'])
def ping():
    return json.dumps({"status": "online"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
