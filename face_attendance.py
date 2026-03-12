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
current_shift = "General"
current_type = "Entry"
last_recognition_time = {}
known_encodings = []
known_ids = []
known_names = {}

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
    global known_encodings, known_ids, known_names
    print("\n--- RELOADING FACE DATABASE ---")
    known_encodings = []
    known_ids = []
    known_names = {}
    
    # 1. Fetch names
    try:
        res = requests.get(PRISONER_LIST_API, timeout=5)
        if res.status_code == 200:
            known_names = res.json()
            print(f"Fetched {len(known_names)} names.")
    except Exception as e:
        print(f"Error fetching names: {e}")

    # 2. Prepare Uploads Dir
    if not os.path.exists(PRISONER_UPLOADS):
        os.makedirs(PRISONER_UPLOADS, exist_ok=True)

    # 3. Sync images from InfinityFree (Crucial for Cloud Deployment)
    print("Syncing images from Cloud Backend...")
    try:
        # We assume get_prisoner_faces.php returns { "P-ID": "path/to/img.jpg", ... }
        faces_api = BASE_URL + "get_prisoner_faces.php"
        faces_res = requests.get(faces_api, timeout=10)
        if faces_res.status_code == 200:
            faces_map = faces_res.json()
            for pid, photo_path in faces_map.items():
                # photo_path might be "../uploads/prisoners/P-ID/img.jpg"
                clean_path = photo_path.replace("../", "")
                img_url = BASE_URL.replace("backend/", "") + clean_path
                
                pid_dir = os.path.join(PRISONER_UPLOADS, pid)
                os.makedirs(pid_dir, exist_ok=True)
                local_img_path = os.path.join(pid_dir, "front.jpg")
                
                if not os.path.exists(local_img_path):
                    print(f"Downloading {pid}...")
                    img_data = requests.get(img_url, timeout=10).content
                    with open(local_img_path, "wb") as f:
                        f.write(img_data)
        else:
            print(f"Could not reach faces API: {faces_res.status_code}")
    except Exception as e:
        print(f"Sync failed (InfinityFree Security?) : {e}")

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
        requests.post(ATTENDANCE_API, json=data, timeout=5)
        return True
    except: return False

@app.route('/recognize_image', methods=['POST'])
def recognize_image():
    if 'image' not in request.files: return json.dumps({"status": "error", "message": "No image"}), 400
    file = request.files['image']
    img = face_recognition.load_image_file(file)
    encodings = face_recognition.face_encodings(img)
    
    results_list = []
    for encoding in encodings:
        results = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.6)
        if True in results:
            idx = results.index(True)
            p_id = known_ids[idx]
            now = datetime.now()
            mark_attendance(p_id, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"))
            results_list.append({"prisoner_id": p_id, "prisoner_name": get_best_name(p_id)})
            
    return json.dumps({"status": "success", "results": results_list})

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
    return json.dumps({"status": "success"})

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
