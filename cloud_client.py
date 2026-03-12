import cv2
import requests
import time
import os

# Configuration
# Replace with your Render URL after deployment
RENDER_URL = "https://facepy.onrender.com/recognize_image"
# If testing locally with the cloud-ready script running:
# RENDER_URL = "http://127.0.0.1:5000/recognize_image"

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print(f"Cloud Recognition Client Started.")
    print(f"Sending frames to: {RENDER_URL}")
    print("Press 'q' to quit.")

    last_request_time = 0
    request_interval = 2  # Send a frame every 2 seconds to avoid overloading

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            
            # Draw local preview
            cv2.putText(frame, "CLOUD RECOGNITION ACTIVE", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Local Feed', frame)

            # Send to cloud every X seconds
            if current_time - last_request_time > request_interval:
                # Convert frame to jpg
                _, img_encoded = cv2.imencode('.jpg', frame)
                
                try:
                    files = {'image': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
                    response = requests.post(RENDER_URL, files=files, timeout=5)
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get('status') == 'success':
                            for res in result.get('results', []):
                                name = res.get('prisoner_name')
                                if name != "Unknown":
                                    print(f"Recognized: {name}")
                        else:
                            print(f"Cloud Error: {result.get('message')}")
                    else:
                        print(f"Server Error: {response.status_code}")
                
                except Exception as e:
                    print(f"Network Error: {e}")
                
                last_request_time = current_time

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
