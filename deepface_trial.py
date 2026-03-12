from deepface import DeepFace
import cv2
import os

# This is a proof-of-concept for an easier, no-dlib alternative
# DeepFace is built on TensorFlow and is much easier to install on Render.

def main():
    # Load some images for a test
    # In your project, this would be your 'uploads/prisoners' folder
    db_path = "../uploads/prisoners"
    
    if not os.path.exists(db_path):
        print(f"Error: Database path {db_path} not found.")
        return

    # To recognize a person:
    # DeepFace.find(img_path = "test.jpg", db_path = "my_db")
    
    print("DeepFace is ready. It provides:")
    print("1. Recognition: DeepFace.find()")
    print("2. Verification: DeepFace.verify()")
    print("3. Analysis (Age, Gender, Emotion): DeepFace.analyze()")
    
    # Example usage (commented out as it needs actual images):
    """
    results = DeepFace.find(
        img_path = "current_frame.jpg", 
        db_path = db_path,
        model_name = "VGG-Face",      # You can choose FaceNet, OpenFace, etc.
        detector_backend = "opencv",   # NO DLIB NEEDED!
        enforce_detection = False
    )
    print(results)
    """

if __name__ == "__main__":
    main()
