import cv2
import sys

def main():
    # Load the pre-trained Haar cascade classifiers for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    
    # Check if the cascade files were loaded successfully
    if face_cascade.empty():
        print("Error: Could not load frontal face cascade classifier")
        return
    
    if profile_cascade.empty():
        print("Error: Could not load profile face cascade classifier")
        return
    
    # Initialize video capture (0 for default camera)
    cap = cv2.VideoCapture(0)
    # Camera Settings
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Press 'q' to quit, 's' to save current frame")
    
    frame_count = 0
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Convert frame to grayscale (face detection works better on grayscale)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect frontal faces in the frame
        frontal_faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,      # How much the image size is reduced at each scale
            minNeighbors=5,       # How many neighbors each face should have to be valid
            minSize=(20, 20),     # Minimum possible face size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Detect profile faces in the frame
        profile_faces = profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Draw rectangles around detected frontal faces
        for (x, y, w, h) in frontal_faces:
            # Draw rectangle around face (green for frontal)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add text label
            cv2.putText(frame, 'Frontal', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw rectangles around detected profile faces
        for (x, y, w, h) in profile_faces:
            # Draw rectangle around face (blue for profile)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Add text label
            cv2.putText(frame, 'Profile', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Calculate total faces detected
        total_faces = len(frontal_faces) + len(profile_faces)
        
        # Display number of faces detected
        face_count_text = f'Total faces: {total_faces} (Frontal: {len(frontal_faces)}, Profile: {len(profile_faces)})'
        cv2.putText(frame, face_count_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display FPS (optional)
        fps_text = f'Frame: {frame_count}'
        cv2.putText(frame, fps_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display the resulting frame
        cv2.imshow('Face Detection - Frontal & Profile', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            filename = f'face_detection_frame_{frame_count}.jpg'
            cv2.imwrite(filename, frame)
            print(f'Frame saved as {filename}')
        
        frame_count += 1
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Face detection stopped")

if __name__ == "__main__":
        main()