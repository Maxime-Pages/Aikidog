import cv2
import numpy as np
import os
from pathlib import Path
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from deepface import DeepFace
# load deepface model
model = DeepFace.build_model("Facenet512")
def detect_faces(cap):
    last_face_pos = None  # "left", "right", or None
    disappearance_dir = None
    face_disappeared_counter = 0  # to add a delay before declaring disappearance

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        frame_height, frame_width = frame.shape[:2]
        frame_center_x = frame_width // 2
        
        try:
            results = DeepFace.extract_faces(frame, detector_backend='opencv', enforce_detection=False)
            face_found = False

            if results:
                for face in results:
                    region = face.get("facial_area", face.get("region", {}))
                    x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)

                    if w >= 0.9 * frame_width and h >= 0.9 * frame_height:
                        print("Skipped false positive (full image size).")
                        continue

                    # Determine face center
                    face_center_x = x + w // 2
                    position = "left" if face_center_x < frame_center_x else "right"

                    if last_face_pos != position:
                        print(f"Face moved to the {position}.")
                    last_face_pos = position
                    disappearance_dir = None  # reset disappearance direction if face is found
                    face_disappeared_counter = 0  # reset disappearance counter

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Face: {position}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    face_img = frame[y:y + h, x:x + w]
                    cv2.imshow("Face", face_img)

                    face_found = True
                    break  # Only track first valid face

            if not face_found:
                face_disappeared_counter += 1
                if face_disappeared_counter >= 5 and last_face_pos:
                    disappearance_dir = last_face_pos
                    print(f"Face disappeared to the {disappearance_dir}.")
                    last_face_pos = None  # Reset so it doesn't repeat

            cv2.imshow("Video", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Error in face detection: {e}")


def main():
    print("Starting face detection...")
    # get stream from webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    # auto focus
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_FOCUS, 0)
    # set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # auto exposure
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    
    # call function to detect faces
    detect_faces(cap)

if __name__ == "__main__":
    main()