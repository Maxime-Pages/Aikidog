
import cv2
import numpy as np
from deepface import DeepFace
import RPi.GPIO as GPIO
import time
from flask import Flask, render_template, Response
import threading
import base64
from io import BytesIO

# Flask app
app = Flask(__name__)

# Global variables for sharing data between threads
current_frame = None
face_data = None
servo_angle = 0
frame_lock = threading.Lock()

# GPIO Setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)
servo = GPIO.PWM(17, 50)  # 50Hz
servo.start(0)
time.sleep(0.5)

# Servo position state
current_angle = 0  # Start at center (0°)
min_angle = -90
max_angle = 45

def angle_to_duty(angle):
    # Convert angle (-90 to 90) to duty cycle (2.5 to 12.5)
    return 2.5 + ((angle + 90) / 180.0) * 10

def set_servo_angle(angle):
    global current_angle, servo_angle
    angle = max(min_angle, min(max_angle, angle))  # Clamp angle

    # Only move if the angle has changed significantly
    if abs(angle - current_angle) < 1:  # Reduced threshold for more precise tracking
        return

    print(f"Moving servo from {current_angle}° to {angle}°")
    duty = angle_to_duty(angle)
    servo.ChangeDutyCycle(duty)
    time.sleep(0.15)  # Give servo time to move
    servo.ChangeDutyCycle(0)  # Stop sending signal to reduce jitter
    current_angle = angle
    servo_angle = angle

# Move to center at start
set_servo_angle(0)
time.sleep(0.5)

def detect_faces_thread(cap):
    global current_frame, face_data

    last_face_pos = None
    face_lost_frames = 0
    scanning_mode = False
    scan_direction = 1  # 1 for right, -1 for left
    last_scan_time = 0

    # Tracking parameters
    FACE_LOST_THRESHOLD = 15  # Frames before starting scan
    SCAN_STEP = 8  # Degrees per scan step
    SCAN_DELAY = 0.4  # Seconds between scan steps
    TRACKING_SENSITIVITY = 30  # Reduced sensitivity for more precise centering

    # PID-like control parameters
    KP = 0.08  # Proportional gain - how aggressively to respond to error
    MAX_MOVEMENT = 8  # Maximum degrees to move per adjustment

    print("Face detection started. Press Ctrl+C to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame_height, frame_width = frame.shape[:2]
        frame_center_x = frame_width // 2
        face_found = False
        detected_face = None

        try:
            # Use a more reliable face detection
            results = DeepFace.extract_faces(frame, detector_backend='opencv', enforce_detection=False)

            if results:
                largest_face = None
                largest_area = 0

                # Find the largest valid face
                for face_data_item in results:
                    region = face_data_item.get("facial_area", face_data_item.get("region", {}))
                    x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)

                    # Skip obviously false positives
                    if w >= 0.8 * frame_width or h >= 0.8 * frame_height:
                        continue

                    # Skip very small detections
                    if w < 50 or h < 50:
                        continue

                    area = w * h
                    if area > largest_area:
                        largest_area = area
                        largest_face = (x, y, w, h)

                if largest_face:
                    x, y, w, h = largest_face
                    detected_face = (x, y, w, h)
                    face_center_x = x + w // 2
                    face_found = True
                    face_lost_frames = 0
                    scanning_mode = False

                    # Calculate error (how far the face is from center)
                    error = face_center_x - frame_center_x  # Positive = face is to the right

                    # Determine face position relative to center for display
                    position = "center"
                    #if error > TRACKING_SENSITIVITY:
                    #    position = "right"
                    #elif error < -TRACKING_SENSITIVITY:
                    #    position = "left"

                    # Update position tracking
                    if last_face_pos != position and position != "center":
                        print(f"Face detected on the {position}, error: {error} pixels")
                        last_face_pos = position

                    # Move servo to track face - FIXED LOGIC
                    if abs(error) > TRACKING_SENSITIVITY:
                        # Calculate servo movement with proportional control
                        # If face is to the RIGHT of center (positive error), servo needs to turn RIGHT (positive angle)
                        # If face is to the LEFT of center (negative error), servo needs to turn LEFT (negative angle)
                        angle_adjustment = error * KP  # Direct proportional relationship

                        # Limit the maximum movement per step
                        angle_adjustment = max(-MAX_MOVEMENT, min(MAX_MOVEMENT, angle_adjustment))

                        new_angle = current_angle + angle_adjustment

                        print(f"Face center: {face_center_x}, Frame center: {frame_center_x}, Error: {error}, Adjustment: {angle_adjustment:.2f}°")
                        set_servo_angle(new_angle)
                    else:
                        print(f"Face centered! Error: {error} pixels (within ±{TRACKING_SENSITIVITY})")

            if not face_found:
                face_lost_frames += 1

                # Start scanning mode after losing face for a while
                if face_lost_frames >= FACE_LOST_THRESHOLD and not scanning_mode:
                    scanning_mode = True
                    print("Face lost - starting scan mode")

                # Perform scanning
                if scanning_mode:
                    current_time = time.time()
                    if current_time - last_scan_time > SCAN_DELAY:
                        next_angle = current_angle + (scan_direction * SCAN_STEP)

                        # Reverse direction if at limits
                        if next_angle >= max_angle:
                            scan_direction = -1
                            next_angle = max_angle
                        elif next_angle <= min_angle:
                            scan_direction = 1
                            next_angle = min_angle

                        set_servo_angle(next_angle)
                        last_scan_time = current_time

                # Add scanning status to console output
                if scanning_mode and face_lost_frames % 10 == 0:  # Reduce console spam
                    print(f"Scanning... current angle: {current_angle:.1f}°")

            # Update global frame data
            with frame_lock:
                current_frame = frame.copy()
                face_data = {
                    'face_rect': detected_face,
                    'scanning': scanning_mode,
                    'servo_angle': current_angle
                }

        except KeyboardInterrupt:
            print("\nStopping face detection...")
            break
        except Exception as e:
            print(f"Error in face detection: {e}")
            time.sleep(0.1)  # Brief pause on error

def generate_frames():
    """Generate frames for Flask streaming"""
    while True:
        with frame_lock:
            if current_frame is not None:
                frame = current_frame.copy()
                face_info = face_data.copy() if face_data else None
            else:
                continue

        # Draw face rectangle if detected
        if face_info and face_info['face_rect']:
            x, y, w, h = face_info['face_rect']
            # Draw green rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Add label
            cv2.putText(frame, 'Face Detected', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Add servo angle info
        if face_info:
            angle_text = f"Servo: {face_info['servo_angle']:.1f}°"
            cv2.putText(frame, angle_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Add scanning status
            if face_info['scanning']:
                cv2.putText(frame, 'SCANNING', (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    with frame_lock:
        if face_data:
            return {
                'servo_angle': face_data['servo_angle'],
                'scanning': face_data['scanning'],
                'face_detected': face_data['face_rect'] is not None
            }
    return {'servo_angle': 0, 'scanning': False, 'face_detected': False}

def create_templates():
    """Create the HTML template"""
    import os

    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')

    # Create the HTML template
    html_content = '''<!DOCTYPE html>
<html>
<head>
    <title>Face Tracking Camera</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .video-container {
            text-align: center;
            margin: 20px 0;
        }
        .video-stream {
            border: 2px solid #333;
            border-radius: 10px;
            max-width: 100%;
            height: auto;
        }
        .status-panel {
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        .status-item {
            text-align: center;
        }
        .status-label {
            font-weight: bold;
            color: #666;
        }
        .status-value {
            font-size: 18px;
            margin-top: 5px;
        }
        .face-detected {
            color: #28a745;
        }
        .no-face {
            color: #dc3545;
        }
        .scanning {
            color: #ffc107;
            animation: blink 1s infinite;
        }
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0.3; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Face Tracking Camera System</h1>

        <div class="video-container">
            <img src="{{ url_for('video_feed') }}" class="video-stream" alt="Camera Feed">
        </div>

        <div class="status-panel">
            <div class="status-item">
                <div class="status-label">Servo Position</div>
                <div class="status-value" id="servo-angle">--°</div>
            </div>
            <div class="status-item">
                <div class="status-label">Face Detection</div>
                <div class="status-value" id="face-status">--</div>
            </div>
            <div class="status-item">
                <div class="status-label">System Status</div>
                <div class="status-value" id="system-status">--</div>
            </div>
        </div>
    </div>

    <script>
        function updateStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    // Update servo angle
                    document.getElementById('servo-angle').textContent = data.servo_angle.toFixed(1) + '°';

                    // Update face detection status
                    const faceStatus = document.getElementById('face-status');
                    if (data.face_detected) {
                        faceStatus.textContent = 'DETECTED';
                        faceStatus.className = 'status-value face-detected';
                    } else {
                        faceStatus.textContent = 'NOT DETECTED';
                        faceStatus.className = 'status-value no-face';
                    }

                    // Update system status
                    const systemStatus = document.getElementById('system-status');
                    if (data.scanning) {
                        systemStatus.textContent = 'SCANNING';
                        systemStatus.className = 'status-value scanning';
                    } else if (data.face_detected) {
                        systemStatus.textContent = 'TRACKING';
                        systemStatus.className = 'status-value face-detected';
                    } else {
                        systemStatus.textContent = 'IDLE';
                        systemStatus.className = 'status-value';
                    }
                })
                .catch(error => {
                    console.error('Error fetching status:', error);
                });
        }

        // Update status every 500ms
        setInterval(updateStatus, 500);

        // Initial status update
        updateStatus();
    </script>
</body>
</html>'''

    with open('templates/index.html', 'w') as f:
        f.write(html_content)

def main():
    print("Initializing face tracking system with Flask web interface...")

    # Create templates
    create_templates()

    cap = cv2.VideoCapture("/dev/video1")

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Configure camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for stability

    # Disable autofocus if possible (may not work on all cameras)
    try:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    except:
        pass

    try:
        # Start face detection in a separate thread
        face_thread = threading.Thread(target=detect_faces_thread, args=(cap,))
        face_thread.daemon = True
        face_thread.start()

        print("Face detection thread started.")
        print("Starting Flask web server...")
        print("Access the camera feed at: http://localhost:5000")
        print("Press Ctrl+C to stop the server.")

        # Start Flask app
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        print("Cleaning up...")
        cap.release()
        servo.stop()
        GPIO.cleanup()
        print("Cleanup complete.")

if __name__ == "__main__":
    main()