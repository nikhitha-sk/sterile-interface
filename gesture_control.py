import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import os
import time
import math

COMMAND_FILE = "/tmp/gesture_command.txt"

def send_command(cmd):
    """Send a command to the viewer via shared file"""
    try:
        with open(COMMAND_FILE, 'w') as f:
            f.write(cmd)
        return True
    except Exception as e:
        print(f"Could not send command: {e}")
    return False

# Download hand landmarker model if not exists
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )

# Initialize webcam - try multiple backends/indices
def init_camera():
    """Try different camera indices and backends"""
    for idx in [0, 1, 2]:
        # Try V4L2 backend first (Linux)
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            print(f"Camera opened with V4L2 backend at index {idx}")
            return cap
        cap.release()
        
        # Try default backend
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"Camera opened at index {idx}")
            return cap
        cap.release()
    
    return None

cap = init_camera()
if cap is None:
    print("ERROR: Could not open any camera!")
    print("Make sure your webcam is connected and not in use by another app.")
    print("Try: ls /dev/video*")
    exit(1)

# Setup hand landmarker with new API
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

# Gesture detection helpers
def get_finger_state(hand, frame_shape):
    """
    Detect which fingers are extended.
    Returns dict with finger names and True/False for extended.
    """
    h, w = frame_shape[:2]
    
    # Landmark indices
    # Thumb: 1-4 (4=tip), Index: 5-8 (8=tip), Middle: 9-12, Ring: 13-16, Pinky: 17-20
    # MCP joints: 5, 9, 13, 17 (base of fingers)
    # Tips: 4, 8, 12, 16, 20
    
    # Get wrist as reference
    wrist_y = hand[0].y
    
    # For fingers (not thumb): tip above PIP means extended
    # Index finger
    index_tip = hand[8]
    index_pip = hand[6]
    index_extended = index_tip.y < index_pip.y
    
    # Middle finger
    middle_tip = hand[12]
    middle_pip = hand[10]
    middle_extended = middle_tip.y < middle_pip.y
    
    # Ring finger
    ring_tip = hand[16]
    ring_pip = hand[14]
    ring_extended = ring_tip.y < ring_pip.y
    
    # Pinky
    pinky_tip = hand[20]
    pinky_pip = hand[18]
    pinky_extended = pinky_tip.y < pinky_pip.y
    
    # Thumb (check x distance from palm for horizontal detection)
    thumb_tip = hand[4]
    thumb_ip = hand[3]
    index_mcp = hand[5]
    # Thumb is extended if tip is far from index MCP
    thumb_extended = abs(thumb_tip.x - index_mcp.x) > 0.08
    
    return {
        'thumb': thumb_extended,
        'index': index_extended,
        'middle': middle_extended,
        'ring': ring_extended,
        'pinky': pinky_extended
    }

def count_extended_fingers(fingers):
    """Count how many fingers are extended"""
    return sum(1 for f in fingers.values() if f)

def is_open_palm(fingers):
    """4-5 fingers extended = open palm"""
    return count_extended_fingers(fingers) >= 4

def is_pinch(hand, fingers):
    """Thumb and index fingertips close together = pinch gesture
       But NOT if index finger is extended/pointing up"""
    # Don't trigger pinch if index finger is pointing up alone
    if is_index_pointing(fingers):
        return False
    
    thumb_tip = hand[4]
    index_tip = hand[8]
    
    # Distance between thumb and index tips
    dist = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
    
    return dist < 0.06  # Close together

def is_two_fingers(fingers):
    """Index and middle extended, others closed = peace sign / 2 fingers"""
    return (fingers['index'] and 
            fingers['middle'] and 
            not fingers['ring'] and 
            not fingers['pinky'])

def is_index_pointing(fingers):
    """Only index finger extended = pointing for next image"""
    return (fingers['index'] and 
            not fingers['middle'] and 
            not fingers['ring'] and 
            not fingers['pinky'])

# State variables  
COOLDOWN_TIME = 0.7  # seconds between gestures
last_next_time = 0
last_zoom_time = 0
last_reset_time = 0

current_gesture = "NONE"

print("=" * 50)
print("GESTURE CONTROL")
print("=" * 50)
print("☝️  INDEX FINGER UP             -> Next image")
print("🤏 PINCH (thumb+index close)   -> Zoom IN")
print("✌️  TWO FINGERS (peace sign)    -> Zoom OUT")  
print("👋 OPEN PALM (4-5 fingers)     -> Reset zoom")
print("Press 'q' or ESC to quit")
print("=" * 50)

while True:

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    frame = cv2.flip(frame, 1)
    current_time = time.time()
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = detector.detect(mp_image)

    gesture_text = "No hand detected"
    color = (128, 128, 128)

    if results.hand_landmarks and len(results.hand_landmarks) > 0:
        hand = results.hand_landmarks[0]
        
        # Get finger states
        fingers = get_finger_state(hand, frame.shape)
        num_fingers = count_extended_fingers(fingers)
        
        # Get index fingertip position
        index_x = int(hand[8].x * w)
        index_y = int(hand[8].y * h)
        
        # Draw all fingertips
        tip_indices = [4, 8, 12, 16, 20]
        tip_colors = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
        for i, idx in enumerate(tip_indices):
            px = int(hand[idx].x * w)
            py = int(hand[idx].y * h)
            cv2.circle(frame, (px, py), 8, tip_colors[i], -1)
        
        # Draw index finger larger when pointing
        if is_index_pointing(fingers):
            cv2.circle(frame, (index_x, index_y), 20, (0, 255, 0), 3)
        
        # ========== GESTURE DETECTION ==========
        
        # PRIORITY 1: INDEX FINGER UP = NEXT IMAGE (no zoom when pointing)
        if is_index_pointing(fingers):
            gesture_text = "INDEX UP -> NEXT IMAGE"
            color = (0, 255, 255)  # Yellow
            current_gesture = "POINTING"
            
            if (current_time - last_next_time) > COOLDOWN_TIME:
                send_command('NEXT')
                print("➡️ NEXT IMAGE")
                last_next_time = current_time
        
        # PRIORITY 2: Check for PINCH (zoom in) - only if NOT index pointing
        elif is_pinch(hand, fingers):
            gesture_text = "PINCH - Zoom IN"
            color = (0, 165, 255)  # Orange
            current_gesture = "PINCH"
            
            if (current_time - last_zoom_time) > COOLDOWN_TIME:
                send_command('ZOOM_IN')
                print("🔍 ZOOM IN")
                last_zoom_time = current_time
        
        # PRIORITY 3: Check for TWO FINGERS (zoom out)
        elif is_two_fingers(fingers):
            gesture_text = "TWO FINGERS - Zoom OUT"
            color = (255, 0, 255)  # Magenta
            current_gesture = "TWO"
            
            if (current_time - last_zoom_time) > COOLDOWN_TIME:
                send_command('ZOOM_OUT')
                print("🔎 ZOOM OUT")
                last_zoom_time = current_time
        
        # PRIORITY 4: Check for OPEN PALM (reset zoom)
        elif is_open_palm(fingers):
            gesture_text = "OPEN PALM - Reset zoom"
            color = (0, 255, 0)  # Green
            current_gesture = "PALM"
            
            if (current_time - last_reset_time) > COOLDOWN_TIME:
                send_command('RESET')
                print("🔄 RESET ZOOM")
                last_reset_time = current_time
        
        else:
            gesture_text = f"Hand ({num_fingers} fingers)"
            color = (200, 200, 200)
            current_gesture = "OTHER"
    else:
        # No hand - reset states
        current_gesture = "NONE"

    # Draw gesture info on frame
    cv2.rectangle(frame, (5, 5), (400, 80), (0, 0, 0), -1)
    cv2.putText(frame, gesture_text, (10, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Show finger states
    if results.hand_landmarks and len(results.hand_landmarks) > 0:
        fingers = get_finger_state(results.hand_landmarks[0], frame.shape)
        finger_str = f"T:{int(fingers['thumb'])} I:{int(fingers['index'])} M:{int(fingers['middle'])} R:{int(fingers['ring'])} P:{int(fingers['pinky'])}"
        cv2.putText(frame, finger_str, (10, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Gesture Camera", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)