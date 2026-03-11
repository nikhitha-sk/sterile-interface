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

def is_pinch(hand):
    """Thumb and index fingertips close together = pinch gesture"""
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

def is_fist_or_one_finger(fingers):
    """Fist or only one finger up - good for swiping"""
    extended = count_extended_fingers(fingers)
    return extended <= 2  # Fist or 1-2 fingers

# State variables  
prev_palm_x = None
prev_palm_time = None
SWIPE_THRESHOLD = 120  # pixels for swipe
SWIPE_MIN_TIME = 0.1   # minimum time for swipe
SWIPE_MAX_TIME = 0.6   # maximum time for swipe

COOLDOWN_TIME = 0.5  # seconds between gestures
last_swipe_time = 0
last_zoom_time = 0
last_reset_time = 0

# Position history for velocity calculation
x_history = []
HISTORY_SIZE = 5

current_gesture = "NONE"

print("=" * 50)
print("GESTURE CONTROL - NEW GESTURES")
print("=" * 50)
print("👋 OPEN PALM (4-5 fingers)     -> Reset zoom")
print("🤏 PINCH (thumb+index close)   -> Zoom IN")
print("✌️  TWO FINGERS (peace sign)    -> Zoom OUT")  
print("👊 FIST/1-2 FINGERS + SWIPE    -> Next/Prev image")
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
        
        # Get palm center for swipe tracking
        palm_x = int(hand[0].x * w)
        palm_y = int(hand[0].y * h)
        
        # Draw all fingertips
        tip_indices = [4, 8, 12, 16, 20]
        tip_colors = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
        for i, idx in enumerate(tip_indices):
            px = int(hand[idx].x * w)
            py = int(hand[idx].y * h)
            cv2.circle(frame, (px, py), 8, tip_colors[i], -1)
        
        # Draw palm center
        cv2.circle(frame, (palm_x, palm_y), 12, (255, 255, 255), 3)
        
        # Add to position history
        x_history.append((palm_x, current_time))
        if len(x_history) > HISTORY_SIZE:
            x_history.pop(0)
        
        # ========== GESTURE DETECTION ==========
        
        # Check for PINCH (zoom in) - thumb and index close
        if is_pinch(hand):
            gesture_text = f"PINCH - Zoom IN ({num_fingers} fingers)"
            color = (0, 165, 255)  # Orange
            current_gesture = "PINCH"
            
            if (current_time - last_zoom_time) > COOLDOWN_TIME:
                send_command('ZOOM_IN')
                print("🔍 ZOOM IN")
                last_zoom_time = current_time
                x_history.clear()  # Clear to prevent accidental swipe
        
        # Check for TWO FINGERS (zoom out)
        elif is_two_fingers(fingers):
            gesture_text = "TWO FINGERS - Zoom OUT"
            color = (255, 0, 255)  # Magenta
            current_gesture = "TWO"
            
            if (current_time - last_zoom_time) > COOLDOWN_TIME:
                send_command('ZOOM_OUT')
                print("🔎 ZOOM OUT")
                last_zoom_time = current_time
                x_history.clear()
        
        # Check for OPEN PALM (reset zoom)
        elif is_open_palm(fingers):
            gesture_text = f"OPEN PALM - Reset ({num_fingers} fingers)"
            color = (0, 255, 0)  # Green
            current_gesture = "PALM"
            
            if (current_time - last_reset_time) > COOLDOWN_TIME:
                send_command('RESET')
                print("🔄 RESET ZOOM")
                last_reset_time = current_time
                x_history.clear()
        
        # Check for FIST or 1-2 fingers (swipe for navigation)
        elif is_fist_or_one_finger(fingers):
            gesture_text = f"SWIPE MODE ({num_fingers} fingers)"
            color = (255, 255, 0)  # Cyan
            current_gesture = "SWIPE"
            
            # Calculate swipe using history
            if len(x_history) >= 3 and (current_time - last_swipe_time) > COOLDOWN_TIME:
                first_x, first_time = x_history[0]
                last_x, last_time = x_history[-1]
                
                time_diff = last_time - first_time
                x_diff = last_x - first_x
                
                if SWIPE_MIN_TIME < time_diff < SWIPE_MAX_TIME:
                    if x_diff > SWIPE_THRESHOLD:
                        send_command('NEXT')
                        print(f"➡️ NEXT IMAGE (moved {x_diff}px)")
                        last_swipe_time = current_time
                        x_history.clear()
                    
                    elif x_diff < -SWIPE_THRESHOLD:
                        send_command('PREV')
                        print(f"⬅️ PREVIOUS IMAGE (moved {x_diff}px)")
                        last_swipe_time = current_time
                        x_history.clear()
        else:
            gesture_text = f"Hand ({num_fingers} fingers)"
            color = (200, 200, 200)
            current_gesture = "OTHER"
    else:
        # No hand - reset states
        x_history.clear()
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