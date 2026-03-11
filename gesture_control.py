import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pyautogui
import urllib.request
import os
import time
import math

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
    thumb_extended = abs(thumb_tip.x - index_mcp.x) > 0.1
    
    return {
        'thumb': thumb_extended,
        'index': index_extended,
        'middle': middle_extended,
        'ring': ring_extended,
        'pinky': pinky_extended
    }

def is_open_palm(fingers):
    """All 5 fingers extended = open palm"""
    return all(fingers.values())

def is_pinch(hand):
    """All fingertips close together = pinch gesture"""
    # Get all fingertip positions
    thumb_tip = hand[4]
    index_tip = hand[8]
    middle_tip = hand[12]
    ring_tip = hand[16]
    pinky_tip = hand[20]
    
    tips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
    
    # Calculate center of all tips
    center_x = sum(t.x for t in tips) / 5
    center_y = sum(t.y for t in tips) / 5
    
    # Check if all tips are close to center
    PINCH_THRESHOLD = 0.08  # Maximum distance from center for pinch
    
    for tip in tips:
        dist = math.sqrt((tip.x - center_x)**2 + (tip.y - center_y)**2)
        if dist > PINCH_THRESHOLD:
            return False
    
    return True

def is_index_pointing(fingers):
    """Only index finger extended = pointing gesture"""
    return (fingers['index'] and 
            not fingers['middle'] and 
            not fingers['ring'] and 
            not fingers['pinky'])

# State variables
swipe_start_x = None
swipe_start_time = None
SWIPE_THRESHOLD = 80  # pixels for swipe
SWIPE_TIME_WINDOW = 0.5  # seconds

COOLDOWN_TIME = 0.6  # seconds between gestures
last_swipe_time = 0
last_zoom_time = 0
last_reset_time = 0

# Track pinch state to detect release
was_pinching = False
current_gesture = "NONE"

print("=" * 50)
print("GESTURE CONTROL - NEW GESTURES")
print("=" * 50)
print("👋 OPEN PALM (all fingers spread) -> Reset zoom")
print("🤏 PINCH (all fingers together)  -> Zoom IN")
print("☝️  INDEX POINTING + SWIPE        -> Next/Prev image")
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
        
        # Draw all fingertips
        tip_indices = [4, 8, 12, 16, 20]
        tip_colors = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
        for i, idx in enumerate(tip_indices):
            px = int(hand[idx].x * w)
            py = int(hand[idx].y * h)
            cv2.circle(frame, (px, py), 8, tip_colors[i], -1)
        
        # Draw palm center
        palm_x = int(hand[0].x * w)
        palm_y = int(hand[0].y * h)
        cv2.circle(frame, (palm_x, palm_y), 10, (255, 255, 255), 2)
        
        # ========== GESTURE DETECTION ==========
        
        # Check for PINCH (zoom in)
        if is_pinch(hand):
            gesture_text = "PINCH - Zooming IN"
            color = (0, 165, 255)  # Orange
            current_gesture = "PINCH"
            was_pinching = True
            
            if (current_time - last_zoom_time) > COOLDOWN_TIME:
                pyautogui.press('+')
                print("🔍 ZOOM IN")
                last_zoom_time = current_time
        
        # Check for OPEN PALM (reset zoom)
        elif is_open_palm(fingers):
            gesture_text = "OPEN PALM - Normal view"
            color = (0, 255, 0)  # Green
            current_gesture = "PALM"
            
            # If we were pinching and now showing palm, reset zoom
            if was_pinching and (current_time - last_reset_time) > COOLDOWN_TIME:
                pyautogui.press('0')  # Send 0 to reset zoom
                print("🔄 RESET ZOOM")
                last_reset_time = current_time
                was_pinching = False
        
        # Check for INDEX POINTING (swipe for navigation)
        elif is_index_pointing(fingers):
            gesture_text = "POINTING - Swipe to navigate"
            color = (255, 255, 0)  # Cyan
            current_gesture = "POINTING"
            
            # Get index tip position for swipe
            index_x = int(hand[8].x * w)
            
            if swipe_start_x is None:
                swipe_start_x = index_x
                swipe_start_time = current_time
            else:
                elapsed = current_time - swipe_start_time
                
                if elapsed > SWIPE_TIME_WINDOW:
                    # Reset swipe tracking
                    swipe_start_x = index_x
                    swipe_start_time = current_time
                elif (current_time - last_swipe_time) > COOLDOWN_TIME:
                    diff = index_x - swipe_start_x
                    
                    if diff > SWIPE_THRESHOLD:
                        pyautogui.press("right")
                        print("➡️ NEXT IMAGE")
                        last_swipe_time = current_time
                        swipe_start_x = None
                        swipe_start_time = None
                    
                    elif diff < -SWIPE_THRESHOLD:
                        pyautogui.press("left")
                        print("⬅️ PREVIOUS IMAGE")
                        last_swipe_time = current_time
                        swipe_start_x = None
                        swipe_start_time = None
        else:
            gesture_text = "Hand detected"
            color = (200, 200, 200)
            current_gesture = "OTHER"
            swipe_start_x = None
            swipe_start_time = None
    else:
        # No hand - reset states
        swipe_start_x = None
        swipe_start_time = None
        current_gesture = "NONE"

    # Draw gesture info on frame
    cv2.rectangle(frame, (5, 5), (350, 45), (0, 0, 0), -1)
    cv2.putText(frame, gesture_text, (10, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Show finger states
    if results.hand_landmarks and len(results.hand_landmarks) > 0:
        fingers = get_finger_state(results.hand_landmarks[0], frame.shape)
        finger_str = f"T:{int(fingers['thumb'])} I:{int(fingers['index'])} M:{int(fingers['middle'])} R:{int(fingers['ring'])} P:{int(fingers['pinky'])}"
        cv2.putText(frame, finger_str, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Gesture Camera", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)