import cv2
import os

image_folder = "images"
COMMAND_FILE = "/tmp/gesture_command.txt"

# Clear any old commands
if os.path.exists(COMMAND_FILE):
    os.remove(COMMAND_FILE)

images = sorted([os.path.join(image_folder, img) for img in os.listdir(image_folder) 
                 if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

if not images:
    print("No images found in 'images' folder!")
    exit(1)

print(f"Loaded {len(images)} images")
print("Controls: Left/Right arrows to navigate, +/- to zoom, 0 to reset zoom, q/ESC to quit")
print("Gesture control: Run gesture_control.py in another terminal")

index = 0
zoom = 1.0
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

cv2.namedWindow("Medical Viewer", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Medical Viewer", WINDOW_WIDTH, WINDOW_HEIGHT)

def check_gesture_command():
    """Check for commands from gesture controller"""
    global index, zoom
    changed = False
    if os.path.exists(COMMAND_FILE):
        try:
            with open(COMMAND_FILE, 'r') as f:
                cmd = f.read().strip()
            os.remove(COMMAND_FILE)  # Clear after reading
            
            if cmd == "NEXT":
                index = (index + 1) % len(images)
                print(f">>> Gesture: Next image ({index+1}/{len(images)})")
                changed = True
            elif cmd == "PREV":
                index = (index - 1) % len(images)
                print(f">>> Gesture: Previous image ({index+1}/{len(images)})")
                changed = True
            elif cmd == "ZOOM_IN":
                old_zoom = zoom
                zoom = min(4.0, zoom + 0.5)
                print(f">>> Gesture: Zoom in ({old_zoom:.1f}x -> {zoom:.1f}x)")
                changed = True
            elif cmd == "ZOOM_OUT":
                old_zoom = zoom
                zoom = max(0.5, zoom - 0.5)
                print(f">>> Gesture: Zoom out ({old_zoom:.1f}x -> {zoom:.1f}x)")
                changed = True
            elif cmd == "RESET":
                zoom = 1.0
                print(">>> Gesture: Zoom reset to 1.0x")
                changed = True
        except Exception as e:
            print(f"Error reading command: {e}")
    return changed

while True:
    
    # Check for gesture commands
    check_gesture_command()

    img = cv2.imread(images[index])
    if img is None:
        print(f"Could not load image: {images[index]}")
        index = (index + 1) % len(images)
        continue
    
    orig_h, orig_w = img.shape[:2]
    
    # Apply zoom by cropping center region and scaling up
    if zoom != 1.0:
        # Calculate crop region (zoom into center)
        crop_w = int(orig_w / zoom)
        crop_h = int(orig_h / zoom)
        
        # Ensure minimum size
        crop_w = max(100, min(crop_w, orig_w))
        crop_h = max(100, min(crop_h, orig_h))
        
        # Center crop
        x1 = (orig_w - crop_w) // 2
        y1 = (orig_h - crop_h) // 2
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        
        cropped = img[y1:y2, x1:x2]
        
        # Scale cropped region to window size
        display = cv2.resize(cropped, (WINDOW_WIDTH, WINDOW_HEIGHT), interpolation=cv2.INTER_CUBIC)
    else:
        # No zoom - fit to window
        display = cv2.resize(img, (WINDOW_WIDTH, WINDOW_HEIGHT), interpolation=cv2.INTER_AREA)

    # Draw info overlay
    overlay = display.copy()
    cv2.rectangle(overlay, (5, 5), (300, 45), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
    
    cv2.putText(display, f"Zoom: {zoom:.1f}x  Image: {index+1}/{len(images)}", 
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Medical Viewer", display)

    key = cv2.waitKey(30) & 0xFF

    if key == ord('q') or key == 27:
        break
    elif key == 81 or key == 2:   # left arrow
        index = (index - 1) % len(images)
    elif key == 83 or key == 3:   # right arrow
        index = (index + 1) % len(images)
    elif key == ord('+') or key == ord('='):
        zoom = min(4.0, zoom + 0.5)
    elif key == ord('-') or key == ord('_'):
        zoom = max(0.5, zoom - 0.5)
    elif key == ord('0'):
        zoom = 1.0
        print("Zoom reset to 1.0x")

# Cleanup
if os.path.exists(COMMAND_FILE):
    os.remove(COMMAND_FILE)
cv2.destroyAllWindows()
cv2.waitKey(1)