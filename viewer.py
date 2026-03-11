import cv2
import os

image_folder = "images"

images = sorted([os.path.join(image_folder, img) for img in os.listdir(image_folder) 
                 if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

if not images:
    print("No images found in 'images' folder!")
    exit(1)

print(f"Loaded {len(images)} images")
print("Controls: Left/Right arrows to navigate, +/- to zoom, 0 to reset zoom, q/ESC to quit")

index = 0
zoom = 1.0

cv2.namedWindow("Medical Viewer", cv2.WINDOW_NORMAL)

while True:

    img = cv2.imread(images[index])
    if img is None:
        index = (index + 1) % len(images)
        continue
        
    h, w = img.shape[:2]
    new_w, new_h = int(w*zoom), int(h*zoom)
    if new_w > 0 and new_h > 0:
        # Use high-quality interpolation for better zoom
        if zoom > 1.0:
            # INTER_CUBIC for zooming in - better quality
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            # INTER_AREA for zooming out - better for shrinking
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = img

    # Show zoom level
    cv2.putText(resized, f"Zoom: {zoom:.1f}x  Image: {index+1}/{len(images)}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow("Medical Viewer", resized)

    key = cv2.waitKey(50) & 0xFF  # Mask to get proper key code

    if key == ord('q') or key == 27:  # q or ESC
        break

    elif key == 81:   # left arrow
        index = (index - 1) % len(images)

    elif key == 83:   # right arrow
        index = (index + 1) % len(images)

    elif key == ord('+') or key == ord('='):  # + or = (for keyboards without numpad)
        zoom = min(5.0, zoom + 0.2)  # Max zoom 5x

    elif key == ord('-') or key == ord('_'):
        zoom = max(0.2, zoom - 0.2)  # Min zoom 0.2x
    
    elif key == ord('0'):  # Reset zoom
        zoom = 1.0
        print("Zoom reset to 1.0x")

cv2.destroyAllWindows()
cv2.waitKey(1)  # Ensure window closes