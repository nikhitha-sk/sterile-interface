# Project Report: Sterile Interface for Surgical Environments

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Libraries and Technologies Used](#3-libraries-and-technologies-used)
4. [Project Structure](#4-project-structure)
5. [Steps to Build and Run](#5-steps-to-build-and-run)
6. [System Architecture and Working](#6-system-architecture-and-working)
7. [Output Description and Snapshots](#7-output-description-and-snapshots)
8. [Conclusion](#8-conclusion)
9. [References](#9-references)

---

## 1. Abstract

The **Sterile Interface** is a touch-free gesture control system designed for surgical operating rooms. Surgeons regularly need to view and navigate medical imaging data (such as MRI scans, CT slices, or X-rays) during procedures, but physical interaction with keyboards, mice, or touchscreens violates sterility protocols. This project addresses that limitation by enabling surgeons to control a medical image viewer entirely through hand gestures captured via a standard webcam.

The system leverages computer vision (OpenCV), AI-based hand landmark detection (MediaPipe), and optional depth estimation (MiDaS via PyTorch/ONNX) to recognize distinct hand poses and translate them into viewer commands — navigating images, zooming in, zooming out, or resetting the view — all without touching any device.

---

## 2. Introduction

Maintaining a sterile field is one of the most critical requirements in any surgical procedure. Any contact with non-sterile objects, including keyboards or touch displays, risks contaminating the surgeon's gloves and potentially endangering the patient. Yet during complex surgeries, surgeons frequently need to refer to preoperative scans or intraoperative imaging data displayed on screens in the operating room.

Existing solutions (voice control, foot pedals, sterile keyboard covers) have practical drawbacks including unreliability, additional equipment, or workflow disruption. A vision-based, touchless gesture interface overcomes these challenges by requiring only a webcam positioned near the display — no additional hardware contact needed.

This project demonstrates:

- **Computer vision** for real-time hand tracking
- **AI-based gesture recognition** using pre-trained neural network models
- **Depth-aware interaction** for more natural gesture sensing
- **Human–computer interaction** tailored for clinical environments

---

## 3. Libraries and Technologies Used

### 3.1 Python (3.10+)

The core programming language. Python's rich ecosystem of computer vision and machine learning libraries makes it ideal for rapid prototyping and deployment of gesture-recognition systems.

---

### 3.2 OpenCV (`opencv-python`)

| Property | Detail |
|----------|--------|
| Version | Latest stable |
| Purpose | Image capture, display, and processing |
| Website | https://opencv.org |

OpenCV (Open Source Computer Vision Library) is the backbone of the image viewer and camera interface. It is used to:

- Capture live frames from the webcam
- Load, resize, and display medical images
- Draw visual overlays (fingertip markers, gesture labels) on camera frames
- Implement zoom by cropping and scaling images

---

### 3.3 MediaPipe (`mediapipe`)

| Property | Detail |
|----------|--------|
| Version | Latest stable |
| Purpose | Hand detection and 21-point landmark tracking |
| Website | https://mediapipe.dev |

MediaPipe is a cross-platform ML framework developed by Google. The project uses the **HandLandmarker** solution, which detects a hand in each camera frame and returns 21 3D keypoints (landmarks), covering the wrist, knuckles, and fingertips.

**Hand landmarks used:**

| Landmark ID | Joint |
|-------------|-------|
| 0 | Wrist |
| 4 | Thumb tip |
| 5 | Index MCP (knuckle) |
| 6 | Index PIP |
| 8 | Index tip |
| 12 | Middle tip |
| 16 | Ring tip |
| 20 | Pinky tip |

The pre-trained model file (`hand_landmarker.task`) is automatically downloaded on first run.

---

### 3.4 NumPy (`numpy`)

| Property | Detail |
|----------|--------|
| Purpose | Numerical array operations |
| Website | https://numpy.org |

NumPy is used for efficient mathematical operations on image arrays (pixel data) and for computing distances between hand landmark coordinates used in gesture classification.

---

### 3.5 PyAutoGUI (`pyautogui`)

| Property | Detail |
|----------|--------|
| Purpose | GUI automation / simulating keyboard input |
| Website | https://pyautogui.readthedocs.io |

PyAutoGUI allows the gesture control module to programmatically simulate keyboard presses. This bridges the gesture controller and the image viewer without requiring a custom network protocol.

---

### 3.6 PyTorch (`torch`) and TorchVision (`torchvision`)

| Property | Detail |
|----------|--------|
| Purpose | Deep learning framework for depth estimation |
| Website | https://pytorch.org |

PyTorch is used to load and run the **MiDaS depth estimation model** in ONNX format via PyTorch's ONNX runtime. Depth information adds an extra dimension to gesture recognition — for example, distinguishing a hand moving toward the camera (zoom in) from a lateral hand wave.

For systems without a GPU, a CPU-only build of PyTorch is sufficient:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

### 3.7 MiDaS Depth Estimation Model (`model-small.onnx`)

| Property | Detail |
|----------|--------|
| Model | MiDaS v2.1 Small |
| Format | ONNX |
| Size | ~64 MB |
| Source | https://github.com/isl-org/MiDaS |

MiDaS (Mixed Dataset for depth estimation) is an AI model that estimates monocular depth from a single RGB camera frame. The `model-small.onnx` variant is optimized for real-time inference on CPU hardware.

---

### 3.8 Pre-trained Hand Model (`hand_landmarker.task`)

| Property | Detail |
|----------|--------|
| Size | ~7.8 MB |
| Provider | Google MediaPipe |
| Auto-downloaded | Yes |

This is MediaPipe's bundled neural network for hand detection and landmark localization. It runs entirely on the CPU without any GPU requirement.

---

### Summary Table

| Library / Model | Role |
|-----------------|------|
| Python 3.10+ | Core programming language |
| OpenCV | Image capture, display, processing |
| MediaPipe | Hand detection and 21-landmark tracking |
| NumPy | Numerical operations on image/landmark data |
| PyAutoGUI | Simulated keyboard input for automation |
| PyTorch + TorchVision | ONNX runtime for MiDaS depth model |
| MiDaS ONNX (model-small.onnx) | Monocular depth estimation |
| hand_landmarker.task | Pre-trained hand landmark detector |

---

## 4. Project Structure

```
sterile-interface/
│
├── viewer.py               # Medical image viewer application
├── gesture_control.py      # Gesture recognition and command dispatch
├── hand_landmarker.task    # MediaPipe hand detection model (auto-downloaded)
├── README.md               # Setup and usage documentation
├── REPORT.md               # This project report
│
├── models/
│   └── model-small.onnx    # MiDaS depth estimation model (~64 MB)
│
└── images/
    ├── img1.jpg            # Sample medical scan image
    ├── img2.jpg            # Sample medical scan image
    └── img3.jpg            # Sample medical scan image
```

---

## 5. Steps to Build and Run

### 5.1 System Requirements

| Requirement | Minimum |
|-------------|---------|
| Operating System | Ubuntu 22.04 or later |
| Python | 3.10+ |
| Hardware | Webcam (USB or built-in) |
| RAM | 2 GB+ recommended |
| GPU | Optional (CPU-only mode supported) |

---

### 5.2 Step 1 — Check Python Installation

```bash
python3 --version
```

If Python is not installed:

```bash
sudo apt update
sudo apt install python3 python3-pip
```

---

### 5.3 Step 2 — Install Virtual Environment Support

```bash
sudo apt install python3-venv
```

For Python 3.13 specifically:

```bash
sudo apt install python3.13-venv
```

---

### 5.4 Step 3 — Create the Project Folder

```bash
mkdir sterile-interface
cd sterile-interface
```

---

### 5.5 Step 4 — Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

Your terminal prompt should update to show:

```
(venv) user@machine:~/sterile-interface$
```

---

### 5.6 Step 5 — Install Required Libraries

**Standard install (with GPU support):**

```bash
pip install opencv-python mediapipe numpy pyautogui torch torchvision
```

**CPU-only install (smaller download, no GPU needed):**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python mediapipe numpy pyautogui
```

---

### 5.7 Step 6 — Download MiDaS Depth Model

```bash
mkdir models
wget https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx -P models
```

---

### 5.8 Step 7 — Add Medical Images

```bash
mkdir images
# Copy or move your scan images into the images/ folder
# Supported formats: .jpg, .jpeg, .png, .bmp
```

Sample images can be any medical scans (MRI, CT, X-ray). The viewer loads all images from the `images/` folder automatically.

---

### 5.9 Step 8 — Run the System (Two Terminals Required)

Open two separate terminal windows.

**Terminal 1 — Start the Image Viewer:**

```bash
source venv/bin/activate
python viewer.py
```

This opens the medical image viewer window displaying the first image.

**Terminal 2 — Start Gesture Control (after viewer is running):**

```bash
source venv/bin/activate
python gesture_control.py
```

This opens the webcam feed and begins gesture detection.

---

### 5.10 Gesture Reference

| Hand Gesture | Action | Cooldown |
|--------------|--------|----------|
| ☝️ Index finger pointing up | Next image | 1.5 seconds |
| 🤏 Pinch (thumb + index close) | Zoom in | 0.7 seconds |
| ✌️ Two fingers (peace sign) | Zoom out | 0.7 seconds |
| 👋 Open palm (4–5 fingers) | Reset zoom to 1.0× | 0.7 seconds |

---

### 5.11 Keyboard Shortcuts (Viewer Window)

| Key | Action |
|-----|--------|
| `→` Right arrow | Next image |
| `←` Left arrow | Previous image |
| `+` or `=` | Zoom in (max 4.0×) |
| `-` or `_` | Zoom out (min 0.5×) |
| `0` | Reset zoom to 1.0× |
| `q` or `ESC` | Quit |

---

## 6. System Architecture and Working

### 6.1 High-Level Architecture

```
 ┌──────────────────────────────────────┐
 │          Webcam (camera feed)        │
 └──────────────────┬───────────────────┘
                    │ Raw video frames
                    ▼
 ┌──────────────────────────────────────┐
 │       gesture_control.py             │
 │  ┌────────────────────────────────┐  │
 │  │  MediaPipe Hand Landmarker     │  │
 │  │  (21 keypoints per hand)       │  │
 │  └──────────────┬─────────────────┘  │
 │                 │                    │
 │  ┌──────────────▼─────────────────┐  │
 │  │  Finger State Detector         │  │
 │  │  (extended / not extended)     │  │
 │  └──────────────┬─────────────────┘  │
 │                 │                    │
 │  ┌──────────────▼─────────────────┐  │
 │  │  Gesture Classifier            │  │
 │  │  (priority-ordered rules)      │  │
 │  └──────────────┬─────────────────┘  │
 │                 │ Command string     │
 │  ┌──────────────▼─────────────────┐  │
 │  │  /tmp/gesture_command.txt      │  │ ◄── IPC shared file
 │  └────────────────────────────────┘  │
 └──────────────────────────────────────┘
                    │
                    ▼
 ┌──────────────────────────────────────┐
 │           viewer.py                  │
 │  ┌────────────────────────────────┐  │
 │  │  Command Poller                │  │
 │  │  (reads command file)          │  │
 │  └──────────────┬─────────────────┘  │
 │                 │                    │
 │  ┌──────────────▼─────────────────┐  │
 │  │  Image Navigator + Zoom Engine │  │
 │  └──────────────┬─────────────────┘  │
 │                 │                    │
 │  ┌──────────────▼─────────────────┐  │
 │  │  OpenCV Display Window         │  │
 │  │  (800×600, info overlay)       │  │
 │  └────────────────────────────────┘  │
 └──────────────────────────────────────┘
```

---

### 6.2 Gesture Detection Logic

Each camera frame is analyzed as follows:

1. **Landmark extraction** — MediaPipe returns the (x, y, z) coordinates of 21 hand joints.
2. **Finger extension check** — For each finger, the tip position is compared to the PIP joint. A finger is considered "extended" if its tip is above its PIP joint (lower y-coordinate in image space).
3. **Thumb check** — The thumb is extended if its tip is horizontally far from the index MCP joint (>0.08 normalized units), since the thumb moves laterally rather than vertically.
4. **Gesture classification** — The combination of extended fingers is matched against rules in priority order to identify the gesture.
5. **Cooldown enforcement** — Once a gesture is triggered, a cooldown timer prevents the same command from firing repeatedly.
6. **Command dispatch** — The recognized command is written to `/tmp/gesture_command.txt`, which the viewer polls continuously.

### 6.3 Zoom Implementation (viewer.py)

Zoom is implemented as a center-crop-and-scale operation:

```
crop_size = original_image_size / zoom_factor
```

At 1.0× zoom, the entire image is shown. At 2.0× zoom, only the central 50% of the image is cropped and then stretched to fill the 800×600 window, creating the effect of zooming in. This approach preserves image quality and is computationally lightweight.

---

## 7. Output Description and Snapshots

### 7.1 Gesture Control Window Output

When `gesture_control.py` is running, the webcam feed is displayed in real time with the following overlays:

- **Colored circles on fingertips** — Each finger is marked with a distinct color:
  - Thumb: Blue
  - Index: Green
  - Middle: Yellow/Cyan
  - Ring: Magenta
  - Pinky: Yellow

- **Enlarged green circle on index fingertip** when pointing gesture is detected

- **Gesture label text** displayed on the frame (e.g., `INDEX POINTING`, `PINCH`, `TWO FINGERS`, `OPEN PALM`)

- **Finger state display** — Shows which fingers are extended: `T:1 I:1 M:0 R:0 P:0`

**Console output during operation:**

```
==================================================
GESTURE CONTROL
==================================================
☝️  INDEX FINGER UP             -> Next image (1.5s delay)
🤏 PINCH (thumb+index close)   -> Zoom IN
✌️  TWO FINGERS (peace sign)    -> Zoom OUT
👋 OPEN PALM (4-5 fingers)     -> Reset zoom
Press 'q' or ESC to quit
==================================================
```

Real-time action messages printed to console:

```
➡️ NEXT IMAGE
🔍 ZOOM IN
🔎 ZOOM OUT
🔄 RESET ZOOM
```

---

### 7.2 Medical Image Viewer Window Output

When `viewer.py` is running, a window titled **"Medical Viewer"** opens showing:

- The current medical scan image scaled to **800×600 pixels**
- A **green info overlay** (bottom-left) showing:
  ```
  Zoom: 1.5x  Image: 2/3
  ```
  with a semi-transparent dark background for readability

**Console output during navigation:**

```
Loaded 3 images
Controls: Left/Right arrows to navigate, +/- to zoom, 0 to reset zoom, q/ESC to quit
Gesture control: Run gesture_control.py in another terminal
>>> Gesture: Next image (2/3)
>>> Gesture: Zoom in (1.0x -> 1.5x)
>>> Gesture: Zoom out (1.5x -> 1.0x)
>>> Gesture: Zoom reset to 1.0x
```

---

### 7.3 Sample Medical Images

The `images/` folder includes three sample scan images used for demonstration:

| File | Size | Purpose |
|------|------|---------|
| `img1.jpg` | ~56 KB | Medical scan — demonstration image 1 |
| `img2.jpg` | ~7 KB | Medical scan — demonstration image 2 |
| `img3.jpg` | ~53 KB | Medical scan — demonstration image 3 |

> **Note on snapshots:** Live screenshots of the running system would show the webcam feed with hand landmark overlays in one window, and the selected medical scan with zoom level indicator in the other. Since this is a real-time system, screenshots are best captured during a live demo by pressing `PrtScr` while both windows are visible side-by-side.

---

### 7.4 Gesture-to-Action Demonstration

| Step | Surgeon Gesture | System Response |
|------|----------------|-----------------|
| 1 | Raises one finger (index up) | Viewer advances to next scan image |
| 2 | Makes pinch gesture | Viewer zooms in (1.0× → 1.5× → 2.0× ...) |
| 3 | Shows peace sign (✌️) | Viewer zooms out (2.0× → 1.5× → 1.0× ...) |
| 4 | Opens hand flat | Viewer resets zoom to 1.0× |

Each action triggers only once per cooldown period, preventing unintentional repeated commands from a held gesture.

---

## 8. Conclusion

The **Sterile Interface** project successfully demonstrates that a practical, touch-free surgical image viewer is achievable using off-the-shelf hardware (a standard webcam) and open-source software libraries. The system meets its core objective: allowing surgeons to navigate and zoom medical images without any physical contact, thereby preserving sterility in the operating room.

### Key Achievements

- **Reliable gesture recognition** using MediaPipe's pre-trained hand landmarker, with a priority-based classification system that avoids false positives between similar gestures.
- **Responsive image navigation** with smooth zoom via center-crop scaling and a configurable zoom range (0.5× to 4.0×).
- **Low hardware requirements** — functions on CPU-only systems with a basic USB webcam, making deployment accessible in resource-constrained hospital settings.
- **Modular architecture** — the gesture detector and image viewer run as independent processes communicating through a shared file, making each component easy to test, replace, or extend.
- **Cooldown system** that prevents accidental repeated commands while still feeling responsive for intentional gestures.

### Limitations

- **Lighting sensitivity** — MediaPipe hand detection performance degrades in very low or overly bright lighting conditions common in some OR setups.
- **Single-hand operation** — the system currently tracks only one hand at a time.
- **Limited gesture vocabulary** — four gestures cover the core use case, but more complex interactions (rotation, panning) require additional gesture types.
- **2D images only** — the viewer displays flat image files; it does not yet support 3D volumetric rendering of DICOM datasets.

### Future Improvements

- Integration of YOLOv11 for faster and more robust hand detection
- Support for DICOM format (real CT/MRI data)
- 3D organ viewer with volumetric rendering
- Gesture smoothing algorithms to reduce jitter
- Multi-hand gesture support for two-handed interactions
- Dedicated GUI interface
- GPU-accelerated inference for higher frame rates
- Expanded gesture vocabulary (panning, rotation)

Overall, this project provides a strong proof-of-concept foundation for touchless human–computer interaction in sterile clinical environments, combining accessible AI tools with a clear, practical healthcare application.

---

## 9. References

1. **MediaPipe Hand Landmarker**  
   Google LLC. *MediaPipe Solutions — Hand Landmark Detection.*  
   https://developers.google.com/mediapipe/solutions/vision/hand_landmarker

2. **OpenCV**  
   Bradski, G. (2000). *The OpenCV Library.* Dr. Dobb's Journal of Software Tools.  
   https://opencv.org

3. **MiDaS — Towards Robust Monocular Depth Estimation**  
   Ranftl, R., Lasinger, K., Hafner, D., Schindler, K., & Koltun, V. (2020).  
   *Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer.*  
   IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI).  
   https://github.com/isl-org/MiDaS

4. **PyTorch**  
   Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.*  
   Advances in Neural Information Processing Systems 32 (NeurIPS 2019).  
   https://pytorch.org

5. **NumPy**  
   Harris, C.R., et al. (2020). *Array programming with NumPy.*  
   Nature, 585, 357–362.  
   https://numpy.org

6. **PyAutoGUI**  
   Sweigart, A. *PyAutoGUI Documentation.*  
   https://pyautogui.readthedocs.io

7. **Sterile Field in Surgery — Clinical Background**  
   Association of periOperative Registered Nurses (AORN). *Guidelines for Perioperative Practice.*  
   https://www.aorn.org

8. **Hand Gesture Recognition Survey**  
   Oudah, M., Al-Naji, A., & Chahl, J. (2020). *Hand Gesture Recognition Based on Computer Vision: A Review of Techniques.* Journal of Imaging, 6(8), 73.  
   https://doi.org/10.3390/jimaging6080073

9. **Python Software Foundation**  
   *Python 3 Documentation.*  
   https://docs.python.org/3/

---

*Report prepared for the Sterile Interface project — a gesture-based, touch-free medical image viewer for sterile surgical environments.*
