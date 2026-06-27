"""
Real-time Valorant enemy detection overlay
Capture screen → ONNX inference → show detections
Safe: read-only screen capture, no game injection
"""
import cv2
import numpy as np
import os
import time
from ultralytics import YOLO

# ─── Config ──────────────────────────────────────────
MODEL_PATH = "D:/aidemo/Valorant-Enemy-Detection-with-YOLO11-main/exported_models/optimized.onnx"
CONF_THRESH = 0.4     # confidence threshold
CAPTURE_FPS = 30      # target capture FPS
SHOW_FPS = True       # show FPS on overlay

# ─── Load model ──────────────────────────────────────
print(f"Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH, task="detect")
print("Model loaded")

# ─── Try to use dxcam for fast capture ───────────────
try:
    import dxcam
    camera = dxcam.create(output_idx=0, output_color="BGR")
    dxcam_available = True
    print("Using dxcam (DirectX capture)")
except:
    dxcam_available = False
    print("dxcam not installed, use pip install dxcam")
    print("Falling back to mss...")
    try:
        import mss
        mss_available = True
    except:
        mss_available = False
        print("mss also not installed. pip install mss dxcam")

# ─── Main loop ───────────────────────────────────────
print("\nPress 'q' to quit")
print("=" * 40)

# Create a named window
window_name = "Valorant Enemy Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 960, 540)

frame_count = 0
fps_timer = time.time()
fps = 0

while True:
    loop_start = time.time()

    # ── Capture screen ──────────────────────────────
    if dxcam_available:
        frame = camera.grab()
        if frame is None:
            time.sleep(0.01)
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR) if frame.shape[2] == 4 else frame
    else:
        # Use mss as fallback
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # primary monitor
            img = sct.grab(monitor)
            frame = np.array(img)[:, :, :3]  # BGRA → BGR

    if frame is None or frame.size == 0:
        continue

    h, w = frame.shape[:2]

    # ── Inference ───────────────────────────────────
    results = model(frame, conf=CONF_THRESH, verbose=False, imgsz=640)

    # ── Draw detections ─────────────────────────────
    annotated = results[0].plot()
    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        # Draw additional info: count
        enemy_count = len(boxes)
        cv2.putText(annotated, f"Enemies: {enemy_count}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3)

    # ── FPS counter ─────────────────────────────────
    frame_count += 1
    if frame_count >= 15:
        fps = frame_count / (time.time() - fps_timer)
        frame_count = 0
        fps_timer = time.time()

    if SHOW_FPS:
        cv2.putText(annotated, f"FPS: {fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 0), 2)

    # ── Show window ─────────────────────────────────
    # Resize for display while keeping aspect ratio
    display = cv2.resize(annotated, (960, 540))
    cv2.imshow(window_name, display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # ── FPS limiter ─────────────────────────────────
    elapsed = time.time() - loop_start
    target_dt = 1.0 / CAPTURE_FPS
    if elapsed < target_dt:
        time.sleep(target_dt - elapsed)

cv2.destroyAllWindows()
if dxcam_available:
    camera.release()
print("Stopped")
