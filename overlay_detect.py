"""
⚠ 透明覆盖版本 - 可能存在风险，请自行判断
在游戏画面上方画检测框（类似准星软件）

安全说明：
- 不修改游戏进程，只创建一个置顶透明窗口画图
- 类似 Discord 内覆盖、MSI Afterburner
- Valorant 官方未明确禁止此类 overlay
- 但使用任何辅助工具都有被封的风险
"""
import cv2
import numpy as np
import os
import time
import threading
from ultralytics import YOLO

# ─── Config ──────────────────────────────────────────
MODEL_PATH = "D:/aidemo/Valorant-Enemy-Detection-with-YOLO11-main/exported_models/optimized.onnx"
CONF_THRESH = 0.4

# ─── Load model ──────────────────────────────────────
print(f"Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH, task="detect")
print("Model loaded")

# ─── Capture ─────────────────────────────────────────
try:
    import dxcam
    camera = dxcam.create(output_idx=0, output_color="BGR")
    print("Using dxcam")
except:
    import mss
    camera = None
    print("Using mss")

# ─── Transparent overlay window ──────────────────────
import tkinter as tk

# Capture the screen region (game window area)
# Adjust to your game window position if not fullscreen
def capture_screen():
    if camera:
        frame = camera.grab()
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)[:, :, :3]
    else:
        import mss
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            img = sct.grab(monitor)
            return np.array(img)[:, :, :3]

# Inference is done in a separate thread, overlay draws results
detections = []
lock = threading.Lock()

def inference_loop():
    global detections
    while True:
        frame = capture_screen()
        if frame is None:
            time.sleep(0.01)
            continue
        h, w = frame.shape[:2]
        results = model(frame, conf=CONF_THRESH, verbose=False, imgsz=640)
        boxes_list = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                # Scale from model inference back to screen coordinates
                boxes_list.append((int(x1), int(y1), int(x2), int(y2), conf))
        with lock:
            detections = boxes_list

threading.Thread(target=inference_loop, daemon=True).start()

# ─── Create overlay window ───────────────────────────
# Get screen size
import ctypes
user32 = ctypes.windll.user32
screen_w = user32.GetSystemMetrics(0)
screen_h = user32.GetSystemMetrics(1)

root = tk.Tk()
root.title("Enemy Detection Overlay")
root.geometry(f"{screen_w}x{screen_h}+0+0")
root.attributes('-topmost', True)
root.attributes('-transparentcolor', 'black')  # black = transparent
root.attributes('-alpha', 0.7)  # semi-transparent
root.wm_attributes('-disabled', True)  # click-through
root.overrideredirect(True)  # no window frame

canvas = tk.Canvas(root, width=screen_w, height=screen_h,
                   bg='black', highlightthickness=0)
canvas.pack()

def update_overlay():
    canvas.delete("all")
    with lock:
        current = detections.copy()
    for x1, y1, x2, y2, conf in current:
        # Draw box
        canvas.create_rectangle(x1, y1, x2, y2,
                                outline='lime', width=3)
        # Label
        canvas.create_text(x1, y1-10, anchor='sw',
                           text=f"Enemy {conf:.2f}",
                           fill='lime', font=('Arial', 14, 'bold'))
    # Enemy count
    canvas.create_text(10, 10, anchor='nw',
                       text=f"Enemies: {len(current)}",
                       fill='lime', font=('Arial', 18, 'bold'))
    root.after(30, update_overlay)

root.after(1000, update_overlay)
print("Overlay started. Press Ctrl+C in terminal to stop.")
root.mainloop()
