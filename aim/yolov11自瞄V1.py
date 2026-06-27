
import os, time, ctypes, numpy as np
import mss
from ultralytics import YOLO
from aim_optimizer import response_curve, AimParams

mode = "gvinput"
area_x, area_y = 320, 320
conf = 0.3
classes_list = [0, 1]
model = r"D:\aidemo\Valorant-Enemy-Detection-with-YOLO11-main\aim\perfect.engine"
move_X = 10
move_Y = 8
ox, oy = 1920, 1080

# ─── 鼠标 ─────────────────
def init_mouse():
    global move_mouse
    if mode == "gvinput":
        from gvinput_wrapper import GVInputMouse
        gv = GVInputMouse()
        def _move(dx, dy): gv.move_relative(dx, dy)
        move_mouse = _move
        print("✓ GVInput")
    elif mode == "interception":
        from interception import Interception, MouseFlag, MouseStroke
        import interception as ic
        ctx = Interception(); dev = None
        for i in range(11, 20):
            if ic.get_mouse(i): dev = i; break
        if dev is None: print("✗ 无 Interception"); raise SystemExit(1)
        def _move(dx, dy):
            ctx.send(dev, MouseStroke(flags=MouseFlag.MOUSE_MOVE_RELATIVE, x=int(dx), y=int(dy), rolling=0, information=0))
        move_mouse = _move
        print(f"✓ Interception ({dev})")
    elif mode == "hardware":
        import serial
        num = input("串口: COM")
        ser = serial.Serial(f"COM{num}", 115200, timeout=0.01)
        time.sleep(2)
        def _move(dx, dy): ser.write(f"{dx} {dy}\n".encode())
        move_mouse = _move
        print(f"✓ 串口 COM{num}")
    else:
        class MI(ctypes.Structure):
            _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                        ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                        ("time", ctypes.c_ulong), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]
        class I(ctypes.Structure):
            _fields_ = [("type", ctypes.c_ulong), ("mi", MI)]
        def _move(dx, dy):
            inp = I(0); inp.mi = MI(int(dx), int(dy), 0, 0x0001, 0, None)
            ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
        move_mouse = _move
        print("○ SendInput")

move_mouse = None

# ─── 加载模型和参数 ────────
model_path = model
if model_path.endswith(".engine"):
    pt = model_path.replace(".engine", ".pt")
    if os.path.exists(pt): model_path = pt

print(f"模型: {model_path}")
yolo = YOLO(model=model_path, task="detect")

aim_params = AimParams.load()
print(f"自瞄参数: {aim_params.to_dict()}")

left = ox // 2 - area_x // 2
right = ox // 2 + area_x // 2
top = oy // 2 - area_y // 2
down = oy // 2 + area_y // 2

sct = mss.mss()
init_mouse()

print(f"截图: {area_x}x{area_y}")
print(f"区域: ({left},{top})-({right},{down})")

frame_count = 0

while True:
    img = sct.grab({"top": top, "left": left, "width": area_x, "height": area_y})
    frame = np.array(img)[:, :, :3]
    results = yolo(source=frame, conf=conf, classes=classes_list, verbose=False)
    boxes = results[0].boxes
    frame_count += 1

    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        best_d = 9999999; best = None
        for c in xyxy:
            fx, fy = (c[0]+c[2])/2, (c[1]+c[3])/2
            ax, ay = left+fx, top+fy
            d = (ox//2-ax)**2 + (oy//2-ay)**2
            if d < best_d: best_d = d; best = (fx, fy, ax, ay)
        if best:
            fx, fy, ax, ay = best
            err_x = ax - ox/2 + move_X
            err_y = ay - oy/2 + move_Y
            dx, dy = response_curve(err_x, err_y, aim_params)
            if dx or dy:
                move_mouse(dx, dy)
    else:
        if frame_count % 60 == 0:
            print(f"× 未检测到 (帧 {frame_count})")
