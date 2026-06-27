
import ctypes, time, threading, json, os
import cv2
import numpy as np
from flask import Flask, render_template_string, Response, request, jsonify
from ultralytics import YOLO
import mss
import torch
from aim_optimizer import response_curve, AimParams, AimRecorder, PARAMS_FILE

# ─── 鼠标 ─────────────────
MOUSE_MODE = "gvinput"
move_mouse = None

def init_mouse():
    global move_mouse
    if MOUSE_MODE == "gvinput":
        from gvinput_wrapper import GVInputMouse
        gv = GVInputMouse()
        def _move(dx, dy): gv.move_relative(dx, dy)
        move_mouse = _move
        print("鼠标: GVInput")
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
        print("鼠标: SendInput")

init_mouse()

# ─── 配置 ─────────────────
aim_params = AimParams.load()
print(f"  自瞄参数: {json.dumps(aim_params.to_dict())}")

def reload_params():
    global aim_params
    aim_params = AimParams.load()
    print(f"  🔄 参数重载: {json.dumps(aim_params.to_dict())}")

class Config:
    def __init__(self):
        self.speed = 0.5; self.predict = 0; self.smooth = 0.85
        self.slow_radius = 60; self.slow_min = 0.12
        self.area_x = 320; self.area_y = 320; self.conf = 0.45
        self.classes_list = [0,1]; self.move_X = 0; self.move_Y = 0
        self.ox = 1920; self.oy = 1080; self.aim_enabled = True
        self.model_path = "perfect.engine"
        if self.model_path.endswith(".engine"):
            pt = self.model_path.replace(".engine", ".pt")
            if os.path.exists(pt): self.model_path = pt
        self.left = self.ox // 2 - self.area_x // 2
        self.top = self.oy // 2 - self.area_y // 2
    def update_area(self):
        self.left = self.ox // 2 - self.area_x // 2
        self.right = self.ox // 2 + self.area_x // 2
        self.top = self.oy // 2 - self.area_y // 2
        self.down = self.oy // 2 + self.area_y // 2
    def to_dict(self):
        return {"speed": self.speed, "predict": self.predict, "smooth": self.smooth,
                "slow_radius": self.slow_radius, "slow_min": self.slow_min,
                "area_x": self.area_x, "area_y": self.area_y,
                "conf": self.conf, "classes_list": self.classes_list,
                "move_X": self.move_X, "move_Y": self.move_Y,
                "ox": self.ox, "oy": self.oy,
                "aim_enabled": self.aim_enabled}

cfg = Config()
cfg_lock = threading.Lock()

print("加载模型中...")
yolo = YOLO(model=cfg.model_path, task="detect")
print("模型加载完成")

recorder = AimRecorder()
sct = None; latest_frame = None; frame_lock = threading.Lock()
target_center = None; target_lock = threading.Lock()
stats_lock = threading.Lock()
yolo_fps = 0.0
yolo_device = "GPU" if torch.cuda.is_available() else "CPU"

def detection_loop():
    global latest_frame, sct, target_center, yolo_fps
    sct = mss.mss()
    fps_cnt = 0
    fps_t0 = time.time()
    while True:
        with cfg_lock:
            ox=cfg.ox; oy=cfg.oy; ax=cfg.area_x; ay=cfg.area_y
            conf=cfg.conf; cls=cfg.classes_list
            aim_on=cfg.aim_enabled
            l=cfg.left; t=cfg.top
        img = sct.grab({"top": t, "left": l, "width": ax, "height": ay})
        frame = np.array(img)[:, :, :3]
        results = yolo(source=frame, conf=conf, classes=cls, device=0, verbose=False)
        boxes = results[0].boxes
        fps_cnt += 1
        if fps_cnt >= 30:
            with stats_lock: yolo_fps = fps_cnt / (time.time() - fps_t0)
            fps_cnt = 0; fps_t0 = time.time()
        display = frame.copy()
        best = None; best_d = 9999999
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            for c in xyxy:
                x1,y1,x2,y2=map(int,c); cx=l+(x1+x2)/2; cy=t+(y1+y2)/2
                d=(ox//2-cx)**2+(oy//2-cy)**2
                if d<best_d: best_d=d; best=(x1,y1,x2,y2)
                cv2.rectangle(display,(x1,y1),(x2,y2),(0,255,0),2)
            if best:
                x1,y1,x2,y2=best; cx=(x1+x2)//2; cy=(y1+y2)//2
                cv2.line(display,(cx-10,cy),(cx+10,cy),(0,0,255),1)
                cv2.line(display,(cx,cy-10),(cx,cy+10),(0,0,255),1)
                with target_lock: target_center = (l+(x1+x2)/2, t+(y1+y2)/2)
            else:
                with target_lock: target_center = None
        else:
            with target_lock: target_center = None
        h,w=display.shape[:2]
        cv2.line(display,(w//2-15,h//2),(w//2+15,h//2),(255,255,255),1)
        cv2.line(display,(w//2,h//2-15),(w//2,h//2+15),(255,255,255),1)
        st = "ON" if aim_on else "OFF"
        cv2.putText(display,st,(10,h-20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0)if aim_on else(0,0,255),2)
        st_fps = f"{yolo_fps:.0f} FPS"
        cv2.putText(display,st_fps,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
        _,j=cv2.imencode(".jpg",display,[cv2.IMWRITE_JPEG_QUALITY,70])
        with frame_lock: latest_frame=j.tobytes()

def mouse_loop():
    smooth_dx = 0.0
    smooth_dy = 0.0
    while True:
        with cfg_lock:
            ox=cfg.ox; oy=cfg.oy; sp=cfg.speed; sm=cfg.smooth
            mx=cfg.move_X; my=cfg.move_Y; aim_on=cfg.aim_enabled
        with target_lock:
            target = target_center
        if target and aim_on:
            cx, cy = target
            err_x = cx - ox/2 + mx
            err_y = cy - oy/2 + my
            dist = np.hypot(err_x, err_y)
            t_now = time.time()
            dx_raw, dy_raw = response_curve(err_x, err_y, aim_params)
            if dx_raw or dy_raw:
                slow_factor = max(cfg.slow_min, min(1.0, dist / cfg.slow_radius))
                dx_raw = dx_raw * sp * slow_factor
                dy_raw = dy_raw * sp * slow_factor
                smooth_dx = sm * smooth_dx + (1 - sm) * dx_raw
                smooth_dy = sm * smooth_dy + (1 - sm) * dy_raw
                sdx = max(-127, min(127, int(round(smooth_dx))))
                sdy = max(-127, min(127, int(round(smooth_dy))))
                if sdx or sdy:
                    move_mouse(sdx, sdy)
                if recorder.recording:
                    recorder.record(t_now, cx, cy, sdx, sdy, ox/2, oy/2)
        else:
            smooth_dx = 0.0
            smooth_dy = 0.0
        time.sleep(0.005)
        

def gen_frames():
    while True:
        with frame_lock:
            if latest_frame is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+latest_frame+b"\r\n"
        time.sleep(0.03)

app = Flask(__name__)

H="""<!DOCTYPE html><html><head><meta charset="utf-8"><title>自瞄</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',sans-serif;background:#1a1a2e;color:#eee;display:flex;height:100vh}
.video{flex:1;display:flex;align-items:center;justify-content:center;background:#0f0f23}
.video img{max-width:100%;max-height:100vh;border:1px solid #333}
h2{font-size:16px;margin:16px 0 8px;color:#0f0}
.row{display:flex;align-items:center;margin:6px 0}
.row label{width:80px;font-size:13px;color:#aaa}
.row .val{width:40px;text-align:right;font-size:13px}
.row input[type=number]{width:60px;background:#0f3460;border:1px solid #333;color:#eee;padding:2px 4px}
.checkbox{display:flex;align-items:center;gap:8px;margin:10px 0}
.checkbox label{font-size:14px}
select{background:#0f3460;border:1px solid #333;color:#eee;padding:4px;width:100%;border-radius:4px}
.section{border-top:1px solid #333;padding-top:12px;margin-top:12px}
.status{font-size:12px;color:#888;margin-top:20px;text-align:center}
</style></head><body>
<h2>⚡ 自瞄</h2>
<div class="checkbox"><input type="checkbox" id="aim_enabled" checked onchange="updateParam('aim_enabled',this.checked)"><label for="aim_enabled">启用</label></div>
<div class="checkbox"><input type="checkbox" id="recording" onchange="toggleRecord(this.checked)"><label for="recording">📹 录制轨迹</label></div>
<div class="section">
<h2>🎯 鼠标</h2>
<div class="row"><label>速度</label><input type="range" id="speed" min="0.1" max="2.0" step="0.05" value="0.5" oninput="updateParam('speed',this.value)"><span class="val" id="speed_val">0.50</span></div>
<div class="row"><label>平滑</label><input type="range" id="smooth" min="0" max="0.98" step="0.02" value="0.85" oninput="updateParam('smooth',this.value)"><span class="val" id="smooth_val">0.85</span></div>
<div class="row"><label>偏移X</label><input type="range" id="move_X" min="-30" max="30" step="1" value="0" oninput="updateParam('move_X',this.value)"><span class="val" id="move_X_val">0</span></div>
<div class="row"><label>偏移Y</label><input type="range" id="move_Y" min="-30" max="30" step="1" value="0" oninput="updateParam('move_Y',this.value)"><span class="val" id="move_Y_val">0</span></div>
</div>
<div class="section">
<h2>🐢 减速</h2>
<div class="row"><label>减速半径</label><input type="range" id="slow_radius" min="10" max="200" step="5" value="60" oninput="updateParam('slow_radius',this.value)"><span class="val" id="slow_radius_val">60</span></div>
<div class="row"><label>最低速度</label><input type="range" id="slow_min" min="0.02" max="0.50" step="0.02" value="0.12" oninput="updateParam('slow_min',this.value)"><span class="val" id="slow_min_val">0.12</span></div>
</div>
<div class="section">
<h2>📷 检测</h2>
<div class="row"><label>置信度</label><input type="range" id="conf" min="0.1" max="0.9" step="0.05" value="0.3" oninput="updateParam('conf',this.value)"><span class="val" id="conf_val">0.30</span></div>
<div class="checkbox"><input type="checkbox" id="cls_head" checked onchange="updCls()"><label>头部(0)</label><input type="checkbox" id="cls_body" checked onchange="updCls()"><label>身体(1)</label></div>
<div style="text-align:center;margin-top:8px"><button onclick="reloadParams()" style="background:#0f3460;border:1px solid #333;color:#eee;padding:6px 16px;border-radius:4px;cursor:pointer">🔄 重载参数</button></div>
<div class="status">响应曲线参数由离线拟合决定</div>
<h2>🖥 截图</h2>
<div class="row"><label>范围X</label><input type="range" id="area_x" min="64" max="640" step="32" value="320" oninput="updateParam('area_x',this.value)"><span class="val" id="area_x_val">320</span></div>
<div class="row"><label>范围Y</label><input type="range" id="area_y" min="64" max="640" step="32" value="320" oninput="updateParam('area_y',this.value)"><span class="val" id="area_y_val">320</span></div>
<div class="row"><label>分辨率X</label><input type="number" id="ox" value="1920" onchange="updateParam('ox',this.value)"></div>
<div class="row"><label>分辨率Y</label><input type="number" id="oy" value="1080" onchange="updateParam('oy',this.value)"></div>
</div>
<div class="status"><span id="stats_text">等待中...</span></div>
</div>
<script>
function updateParam(k,v){const n=parseFloat(v);fetch('/update',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key:k,value:isNaN(n)?v:n})});const e=document.getElementById(k+'_val');if(e)e.textContent=typeof n==='number'?n.toFixed(n%1===0?0:2):v}
function reloadParams(){fetch('/reload_params').then(r=>r.json()).then(d=>{console.log('Params reloaded',d.params);alert('参数已重载')})}
function updCls(){const c=[];if(document.getElementById('cls_head').checked)c.push(0);if(document.getElementById('cls_body').checked)c.push(1);fetch('/update',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key:'classes_list',value:c})})}
function toggleRecord(on){fetch('/record',{method:'POST',body:JSON.stringify({recording:on}),headers:{'Content-Type':'application/json'}})}
setInterval(async()=>{const r=await fetch('/params');const p=await r.json();for(const[k,v]of Object.entries(p)){const e=document.getElementById(k+'_val');if(e)e.textContent=typeof v==='number'?v.toFixed(v%1===0?0:2):v;const s=document.getElementById(k);if(s&&s.tagName==='SELECT')s.value=v};const s=await fetch('/stats');const d=await s.json();document.getElementById('stats_text').textContent=d.device+' · '+d.fps+' FPS'},1000)
</script></body></html>"""

@app.route("/")
def idx(): return render_template_string(H)
@app.route("/video_feed")
def vf(): return Response(gen_frames(),mimetype="multipart/x-mixed-replace; boundary=frame")
@app.route("/update",methods=["POST"])
def upd():
    d=request.get_json(); k=d.get("key"); v=d.get("value")
    with cfg_lock:
        if hasattr(cfg,k): setattr(cfg,k,v)
        if k in ("area_x","area_y","ox","oy"): cfg.update_area()
    return jsonify({"ok":True})
@app.route("/reload_params")
def rlp():
    reload_params()
    return jsonify({"ok":True, "params": aim_params.to_dict()})
@app.route("/record",methods=["POST"])
def rec():
    d=request.get_json()
    if d.get("recording"):
        recorder.start()
    else:
        recorder.stop()
    return jsonify({"ok":True})
@app.route("/params")
def prm():
    with cfg_lock: return jsonify(cfg.to_dict())
@app.route("/stats")
def stt():
    with stats_lock: fps = yolo_fps
    return jsonify({"fps": round(fps, 1), "device": yolo_device})
if __name__=="__main__":
    threading.Thread(target=detection_loop,daemon=True).start()
    threading.Thread(target=mouse_loop,daemon=True).start()
    print("打开 http://127.0.0.1:5000")
    app.run(host="127.0.0.1",port=5000,debug=False,threaded=True)
