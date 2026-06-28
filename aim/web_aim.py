
import ctypes, time, threading, json, os
import logging
logging.getLogger("werkzeug").setLevel(logging.ERROR)
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

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
os.makedirs(CONFIG_DIR, exist_ok=True)
_CFG_NONSAVE = {"model_path","left","top","right","down"}

class Config:
    def __init__(self):
        self.speed = 0.5; self.predict = 0; self.smooth = 0.85
        self.slow_radius = 60; self.slow_min = 0.12
        self.aim_timeout = 0.3
        self.area_x = 320; self.area_y = 320; self.conf = 0.45
        self.classes_list = [0,1]; self.move_X = 0; self.move_Y = 0
        self.hit_ox = 0; self.hit_oy = 0
        self.ox = 1920; self.oy = 1080; self.aim_enabled = True
        self.model_path = "perfect.engine"
        if self.model_path.endswith(".engine"):
            pt = self.model_path.replace(".engine", ".pt")
            if os.path.exists(pt): self.model_path = pt
        self._profile = "default"
        self._load("default")
        self.update_area()
    def update_area(self):
        self.left = self.ox // 2 - self.area_x // 2
        self.right = self.ox // 2 + self.area_x // 2
        self.top = self.oy // 2 - self.area_y // 2
        self.down = self.oy // 2 + self.area_y // 2
    def to_dict(self):
        return {"speed": self.speed, "predict": self.predict, "smooth": self.smooth,
                "slow_radius": self.slow_radius, "slow_min": self.slow_min,
                "aim_timeout": self.aim_timeout,
                "area_x": self.area_x, "area_y": self.area_y,
                "conf": self.conf, "classes_list": self.classes_list,
                "move_X": self.move_X, "move_Y": self.move_Y,
                "ox": self.ox, "oy": self.oy,
                "aim_enabled": self.aim_enabled, "hit_ox": self.hit_ox, "hit_oy": self.hit_oy,
                "_profile": self._profile}
    def _path(self, name): return os.path.join(CONFIG_DIR, name+".json")
    def save(self, name=None):
        name = name or self._profile
        d = {k: v for k, v in self.to_dict().items() if k not in _CFG_NONSAVE and not k.startswith("_")}
        with open(self._path(name), "w") as f: json.dump(d, f, indent=2)
        self._profile = name
    def _load(self, name):
        p = self._path(name)
        if not os.path.exists(p): return
        with open(p) as f: d = json.load(f)
        for k, v in d.items():
            if hasattr(self, k) and k not in _CFG_NONSAVE:
                setattr(self, k, v)
        self._profile = name
    def list_profiles(self):
        names = []
        for fn in os.listdir(CONFIG_DIR):
            if fn.endswith(".json"): names.append(fn[:-5])
        return sorted(names) if names else ["default"]
    def delete_profile(self, name):
        p = self._path(name)
        if os.path.exists(p): os.remove(p)
cfg = Config()
cfg_lock = threading.Lock()

print("加载模型中...")
yolo = YOLO(model=cfg.model_path, task="detect")
print("模型加载完成")

recorder = AimRecorder()
sct = None; latest_frame = None; frame_lock = threading.Lock()
target_info = None; target_lock = threading.Lock()
stats_lock = threading.Lock()
yolo_fps = 0.0
yolo_device = "GPU" if torch.cuda.is_available() else "CPU"

def detection_loop():
    global latest_frame, sct, target_info, yolo_fps
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
                with target_lock: target_info = (l+(x1+x2)/2, t+(y1+y2)/2, l+x1, t+y1, l+x2, t+y2)
            else:
                with target_lock: target_info = None
        else:
            with target_lock: target_info = None
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
    last_on_target = 0.0
    while True:
        with cfg_lock:
            ox=cfg.ox; oy=cfg.oy; sp=cfg.speed; sm=cfg.smooth
            mx=cfg.move_X; my=cfg.move_Y; aim_on=cfg.aim_enabled
            to_cfg=cfg.aim_timeout; hx=cfg.hit_ox; hy=cfg.hit_oy
        with target_lock:
            info = target_info
        if info and aim_on:
            cx, cy, x1, y1, x2, y2 = info
            err_x = cx - ox/2 + mx
            err_y = cy - oy/2 + my
            dist = np.hypot(err_x, err_y)
            on_target = (x1-hx <= ox/2 <= x2+hx and y1-hy <= oy/2 <= y2+hy)
            t_now = time.time()
            if on_target:
                last_on_target = t_now
                smooth_dx = 0.0
                smooth_dy = 0.0
            else:
                dx_raw, dy_raw = response_curve(err_x, err_y, aim_params)
                if dx_raw or dy_raw:
                    slow_factor = max(cfg.slow_min, min(1.0, dist / cfg.slow_radius))
                    dx_raw = dx_raw * sp * slow_factor
                    dy_raw = dy_raw * sp * slow_factor
                    if t_now - last_on_target >= to_cfg:
                        sdx = max(-127, min(127, int(round(dx_raw))))
                        sdy = max(-127, min(127, int(round(dy_raw))))
                    else:
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

H="""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Valorant AI 自瞄</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,'Segoe UI',sans-serif;background:#0d1117;color:#c9d1d9;display:flex;height:100vh;overflow:hidden}

/* video area */
.video-wrap{flex:1;display:flex;align-items:center;justify-content:center;background:#010409;position:relative;min-width:0}
.video-wrap img{max-width:100%;max-height:100vh;display:block;image-rendering:pixelated}
.video-wrap .overlay{position:absolute;top:12px;left:12px;font-size:13px;background:rgba(0,0,0,0.7);padding:4px 10px;border-radius:6px;color:#8b949e;pointer-events:none}
.video-wrap .overlay span{color:#58a6ff}

/* sidebar */
.panel{width:340px;background:#161b22;border-left:1px solid #30363d;display:flex;flex-direction:column;overflow:hidden;flex-shrink:0}
.panel-header{padding:16px 16px 12px;border-bottom:1px solid #30363d}
.panel-header h1{font-size:16px;color:#f0f6fc;font-weight:600;letter-spacing:-0.3px}
.panel-header .sub{font-size:11px;color:#8b949e;margin-top:2px}
.panel-body{flex:1;overflow-y:auto;padding:8px 16px 16px}
.panel-body::-webkit-scrollbar{width:6px}
.panel-body::-webkit-scrollbar-thumb{background:#30363d;border-radius:3px}

/* sections */
.section{margin-top:14px}
.section-title{font-size:12px;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;display:flex;align-items:center;gap:6px}

/* rows */
.row{display:flex;align-items:center;margin:5px 0;gap:8px}
.row label{width:64px;font-size:12px;color:#8b949e;flex-shrink:0}
.row .val{min-width:36px;text-align:right;font-size:12px;color:#c9d1d9;font-variant-numeric:tabular-nums;flex-shrink:0}

/* range slider */
input[type=range]{-webkit-appearance:none;appearance:none;flex:1;height:4px;background:#21262d;border-radius:2px;outline:none;cursor:pointer}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:14px;height:14px;border-radius:50%;background:#58a6ff;border:2px solid #1f6feb;cursor:pointer;transition:.15s}
input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.15)}
input[type=range]::-moz-range-thumb{width:14px;height:14px;border-radius:50%;background:#58a6ff;border:2px solid #1f6feb;cursor:pointer}

/* number input */
input[type=number]{width:64px;background:#0d1117;border:1px solid #30363d;color:#c9d1d9;padding:3px 6px;border-radius:4px;font-size:12px;font-family:inherit}
input[type=number]:focus{border-color:#58a6ff;outline:none}

/* checkbox */
.checkbox{display:flex;align-items:center;gap:8px;margin:6px 0}
.checkbox label{font-size:13px;cursor:pointer;color:#c9d1d9}
input[type=checkbox]{accent-color:#58a6ff;width:15px;height:15px;cursor:pointer}

/* button */
.btn{background:#21262d;border:1px solid #30363d;color:#c9d1d9;padding:5px 14px;border-radius:6px;cursor:pointer;font-size:12px;transition:.15s}
.btn:hover{background:#30363d;border-color:#8b949e}
.btn-primary{background:#1f6feb;border-color:#1f6feb;color:#fff}
.btn-primary:hover{background:#388bfd}

/* status */
.status-bar{padding:8px 16px;border-top:1px solid #30363d;display:flex;justify-content:space-between;font-size:11px;color:#8b949e;flex-shrink:0}
.status-bar .item{display:flex;align-items:center;gap:4px}
.status-bar .dot{width:7px;height:7px;border-radius:50%;display:inline-block}
.dot-green{background:#3fb950}
.dot-gray{background:#30363d}

/* toggle switch */
.toggle-wrap{display:flex;align-items:center;gap:10px;margin:4px 0}
.toggle{position:relative;width:36px;height:20px;cursor:pointer}
.toggle input{opacity:0;width:0;height:0}
.toggle .slider{position:absolute;inset:0;background:#21262d;border-radius:10px;transition:.2s}
.toggle .slider::before{content:'';position:absolute;width:14px;height:14px;left:3px;bottom:3px;background:#8b949e;border-radius:50%;transition:.2s}
.toggle input:checked+.slider{background:#1f6feb}
.toggle input:checked+.slider::before{transform:translateX(16px);background:#fff}
.toggle-label{font-size:13px;cursor:pointer;color:#c9d1d9}

/* class tag */
.tag-group{display:flex;gap:6px;margin:4px 0}
.tag{display:flex;align-items:center;gap:4px;background:#21262d;border:1px solid #30363d;padding:3px 8px;border-radius:12px;font-size:11px;cursor:pointer;transition:.15s;color:#8b949e;user-select:none}
.tag.active{border-color:#58a6ff;color:#58a6ff;background:#0d1117}
.tag input{display:none}
</style></head><body>
<div class="video-wrap">
  <img src="/video_feed" id="feed" alt="feed">
  <div class="overlay"><span id="fps_display">--</span> FPS</div>
</div>
<div class="panel">
  <div class="panel-header">
    <h1>⚡ 自瞄控制</h1>
    <div class="sub">Valorant AI · YOLO11</div>
  </div>
  <div class="panel-body">

    <div class="toggle-wrap">
      <label class="toggle"><input type="checkbox" id="aim_enabled" checked onchange="updateParam('aim_enabled',this.checked)"><span class="slider"></span></label>
      <label class="toggle-label" for="aim_enabled">自瞄</label>
    </div>
    <div class="toggle-wrap">
      <label class="toggle"><input type="checkbox" id="recording" onchange="toggleRecord(this.checked)"><span class="slider"></span></label>
      <label class="toggle-label" for="recording">录制轨迹</label>
    </div>

    <div class="section">
      <div class="section-title">🎯 鼠标</div>
      <div class="row"><label>速度</label><input type="range" id="speed" min="0.1" max="2.0" step="0.05" value="0.5" oninput="updateParam('speed',this.value)"><span class="val" id="speed_val">0.50</span></div>
      <div class="row"><label>平滑</label><input type="range" id="smooth" min="0" max="0.98" step="0.02" value="0.85" oninput="updateParam('smooth',this.value)"><span class="val" id="smooth_val">0.85</span></div>
      <div class="row"><label>偏移X</label><input type="range" id="move_X" min="-30" max="30" step="1" value="0" oninput="updateParam('move_X',this.value)"><span class="val" id="move_X_val">0</span></div>
      <div class="row"><label>偏移Y</label><input type="range" id="move_Y" min="-30" max="30" step="1" value="0" oninput="updateParam('move_Y',this.value)"><span class="val" id="move_Y_val">0</span></div>
    </div>

    <div class="section">
      <div class="section-title">🐢 减速</div>
      <div class="row"><label>减速半径</label><input type="range" id="slow_radius" min="10" max="200" step="5" value="60" oninput="updateParam('slow_radius',this.value)"><span class="val" id="slow_radius_val">60</span></div>
      <div class="row"><label>最低速度</label><input type="range" id="slow_min" min="0.02" max="0.50" step="0.02" value="0.12" oninput="updateParam('slow_min',this.value)"><span class="val" id="slow_min_val">0.12</span></div>
      <div class="row"><label>瞄准超时</label><input type="range" id="aim_timeout" min="0.05" max="1.0" step="0.05" value="0.3" oninput="updateParam('aim_timeout',this.value)"><span class="val" id="aim_timeout_val">0.30</span></div>
      <div class="row"><label>命中范围X</label><input type="range" id="hit_ox" min="-50" max="50" step="1" value="0" oninput="updateParam('hit_ox',this.value)"><span class="val" id="hit_ox_val">0</span></div>
      <div class="row"><label>命中范围Y</label><input type="range" id="hit_oy" min="-50" max="50" step="1" value="0" oninput="updateParam('hit_oy',this.value)"><span class="val" id="hit_oy_val">0</span></div>
    </div>

    <div class="section">
      <div class="section-title">📷 检测</div>
      <div class="row"><label>置信度</label><input type="range" id="conf" min="0.1" max="0.9" step="0.05" value="0.3" oninput="updateParam('conf',this.value)"><span class="val" id="conf_val">0.30</span></div>
      <div class="tag-group">
        <label class="tag active" id="tag_0"><input type="checkbox" checked onchange="updCls()">身体</label>
        <label class="tag active" id="tag_1"><input type="checkbox" checked onchange="updCls()">头部</label>
      </div>
      <div style="margin-top:8px"><button class="btn" onclick="reloadParams()">🔄 重载参数</button></div>
    </div>

    <div class="section">
      <div class="section-title">🖥 截图</div>
      <div class="row"><label>范围X</label><input type="range" id="area_x" min="64" max="640" step="32" value="320" oninput="updateParam('area_x',this.value)"><span class="val" id="area_x_val">320</span></div>
      <div class="row"><label>范围Y</label><input type="range" id="area_y" min="64" max="640" step="32" value="320" oninput="updateParam('area_y',this.value)"><span class="val" id="area_y_val">320</span></div>
      <div class="row"><label>分辨率X</label><input type="number" id="ox" value="1920" onchange="updateParam('ox',this.value)"></div>
      <div class="row"><label>分辨率Y</label><input type="number" id="oy" value="1080" onchange="updateParam('oy',this.value)"></div>
    </div>

    <div class="section">
      <div class="section-title">📂 配置管理</div>
      <div class="row" style="gap:4px">
        <select id="profile_select" style="flex:1;background:#0d1117;border:1px solid #30363d;color:#c9d1d9;padding:4px 6px;border-radius:4px;font-size:12px"></select>
        <button class="btn" onclick="loadProfile()" title="加载">📂</button>
        <button class="btn" onclick="saveProfile()" title="保存">💾</button>
        <button class="btn" onclick="deleteProfile()" title="删除">🗑</button>
      </div>
      <div style="margin-top:4px;display:flex;gap:4px">
        <input type="text" id="profile_name" placeholder="配置名称" style="flex:1;background:#0d1117;border:1px solid #30363d;color:#c9d1d9;padding:4px 6px;border-radius:4px;font-size:12px">
        <button class="btn btn-primary" onclick="saveAsProfile()">另存为</button>
      </div>
    </div>
<script>
function updateParam(k,v){const n=parseFloat(v);fetch('/update',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key:k,value:isNaN(n)?v:n})});const e=document.getElementById(k+'_val');if(e)e.textContent=typeof n==='number'?n.toFixed(n%1===0?0:2):v}
function reloadParams(){fetch('/reload_params').then(r=>r.json()).then(d=>{console.log('Params reloaded',d.params);alert('参数已重载')})}
function updCls(){const c=[];if(document.getElementById('tag_0').querySelector('input').checked)c.push(0);if(document.getElementById('tag_1').querySelector('input').checked)c.push(1);fetch('/update',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key:'classes_list',value:c})});document.getElementById('tag_0').classList.toggle('active',c.includes(0));document.getElementById('tag_1').classList.toggle('active',c.includes(1))}
function toggleRecord(on){fetch('/record',{method:'POST',body:JSON.stringify({recording:on}),headers:{'Content-Type':'application/json'}})}
async function refreshProfiles(){const r=await fetch('/configs');const d=await r.json();const sel=document.getElementById('profile_select');const cur=d.current;sel.innerHTML='';d.profiles.forEach(p=>{const o=document.createElement('option');o.value=p;o.text=p;if(p===cur)o.selected=true;sel.appendChild(o)});document.getElementById('profile_name').value=''}
async function loadProfile(){const name=document.getElementById('profile_select').value;if(!name)return;const r=await fetch('/config_load',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});const d=await r.json();if(d.ok){for(const[k,v]of Object.entries(d.profile)){const e=document.getElementById(k+'_val');if(e)e.textContent=typeof v==='number'?v.toFixed(v%1===0?0:2):v;const s=document.getElementById(k);if(s&&s.tagName==='SELECT')s.value=v;const c=document.getElementById(k);if(c&&c.type==='checkbox')c.checked=v};if(document.getElementById('tag_0')){document.getElementById('tag_0').classList.toggle('active',d.profile.classes_list.includes(0));document.getElementById('tag_1').classList.toggle('active',d.profile.classes_list.includes(1))};refreshProfiles()}}
async function saveAsProfile(){const name=document.getElementById('profile_name').value.trim();if(!name)return alert('输入配置名称');await fetch('/config_save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});refreshProfiles()}
async function saveProfile(){await fetch('/config_save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:document.getElementById('profile_select').value})});refreshProfiles()}
async function deleteProfile(){const name=document.getElementById('profile_select').value;if(!name||name==='default')return;if(!confirm('删除配置 "'+name+'"？'))return;await fetch('/config_delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});refreshProfiles()}
setInterval(async()=>{
  const r=await fetch('/params');const p=await r.json();
  for(const[k,v]of Object.entries(p)){const e=document.getElementById(k+'_val');if(e)e.textContent=typeof v==='number'?v.toFixed(v%1===0?0:2):v;const s=document.getElementById(k);if(s&&s.tagName==='SELECT')s.value=v}
  const s=await fetch('/stats');const d=await s.json();
  document.getElementById('stats_text').textContent=d.device;document.getElementById('fps_text').textContent=d.fps;document.getElementById('fps_display').textContent=d.fps;
  const dot=document.getElementById('status_dot');dot.className='dot '+(d.fps>0?'dot-green':'dot-gray')
},1000)
refreshProfiles()
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
        cfg.save()
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
@app.route("/configs")
def configs():
    return jsonify({"profiles": cfg.list_profiles(), "current": cfg._profile})
@app.route("/config_save",methods=["POST"])
def config_save():
    d=request.get_json(); name=d.get("name","default")
    with cfg_lock: cfg.save(name)
    return jsonify({"ok":True, "current": name})
@app.route("/config_load",methods=["POST"])
def config_load():
    d=request.get_json(); name=d.get("name","default")
    with cfg_lock:
        cfg._load(name)
        cfg.update_area()
    return jsonify({"ok":True, "profile": cfg.to_dict()})
@app.route("/config_delete",methods=["POST"])
def config_delete():
    d=request.get_json(); name=d.get("name")
    with cfg_lock: cfg.delete_profile(name)
    return jsonify({"ok":True})
if __name__=="__main__":
    threading.Thread(target=detection_loop,daemon=True).start()
    threading.Thread(target=mouse_loop,daemon=True).start()
    print("打开 http://127.0.0.1:5000")
    app.run(host="127.0.0.1",port=5000,debug=False,threaded=True)
