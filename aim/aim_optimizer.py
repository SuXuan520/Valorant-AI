"""
Aim optimizer: offline parameter fitting for simple response-curve aim

Usage:
  python aim/aim_optimizer.py --record   # record detection trace (called from web_aim.py)
  python aim/aim_optimizer.py --fit      # offline fit params from recorded trace
  python aim/aim_optimizer.py --simulate # simulate fitted params on trace
"""

import json, os, time, argparse
from dataclasses import dataclass, asdict
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
PARAMS_FILE = os.path.join(BASE, "fitted_params.json")
TRACE_FILE = os.path.join(BASE, "aim_trace.jsonl")

# ─── Parameters ──────────────────────────────────────

@dataclass
class AimParams:
    """Fittable aim parameters.

    Response curve:  dx = sign(err_x) * max_delta * (|err| / scale)^power
    """
    response_power: float = 0.65       # <1 = more precision near center
    response_scale: float = 8.0        # error (pixels) where factor = 1.0
    deadzone_pixels: float = 2.0       # ignore errors below this
    max_delta: float = 28.0            # max single-frame mouse delta

    def to_dict(self):
        return asdict(self)

    def save(self, path=PARAMS_FILE):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"  ✓ Params saved: {path}")

    @classmethod
    def load(cls, path=PARAMS_FILE):
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        return cls()


# ─── Response Curve ───────────────────────────────────

def response_curve(err_x: float, err_y: float, params: AimParams) -> tuple:
    """Nonlinear power-law response. Returns (dx, dy) as ints, clamped ±127."""
    dist = np.hypot(err_x, err_y)
    if dist < params.deadzone_pixels:
        return (0, 0)

    norm = dist / params.response_scale
    factor = min(norm ** params.response_power, 1.0)

    raw_dx = np.sign(err_x) * params.max_delta * factor * (abs(err_x) / dist)
    raw_dy = np.sign(err_y) * params.max_delta * factor * (abs(err_y) / dist)

    dx = int(round(raw_dx))
    dy = int(round(raw_dy))
    return (max(-127, min(127, dx)), max(-127, min(127, dy)))


# ─── Trace Recorder ───────────────────────────────────

class AimRecorder:
    """Records detection traces for offline fitting."""

    def __init__(self, path=TRACE_FILE):
        self.path = path
        self.entries = []
        self.recording = False

    def start(self):
        self.entries.clear()
        self.recording = True

    def stop(self):
        self.recording = False
        if self.entries:
            with open(self.path, 'w') as f:
                for e in self.entries:
                    f.write(json.dumps(e) + '\n')
            print(f"  ✓ Saved {len(self.entries)} frames to {self.path}")

    def record(self, t: float, cx: float, cy: float, dx: int, dy: int,
               screen_cx: float, screen_cy: float):
        if not self.recording:
            return
        self.entries.append({
            "t": round(t, 4),
            "cx": round(cx, 1), "cy": round(cy, 1),
            "dx": dx, "dy": dy,
            "scx": round(screen_cx, 1), "scy": round(screen_cy, 1),
        })


# ─── Offline Fitter ───────────────────────────────────

class AimFitter:
    """Offline parameter optimizer. Simulates response_curve on recorded trace."""

    def __init__(self, trace_path=TRACE_FILE):
        self.trace = []
        if os.path.exists(trace_path):
            with open(trace_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.trace.append(json.loads(line))

    def _compute_cost(self, params: AimParams) -> float:
        """Lower is better. Components: jitter + accuracy."""
        if len(self.trace) < 10:
            return 1e9

        screen_cx = self.trace[0].get("scx", 960)
        screen_cy = self.trace[0].get("scy", 540)

        sim_outputs = []
        sim_errors = []
        for e in self.trace:
            scx = e.get("scx", screen_cx)
            scy = e.get("scy", screen_cy)
            err_x = e["cx"] - scx
            err_y = e["cy"] - scy
            dx, dy = response_curve(err_x, err_y, params)
            sim_outputs.append((dx, dy))
            # residual error after movement
            sim_errors.append(np.hypot(e["cx"] - scx - dx, e["cy"] - scy - dy))

        if not sim_outputs:
            return 1e9

        # 1. Jitter: std of output deltas
        deltas = np.array(sim_outputs)
        jitter = float(np.std(deltas[:, 0]) + np.std(deltas[:, 1]))

        # 2. Accuracy: mean residual error
        accuracy = float(np.mean(sim_errors))

        # Weighted sum (jitter 0.4, accuracy 0.6)
        return 0.4 * jitter + 0.6 * accuracy

    def fit(self, n_iter: int = 200) -> AimParams:
        """Simple random search over param space."""
        if len(self.trace) < 10:
            print("  ✗ Not enough trace data (need >= 10 frames)")
            return AimParams()

        best_params = AimParams()
        best_cost = self._compute_cost(best_params)
        print(f"  Baseline cost: {best_cost:.4f}")

        param_bounds = {
            "response_power": (0.3, 1.0),
            "response_scale": (3.0, 20.0),
            "deadzone_pixels": (0.5, 8.0),
            "max_delta": (10.0, 40.0),
        }

        for it in range(n_iter):
            candidate = AimParams()
            for k, (lo, hi) in param_bounds.items():
                setattr(candidate, k, np.random.uniform(lo, hi))
            cost = self._compute_cost(candidate)
            if cost < best_cost:
                best_cost = cost
                best_params = candidate
            if (it + 1) % 50 == 0:
                print(f"    Iter {it+1}/{n_iter}  best cost: {best_cost:.4f}")

        print(f"\n  ✓ Best cost: {best_cost:.4f}")
        print(f"  Best params: {json.dumps(best_params.to_dict(), indent=2)}")
        return best_params


# ─── CLI ──────────────────────────────────────────────

def cmd_record():
    recorder = AimRecorder()
    recorder.start()
    print("  Recording aim trace... Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        recorder.stop()

def cmd_fit():
    fitter = AimFitter()
    best = fitter.fit(n_iter=200)
    best.save()

def cmd_simulate():
    if not os.path.exists(TRACE_FILE):
        print("  ✗ No trace file")
        return
    params = AimParams.load()
    print(f"  Loaded params: {json.dumps(params.to_dict(), indent=2)}")
    fitter = AimFitter()
    cost = fitter._compute_cost(params)
    print(f"  Cost on trace: {cost:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aim optimizer")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()

    if args.record:
        cmd_record()
    elif args.fit:
        cmd_fit()
    elif args.simulate:
        cmd_simulate()
    else:
        cmd_fit()
        cmd_simulate()
