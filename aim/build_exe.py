#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-click build script for Valorant AI WebAim

Usage:
  python build_exe.py          # GPU version (default, ~4 GB)
  python build_exe.py --cpu    # CPU version (smaller, no NVIDIA GPU needed)

Output:
  dist/WebAim/         GPU version
  dist/WebAim-CPU/     CPU version

Creates .venv-cpu/ for CPU-only torch on first --cpu build.
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_FILES = ["perfect.pt", "perfect.onnx", "perfect.engine"]

parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true", help="Build CPU-only version")
args = parser.parse_args()

CPU_MODE = args.cpu
DIST_DIR = PROJECT_DIR / "dist" / ("WebAim-CPU" if CPU_MODE else "WebAim")
BUILD_DIR = PROJECT_DIR / "build"


def find_unicodedata_pyd(python_exe=None):
    py = python_exe or sys.executable
    result = subprocess.run(
        [py, "-c", "import unicodedata; print(unicodedata.__file__)"],
        capture_output=True, text=True, cwd=str(PROJECT_DIR),
    )
    p = Path(result.stdout.strip())
    return str(p) if p.exists() else None


def step(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def run(cmd, **kwargs):
    print(f"  $ {' '.join(cmd)}")
    kwargs.setdefault("cwd", str(PROJECT_DIR))
    return subprocess.run(cmd, check=True, **kwargs)


PYTHON_EXE = sys.executable

if CPU_MODE:
    CPU_VENV = PROJECT_DIR / ".venv-cpu"
    PYTHON_EXE = str(CPU_VENV / "Scripts" / "python.exe")

    if not CPU_VENV.exists():
        step("Creating CPU virtual environment")
        run([sys.executable, "-m", "venv", str(CPU_VENV), "--clear"])
        run([PYTHON_EXE, "-m", "pip", "install", "--upgrade", "pip"])
        run([PYTHON_EXE, "-m", "pip", "install",
             "torch", "torchvision", "--index-url",
             "https://download.pytorch.org/whl/cpu"])

    step("Installing deps to CPU venv")
    run([PYTHON_EXE, "-m", "pip", "install", "--quiet",
         "pyinstaller",
         "ultralytics", "opencv-python", "numpy",
         "flask", "mss", "pywinusb", "matplotlib",
         "pillow", "psutil", "pyyaml",
    ])

# 1. Check PyInstaller
step("1/5 Checking PyInstaller")
run([PYTHON_EXE, "-c", "import PyInstaller"])
print("  [OK] PyInstaller ready")

# 2. Prepare spec
step("2/5 Preparing spec")
spec_path = PROJECT_DIR / "web_aim.spec"
if not spec_path.exists():
    print("  Generating spec ...")
    run([
        PYTHON_EXE, "-m", "PyInstaller",
        "--onedir", "--name", "WebAim", "--console",
        "--add-data", f"configs{os.pathsep}configs",
        "--add-data", f"fitted_params.json{os.pathsep}.",
        "--hidden-import", "ultralytics",
        "--hidden-import", "torch",
        "--hidden-import", "torchvision",
        "--hidden-import", "cv2",
        "--hidden-import", "unicodedata",
        "--hidden-import", "pywinusb.hid",
        "web_aim.py",
    ])
    print("  [OK] spec generated. Review and re-run.")
    sys.exit(0)
print("  [OK] spec found")

# 3. Build
step("3/5 Building (may take 5-10 min)")

shutil.rmtree(BUILD_DIR, ignore_errors=True)
shutil.rmtree(DIST_DIR, ignore_errors=True)

run([
    PYTHON_EXE, "-m", "PyInstaller",
    str(spec_path),
    "--distpath", str(PROJECT_DIR / "dist"),
    "--workpath", str(BUILD_DIR),
    "--noconfirm",
])

built_dir = PROJECT_DIR / "dist" / "WebAim"
if CPU_MODE and built_dir.exists() and built_dir != DIST_DIR:
    shutil.rmtree(DIST_DIR, ignore_errors=True)
    shutil.move(str(built_dir), str(DIST_DIR))

exe_path = DIST_DIR / "WebAim.exe"
if not exe_path.exists():
    print(f"  [FAIL] {exe_path} not found")
    sys.exit(1)
print(f"  [OK] Build complete: {exe_path}")

# 4. Patch unicodedata.pyd
step("4/5 Patching Python 3.14 unicodedata.pyd")
src = find_unicodedata_pyd(PYTHON_EXE)
if src:
    dst = DIST_DIR / "_internal" / "unicodedata.pyd"
    shutil.copy2(src, dst)
    print(f"  [OK] unicodedata.pyd -> {dst}")
else:
    print("  [WARN] unicodedata.pyd not found, skipping")

# 5. Copy model files + cleanup
step("5/5 Copying model files + cleanup")
for mf in MODEL_FILES:
    src = PROJECT_DIR / mf
    if src.exists():
        dst = DIST_DIR / mf
        if not dst.exists():
            shutil.copy2(src, dst)
            size_mb = dst.stat().st_size / (1024 * 1024)
            print(f"  [OK] {mf} ({size_mb:.0f} MB)")
    else:
        print(f"  [SKIP] {mf} not found")

shutil.rmtree(BUILD_DIR, ignore_errors=True)
total = sum(f.stat().st_size for f in DIST_DIR.rglob("*") if f.is_file())
print(f"  Dist size: {total / (1024**3):.2f} GB")

variant = "CPU" if CPU_MODE else "GPU"
label = "WebAim-CPU (CPU inference)" if CPU_MODE else "WebAim (GPU inference)"
print(f"""
{'='*60}
  BUILD SUCCESS  [{variant} VERSION]
  Output: {DIST_DIR}
  Launch: {exe_path}
  URL:    http://127.0.0.1:5000
{'='*60}
""")
