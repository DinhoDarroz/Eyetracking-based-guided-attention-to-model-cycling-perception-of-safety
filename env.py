#!/usr/bin/env python3
"""
setup_env.py

Creates a virtual environment and installs all dependencies required by:
 - train.py
 - train_script.py

USAGE:
    python3 setup_env.py

NOTES:
 - Assumes Python >= 3.8
 - If CUDA is available, installs GPU PyTorch automatically
 - Otherwise installs CPU-only PyTorch
"""

import os
import subprocess
import sys
import shutil


VENV_DIR = ".venv"


def run(cmd):
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.check_call(cmd)


def has_cuda():
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def main():
    if sys.version_info < (3, 8):
        raise RuntimeError("Python >= 3.8 is required")

    # ------------------------------------------------------------------
    # 1. Create venv
    # ------------------------------------------------------------------
    if not os.path.exists(VENV_DIR):
        print("Creating virtual environment...")
        run([sys.executable, "-m", "venv", VENV_DIR])
    else:
        print("Virtual environment already exists")

    python_bin = os.path.join(VENV_DIR, "bin", "python")
    pip_bin = os.path.join(VENV_DIR, "bin", "pip")

    # ------------------------------------------------------------------
    # 2. Upgrade pip toolchain
    # ------------------------------------------------------------------
    run([python_bin, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    # ------------------------------------------------------------------
    # 3. Install PyTorch
    # ------------------------------------------------------------------
    if has_cuda():
        print("CUDA detected → installing GPU PyTorch")
        run([
            pip_bin, "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ])
    else:
        print("CUDA not detected → installing CPU PyTorch")
        run([
            pip_bin, "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ])

    # ------------------------------------------------------------------
    # 4. Core dependencies (from train.py + train_script.py)
    # ------------------------------------------------------------------
    core_packages = [
        "pytorch-ignite",
        "timm",
        "wandb",
        "optuna",
        "numpy",
        "pandas",
        "scikit-learn",
        "pillow",
        "tqdm"
    ]

    run([pip_bin, "install", *core_packages])

    # ------------------------------------------------------------------
    # 5. Freeze requirements
    # ------------------------------------------------------------------
    run([pip_bin, "freeze"])
    run([pip_bin, "freeze", ">", "requirements.txt"])

    print("\nSetup completed successfully.")
    print(f"Activate with: source {VENV_DIR}/bin/activate")


if __name__ == "__main__":
    main()
