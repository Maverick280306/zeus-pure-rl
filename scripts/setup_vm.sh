#!/usr/bin/env bash
# Prepare a fresh Linux box (RunPod pod, GCP VM, any NVIDIA/CPU machine) for training.
#   bash scripts/setup_vm.sh            # installs into the current python (RunPod/GCP DL images)
#   VENV=1 bash scripts/setup_vm.sh     # or into ./.venv (bare Ubuntu)
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "${VENV:-0}" = "1" ]; then
  python3 -m venv .venv && source .venv/bin/activate
fi
python -c "import torch" 2>/dev/null || { echo "[setup] installing torch"; pip install torch; }
pip install -r requirements.txt
python - <<'PY'
import os, torch, mujoco, stable_baselines3 as sb3
cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
print(f"[setup] usable CPUs: {cpus}   torch {torch.__version__}   mujoco {mujoco.__version__}   sb3 {sb3.__version__}")
print("[setup] cuda:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NOT available (training will use the CPU)")
PY
python tests/golden.py check
echo "[setup] ok. Next: bash scripts/train_cloud.sh   (env: RUN STEPS ENVS DEVICE ROLLOUT BATCH)"
