#!/usr/bin/env bash
set -uo pipefail
printf '\n== GPU ==\n'; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
printf '\n== Runtime ==\n'; cat /etc/os-release | head -4; echo "DISPLAY=${DISPLAY:-unset}"
printf '\n== Storage ==\n'; df -h / /workspace /runpod-volume 2>/dev/null || true
printf '\n== Isaac Python installations ==\n'; command -v python3; find /opt /root -maxdepth 6 -name 'isaaclab.sh' -o -name 'isaac-sim.sh' 2>/dev/null | head -20
printf '\n== Existing scene processes ==\n'; pgrep -af 'isaac-sim|kit|isaaclab' | head -15 || true
