#!/usr/bin/env bash
# Launch (or resume) a training run. Configure with environment variables; extra args go to train.py.
#   RUN=first STEPS=50000000 bash scripts/train_cloud.sh
#   RUN=first STEPS=50000000 DEVICE=cuda ENVS=32 ROLLOUT=32768 BATCH=8192 bash scripts/train_cloud.sh
# Re-running with the same RUN continues from the newest checkpoint, so a preempted spot
# instance only needs this same command again.  Output: $RUNS_DIR/$RUN/
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${RUNS_DIR:-}" ]; then
  if [ -d /workspace ]; then RUNS_DIR=/workspace/runs; else RUNS_DIR="$PWD/runs"; fi
fi
export RUNS_DIR
mkdir -p "$RUNS_DIR"
exec python train.py --run "${RUN:-zeus_pure}" --steps "${STEPS:-20000000}" --envs "${ENVS:-auto}" \
     --device "${DEVICE:-auto}" --rollout "${ROLLOUT:-16384}" --batch-size "${BATCH:-4096}" "$@"
