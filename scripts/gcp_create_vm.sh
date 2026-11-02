#!/usr/bin/env bash
# Create a Compute Engine VM for training. NOT TESTED by the author (no gcloud on the dev box):
# if the image family is rejected, list current ones with
#   gcloud compute images list --project deeplearning-platform-release --no-standard-images
#
#   PROJECT=my-proj bash scripts/gcp_create_vm.sh                         # 1x L4 GPU, 16 vCPU
#   PROJECT=my-proj MODE=cpu MACHINE=c3-standard-44 bash scripts/gcp_create_vm.sh   # many-core, no GPU
#   SPOT=1 ...                                                            # ~60-90% cheaper, can be reclaimed
# The simulator is CPU work: vCPUs set the speed, the GPU only accelerates the network update
# (about 10% of an iteration here). Run `python tools/bench_ppo.py` on the VM before committing budget.
set -euo pipefail
: "${PROJECT:?set PROJECT=<gcp project id>}"
NAME="${NAME:-zeus-pure-rl}"; ZONE="${ZONE:-us-central1-a}"; MODE="${MODE:-gpu}"
args=(--project "$PROJECT" --zone "$ZONE" --boot-disk-size 100GB --scopes cloud-platform)
if [ "$MODE" = "gpu" ]; then
  args+=(--machine-type "${MACHINE:-g2-standard-16}" --image-family pytorch-latest-gpu
         --image-project deeplearning-platform-release --maintenance-policy TERMINATE
         --metadata install-nvidia-driver=True)
else
  args+=(--machine-type "${MACHINE:-c3-standard-44}" --image-family ubuntu-2204-lts --image-project ubuntu-os-cloud)
fi
[ "${SPOT:-0}" = "1" ] && args+=(--provisioning-model SPOT --instance-termination-action STOP)
gcloud compute instances create "$NAME" "${args[@]}"
echo "ssh:  gcloud compute ssh $NAME --project $PROJECT --zone $ZONE"
echo "then: git clone <repo> && cd zeus-pure-rl && VENV=1 bash scripts/setup_vm.sh && nohup bash scripts/train_cloud.sh > train.log 2>&1 &"
