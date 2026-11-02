# Pure-RL Zeus trainer. One image for RunPod, GCP (Compute Engine / GKE) or any NVIDIA box.
#   docker build -t zeus-pure-rl .
#   docker run --rm --gpus all -v $PWD/runs:/workspace/runs -e STEPS=20000000 zeus-pure-rl
# Code lives in /opt (a volume mounted at /workspace, as RunPod does, would hide it); runs go to
# /workspace/runs.  CPU-only box: build with --build-arg BASE_IMAGE=python:3.12-slim and
# add `pip install torch --index-url https://download.pytorch.org/whl/cpu` (see README).
ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RUNS_DIR=/workspace/runs
WORKDIR /opt/zeus-pure-rl
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /workspace/runs
# python becomes PID 1 (scripts/train_cloud.sh uses exec), so `docker stop` / spot reclaim
# delivers SIGTERM to the trainer, which checkpoints and exits; relaunching resumes.
CMD ["bash", "scripts/train_cloud.sh"]
