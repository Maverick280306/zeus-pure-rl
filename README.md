# zeus-pure-rl

Pure-RL PPO training for the **Zeus** biped (8 actuated joints), packaged to run unattended on
**RunPod**, **Google Cloud**, or any Linux box with or without an NVIDIA GPU.

* **Pure RL**: no gait library, no trajectory optimiser, no Pinocchio / CasADi. The policy gets
  only the reward in `env/zeus_pure_env.py` and has to find a gait itself.
* **Cheap to step**: the environment costs about half as much CPU time per step as the Task 2
  reference environment it was derived from, and about 1/30 of its memory (numbers below).
* **Cloud-safe**: checkpoints policy + optimiser + normaliser, traps SIGTERM (spot reclaim,
  `docker stop`), and resumes by re-running the same command.

Behaviour (observation, action, reward, termination, physics) is unchanged from the reference
environment. Only the cost of computing it changed, and `tests/golden.py` proves that against
the original.

---

## 1. Why a GPU box, and what it does and does not speed up

Every training step is a physics simulation, and that is **CPU work in worker processes**. The
GPU only runs the PPO network update. So an iteration has two parts:

| Part | Runs on | Scales with |
|---|---|---|
| **collect**: simulator + pipes + policy forward pass | CPU cores | number of vCPUs |
| **update**: gradient steps on the network | GPU (or CPU) | the device |

On the 4-core laptop this was developed on, the update was **8–12 % of an iteration**, so a GPU
there would have bought almost nothing. On a many-core cloud box collection gets much faster
(it scales with cores) while the CPU update stays the same, so the update's share grows and a
GPU starts to matter. That is an estimate, not a measurement on cloud hardware. **Measure on the
machine you are about to rent**, it takes a few minutes:

```bash
python tools/bench_ppo.py --envs 32            # prints collect s vs update s, on cpu and (if present) cuda
```

Rule of thumb, reasoned from the table above and not yet measured on cloud hardware: pay for
**vCPUs first, GPU second**. A GPU on a machine with only a handful of vCPUs leaves the simulator
as the bottleneck.

## 2. Quick start

```bash
pip install torch                   # the build for your machine; skip if it is already installed
pip install -r requirements.txt
python tests/golden.py check        # sanity: environment matches the reference
python train.py --run first --steps 20_000_000
python eval.py  --run first
```

`train.py` picks the device (`--device auto|cpu|cuda`) and the number of worker processes
(`--envs auto` = every CPU the container may use, max 64). Output goes to
`runs/<run>/` (`$RUNS_DIR` or `--out-dir` to change): `ckpt/`, `progress.csv`, TensorBoard
events (`tensorboard --logdir runs`), `config.json`, `final.zip`, `final_vecnorm.pkl`.

Key flags: `--rollout` (samples per PPO iteration = envs × n_steps, default 16384),
`--batch-size` (default 4096; raise to 8192–16384 with a big GPU), `--net 256,256,128`,
`--lr`, `--epochs`, `--v-cmd` (commanded speed, m/s), `--no-randomize`, `--save-every`.

**Resume / preemption.** Re-run the identical command. It loads the newest checkpoint in the run
folder with its matching normaliser; if none exists it starts fresh. `--resume no` forces a fresh
start, `--resume path/to/ppo_5000000_steps.zip` picks one. A checkpoint with no matching
normaliser is refused rather than silently resumed with the wrong statistics. (Tested here by
sending SIGTERM mid-run and re-launching.)

## 3. RunPod

1. **Pick hardware by vCPUs.** Deploy a Pod from the *RunPod PyTorch* template with a network
   volume mounted at `/workspace`. Choose a GPU for the update, but check the vCPU count of the
   instance, since that sets the simulator speed.
2. In the pod terminal:
   ```bash
   cd /workspace
   git clone https://github.com/Maverick280306/zeus-pure-rl.git   # private repo: use a token, see below
   cd zeus-pure-rl && bash scripts/setup_vm.sh
   python tools/bench_ppo.py --envs 32                             # optional: is the GPU worth it here?
   RUN=first STEPS=50000000 nohup bash scripts/train_cloud.sh > /workspace/first.log 2>&1 &
   ```
   Runs land in `/workspace/runs` (the persistent volume). Stopped or preempted pod: start it
   again and re-run the last command, it resumes.
3. **Alternative, Docker template:** build and push the image (below), then create a pod
   template with that image; set env vars `RUN`, `STEPS` (and optionally `ENVS`, `DEVICE`);
   leave the start command empty. The image's default command runs `scripts/train_cloud.sh`.

Private repo access: `git clone https://<GITHUB_TOKEN>@github.com/Maverick280306/zeus-pure-rl.git`
with a fine-grained read-only token, or add the pod's SSH key as a deploy key.

## 4. Google Cloud

```bash
PROJECT=my-project bash scripts/gcp_create_vm.sh                                    # 1x L4, 16 vCPU (Deep Learning VM)
PROJECT=my-project MODE=cpu MACHINE=c3-standard-44 bash scripts/gcp_create_vm.sh    # many cores, no GPU
SPOT=1 PROJECT=my-project bash scripts/gcp_create_vm.sh                             # preemptible, much cheaper
gcloud compute ssh zeus-pure-rl --project my-project --zone us-central1-a
# on the VM:
git clone https://github.com/Maverick280306/zeus-pure-rl.git && cd zeus-pure-rl
VENV=1 bash scripts/setup_vm.sh
RUN=first STEPS=50000000 nohup bash scripts/train_cloud.sh > train.log 2>&1 &
```

If a Spot VM is reclaimed, `gcloud compute instances start`, ssh in, and run the same
`train_cloud.sh` line: it resumes. Copy results back with
`gcloud compute scp --recurse zeus-pure-rl:~/zeus-pure-rl/runs ./`.

## 5. Docker (any provider)

```bash
docker build -t zeus-pure-rl .
docker run --rm --gpus all -v $PWD/runs:/workspace/runs -e RUN=first -e STEPS=50000000 zeus-pure-rl
docker push <registry>/zeus-pure-rl          # to use it as a RunPod template or on GKE / Cloud Run jobs
```

The code is in `/opt/zeus-pure-rl` (a volume mounted at `/workspace`, as RunPod does, would hide
it); runs are written to `/workspace/runs`. Python is PID 1, so `docker stop` triggers a
checkpoint.

> **Not tested on cloud.** Everything in sections 3–5 was written to be correct but could not be
> run where this was developed: no GPU, no `gcloud`, and no Docker daemon access. What *was*
> run locally: `setup_vm.sh`, `train_cloud.sh`, `train.py` (cpu, auto-resume, SIGTERM save),
> the `--device cuda` failure path, `eval.py`, `tools/bench_ppo.py`, and the test suite. Expect
> to adjust the GCP image family or the base-image tag on first use; the first CUDA run is the
> real test of the GPU path.

## 6. What was optimised, and the proof

Reference: Task 2's `zeus_pure_env.py` (398–462 env-steps/s on one core of a 4-core i7).

| Change | Effect |
|---|---|
| **Collision hulls.** MuJoCo collides each limb as its convex hull, but also builds and clears a bounding tree over every triangle of the CAD mesh (1.74 M nodes, 40 MB of STL). `model/zeus_fast.xml` gives it the hull vertices inline instead: 54 k nodes, 460 KB, no STL files. | the large majority of the saving |
| PD loop and observation write into preallocated buffers; contacts scanned in plain Python over the 1–5 active pairs; gravity projected once per step | smaller (Python is now 13 % of a step) |

| Measured here | Reference | This repo |
|---|---:|---:|
| One env, one core (`tools/bench_env.py --compare`, 3 runs) | 424–462 steps/s | **826–861 steps/s (1.86–1.95×)** |
| Memory added per worker process | 890 MB | **28 MB** |
| Model on disk | 42 MB + STL files | 460 KB, one file |
| PPO iteration, 8 envs, this 8-thread laptop (`tools/bench_ppo.py`) | – | 1222 steps/s, update 8 % of the iteration |

Where a step goes now: MuJoCo C code 87 % (1039 µs), Python 13 % (152 µs).

**Correctness** (`python tests/golden.py check`, also run by `scripts/setup_vm.sh`):

* *Tier 1, original STL model:* the optimised code reproduces the reference **bit-for-bit** over
  1000 steps (qpos, qvel, observation, reward, done). Run it with
  `python tests/golden.py check --model-xml <Task_2>/model/zeus.xml`.
* *Tier 2, shipped hull model:* collision shapes agree to **1.4e-8 m**; mass and inertia are
  identical (the XML sets them explicitly). Over the first 90 steps the trajectories agree to
  1.8e-7 m. After that they drift, bounded at 2.3e-3 m over 1000 steps, with all episode ends on
  the same steps. The reason is a property of this model, not of the change: **in the standing
  pose the two ankle meshes already overlap by 23 mm**, a permanent deep leg-vs-leg contact
  whose solver output moves ~1e-7 per step with the hull's vertex order and then grows chaotically,
  as in any contact-rich simulation.
* *Statistical check* (40 seeds, random actions, 600 steps, original vs hull model): mean episode
  length 266.4 vs 266.4 steps, mean return 146.6 vs 146.4; 37 of 40 episodes identical in length.

That overlapping-ankle contact looks like a modelling flaw in the reference model (the two legs
touch at rest), which is worth fixing at the source; it is left as is here so behaviour matches.

## 7. Environment summary

Observation (38): projected gravity 3, base angular velocity 3, joint pos 8, joint vel 8,
previous action 8, commanded speed 1, phase sin/cos 2, foot contacts 2, base height 1, base xy
velocity 2. Action (8) in [-1, 1] maps to joint-position targets over each joint's range; a PD
loop (Kp 60/60/60/30, Kd 2/2/2/1 per leg) runs at 1 kHz under a 50 Hz policy. Reward:
`-|vx - v_cmd| - 0.5|z - 0.6643| + 0.5·upright - 1e-4·Σctrl² + 0.5`; an episode ends when
height < 0.45 m or uprightness < 0.6, or at 1000 steps. Base velocity is observed, which hardware
cannot measure; this environment is for throughput, not sim-to-real.

## 8. Layout and regenerating the model

```
env/zeus_pure_env.py     the environment
model/zeus_fast.xml      self-contained robot model (hull collision, inline vertices)
train.py  eval.py        training (device/worker auto, resume, SIGTERM-safe) and evaluation
tools/                   bench_env.py, bench_ppo.py, make_fast_model.py
tests/                   golden.py + golden_pure.npz (reference trajectory), test_env.py
scripts/                 setup_vm.sh, train_cloud.sh, gcp_create_vm.sh
Dockerfile  requirements.txt  requirements-dev.txt
```

`model/zeus_fast.xml` is generated from the full-mesh CAD model, which is not in this repo:
`pip install -r requirements-dev.txt && python tools/make_fast_model.py --src <folder with zeus.xml + meshes/>`.
If the robot geometry changes, regenerate it and re-record the reference with
`python tests/golden.py record --orig <reference repo>`.
