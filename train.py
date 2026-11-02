"""
train.py -- pure-RL PPO for Zeus (no gait library), built to run unattended on a cloud box.

    python train.py --run first --steps 20_000_000            # auto device, auto worker count
    python train.py --run first --steps 20_000_000 --device cuda --envs 32 --rollout 32768

* The simulator runs on CPU in worker processes; only the PPO network update uses the GPU.
  --envs auto uses every CPU the container may use.
* Preemption-safe: SIGTERM/SIGINT (spot reclaim, `docker stop`, ctrl-C) saves a checkpoint and
  exits 0.  Re-running the SAME command resumes from the newest checkpoint in the run folder
  (policy, optimiser, and observation/reward normaliser together).
* Output: <out-dir>/<run>/{ckpt/, progress.csv, events.out.tfevents.*, config.json, final.zip, final_vecnorm.pkl}
  with <out-dir> from --out-dir or $RUNS_DIR (default ./runs).
"""
import argparse
import glob
import json
import os
import re
import signal
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.join(HERE, "env")
sys.path.insert(0, ENV_DIR)

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

CKPT_RE = re.compile(r"ppo_(\d+)_steps\.zip$")


def make_env(rank, seed, v_cmd, randomize):
    def f():
        import sys as _sys
        _sys.path.insert(0, ENV_DIR)
        from zeus_pure_env import ZeusPureEnv
        return Monitor(ZeusPureEnv(v_cmd=v_cmd, randomize=randomize))
    return f


def usable_cpus():
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


class StopOnSignal(BaseCallback):
    """Stops learn() cleanly after SIGTERM / SIGINT so the caller can checkpoint."""

    def __init__(self):
        super().__init__()
        self.stop = False
        for s in (signal.SIGTERM, signal.SIGINT):
            signal.signal(s, self._handler)

    def _handler(self, signum, frame):
        if not self.stop:
            print(f"\n[train] signal {signum}: finishing the current step, then saving a checkpoint...", flush=True)
        self.stop = True

    def _on_step(self):
        return not self.stop


def latest_checkpoint(run_dir):
    best = None
    for p in glob.glob(os.path.join(run_dir, "ckpt", "ppo_*_steps.zip")):
        m = CKPT_RE.search(p)
        if m and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), p)
    if best is None:
        return None
    steps, zip_path = best
    pkl = os.path.join(run_dir, "ckpt", f"ppo_vecnormalize_{steps}_steps.pkl")
    return zip_path, (pkl if os.path.exists(pkl) else None), steps


def save_ckpt(model, venv, run_dir):
    n = model.num_timesteps
    d = os.path.join(run_dir, "ckpt")
    os.makedirs(d, exist_ok=True)
    model.save(os.path.join(d, f"ppo_{n}_steps.zip"))
    venv.save(os.path.join(d, f"ppo_vecnormalize_{n}_steps.pkl"))
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="pure_rl")
    ap.add_argument("--out-dir", default=os.environ.get("RUNS_DIR", os.path.join(HERE, "runs")))
    ap.add_argument("--steps", type=int, default=20_000_000, help="total env steps for the run")
    ap.add_argument("--envs", default="auto", help="worker processes, or 'auto' (= usable CPUs, max 64)")
    ap.add_argument("--rollout", type=int, default=16384, help="samples per PPO iteration (= envs x n_steps)")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--log-std-init", type=float, default=0.0)
    ap.add_argument("--net", default="256,256,128", help="hidden layer sizes of the policy and value nets")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--v-cmd", type=float, default=0.3, help="commanded forward speed, m/s")
    ap.add_argument("--no-randomize", action="store_true", help="no start-pose noise")
    ap.add_argument("--save-every", type=int, default=1_000_000, help="checkpoint period in env steps")
    ap.add_argument("--resume", default="auto", help="'auto' (newest checkpoint in the run folder, if any), 'no', or a .zip path")
    a = ap.parse_args()

    if a.device == "cuda" and not torch.cuda.is_available():
        sys.exit("[train] --device cuda requested but torch.cuda.is_available() is False "
                 "(install a CUDA build of torch, or use --device cpu / auto)")
    device = "cuda" if (a.device == "auto" and torch.cuda.is_available()) else ("cpu" if a.device == "auto" else a.device)
    torch.set_num_threads(a.torch_threads)

    n_envs = min(usable_cpus(), 64) if a.envs == "auto" else int(a.envs)
    n_steps = max(a.rollout // n_envs, 1)
    rollout = n_envs * n_steps
    batch = min(a.batch_size, rollout)
    run_dir = os.path.join(a.out_dir, a.run)
    os.makedirs(os.path.join(run_dir, "ckpt"), exist_ok=True)

    found = None
    if a.resume == "auto":
        found = latest_checkpoint(run_dir)
    elif a.resume != "no":
        d, base = os.path.split(a.resume)
        stem = base[:-4] if base.endswith(".zip") else base
        m = re.fullmatch(r"ppo_(\d+)_steps", stem)
        pkl = os.path.join(d, f"ppo_vecnormalize_{m.group(1)}_steps.pkl" if m else f"{stem}_vecnorm.pkl")
        found = (os.path.join(d, stem + ".zip"), pkl if os.path.exists(pkl) else None, 0)
    if found and found[1] is None:
        sys.exit(f"[train] checkpoint {found[0]} has no matching normaliser (.pkl); refusing to resume with a mismatched one")

    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "-"
    print(f"[train] run={a.run} device={device} ({gpu}) envs={n_envs} n_steps={n_steps} "
          f"rollout={rollout} batch={batch} cpus={usable_cpus()} torch={torch.__version__}", flush=True)
    json.dump(vars(a) | dict(device_used=device, n_envs=n_envs, n_steps=n_steps, rollout=rollout,
                             torch=torch.__version__, cuda_device=gpu),
              open(os.path.join(run_dir, "config.json"), "w"), indent=1)

    venv = SubprocVecEnv([make_env(i, a.seed, a.v_cmd, not a.no_randomize) for i in range(n_envs)])
    venv.seed(a.seed)
    if found:
        venv = VecNormalize.load(found[1], venv)
        venv.training = True
        model = PPO.load(found[0], env=venv, device=device,
                         custom_objects=dict(n_steps=n_steps, batch_size=batch, learning_rate=a.lr))
        print(f"[train] resumed from {found[0]} at {model.num_timesteps:,} steps", flush=True)
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=a.gamma)
        hidden = [int(x) for x in a.net.split(",")]
        model = PPO("MlpPolicy", venv, device=device, verbose=0, seed=a.seed,
                    n_steps=n_steps, batch_size=batch, n_epochs=a.epochs, learning_rate=a.lr,
                    gamma=a.gamma, gae_lambda=a.gae_lambda, clip_range=a.clip, ent_coef=a.ent_coef,
                    policy_kwargs=dict(net_arch=dict(pi=hidden, vf=hidden), activation_fn=nn.ELU,
                                       log_std_init=a.log_std_init))
    try:
        import tensorboard  # noqa: F401
        fmts = ["stdout", "csv", "tensorboard"]
    except ImportError:
        fmts = ["stdout", "csv"]
    model.set_logger(configure(run_dir, fmts))

    todo = a.steps - model.num_timesteps
    if todo <= 0:
        print(f"[train] already at {model.num_timesteps:,} >= {a.steps:,} steps; nothing to do", flush=True)
        venv.close()
        return
    stopper = StopOnSignal()
    ckpt = CheckpointCallback(max(a.save_every // n_envs, 1), os.path.join(run_dir, "ckpt"),
                              name_prefix="ppo", save_vecnormalize=True)
    t0, s0 = time.time(), model.num_timesteps
    model.learn(total_timesteps=todo, callback=[ckpt, stopper], reset_num_timesteps=False,
                log_interval=1)
    n = save_ckpt(model, venv, run_dir)
    dt = time.time() - t0
    fps = (n - s0) / max(dt, 1e-9)
    if stopper.stop:
        print(f"[train] interrupted at {n:,} steps; checkpoint saved. Re-run the same command to resume.", flush=True)
    else:
        model.save(os.path.join(run_dir, "final.zip"))
        venv.save(os.path.join(run_dir, "final_vecnorm.pkl"))
        print(f"[train] done: {n:,} steps in {dt/3600:.2f} h ({fps:,.0f} steps/s this session)", flush=True)
    json.dump(dict(steps=int(n), complete=not stopper.stop, steps_per_s=float(fps)),
              open(os.path.join(run_dir, "status.json"), "w"))
    venv.close()


if __name__ == "__main__":
    main()
