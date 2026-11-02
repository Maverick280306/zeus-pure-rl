"""
bench_ppo.py -- one PPO iteration, split into data collection and network update, per device.

    python tools/bench_ppo.py --envs 16                       # cpu, then cuda if present
    python tools/bench_ppo.py --envs 32 --rollout 32768 --devices cpu cuda

Use it on the machine you are about to pay for: it shows how much of an iteration the GPU can
possibly speed up ("update"), and how much stays on the CPUs ("collect": simulator + policy).
"""
import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "env"))

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from train import make_env, usable_cpus


def run(device, n_envs, rollout, batch, iters):
    torch.set_num_threads(2)
    venv = VecNormalize(SubprocVecEnv([make_env(i, 0, 0.3, True) for i in range(n_envs)]),
                        norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
    n_steps = max(rollout // n_envs, 1)
    m = PPO("MlpPolicy", venv, device=device, verbose=0, n_steps=n_steps, batch_size=min(batch, n_steps * n_envs),
            n_epochs=5, learning_rate=3e-4,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]), activation_fn=nn.ELU))
    acc = dict(collect=0.0, train=0.0)
    for name in ("collect_rollouts", "train"):
        f = getattr(m, name)
        def wrap(f=f, key="collect" if name == "collect_rollouts" else "train"):
            def g(*a, **k):
                t = time.perf_counter(); r = f(*a, **k); acc[key] += time.perf_counter() - t; return r
            return g
        setattr(m, name, wrap())
    m.learn(n_steps * n_envs)                       # warm-up iteration (CUDA init, worker start)
    acc.update(collect=0.0, train=0.0)
    t = time.perf_counter()
    m.learn(n_steps * n_envs * iters, reset_num_timesteps=False)
    wall = time.perf_counter() - t
    venv.close()
    tot = acc["collect"] + acc["train"]
    print(f"{device:5s} envs={n_envs:3d}  {n_steps * n_envs * iters / wall:7.0f} steps/s   collect {acc['collect']/iters:6.2f} s"
          f"   update {acc['train']/iters:5.2f} s   ({100 * acc['train'] / tot:.0f}% of an iteration is the update)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=min(usable_cpus(), 16))
    ap.add_argument("--rollout", type=int, default=16384)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--devices", nargs="+", default=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
    a = ap.parse_args()
    print(f"usable CPUs: {usable_cpus()}   cuda: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available'}")
    for d in a.devices:
        run(d, a.envs, a.rollout, a.batch_size, a.iters)
