"""
bench_env.py -- environment throughput (env-steps/s), no neural network involved.

    python tools/bench_env.py                          # single env, one core
    python tools/bench_env.py --n-envs 16 --subproc    # 16 workers via SubprocVecEnv
    python tools/bench_env.py --compare /path/to/Task_2_GPU_Accelerated_RL
        # interleaved A/B on one core: original reference env vs this one

Random actions, so the absolute rate is lower than with a trained policy (many resets/contacts);
use it for ratios and for sizing a machine.
"""
import argparse
import importlib.util
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "env"))
from zeus_pure_env import ZeusPureEnv


def load_orig(root):
    spec = importlib.util.spec_from_file_location("orig_env", os.path.join(root, "env", "zeus_pure_env.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ZeusPureEnv


def block(env, rng, n):
    t = time.perf_counter()
    for _ in range(n):
        _, _, te, tr, _ = env.step((rng.uniform(-1, 1, 8) * 0.1).astype(np.float32))
        if te or tr:
            env.reset()
    return n / (time.perf_counter() - t)


def compare(orig_root, rounds=12, n=300):
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[0]})
    a, b = load_orig(orig_root)(), ZeusPureEnv()
    a.reset(seed=0); b.reset(seed=0)
    ra, rb = np.random.default_rng(0), np.random.default_rng(0)
    block(a, ra, 200); block(b, rb, 200)
    ta, tb = [], []
    for _ in range(rounds):
        ta.append(block(a, ra, n)); tb.append(block(b, rb, n))
    ma, mb = np.median(ta), np.median(tb)
    print(f"original reference env : {ma:7.0f} steps/s  (one core, median of {rounds} blocks)")
    print(f"this env               : {mb:7.0f} steps/s")
    print(f"speed-up               : {mb/ma:.2f}x")


def subproc(n_envs, n_steps):
    from stable_baselines3.common.vec_env import SubprocVecEnv
    v = SubprocVecEnv([ZeusPureEnv for _ in range(n_envs)], start_method="forkserver")
    v.reset()
    rng = np.random.default_rng(0)
    for _ in range(20):
        v.step((rng.uniform(-1, 1, (n_envs, 8)) * 0.1).astype(np.float32))
    t = time.perf_counter()
    for _ in range(n_steps):
        v.step((rng.uniform(-1, 1, (n_envs, 8)) * 0.1).astype(np.float32))
    r = n_envs * n_steps / (time.perf_counter() - t)
    v.close()
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, default=1)
    ap.add_argument("--n-steps", type=int, default=1000)
    ap.add_argument("--subproc", action="store_true")
    ap.add_argument("--compare", metavar="TASK2_DIR")
    a = ap.parse_args()
    if a.compare:
        compare(a.compare)
    elif a.n_envs == 1 and not a.subproc:
        e = ZeusPureEnv(); e.reset(seed=0)
        print(f"[single env] {block(e, np.random.default_rng(0), a.n_steps * 2):.0f} env-steps/s")
    elif a.subproc:
        print(f"[{a.n_envs} envs, SubprocVecEnv] {subproc(a.n_envs, a.n_steps):.0f} env-steps/s")
    else:
        envs = [ZeusPureEnv() for _ in range(a.n_envs)]
        for e in envs: e.reset()
        rng = np.random.default_rng(0)
        t = time.perf_counter()
        for _ in range(a.n_steps):
            for e in envs:
                _, _, te, tr, _ = e.step((rng.uniform(-1, 1, 8) * 0.1).astype(np.float32))
                if te or tr: e.reset()
        print(f"[{a.n_envs} envs, sequential] {a.n_envs * a.n_steps / (time.perf_counter() - t):.0f} env-steps/s")
