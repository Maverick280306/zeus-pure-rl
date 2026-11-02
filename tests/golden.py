"""
golden.py -- regression gate: the optimised env must reproduce the ORIGINAL env.

    python tests/golden.py record --orig /path/to/Task_2_GPU_Accelerated_RL   # once
    python tests/golden.py check                                              # any time
    python tests/golden.py check --model-xml /path/to/original/zeus.xml       # tier 1

1000 steps, seeded random actions, randomised start pose, resets included.
tier 1 (original STL model): qpos, qvel, obs, reward, done must be bit-identical.
tier 2 (shipped hull model): collision shape equal to 1e-8 m; only the leg-vs-leg contact
solution moves, so the run is compared tightly early and loosely (bounded chaos) overall.
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "golden_pure.npz")
N = 1000
EARLY = 90          # steps before the first episode end of the recorded run (step 93)
# tier 2, early window: before chaos has amplified anything, the hull model must track tightly
TOL_EARLY = dict(qpos=1e-6, qvel=1e-4, obs=1e-3, rew=1e-5)
# tier 2, whole run: this model's standing pose has the two ankle meshes interpenetrating by 23 mm
# (a persistent leg-vs-leg contact), whose solution moves ~1e-7 per step with the hull vertex order and
# then grows chaotically; bound it, and require identical episode ends.
TOL_RUN = dict(qpos=1e-2, qvel=0.5, obs=0.5, rew=1e-1)


def rollout(env):
    rng = np.random.default_rng(0)
    acts = rng.uniform(-1, 1, (N, 8)).astype(np.float32) * 0.3
    env.reset(seed=123)
    Q, V, O, R, D = [], [], [], [], []
    for k in range(N):
        o, r, te, tr, _ = env.step(acts[k])
        Q.append(env.data.qpos.copy()); V.append(env.data.qvel.copy())
        O.append(o); R.append(r); D.append(te or tr)
        if te or tr:
            env.reset()
    return dict(qpos=np.array(Q), qvel=np.array(V), obs=np.array(O), rew=np.array(R), done=np.array(D))


def load_orig(root):
    path = os.path.join(root, "env", "zeus_pure_env.py")
    spec = importlib.util.spec_from_file_location("orig_zeus_pure_env", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["record", "check"])
    ap.add_argument("--orig")
    ap.add_argument("--model-xml", help="check with this model (original STL model => tier 1, bit-identical)")
    a = ap.parse_args()
    if a.mode == "record":
        env = load_orig(a.orig).ZeusPureEnv(randomize=True)
        g = rollout(env)
        np.savez_compressed(OUT, **g)
        print(f"recorded {N} steps ({int(g['done'].sum())} episode ends) -> {OUT}")
        return
    sys.path.insert(0, os.path.join(HERE, "..", "env"))
    from zeus_pure_env import ZeusPureEnv
    g = np.load(OUT)
    exact = a.model_xml is not None
    env = ZeusPureEnv(randomize=True, model_xml=a.model_xml) if exact else ZeusPureEnv(randomize=True)
    n = rollout(env)
    ok = True
    if exact:
        print("tier 1 (original STL model): bit-identical")
        for k in g.files:
            d = np.abs(g[k].astype(float) - n[k].astype(float))
            good = bool((d == 0).all()); ok &= good
            print(f"  {k:5s} max|diff| {np.nanmax(d):.2e}  {'ok' if good else 'FAIL'}")
    else:
        print("tier 2 (hull model): early window, then whole run")
        for k in TOL_RUN:
            d = np.abs(g[k].astype(float) - n[k].astype(float))
            e, w = d[:EARLY].max(), d.max()
            good = bool(e <= TOL_EARLY[k] and w <= TOL_RUN[k]); ok &= good
            print(f"  {k:5s} first {EARLY} steps {e:.2e} (<= {TOL_EARLY[k]:.0e})   whole run {w:.2e} (<= {TOL_RUN[k]:.0e})  {'ok' if good else 'FAIL'}")
        same = bool(np.array_equal(g["done"], n["done"])); ok &= same
        print(f"  done  episode ends identical: {same}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
