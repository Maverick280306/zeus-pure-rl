"""
eval.py -- run a trained policy and report how it walks.

    python eval.py --run first                  # newest checkpoint (or final.zip) of runs/first
    python eval.py --run first --episodes 20 --v-cmd 0.4
    python eval.py --model path/to/ppo_5000000_steps.zip   # the matching normaliser is found beside it
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "env"))
sys.path.insert(0, HERE)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from train import latest_checkpoint
from zeus_pure_env import CTRL_DT, ZeusPureEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pure_rl")
    ap.add_argument("--out-dir", default=os.environ.get("RUNS_DIR", os.path.join(HERE, "runs")))
    ap.add_argument("--model", help="explicit .zip (normaliser inferred from its name)")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--v-cmd", type=float, default=0.3)
    ap.add_argument("--stochastic", action="store_true")
    a = ap.parse_args()

    run_dir = os.path.join(a.out_dir, a.run)
    if a.model:
        d, base = os.path.split(a.model)
        m = re.fullmatch(r"ppo_(\d+)_steps\.zip", base)
        zip_path = a.model
        pkl = os.path.join(d, f"ppo_vecnormalize_{m.group(1)}_steps.pkl" if m else base[:-4] + "_vecnorm.pkl")
    elif os.path.exists(os.path.join(run_dir, "final.zip")):
        zip_path, pkl = os.path.join(run_dir, "final.zip"), os.path.join(run_dir, "final_vecnorm.pkl")
    else:
        c = latest_checkpoint(run_dir)
        if c is None:
            sys.exit(f"no checkpoint in {run_dir}")
        zip_path, pkl = c[0], c[1]
    env = ZeusPureEnv(v_cmd=a.v_cmd)
    vn = VecNormalize.load(pkl, DummyVecEnv([lambda: env]))
    vn.training = False
    model = PPO.load(zip_path, device="cpu")
    print(f"policy: {zip_path}  ({model.num_timesteps:,} training steps)")
    lens, vxs, falls = [], [], 0
    for ep in range(a.episodes):
        obs, _ = env.reset(seed=1000 + ep)
        vx = []
        for k in range(env.max_ep_steps):
            act, _ = model.predict(vn.normalize_obs(obs[None]), deterministic=not a.stochastic)
            obs, r, te, tr, info = env.step(act[0])
            vx.append(env.data.qvel[0])
            if te or tr:
                break
        lens.append((k + 1) * CTRL_DT); vxs.append(float(np.mean(vx))); falls += int(te)
        print(f"  episode {ep:2d}: {lens[-1]:5.1f} s alive, mean vx {vxs[-1]:+.3f} m/s {'(fell)' if te else ''}")
    print(f"mean {np.mean(lens):.1f} s alive of {env.max_ep_steps * CTRL_DT:.0f} s, mean vx {np.mean(vxs):+.3f} m/s "
          f"(command {a.v_cmd}), falls {falls}/{a.episodes}")


if __name__ == "__main__":
    main()
