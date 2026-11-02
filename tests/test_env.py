"""pytest tests/   (or: python tests/test_env.py)"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "env"))
from zeus_pure_env import OBS_DIM, ZeusPureEnv


def test_spaces_and_step():
    e = ZeusPureEnv()
    o, _ = e.reset(seed=0)
    assert o.shape == (OBS_DIM,) == e.observation_space.shape and o.dtype == np.float32
    o, r, te, tr, info = e.step(np.zeros(8, np.float32))
    assert o.shape == (OBS_DIM,) and np.isfinite(o).all() and np.isfinite(r) and "fell" in info


def test_deterministic_with_seed():
    def run():
        e = ZeusPureEnv(randomize=True); e.reset(seed=7)
        return np.array([e.step(np.full(8, 0.1 * k % 1, np.float32))[0] for k in range(30)])
    assert np.array_equal(run(), run())


def test_falls_terminate():
    e = ZeusPureEnv(); e.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(400):
        _, _, te, tr, _ = e.step(rng.uniform(-1, 1, 8).astype(np.float32))
        if te:
            return
    raise AssertionError("random full-range actions never produced a fall")


def test_golden_hull_model():
    r = subprocess.run([sys.executable, os.path.join(HERE, "golden.py"), "check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
