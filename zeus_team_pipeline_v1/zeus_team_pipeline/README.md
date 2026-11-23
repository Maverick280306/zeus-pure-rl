# Robot-to-Isaac-Lab intake and validation kit

## Scope and limits

This package is **not a universally working native RL trainer**: URDF, MJCF, USD and CAD are asset formats, not interchangeable ready-made training tasks. Native Isaac Lab training requires a PhysX articulation, correct controller, reset and rewards for each robot. This kit automates **intake, asset conversion for inline-MJCF meshes, static checks and stage diagnostics**; it gives a repeatable handoff process for any robot team.

The provided Zeus example is grounded in the earlier `zeus_fast.xml` repository snapshot. It targets 8 named actuated joints, 38 observations, 8 actions, 1 kHz physics and 50 Hz policy. It is not a validated 3.0-EA PPO environment. **Do not use the old `zeus_direct_env.py` for Isaac Lab 3.0 EA**.

## A. Before buying GPU time

1. Ask the team for the primary model (`robot.xml`, `.urdf`, or `.usd`); every referenced mesh and texture; actuator and joint specification; free/fixed base; link masses and inertias; joint limits; units; controller targets; intended ground-contact links; zero/standing pose; and version/date.
2. Put it under `model/` or its own team folder, keep source untouched.
3. Copy `templates/robot_manifest.template.json` to `robot_manifest.json` and fill **real** joint names. For Zeus use `examples/zeus_manifest.json` as a starting reference and correct it if the source changed.
4. Run:

```bash
python3 tools/intake_robot.py model/zeus_fast.xml --out intake.json
python3 tools/check_robot_contract.py --manifest robot_manifest.json
```

## B. Prepare geometry

If MJCF uses inline `vertex=` mesh assets:

```bash
python3 -m venv /tmp/robot-prep
/tmp/robot-prep/bin/pip install numpy scipy
/tmp/robot-prep/bin/python tools/convert_mjcf_inline.py \
  model/zeus_fast.xml --out model/zeus_isaac.xml
python3 tools/intake_robot.py model/zeus_isaac.xml --out converted_intake.json
```

The converter bakes MuJoCo mesh scale into external STL convex hulls. This is not a recreation of the original CAD surface, and it does not verify link transforms or collision fidelity. Prefer original team CAD-derived mesh assets when available. For URDF with `package://` paths, resolve the ROS package resources first. For USD, skip conversion and inspect the stage directly.

## C. USD export using Isaac Sim GUI (proven entry point)

Start your Runpod GUI using the versioned community image you previously verified for KasmVNC port 6901. Open Isaac Sim. In **File > Import**, select the *converted* MJCF or the team URDF. If needed, enable importer extensions in Window > Extensions. Keep source base type, do not merge meshes on the first pass, set Add Reference to Stage. Save the resulting USD somewhere you will back up. If the GUI importer fails, inspect Console logs; do not confuse an Xform hierarchy with a visible robot.

Isaac Sim 6.1 also documents an MJCF CLI importer. `tools/import_mjcf_cli.sh ABS_MJCF ABS_OUTPUT_DIR` works only if the `python.sh` and bundled `standalone_examples` exist at a known installation path, and may require your image-specific path adjustment.

After opening the exported USD, create `tools/inspect_stage.py` on the Pod, then in **Window > Script Editor** run:

```python
exec(open('/root/zeus-pure-rl/tools/inspect_stage.py').read())
```

**Gate 1:** At least one nonempty, visible render mesh, valid rigid bodies/colliders, valid articulation root, correct controlled joints. The script checks geometry and articulation only; engineers must inspect mass, collision and dynamics. If geometry is missing, ask for better CAD/URDF source or repair the exporter, not the policy.

**Gate 2:** Single robot can spawn off the ground, initialize under gravity, stand/reset, and accept bounded actions without NaN or exploding joints. Evaluate joint order and left/right sides by moving one joint at a time.

## D. Native Isaac Lab smoke test using Isaac Sim 6.1

Use the **matching Isaac Lab v3.0.0-EA** checkout. Never mix `v2.3.x` custom trainers with this 6.1 runtime. Check release-specific quickstart and install a matching Python environment. Do not run the standalone Isaac Sim GUI alongside another Kit visualizer on limited GPU VRAM.

A documented Isaac Lab 3 workflow for initial NVIDIA task validation is:

```bash
cd /root/IsaacLab
uv run --extra isaacsim isaaclab train \
  --rl_library rsl_rl --task Isaac-Cartpole --num_envs 16 \
  physics=isaacsim_physx --viz kit
```

If your pinned EA checkout uses a different extra name, check `uv run isaaclab train --help` and its `pyproject.toml`; do not assume an incompatible command will work. This Cartpole test proves Isaac Lab and GUI training, **not** the Zeus task.

## E. Custom task implementation (required for EACH robot)

Build from the **version-pinned** official DirectRLEnv tutorial or a humanoid example. Register a unique Gymnasium task ID (e.g. `Nexus-Zeus-Direct-v0`) and provide:

- `ArticulationCfg` with validated USD and exact eight joint names/ordering;
- action processing at 50 Hz and bounded effort/position targets at 1 kHz;
- 38 observation channels, including **actual** contact measurements (do not replace with silent zeros);
- reward terms with documented dimensions and scale;
- fall/reset logic with stable initial pose;
- PPO configuration + checkpoint directory; one-GUI-process playback using `--viz kit`.

**Gate 3:** 1 env × 200 physics steps, zero NaNs, distinct joint control, no unbounded torques. **Gate 4:** 16 envs × 2 PPO updates, checkpoints generated. **Gate 5:** 64+ env headless training, compare survival/reward and record a 1-env GUI playback.

Reference: https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/tutorials/03_envs/create_direct_rl_env.html

## F. Reusable team handoff

1. Add a new `robots/<robot_id>/` directory with manifest and immutable source plus mesh bundle.
2. Run intake on every source; record every unresolved file, unit ambiguity and joint-name mismatch.
3. Convert or import without overwriting originals.
4. Produce signed-off `robot.usd` and record its absolute path, root prim, contact links, joint names and action order.
5. Run USD stage audit, single robot physical acceptance, and native Isaac Lab example before registering a team-specific RL task.
6. Save `runs/<robot_id>/<experiment>/` plus manifests and imported USD to persistent storage. Keep version hashes so the team can reproduce the experiment.

## Troubleshooting

- `Mesh ... has no file`: MJCF mesh uses inline vertices; convert external mesh assets and inspect paths/scaling.
- Robot hierarchy present, but zero USD Mesh prims: **visual import failed**. Do not train invisible/incomplete asset.
- Robot visible but explodes: check joint frames, inertia, self-collision, ground penetration, drive stiffness and torque caps.
- GPU training works but no UI: remove `--headless`, use `--viz kit`, launch in the KasmVNC X session; avoid two simultaneous Isaac Sim apps.
- Only one robot appears: confirm num_envs and cloned environment configuration. A single active viewport is not proof of parallel training.
- `ModuleNotFoundError`: verify version-pinned Isaac Lab environment and not `/root/zeus-venv`.
- Pod terminated: container disk is gone unless results were exported to persistent storage or downloaded before termination.
