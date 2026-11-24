# NEXUS | Robot Simulation & RL Training on RunPod

**IIT Madras · CFI iBot Club**  
Operating procedure — 24 November 2026 · Isaac Sim 6.1 / Isaac Lab 3.0 EA

## Purpose and scope

Receive a robot URDF (not a URL; a download URL may be used to obtain the files), MJCF or USD, validate it, train a policy, evaluate it, send results to the team and terminate the GPU Pod. No mechanical redesign is part of this workflow.

Zeus is the reference robot, not a universal specification. Every robot can have different joints, actuators, dimensions, observations, control rates and tasks. The existing Zeus walking task is a starting point for **floating-base bipeds**, not a universal trainer for arms, wheeled robots or arbitrary mechanisms.

**Evidence:** On 24 November, 16 Zeus robots were visibly running in Isaac Lab with GPU PhysX on RTX 4090 after correcting USD dependencies and separating configuration from runtime imports. The operator subsequently reported completing the workflow. The final results archive and V2 training metrics were not supplied for independent inspection. The new-URDF adaptation below is an operational procedure requiring validation on each model, not a claim that arbitrary URDF training has already been tested.

## 0. Before deploying: permanent files

Keep these outside RunPod (private GitHub or team Drive):

- Repository: `https://github.com/Maverick280306/zeus-pure-rl.git`.
- `NEXUS_RESULTS.zip` from the successful session. It should contain `trainer/zeus_native.py`, `zeus_native_cfg.py`, `zeus_walk.py`, `zeus_walk_cfg.py`, `trainer/__init__.py`, `zeus_training.usda`, `isaaclab_commit.txt`, `requirements_working.txt` and logs. Check the archive before starting; if V2 files are absent, recover them from the successful Pod or the team backup. Do not assume the old GitHub repository contains them.
- This README, original robot submissions, all meshes and robot-specific configuration files.

Suggested GitHub layout (add it to the existing repository, not a replacement for MuJoCo):

```text
zeus-pure-rl/
  env/                         # existing MuJoCo environment
  model/                       # original Zeus assets
  train.py
  eval.py
  zeus_team_pipeline_v1/       # existing intake and conversion helpers
  isaac_backend/
    reference/                 # copy of working trainer files
  robots/
    robot_revision/
      source/                  # original URDF and meshes
      robot_manifest.json
      training_config.json
      generated/               # USD and import diagnostics
```

Never commit credentials, private access tokens or unnecessary multi-GB checkpoints. Store large assets and checkpoints in approved team storage. The RunPod instance itself is disposable.

# STAGE 1 — Setup and import

## 1. Deploy a fresh Pod from the RunPod UI

1. Open https://www.runpod.io/console/pods and click Deploy.
2. Select one RTX 4090 (24 GB); RTX 3090 or L4 were also used in earlier sessions. Prefer adequate CPU and RAM (approximately 32–64 GB RAM).
3. Use the previously successful image `hrithik108/ubuntu-isaac-sim:6.1.0`. It is a community image; review its trustworthiness before using team credentials. Record its digest if available.
4. Suggested container disk: 100 GB; Pod volume: 50 GB if needed; no separate network volume for a disposable session. Verify available disk in the actual container.
5. Keep template HTTP services, including KasmVNC **6901**. Do not replace with port 8080. Enable SSH or authenticated file transfer if available.
6. Deploy; open Connect → KasmVNC and the Web Terminal in separate tabs. Use the desktop terminal for graphical training; Web Terminal for file operations.

| Component | Version / choice |
|---|---|
| Isaac Sim | 6.1.0 |
| Isaac Lab | `v3.0.0-EA` (not latest main) |
| Isaac Lab Python | 3.12, managed by uv |
| Physics | `isaacsim_physx` |
| PPO | RSL-RL |
| Visualizer | `kit` |
| Working directory | `/root` (do not assume `/workspace`) |

### 1.1 Preflight (Web Terminal)

```bash
cd /root
nvidia-smi
free -h
df -h /root
nproc
command -v git
command -v uv
ls /opt/conda/envs/ 2>/dev/null
```

In the KasmVNC desktop terminal run `echo "$DISPLAY"`. Open the bundled Isaac Sim once and verify that a cube renders. Close it before launching a separate Kit training application. Do not install new GPU drivers inside the Pod.

## 2. Pull the repository and restore today's trainer

```bash
cd /root
if [ ! -d zeus-pure-rl/.git ]; then
  git clone https://github.com/Maverick280306/zeus-pure-rl.git
fi
cd /root/zeus-pure-rl
git rev-parse HEAD
find . -name intake_robot.py
find . -name zeus_fast.xml
```

For a private repository use an approved read-only token or SSH key; never paste secrets in chat or embed them in a URL. If the directory already exists, do not delete or clone over it. Check `git status` before changing branches. Pin a team-approved commit for reproducibility.

Transfer `NEXUS_RESULTS.zip` from your laptop via authenticated Jupyter or SSH/SCP. On the Pod place it under `/root`. Extract without installing zip tools:

```bash
cd /root
python3 -m zipfile -e NEXUS_RESULTS.zip /root/restore
ls /root/restore/NEXUS_RESULTS/trainer
```

If your backup uses tar.gz instead, use `mkdir -p /root/restore && tar -xzf NEXUS_RESULTS.tar.gz -C /root/restore` for your trusted archive. Set `BACKUP=/root/restore/NEXUS_RESULTS` in commands below (adjust only if the archive layout differs).

## 3. Install the compatible Isaac Lab release

```bash
cd /root
if [ ! -d IsaacLab/.git ]; then
  git clone --depth 1 --branch v3.0.0-EA https://github.com/isaac-sim/IsaacLab.git
fi
cd /root/IsaacLab
git describe --tags --always
```

Compare `git rev-parse HEAD` with the saved `isaaclab_commit.txt`. Do not mix 2.x scripts or the later Isaac Sim 6.2 release with this environment. If uv is absent, install using `python3 -m pip install uv` (or the official uv installer if pip is unavailable). Review and accept the NVIDIA license when prompted.

```bash
cd /root/IsaacLab
uv run --extra isaacsim isaaclab train --help
```

This initial command creates/synchronizes `.venv` and may download dependencies. It does not prove physics works.

### 3.1 Critical fix from the successful Zeus session

The previous crashes included `TfNotice`, `UsdAPISchemaBase`, `omni.usd has no attribute get_context`, and exit code 139. Standalone USD packages conflicted with Kit's own USD runtime. Remove them only from the Isaac Lab environment:

```bash
cd /root/IsaacLab
uv pip list --python .venv/bin/python | grep -Ei 'usd|isaacsim'
uv pip uninstall --python .venv/bin/python usd-core usd-exchange
```

From now on use **`uv run --no-sync --extra isaacsim`** for training/playback. A normal sync can reinstall the conflicting packages. Keep runtime imports (`DirectRLEnv`, `CartpoleEnv`, Omni/USD modules) out of pre-launch configuration modules. Use separate Python environments for MuJoCo and standalone USD conversion tools.

### 3.2 Restore the working task registration

```bash
BACKUP=/root/restore/NEXUS_RESULTS
TASK_DIR=/root/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/core/cartpole
ls "$TASK_DIR/cartpole_direct_env.py" "$TASK_DIR/cartpole_direct_env_cfg.py"
ls "$BACKUP/trainer"
```

Only proceed if paths exist and the saved Lab commit matches. For the reference run, restore the saved task files and registration from the same revision:

```bash
cp "$BACKUP/trainer/"zeus*.py "$TASK_DIR/"
cp "$TASK_DIR/__init__.py" "$TASK_DIR/__init__.py.original"
cp "$BACKUP/trainer/__init__.py" "$TASK_DIR/__init__.py"
cp "$BACKUP/zeus_training.usda" /root/zeus-pure-rl/
```

Inspect `grep -A12 'Nexus-Zeus' "$TASK_DIR/__init__.py"`. The registration must use `zeus_native_cfg:ZeusCfg`, not `zeus_native:ZeusCfg`; the walking task should point to the saved walking config and runtime modules. This is a temporary integration into upstream Cartpole infrastructure; migrate to a separate NEXUS task package later.

Optional short installation test, especially if the image changed (desktop terminal; no custom robot needed):

```bash
cd /root/IsaacLab
uv run --no-sync --extra isaacsim isaaclab train --rl_library rsl_rl \
  --task Isaac-Cartpole-Direct --num_envs 16 physics=isaacsim_physx \
  --viz kit agent.max_iterations=2
```

Close the test application before starting another.

## 4. Receive a new robot URDF

Require the following from the submitting team:

- URDF with all visual and collision mesh files, mass/inertia, joint axes and limits; no missing ROS package dependencies.
- Actuated joint names **in policy order**, passive joints, effort/velocity limits, actuator type, gear ratios where applicable, spring stiffness/damping, nominal joint positions, base link, foot links, standing base height, units and coordinate convention.
- Task objective, desired command ranges, physics/policy periods and any motion reference data.

For example, upload `robot.urdf` and `meshes/` into `/root/zeus-pure-rl/robots/new_biped/source/`. If supplied as a trusted download URL, download the entire archive (not a GitHub HTML page) via `curl -fL -o robot.zip 'DIRECT_DOWNLOAD_URL'`, check its contents and extract it; never execute downloaded scripts blindly.

```bash
cd /root/zeus-pure-rl
mkdir -p robots/new_biped/{source,generated}
TOOLS="$(dirname "$(find "$PWD" -name intake_robot.py -type f -print -quit)")"
test -f "$TOOLS/intake_robot.py" || echo 'STOP: intake tool missing'
python3 "$TOOLS/intake_robot.py" robots/new_biped/source/robot.urdf \
  --out robots/new_biped/generated/intake.json
cp "$TOOLS/../templates/robot_manifest.template.json" robots/new_biped/robot_manifest.json
```

Edit the manifest with the new robot's values. Set `format` to `urdf` and `source` to the actual path relative to repository root. Run from repository root:

```bash
python3 "$TOOLS/check_robot_contract.py" --manifest robots/new_biped/robot_manifest.json
```

**Limitation:** this existing checker verifies joint names only for MJCF; for URDF also check names manually or run the following. The intake helper does not reliably validate absolute or `package://` references. Resolve them into portable relative mesh paths in an import copy, leaving original files unchanged.

```bash
python3 - <<'PY'
from pathlib import Path
import json, xml.etree.ElementTree as ET
p=Path('robots/new_biped/source/robot.urdf').resolve()
m=json.loads(Path('robots/new_biped/robot_manifest.json').read_text())
r=ET.parse(p).getroot()
assert r.tag=='robot'
joints={j.get('name'):j.get('type') for j in r.findall('joint')}
a=m['actuated_joint_names']
assert len(a)==len(set(a))==m['action_dimension']
for n in a:
    assert joints.get(n) in ('revolute','continuous','prismatic'), n
for mesh in r.findall('.//mesh'):
    f=mesh.get('filename','')
    assert f and not f.startswith('package://'), f
    q=p.parent/f
    assert q.is_file() and q.stat().st_size>0, str(q)
print('PASS: controlled joint names and mesh paths; inspect inertia and limits separately')
PY
```

## 5. Import in Isaac Sim GUI

1. Open the bundled Isaac Sim Full 6.1 in KasmVNC, File → New, then File → Import. Select the URDF using its absolute path.
2. Choose floating base for a free-standing biped; fixed base only for a task requiring it. Preserve joint axes, inertials, limits and intended collision geometry. Do not replace valid inertias or actuate passive spring joints blindly.
3. Use a fresh output folder, e.g. `robots/new_biped/generated/import_01`. Keep associated USD payload files. For diagnostic import disable mesh merging if offered. Do not change model scale without evidence.
4. Select the robot base in Stage and press F with pointer over viewport.
5. Open Window → Script Editor; use one-line `exec(open(...).read())` to execute scripts created via Web Terminal, avoiding KasmVNC multiline paste errors.

For MJCF only: if inline vertices cause `Mesh ... has no file`, use the existing converter in a separate preparation venv with numpy/scipy. It creates convex hull approximations, not exact CAD surfaces.

```bash
cd /root/zeus-pure-rl
python3 -m venv /root/robot-prep
/root/robot-prep/bin/python -m pip install numpy scipy
/root/robot-prep/bin/python "$TOOLS/convert_mjcf_inline.py" model/zeus_fast.xml --out model/zeus_isaac.xml
find model/zeus_isaac_meshes -name '*.stl' | wc -l
```

Zeus should produce 22 files. New robots will have different counts. Import the converted XML, not the original inline version. Do not regenerate if the conversion is already valid.

### 5.1 Audit USD geometry (required)

Create this script in Web Terminal:

```bash
cat > /root/audit_robot.py <<'PY'
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics
s=omni.usd.get_context().get_stage()
print('Stage:',s.GetRootLayer().identifier,'meters/unit:',UsdGeom.GetStageMetersPerUnit(s))
roots=[]; meshes=[]
for p in s.Traverse(Usd.TraverseInstanceProxies()):
    if p.HasAPI(UsdPhysics.ArticulationRootAPI): roots.append(str(p.GetPath()))
    if p.IsA(UsdPhysics.RevoluteJoint): print('Joint:',p.GetPath())
    if p.IsA(UsdGeom.Mesh):
        g=UsdGeom.Mesh(p); points=g.GetPointsAttr().Get(); faces=g.GetFaceVertexCountsAttr().Get()
        if points is not None and len(points) and faces is not None and len(faces):
            meshes.append(str(p.GetPath()))
            print('Mesh:',p.GetPath(),g.ComputeVisibility(),g.ComputePurpose())
print('Valid meshes:',len(meshes),'Articulation roots:',roots)
assert roots, 'No articulation'
PY
```

In Script Editor run `exec(open('/root/audit_robot.py').read())`. A robot with valid primitives instead of meshes can be valid too; inspect visually. A joint tree alone is not proof of renderable geometry. Compare dimensions with the team specs. If meshes are hidden, inspect visibility/purpose, camera framing, materials and floor occlusion. Zeus had 22 collision meshes; a temporary visibility override using `MakeVisible()`, purpose `default`, and double-sided rendering made them visible. Only use that as a diagnostic; do not indiscriminately expose collision shapes on robots with proper visual meshes.

### 5.2 Save a clean training asset

In Script Editor export a composed copy (change destination for the current robot):

```python
import omni.usd; omni.usd.get_context().get_stage().Flatten().Export('/root/zeus-pure-rl/robots/new_biped/generated/imported.usda')
```

Remove imported floor and physics scene from the **copy** (Isaac Lab creates its own). Inspect for other nonrobot prims or external dependencies before use. The default prim must contain the intended articulation; preserve link transforms and physics schemas. Run the following via a terminal-created script in Script Editor:

```bash
cat > /root/clean_robot.py <<'PY'
from pxr import Usd, UsdPhysics
p='/root/zeus-pure-rl/robots/new_biped/generated/'
s=Usd.Stage.Open(p+'imported.usda')
assert s and s.GetDefaultPrim().IsValid(), 'Missing stage/default prim'
remove=[x.GetPath() for x in s.Traverse() if x.IsA(UsdPhysics.Scene) or x.GetName()=='floor']
for x in remove: s.RemovePrim(x)
roots=[x for x in s.Traverse(Usd.TraverseInstanceProxies()) if x.HasAPI(UsdPhysics.ArticulationRootAPI)]
assert len(roots)==1, 'Expected one robot articulation'
s.Export(p+'robot_training.usda')
print('Saved',p+'robot_training.usda')
PY
```

Run `exec(open('/root/clean_robot.py').read())`. Check the new file exists. If a real robot link is called `floor`, adjust the filter rather than deleting that link. Verify the new asset opens, then close standalone Isaac Sim.

# STAGE 2 — Training

## 6. Reference Zeus run (unchanged robot)

Use the saved Zeus task first to establish reproducibility. Do not run two trainers simultaneously.

```bash
cd /root/IsaacLab
set -o pipefail
uv run --no-sync --extra isaacsim isaaclab train --rl_library rsl_rl \
  --task Nexus-Zeus-Direct --num_envs 16 physics=isaacsim_physx \
  --viz kit agent.max_iterations=10 2>&1 | tee /root/zeus_smoke.log
```

If the saved V2 walking task exists, use `Nexus-Zeus-Walk` in the same command. V1 is a standing task. V2 is preliminary velocity tracking and PD control, not a validated hardware policy.

## 7. Adapt the trainer to a NEW biped (required, not just a filename swap)

The restored task reads Zeus's MJCF motor list and has Zeus-specific gains and height. A bare URDF cannot supply the task or determine which movable joints are powered. Follow the steps below to create a separate task for each new model without modifying the reference. This adaptation is a template procedure; the actual saved V2 source must be checked, and GPU validation is mandatory.

Create `robots/new_biped/training_config.json` using this schema. Example values are illustrative only, not defaults approved for any new robot:

```json
{
  "usd_path": "/root/zeus-pure-rl/robots/new_biped/generated/robot_training.usda",
  "physics_dt": 0.002,
  "policy_dt": 0.02,
  "base_height": 0.65,
  "fall_height": 0.35,
  "target_velocity": 0.2,
  "episode_seconds": 20,
  "actuators": [
    {"name":"joint_1", "effort":40, "kp":30, "kd":1, "q0":0, "action_scale":0.3},
    {"name":"joint_2", "effort":40, "kp":30, "kd":1, "q0":0, "action_scale":0.3}
  ],
  "passive_joints": [],
  "foot_links": []
}
```

Replace every example joint with real actuators; include passive joint properties (name, stiffness, damping, effort limit and rest position), foot links, and any additional task inputs. Units: m, s, rad, Nm for revolute joints. Prismatic joints need force and displacement units and task-specific controller changes. Verify policy/physics ratio is a positive integer. Do not infer effort limits from a different robot.

### 7.1 Copy and adapt the known runtime/config files

```bash
cd /root/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/core/cartpole
cp zeus_native_cfg.py new_biped_native_cfg.py
cp zeus_native.py new_biped_native.py
cp zeus_walk_cfg.py new_biped_walk_cfg.py
cp zeus_walk.py new_biped_walk.py
```

Use a code editor (e.g. `nano` or desktop editor). In the copied files replace relative module imports `.zeus_native_cfg`, `.zeus_native`, `.zeus_walk_cfg` with `.new_biped_native_cfg`, `.new_biped_native`, `.new_biped_walk_cfg` respectively. Keep class names if desired; module names isolate them. Never import runtime modules from config files.

In `new_biped_native_cfg.py`, replace the XML parsing, motor list and actuator construction section with this pattern (retain the existing Isaac Lab imports and class definition):

```python
import json
from pathlib import Path
C = json.loads(Path('/root/zeus-pure-rl/robots/new_biped/training_config.json').read_text())
A = C['actuators']
JOINTS = [a['name'] for a in A]
LIMITS = {a['name']: a['effort'] for a in A}
N = len(JOINTS)
assert N > 0 and len(set(JOINTS)) == N and all(v > 0 for v in LIMITS.values())
actuators = {a['name']: ImplicitActuatorCfg(joint_names_expr=[a['name']],
    joint_effort_limit=a['effort'], stiffness=0.0, damping=0.0) for a in A}
for p in C['passive_joints']:
    assert p['name'] not in actuators
    actuators[p['name']] = ImplicitActuatorCfg(joint_names_expr=[p['name']],
        joint_effort_limit=p['effort'], stiffness=p['stiffness'], damping=p['damping'])
Q0 = {a['name']: a['q0'] for a in A}
Q0.update({p['name']:p['q0'] for p in C['passive_joints']})
DECIMATION = round(C['policy_dt']/C['physics_dt'])
assert DECIMATION > 0 and abs(DECIMATION*C['physics_dt']-C['policy_dt']) < 1e-9
```

Then edit the config class: `usd_path=C['usd_path']`, initial base position `(0,0,C['base_height'])`, `joint_pos=Q0`, `decimation=DECIMATION`, `self.sim.dt=C['physics_dt']`, `self.sim.render_interval=self.decimation`. Set spacing to exceed the robot footprint with room to walk. Keep floating base and one articulation per environment. Remove inherited Cartpole-specific assumptions that conflict with the new model.

In `new_biped_walk_cfg.py` import `C` alongside `N`, and set `target_height=C['base_height']`, `fall_height=C['fall_height']`, `target_velocity=C['target_velocity']`, `episode_length_s=C['episode_seconds']`. Set a unique `experiment_name='nexus_new_biped_walk'`. Keep `action_space=N` and `observation_space=3*N+13` **only if the exact existing observation layout is retained** (joint positions/velocities/actions + gravity, linear/angular velocity, height, command and phase). New sensors or tasks change this dimension. V1 uses `3*N+10`; do not reuse its checkpoints for V2.

In `new_biped_walk.py` import `C` alongside `N`. Replace the gain initialization with a name-based mapping:

```python
by_name = {a['name']: a for a in C['actuators']}
names = [self.cartpole.joint_names[i] for i in self.ids]
self.kp = torch.tensor([by_name[n]['kp'] for n in names], device=self.device)
self.kd = torch.tensor([by_name[n]['kd'] for n in names], device=self.device)
self.action_scales = torch.tensor([by_name[n]['action_scale'] for n in names], device=self.device)
```

Replace `self.cfg.action_scale * self.actions` with `self.action_scales * self.actions`. Clamp target positions to the new robot's verified joint limits and torques to its configured effort limits. Ensure `self.ids` preserves configured actuator ordering. Validate imported drive stiffness/damping and passive spring behavior; avoid double PD control. Audit reset pose and root height, base orientation, foot contacts, and termination conditions. If body/link orientation differs, correct the observation/velocity convention, not the CAD.

Register the copied walking task in the same `__init__.py` using the saved Zeus walking registration as reference:

```python
gym.register(
    id='Nexus-NewBiped-Walk',
    entry_point=f'{__name__}.new_biped_walk:ZeusWalkEnv',
    disable_env_checker=True,
    kwargs={
        'env_cfg_entry_point': f'{__name__}.new_biped_walk_cfg:ZeusWalkCfg',
        'rsl_rl_cfg_entry_point': f'{__name__}.new_biped_walk_cfg:ZeusWalkPPOCfg',
        'default_agent': 'rsl_rl',
    },
)
```

Use the actual class names from the restored files if they differ. Register once. Syntax-check with `/root/IsaacLab/.venv/bin/python -m py_compile new_biped*.py`. A syntax pass is not a GPU test. For arms, quadrupeds or different control objectives, implement the relevant task and observations/rewards instead of using this biped template. Do not claim automatic arbitrary-URDF training until a generic backend is implemented and tested.

## 8. Validation gates and scale-up

From the KasmVNC desktop terminal (set `TASK=Nexus-Zeus-Walk` for the reference or `TASK=Nexus-NewBiped-Walk` for the adapted task):

```bash
cd /root/IsaacLab
TASK=Nexus-NewBiped-Walk
set -o pipefail
uv run --no-sync --extra isaacsim isaaclab train --rl_library rsl_rl \
  --task "$TASK" --num_envs 1 physics=isaacsim_physx --viz kit \
  agent.max_iterations=2 2>&1 | tee /root/robot_single.log
```

A one-environment PPO test may fail if minibatches/rollouts are incompatible; adjust the batch configuration or use a 16-environment smoke run. Before interpreting learning, inspect the single robot with zero and small bounded joint commands; confirm expected axes, stable contact, finite observations, joint limits, reset pose, and correct action mapping. No explosions or implausible falls through the floor.

```bash
uv run --no-sync --extra isaacsim isaaclab train --rl_library rsl_rl \
  --task "$TASK" --num_envs 16 physics=isaacsim_physx --viz kit \
  agent.max_iterations=10 2>&1 | tee /root/robot_visual.log
```

Pass: multiple actual robots visible, PPO iterations increasing, finite rewards/losses, checkpoints created. Falling early in training is normal; repeated immediate resets can indicate a model problem. Close the application after the run. For throughput benchmark use 64/256/512/1024 environments, one run at a time, measuring aggregate steps/sec and GPU memory (`nvidia-smi`). Choose best useful throughput, not simply highest environment count.

```bash
uv run --no-sync --extra isaacsim isaaclab train --rl_library rsl_rl \
  --task "$TASK" --num_envs 512 physics=isaacsim_physx \
  agent.max_iterations=2000 2>&1 | tee /root/robot_full.log
```

No `--viz kit` means headless for this release. 512 and 2000 are suggested starting values, not convergence guarantees. For visible training add `--viz kit` and use 16–64 environments. Monitor reward, episode length, tracking error, fall rate, torque and actual gait. Save intermediate checkpoints. New policies must be trained from scratch unless a compatible observation/action contract and explicit transfer plan exist.

## 9. MuJoCo independent baseline

Do not install MuJoCo in Isaac Lab's `.venv`.

```bash
cd /root/zeus-pure-rl
python3 -m venv /root/zeus-venv
source /root/zeus-venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install torch
python train.py --run quick_test --steps 8192 --envs 4 --device cpu \
  --rollout 2048 --batch-size 512 --save-every 2048
python eval.py --run quick_test --episodes 5
deactivate
```

These commands are for the original Zeus repository. For new URDFs, first load with `mujoco.MjModel.from_xml_path(path)` in the MuJoCo environment; verify conversion and add a free base, floor, actuators and sensors to the simulation representation as required (URDF alone may not include them). The current `ZeusPureEnv` hardcodes eight joint names, foot geoms, 38 observations, gains and timing. Its training CLI has no generic URDF option. Adapt a separate Gymnasium environment and runner to use the new robot contract before calling SB3 PPO. Do not overwrite the original reference. MuJoCo CPU physics is not native Isaac Lab GPU training. Policies cannot be exchanged without matching observation/action order, scaling, control mode and timestep.

# STAGE 3 — Evaluation, delivery and shutdown

## 10. Find and play checkpoints

```bash
cd /root/IsaacLab
find logs -type f -name 'model_*.pt' -printf '%T@ %p\n' | sort -nr | head -10
uv run --no-sync --extra isaacsim isaaclab play --help
```

Choose the checkpoint from the intended experiment (not just the newest unrelated run). In desktop terminal use its actual absolute path:

```bash
TASK=Nexus-NewBiped-Walk
CHECKPOINT=/absolute/path/to/model_100.pt  # replace this
uv run --no-sync --extra isaacsim isaaclab play --rl_library rsl_rl \
  --task "$TASK" --num_envs 1 --checkpoint "$CHECKPOINT" \
  physics=isaacsim_physx --viz kit
```

Verify play CLI against `--help` for the pinned release. Do not change the robot or observation/action mapping during playback. Record GUI video or use supported recording options after checking help. Evaluate multiple episodes with fixed seeds and disturbances; report average velocity, episode duration, falls, reward and torque. Export ONNX/TorchScript when supported by the saved runner and record normalization and joint order. A PPO checkpoint alone does not prove successful walking.

TensorBoard (if installed): `cd /root/IsaacLab && .venv/bin/tensorboard --logdir logs --host 127.0.0.1 --port 6006`. Use an authenticated SSH tunnel to view locally. Do not expose an unauthenticated service publicly.

## 11. Package results AND the working pipeline

Finish training or stop gracefully after a completed checkpoint. Run in Web Terminal:

```bash
cd /root
OUT="/root/NEXUS_RESULTS_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$OUT/trainer" "$OUT/repository"
TASK_DIR=/root/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/core/cartpole
cp -a /root/IsaacLab/logs "$OUT/"
find /root -maxdepth 1 -type f -name '*.log' -exec cp {} "$OUT/" \;
find "$TASK_DIR" -maxdepth 1 -type f \( -name 'zeus*.py' -o -name 'new_biped*.py' -o -name '__init__.py' \) -exec cp {} "$OUT/trainer/" \;
# Copy repository without git metadata or credentials; inspect untracked files before sharing.
tar -C /root/zeus-pure-rl --exclude=.git --exclude=.venv --exclude=__pycache__ -cf - . | tar -C "$OUT/repository" -xf -
if [ -f /root/zeus-pure-rl/zeus_training.usda ]; then
  cp /root/zeus-pure-rl/zeus_training.usda "$OUT/"
fi
git -C /root/IsaacLab rev-parse HEAD > "$OUT/isaaclab_commit.txt"
git -C /root/zeus-pure-rl rev-parse HEAD > "$OUT/robot_repo_commit.txt"
uv pip freeze --python /root/IsaacLab/.venv/bin/python > "$OUT/requirements_working.txt"
nvidia-smi > "$OUT/gpu_info.txt"
find "$OUT" -name 'model_*.pt'
python3 -m zipfile -c "$OUT.zip" "$OUT"
python3 -m zipfile -t "$OUT.zip"
sha256sum "$OUT.zip" > "$OUT.zip.sha256"
ls -lh "$OUT.zip" "$OUT.zip.sha256"
```

Python's standard library avoids the missing `zip` package from the previous session. Alternative: `tar -czf "$OUT.tar.gz" -C /root "$(basename "$OUT")"` and `tar -tzf "$OUT.tar.gz"`. Check that files are nonempty and the expected checkpoint is included. Include evaluation CSV, videos, task config, robot meshes/USD dependencies, README and notes if created separately. Review the archive for secrets before sharing.

## 12. Download, share, terminate

1. Download via authenticated Jupyter file browser (if accessible) or SCP using the exact SSH host/port shown under RunPod Connect. Example from your laptop: `scp -P PORT root@HOST:/root/NEXUS_RESULTS_TIMESTAMP.zip .` Replace placeholders with actual values and use approved SSH authentication. Do not run an open HTTP server exposing `/root`.
2. Verify ZIP extraction locally and checksum (`Get-FileHash filename.zip -Algorithm SHA256` in Windows PowerShell). Check the checkpoint, source, USD asset, logs and trainer files.
3. Upload the results to the team Drive. Commit reproducibility scripts and reviewed configs to the private repository. Include the experiment ID, model revision, target task, training steps, seed, metrics, playback video and known limitations in the team message.
4. Only after verifying external copies: RunPod console → Pod → Terminate → Confirm. Closing the browser does not stop billing. Check that the Pod and any separately billed volumes are gone; external registry/storage may still cost money.

## Troubleshooting — do not repeat the old failures

| Symptom | Action |
|---|---|
| Git clone says destination exists | Use existing checkout and inspect commit/status. |
| `/workspace` missing | Use `/root` for this image. |
| `intake_robot.py` missing | Locate with `find`; old repo nests it under `zeus_team_pipeline_v1/zeus_team_pipeline/tools`. |
| Inline MJCF mesh has no file | Convert external mesh copy; verify scale and paths. |
| 44 STL files for Zeus | Count only the current conversion folder; expected 22. |
| Zero USD meshes | Inspect instance proxies, payloads and latest importer logs; use original URDF if needed. |
| Robot outline but invisible legs | Check floor occlusion, visibility/purpose and camera; do not rescale blindly. |
| Kit crash 139 / TfNotice / UsdAPISchemaBase | Remove conflicting USD distributions in Lab venv, separate runtime imports, use `--no-sync`. |
| Task not registered | Restore correct registration and verify module/class names. |
| No joint mapping / shape mismatch | Match names and actuator order, compute observation size from actual features. |
| Immediate falls or NaN | Check reset pose, inertias, collision, torque, passive springs and timestep; reduce to small test. |
| GPU OOM | Close duplicate Kit apps and lower environment count. |
| ZIP missing | Use `python3 -m zipfile` or tar. |

Read the first actual error, not just crashreporter noise:

```bash
head -80 /root/robot_visual.log
grep -inE 'error|exception|traceback|failed' /root/robot_visual.log | head -40
tail -50 /root/robot_visual.log
```

## Final checklist

- [ ] Fresh Pod deployed; correct image and pinned Isaac Lab revision.
- [ ] Original asset and robot contract recorded; all mesh references resolved.
- [ ] USD visible, correct scale, articulation and physics verified.
- [ ] Small graphical PPO run produces finite metrics and a checkpoint.
- [ ] Throughput benchmark and full training completed to agreed criteria.
- [ ] Checkpoint playback, metrics and limitations documented.
- [ ] Code, configuration, model and results backed up outside RunPod.
- [ ] Team can access verified results; Pod terminated and billing checked.

## References

- [Isaac Lab v3.0.0-EA release / compatibility](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-EA)
- [Pinned release installation guide](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/setup/installation/index.html)
- [Pinned release asset import guide](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/how-to/import_new_asset.html)
- [USD conflict issue #5025](https://github.com/isaac-sim/IsaacLab/issues/5025)
- Internal sources: NEXUS Zeus RunPod full handoff (24 Nov), original repository ZIP, team pipeline ZIP, and the successful session commands/screenshots. Keep the final working backup as the authoritative source of trainer code.

**Next engineering milestone:** move task configuration and registration into a standalone NEXUS package, create an automated manifest-driven backend for each supported robot/task family, and build a pinned custom container image. Until these are tested, this README is the repeatable operator SOP, not an automatic universal training product.
