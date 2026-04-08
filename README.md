# Walker2d Gait Imitation — RL Pipeline

Reinforcement learning pipeline for learning human-like walking gait from
Ulrich IK reference data, using MuJoCo's Walker2d-v4 torque-actuated model.

The primary scientific goal is to train separate agents on **Ulrich baseline**
vs **feedback_ON** walking conditions and compare the learned gait kinematics
(ground truth: soleus activation +24%, ankle ROM +6° in feedback_ON).

A secondary muscle-actuated track using MyoAssist (22-muscle model) is also
included for future experimentation.

---

## Approach

**Phase-aware DeepMimic imitation** (`ppo_walker2d_phase.py`):
- Reference joint angles from Ulrich IK data, resampled from 50 Hz → 125 Hz
  to match Walker2d's control frequency
- Phase-conditioned observations: `[base_obs(17) | q_ref(6) | sin φ | cos φ]`
- Adaptive phase tracking: searches forward up to N frames per step, locks to
  best-matching frame — phase always advances, never regresses
- DeepMimic-style reward: `exp(-k * ||q_sim - q_ref||²)` per joint, optionally
  combined as a product of exponentials (`--product_reward`) so all joints must
  track simultaneously

See [walker2d_pretrain_runs.md](walker2d_pretrain_runs.md) for the history of
earlier approaches (phase-blind imitation, reward shaping, GAIL) and why they
failed to produce walking.

---

## Environment setup

**Python 3.11 required.** Python 3.12+ breaks MyoSuite's C++ dependencies
(`dm-tree`, `labmaze`). Not required for Walker2d-only work.

```bash
conda create -n OpenCap_RL python=3.11
conda activate OpenCap_RL
```

### Option A — NVIDIA RTX 5090 (CUDA 12.8)

Requires [CUDA 12.8 toolkit](https://developer.nvidia.com/cuda-12-8-0-download-archive).

```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements_windows_5090.txt
```

### Option B — CPU only / other machines

```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements_windows_cpu.txt
```

### Option C — macOS (Apple Silicon / Intel)

```bash
pip install -r requirements_macos.txt
```

### MyoAssist (optional)

MyoAssist requires additional system build tools:

| Package | Error | Fix |
|---------|-------|-----|
| `dm-tree` | `CMake must be installed` | Install CMake, add to PATH, restart terminal |
| `labmaze` | `command 'bazel' failed` | Install [Bazelisk](https://github.com/bazelbuild/bazelisk/releases), rename to `bazel.exe`, add to PATH |

Or try skipping the source build: `pip install dm-tree --only-binary=:all:`

---

## File layout

```
6955_Project/
│
│  Walker2d — primary track
├── ppo_walker2d_phase.py       ← main training: phase-aware DeepMimic imitation
├── ppo_walker2d.py             ← reference loader (imported by phase script)
├── pretrain_walker2d.py        ← reward-shaping pretrainer (no reference needed)
├── gail_walker2d.py            ← GAIL approach (discriminator-based)
├── extract_gait_cycle.py       ← extract single gait cycle → gait_cycle_reference.npy
├── render_walker.py            ← visualize pretrain checkpoints (vanilla Walker2d)
├── render_phase.py             ← visualize phase-aware policy
│
│  MyoAssist — secondary track (muscle-actuated, 22 muscles)
├── ppo_myoassist.py            ← PPO training on MyoAssist env
├── ppo_walk.py                 ← MyoSuite walking baseline (myoLegWalk-v0)
├── render_myoassist.py         ← visualize MyoAssist policy
├── train.py                    ← BC + GAIL pipeline entry point
├── bc_policy.py                ← behavioural cloning policy + trainer
├── gail.py                     ← discriminator + PPO GAIL loop
├── data_utils.py               ← Ulrich data loading and preprocessing
├── evaluate.py                 ← quantitative policy evaluation
│
│  Data / reference
├── gait_cycle_reference.npy    ← single Ulrich gait cycle @ 50Hz (6 joints)
├── Ulrich_Treadmill_Data/      ← Ulrich IK trials, Subject1-10, baseline + feedback_ON
├── OpenCap_data/               ← OpenCap markerless motion capture data
│
│  Results
├── results/
│   ├── walker2d_phase_cycle_20260408-115434/   ← first walking policy (base for finetune)
│   ├── walker2d_pretrain_symmetry_*/           ← 4 pretrain keeper runs (see run log)
│   └── walker2d_ulrich_all_20260406-221644/    ← pre-phase failure demo
│
├── walker2d_pretrain_runs.md   ← run log with training curves + render commands
├── requirements_windows_5090.txt
├── requirements_windows_cpu.txt
└── requirements_macos.txt
```

---

## Data sources

### `Ulrich_Treadmill_Data/`
Treadmill walking IK data from the Ulrich dataset. Used as the primary reference
for Walker2d imitation training. Contains two conditions of interest:

- **baseline** — normal treadmill walking
- **feedback_ON** — walking with real-time biofeedback (ground truth: soleus activation
  +24%, ankle ROM +6° vs baseline)

Layout: `Subject{N}/IK/walking_{condition}{trial}/output/results_ik.sto`

Loaded by `ppo_walker2d.py::load_ulrich_reference()` and `extract_gait_cycle.py`.

### `OpenCap_data/`
Markerless motion capture data from OpenCap (video-based pose estimation → OpenSim IK).
Used by the MyoAssist BC+GAIL pipeline (`train.py`, `data_utils.py`).

Layout:
```
OpenCap_data/
└── subject{N}/
    ├── OpenSimData/
    │   └── {source}/IK/{trial}.mot     ← joint angles
    ├── EMGData/{trial}_EMG.sto          ← muscle activations
    └── ForceData/{trial}_forces.mot     ← ground reaction forces
```

Sources: `Mocap` (markered gold standard), `Video/HRNet`, `Video/OpenPose_default`,
`Video/OpenPose_highAccuracy`.

---

## Quickstart — Walker2d

### 1. Extract gait cycle from Ulrich data

```bash
python extract_gait_cycle.py --subject 1 --trial baseline
# → saves gait_cycle_reference.npy
```

### 2. Train phase-aware imitation (from scratch)

```bash
python ppo_walker2d_phase.py --ref_cycle gait_cycle_reference.npy \
    --num_envs 32 --total_steps 5e6
```

### 3. Finetune from the walking checkpoint

```bash
python ppo_walker2d_phase.py --ref_cycle gait_cycle_reference.npy \
    --finetune results/walker2d_phase_cycle_20260408-115434/model.zip \
    --num_envs 32 --total_steps 5e6
```

### 4. Render a trained policy

```bash
# Phase-aware model
python render_phase.py --model results/walker2d_phase_cycle_20260408-115434/model.zip \
    --ref_cycle gait_cycle_reference.npy --steps 500

# Pretrain / vanilla Walker2d model
python render_walker.py --model results/walker2d_pretrain_symmetry_<timestamp>/model.zip \
    --vanilla --steps 500
```

### Key flags — `ppo_walker2d_phase.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--ref_cycle` | — | Path to gait cycle `.npy` (required unless `--ref_all`) |
| `--num_envs` | 32 | Parallel environments |
| `--total_steps` | 5e6 | Total env steps |
| `--finetune` | None | Pretrained `.zip` to finetune from |
| `--product_reward` | off | DeepMimic product-of-exps reward (stricter) |
| `--imit_weight` | 4.0 | Joint imitation reward weight |
| `--forward_weight` | 1.0 | Forward velocity reward weight |
| `--contact_weight` | 2.0 | Foot contact reward weight |
| `--pose_term` | 0.9 rad | Hip/knee deviation termination threshold |
| `--ankle_term` | 0.40 rad | Ankle deviation termination threshold |

---

## Quickstart — MyoAssist

```bash
# Train
python ppo_myoassist.py --num_envs 16 --total_steps 1e7

# Evaluate
python evaluate.py --policy_ckpt checkpoints/policy_final.pt \
    --subject subject10 --trial walking1

# Render
python render_myoassist.py --model <checkpoint>
```

The MyoAssist pipeline also supports BC pre-training + GAIL fine-tuning:

```bash
# Stage 1 — behavioural cloning
python train.py --mode bc --subject subject10 --trial walking1 --bc_epochs 200

# Stage 2 — GAIL
python train.py --mode gail --subject subject10 --trial walking1 \
    --bc_ckpt checkpoints/bc_policy_best.pt --gail_steps 500000
```
