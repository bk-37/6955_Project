# Muscle Activation RL Pipeline

A two-stage training pipeline:

1. **Behavioural Cloning (BC)** — supervised pre-training of a policy from IK + EMG data.
2. **GAIL fine-tuning** — adversarial imitation learning in a physics environment (MyoSuite), refining the policy via RL.

The core framing is **RL**: the agent observes joint angles (state) and outputs muscle activations (action). BC provides a warm-start; GAIL handles the RL fine-tuning.

---

## File layout

```
AI_Project/
├── data/
│   └── subject{N}/                        ← 10 subjects (subject2–subject11)
│       ├── sessionMetadata.yaml
│       ├── EMGData/
│       │   └── {trial}_EMG.sto            ← processed EMG activations
│       ├── ForceData/
│       │   └── {trial}_forces.mot         ← ground reaction forces
│       ├── MarkerData/
│       │   ├── Mocap/
│       │   │   └── {trial}.trc            ← markered mocap
│       │   └── Video/
│       │       ├── HRNet/
│       │       │   ├── 2-cameras/
│       │       │   │   └── {trial}_videoAndMocap.trc
│       │       │   └── 3-cameras/
│       │       │       └── {trial}_videoAndMocap.trc
│       │       ├── OpenPose_default/
│       │       │   └── {trial}_videoAndMocap.trc
│       │       └── OpenPose_highAccuracy/
│       │           └── {trial}_videoAndMocap.trc
│       └── OpenSimData/
│           ├── Mocap/
│           │   ├── IK/   → {trial}.mot    ← joint angles (markered IK)
│           │   ├── ID/   → {trial}.sto    ← inverse dynamics
│           │   ├── SO/   → {trial}.sto    ← static optimization activations
│           │   ├── JR/   → {trial}.sto    ← joint reaction forces
│           │   └── Model/                 ← scaled .osim model
│           └── Video/
│               ├── HRNet/     → IK, SO, etc. from video-based pose estimation
│               ├── OpenPose_default/
│               └── OpenPose_highAccuracy/
├── data_utils.py       ← data loading, alignment, preprocessing
├── bc_policy.py        ← policy network + BC trainer
├── gail.py             ← discriminator + PPO + GAIL loop
├── train.py            ← entry point
└── evaluate.py         ← quantitative evaluation
```

### Trial names (per subject)

| Trial | Description |
|---|---|
| `walking1/2/3` | Normal walking |
| `walkingTS1/2/3` | Treadmill/speed-varied walking |
| `squats1` | Squat |
| `squatsAsym1` | Asymmetric squat |
| `STS1` | Sit-to-stand |
| `STSweakLegs1` | Sit-to-stand with weakened legs |
| `DJ1/2/3` | Drop jump |
| `DJAsym1/2/3` | Asymmetric drop jump |
| `static1` | Static calibration trial |

---

## RL framing

### State (S)

```
Joint angles from IK    (N DOFs, radians)
Joint angular velocities (N DOFs, rad/s via central differences)
─────────────────────────────────────────
Total S = 2N
```

### Action (A)

Muscle activations ∈ [0, 1] — one per EMG channel, mapped to MyoSuite actuators.

### Reward

During GAIL, the reward is provided by the discriminator:
`r = log(D(s, a))` — how indistinguishable the agent's (state, action) pairs are from expert data.

---

## Architecture

### Policy π_θ  (`bc_policy.py`)

Used for both BC pre-training and as the RL actor:

```
Linear(S→256) → LayerNorm → ELU → Dropout
Linear(256→256) → LayerNorm → ELU → Dropout
Linear(256→128) → LayerNorm → ELU → Dropout
Linear(128→A) → Sigmoid
```

Plus a learnable `log_std` vector for PPO stochastic sampling.

### Discriminator D_φ  (`gail.py`)

```
SpectralNorm-Linear(S+A→256) → LeakyReLU(0.2)
SpectralNorm-Linear(256→256) → LeakyReLU(0.2)
Linear(256→1)
```

---

## Data sources per trial

Each trial has three parallel expert data sources for the IK state:

| Source | Path | Notes |
|---|---|---|
| **Markered mocap** | `OpenSimData/Mocap/IK/{trial}.mot` | Gold standard |
| **HRNet video** | `OpenSimData/Video/HRNet/{2,3}-cameras/IK/{trial}.mot` | Markerless |
| **OpenPose** | `OpenSimData/Video/OpenPose_*/IK/{trial}.mot` | Markerless alt |

EMG activations (`EMGData/{trial}_EMG.sto`) are shared across IK sources.
Static optimization activations (`OpenSimData/Mocap/SO/{trial}.sto`) are also available as an alternative expert action source.

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install myosuite
```

### 2. Phase 1 — BC pre-training

```bash
python train.py --mode bc --subject subject10 --trial walking1 --bc_epochs 200
# → saves checkpoints/bc_policy_best.pt
```

### 3. Phase 2 — GAIL (RL fine-tuning)

```bash
python train.py --mode gail --subject subject10 --trial walking1 \
    --source mocap --gail_steps 200000
```

### 4. Train across multiple subjects/trials

```bash
python train.py --mode gail --subject all --trial walking1 walking2 walking3 \
    --source mocap --gail_steps 500000
```

### 5. Evaluate

```bash
python evaluate.py \
    --policy_ckpt checkpoints/policy_final.pt \
    --disc_ckpt   checkpoints/disc_final.pt \
    --subject subject10 --trial walking1
```

---

## Hyperparameter reference

| Parameter | Default | Notes |
|---|---|---|
| `bc_epochs` | 200 | BC pre-training epochs |
| `l1_lambda` | 1e-3 | Muscle sparsity penalty |
| `gail_steps` | 200 000 | Total env steps for GAIL/PPO |
| `rollout_len` | 512 | Steps per PPO rollout |
| `ppo_clip` | 0.2 | PPO ε |
| `disc_epochs` | 3 | Discriminator updates per rollout |
| `gp_lambda` | 10.0 | WGAN-GP gradient penalty (0 = off) |
| `smooth_emg_hz` | 10 Hz | Low-pass cutoff for EMG smoothing |
| `smooth_ik_hz` | 6 Hz | Low-pass cutoff for IK smoothing |
