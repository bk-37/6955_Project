"""
train.py
────────
Entry point for:

  Phase 1 → Behavioural Cloning  (BC)
  Phase 2 → GAIL fine-tuning      (markered or markerless IK as expert)

Usage
─────
  # BC only (single subject/trial):
  python train.py --mode bc --subject subject10 --trial walking1

  # BC → GAIL with markered mocap IK:
  python train.py --mode gail --subject subject10 --trial walking1 --source Mocap

  # BC → GAIL with markerless HRNet IK:
  python train.py --mode gail --subject subject10 --trial walking1 --source Video/HRNet/2-cameras

  # Skip BC and load existing checkpoint:
  python train.py --mode gail --subject subject10 --trial walking1 --bc_ckpt checkpoints/bc_policy.pt

  # Train across multiple subjects/trials:
  python train.py --mode gail --subject subject10 subject11 --trial walking1 walking2 walking3

Data layout (relative to this script):
  data/{subject}/EMGData/{trial}_EMG.sto
  data/{subject}/OpenSimData/{source}/IK/{trial}.mot
  data/{subject}/ForceData/{trial}_forces.mot   ← optional GRFs

MyoSuite environment
────────────────────
  Set ENV_ID to a registered MyoSuite env, e.g. "myoLegWalk-v0".
  If MyoSuite is not installed the script falls back to a lightweight
  DummyEnv that mirrors the data dimensions — useful for smoke-testing
  the pipeline without a physics engine.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# ── local modules ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from data_utils  import ExpertData, GAILDataset, make_dataloaders, load_multi, load_ulrich_multi, IK_ALL_COLS, N_MUSCLES
from bc_policy   import BCPolicy, BCTrainer
from gail        import Discriminator, GAILTrainer


# ── environment ───────────────────────────────────────────────────────────────

ENV_ID = "myoLegWalk-v0"   # ← change to your MyoSuite env


class IKObsWrapper:
    """
    Wraps a MyoSuite env so its observation matches the BC state space (S=32):
      [ 16 joint angles (rad),  16 joint angular velocities (rad/s) ]

    MyoSuite myoleg has no pelvis_tilt/list/rotation or lumbar_* joints, so:
      • Pelvis angles  ← root free-joint quaternion converted to ZYX Euler
      • Pelvis angvels ← root qvel[3:6]  (body-frame angular velocity)
      • Lumbar angles/vels ← zero-padded  (myoleg is a legs-only model)
      • 10 leg joints  ← direct qpos/qvel lookup by name
    """

    # Directly mapped leg joints (present in both OpenSim IK and myoleg)
    LEG_JOINTS = [
        "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
        "knee_angle_r",  "ankle_angle_r",
        "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
        "knee_angle_l",  "ankle_angle_l",
    ]
    N_PELVIS = 3   # pelvis_tilt, pelvis_list, pelvis_rotation
    N_LUMBAR = 3   # lumbar_extension, lumbar_bending, lumbar_rotation
    N_TOTAL  = 16  # = N_PELVIS + len(LEG_JOINTS) + N_LUMBAR

    def __init__(self, env):
        self._env = env
        sim = env.unwrapped.sim
        all_joints = [sim.model.joint(i).name for i in range(sim.model.njnt)]

        # root free joint indices
        root_id = all_joints.index("root")
        root_j  = sim.model.joint(root_id)
        self._root_qpos_adr = root_j.qposadr[0]   # qpos[adr:adr+7] = [x,y,z, qw,qx,qy,qz]
        self._root_qvel_adr = root_j.dofadr[0]    # qvel[adr:adr+6] = [vx,vy,vz, wx,wy,wz]

        # leg joint indices
        self._leg_qpos = []
        self._leg_qvel = []
        for jname in self.LEG_JOINTS:
            if jname not in all_joints:
                raise ValueError(f"[IKObsWrapper] Expected joint '{jname}' not in model.")
            jid = all_joints.index(jname)
            self._leg_qpos.append(sim.model.joint(jid).qposadr[0])
            self._leg_qvel.append(sim.model.joint(jid).dofadr[0])

        # action mapping: 16 EMG muscles → MyoSuite actuator indices
        from data_utils import MYOSUITE_MUSCLE_MAP
        n_act = sim.model.nu
        actuator_names = [sim.model.actuator(i).name for i in range(n_act)]
        self._full_action_dim = n_act
        self._act_indices = []
        for myo_name in MYOSUITE_MUSCLE_MAP.values():
            if myo_name not in actuator_names:
                raise ValueError(
                    f"[IKObsWrapper] Actuator '{myo_name}' not found in MyoSuite model.\n"
                    f"Available actuators: {actuator_names}"
                )
            self._act_indices.append(actuator_names.index(myo_name))

        print(f"[IKObsWrapper] pelvis←root quat, {len(self.LEG_JOINTS)} leg joints, "
              f"{self.N_LUMBAR} lumbar zero-padded → obs dim {2 * self.N_TOTAL}")
        print(f"[IKObsWrapper] action: 16 EMG muscles → {n_act}-dim actuator space")

    @staticmethod
    def _quat_to_euler_zyx(q: np.ndarray) -> np.ndarray:
        """
        MuJoCo quaternion [qw, qx, qy, qz] → ZYX Euler [tilt, list, rotation].
        Matches the OpenSim pelvis convention (Z=tilt, X=list, Y=rotation).
        """
        w, x, y, z = q
        tilt     = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        lst      = np.arcsin( np.clip(2*(w*x - z*y), -1.0, 1.0))
        rotation = np.arctan2(2*(w*y + z*x), 1 - 2*(x*x + y*y))
        return np.array([tilt, lst, rotation], dtype=np.float32)

    def _extract(self) -> np.ndarray:
        sim  = self._env.unwrapped.sim
        qpos = sim.data.qpos
        qvel = sim.data.qvel

        # pelvis (3 angles from root quaternion, 3 angvels from root qvel)
        root_quat    = qpos[self._root_qpos_adr + 3 : self._root_qpos_adr + 7]
        pelvis_ang   = self._quat_to_euler_zyx(root_quat)
        pelvis_vel   = qvel[self._root_qvel_adr + 3 : self._root_qvel_adr + 6].astype(np.float32)

        # leg joints
        leg_ang = qpos[self._leg_qpos].astype(np.float32)
        leg_vel = qvel[self._leg_qvel].astype(np.float32)

        # lumbar: zero-padded
        lumbar_ang = np.zeros(self.N_LUMBAR, dtype=np.float32)
        lumbar_vel = np.zeros(self.N_LUMBAR, dtype=np.float32)

        # assemble in same order as IK_ROTATIONAL_COLS:
        # [pelvis×3, hip_r×3, knee_r, ankle_r, hip_l×3, knee_l, ankle_l, lumbar×3]
        angles = np.concatenate([pelvis_ang, leg_ang, lumbar_ang])
        vels   = np.concatenate([pelvis_vel, leg_vel, lumbar_vel])
        return np.concatenate([angles, vels])

    def reset(self, **kwargs):
        self._env.reset(**kwargs)
        return self._extract(), {}

    def _expand_action(self, action: np.ndarray) -> np.ndarray:
        """Map 16-dim policy action → full MyoSuite actuator vector."""
        full = np.zeros(self._full_action_dim, dtype=np.float32)
        full[self._act_indices] = action
        return full

    def step(self, action):
        _, reward, terminated, truncated, info = self._env.step(self._expand_action(action))
        return self._extract(), reward, terminated, truncated, info

    def close(self):
        self._env.close()

    @property
    def observation_space(self):
        import gymnasium as gym
        n = 2 * self.N_TOTAL
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(n,), dtype=np.float32)

    @property
    def action_space(self):
        import gymnasium as gym
        n = len(self._act_indices)
        return gym.spaces.Box(low=0.0, high=1.0, shape=(n,), dtype=np.float32)

    @property
    def unwrapped(self):
        return self._env.unwrapped


def make_env(state_dim: int, action_dim: int):
    """
    Try to load MyoSuite; fall back to a DummyEnv with matching dimensions.
    Wraps the MyoSuite env with IKObsWrapper to align obs space with BC state.
    """
    try:
        import myosuite  # noqa: F401
        import gymnasium as gym
        env = gym.make(ENV_ID)
        env = IKObsWrapper(env)
        print(f"[env] Loaded MyoSuite env: {ENV_ID}")
        return env
    except ImportError:
        print("[env] MyoSuite / gymnasium not found — using DummyEnv "
              "(pipeline smoke-test only)")
        return DummyEnv(state_dim, action_dim)


class DummyEnv:
    """
    Minimal gym-compatible environment that echoes random states.
    Allows the full BC→GAIL code to run without MyoSuite installed.
    """

    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self._step_n    = 0
        self.max_steps  = 200

    def reset(self, **kwargs):
        self._step_n = 0
        obs = np.random.randn(self.state_dim).astype(np.float32) * 0.1
        return obs, {}

    def step(self, action):
        self._step_n += 1
        obs      = np.random.randn(self.state_dim).astype(np.float32) * 0.1
        reward   = float(np.random.randn())
        done     = self._step_n >= self.max_steps
        return obs, reward, done, False, {}

    @property
    def observation_space(self):
        class _Space:
            shape = (self.state_dim,)
        return _Space()

    @property
    def action_space(self):
        class _Space:
            shape = (self.action_dim,)
        return _Space()


# ── helpers ───────────────────────────────────────────────────────────────────

def build_expert(
    subjects: list,
    trials:   list,
    source:   str  = "Mocap",
    use_grf:  bool = False,
) -> ExpertData:
    """
    Load and concatenate expert data for one or more subjects/trials.

    source examples: "Mocap", "Video/HRNet/2-cameras", "Video/HRNet/3-cameras",
                     "Video/OpenPose_default", "Video/OpenPose_highAccuracy"
    """
    if len(subjects) == 1 and len(trials) == 1:
        return ExpertData(subjects[0], trials[0], source=source, use_grf=use_grf)
    return load_multi(subjects, trials, source=source, use_grf=use_grf)


def state_dim_from_expert(expert: ExpertData) -> int:
    return expert.S   # IK rotations + translations + velocities


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BC → GAIL training for MyoSuite")
    parser.add_argument("--mode",    choices=["bc", "gail"], default="bc")
    parser.add_argument("--subject", nargs="+", default=["subject10"],
                        help="Subject folder name(s), e.g. subject10 subject11")
    parser.add_argument("--trial",   nargs="+", default=["walking1"],
                        help="Trial name(s), e.g. walking1 walking2 walking3")
    parser.add_argument("--source",  default="Mocap",
                        help='IK source: "Mocap", "Video/HRNet/2-cameras", '
                             '"Video/HRNet/3-cameras", "Video/OpenPose_default", '
                             '"Video/OpenPose_highAccuracy"')
    parser.add_argument("--bc_ckpt",  type=str, default=None,
                        help="Path to existing BC checkpoint (skips Phase 1)")
    parser.add_argument("--bc_epochs", type=int, default=200)
    parser.add_argument("--bc_patience", type=int, default=30)
    parser.add_argument("--gail_steps", type=int, default=200_000)
    parser.add_argument("--use_grf",   action="store_true")
    parser.add_argument("--ulrich",    action="store_true",
                        help="Supplement OpenCap data with Ulrich static-opt data")
    parser.add_argument("--source_aware", action="store_true",
                        help="Condition discriminator on mocap source tag")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = args.device
    ckpt_dir = Path("checkpoints")

    # ── load expert data ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Loading expert data  (subjects: {args.subject}  trials: {args.trial}  source: {args.source})")
    print(f"{'='*60}")
    expert = build_expert(args.subject, args.trial, source=args.source, use_grf=args.use_grf)

    if args.ulrich:
        print(f"\n{'='*60}")
        print(f"  Loading Ulrich static-opt data")
        print(f"{'='*60}")
        ulrich_subjects = [f"Subject{i}" for i in range(1, 11)]
        ulrich_trials   = [
            "walking_baseline1", "walking_retention1",
            "walking_FBcolor1_finalFB1", "walking_FBcolor1_finalNoFB1",
            "walking_FBcolor2_finalFB1", "walking_FBcolor2_finalNoFB1",
            "walking_FBexp1_finalFB1",   "walking_FBexp1_finalNoFB1",
        ]
        ulrich = load_ulrich_multi(ulrich_subjects, ulrich_trials)
        expert.states  = np.concatenate([expert.states,  ulrich.states],  axis=0)
        expert.actions = np.concatenate([expert.actions, ulrich.actions], axis=0)
        expert.T       = expert.states.shape[0]
        print(f"  Combined dataset: T={expert.T}")

    S = state_dim_from_expert(expert)
    A = expert.A
    print(f"  State dim  S = {S}")
    print(f"  Action dim A = {A}")

    # ── build models ──────────────────────────────────────────────────────
    policy        = BCPolicy(state_dim=S, action_dim=A, hidden_dims=(256, 256, 128))
    discriminator = Discriminator(
        state_dim=S, action_dim=A,
        source_aware=args.source_aware,
    )

    # ── phase 1: BC ───────────────────────────────────────────────────────
    if args.bc_ckpt:
        print(f"\n[Phase 1] Loading BC checkpoint: {args.bc_ckpt}")
        policy.load_state_dict(torch.load(args.bc_ckpt, map_location=device))
    else:
        print(f"\n{'='*60}")
        print(f"  Phase 1: Behavioural Cloning")
        print(f"{'='*60}")

        train_loader, val_loader = make_dataloaders(expert, batch_size=32, val_frac=0.1)

        bc_trainer = BCTrainer(policy, lr=3e-4, l1_lambda=1e-3, device=device)
        bc_trainer.fit(
            train_loader, val_loader,
            epochs=args.bc_epochs,
            patience=args.bc_patience,
            save_path=ckpt_dir / "bc_policy_best.pt",
            verbose_every=50,
        )

    if args.mode == "bc":
        print("\n[Done] BC training only. Exiting.")
        return

    # ── phase 2: GAIL ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Phase 2: GAIL  (expert source: {args.source})")
    print(f"{'='*60}")

    expert_dataset = GAILDataset(expert)
    env            = make_env(state_dim=S, action_dim=A)

    gail_trainer = GAILTrainer(
        env=env,
        policy=policy,
        discriminator=discriminator,
        expert_dataset=expert_dataset,
        device=device,
        ppo_epochs=5,
        ppo_clip=0.2,
        vf_coef=0.5,
        ent_coef=0.05,
        disc_epochs=3,
        disc_batch=64,
        gp_lambda=10.0,
        rollout_len=2048,
        lr_policy=3e-4,
        lr_disc=1e-4,
        source_aware=args.source_aware,
    )

    source_tag = 0 if args.source == "Mocap" else 1
    metrics    = gail_trainer.train(
        total_steps=args.gail_steps,
        log_every=5_000,
        save_every=20_000,
        checkpoint_dir=ckpt_dir,
        mocap_source=source_tag,
    )

    # ── save final ────────────────────────────────────────────────────────
    torch.save(policy.state_dict(),        ckpt_dir / "policy_final.pt")
    torch.save(discriminator.state_dict(), ckpt_dir / "disc_final.pt")
    print(f"\n[Done] Final models saved to {ckpt_dir}/")


if __name__ == "__main__":
    main()
