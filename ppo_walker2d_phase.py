"""
ppo_walker2d_phase.py
─────────────────────
Phase-aware imitation learning for Walker2d-v4.

Fixes three core problems with ppo_walker2d.py:

  1. Policy was blind to phase.
     FIX: append [q_ref(6), sin(φ), cos(φ)] to obs → 25-dim.
          The policy can now condition actions on where it is in the gait
          cycle AND see the target joints directly.

  2. Phase was open-loop (tick +1 per step regardless of agent state).
     FIX: adaptive phase — each step, search forward up to
          max_phase_advance frames and lock to the best-matching frame.
          Phase always moves forward (no regression), but can skip frames
          if the agent is slightly ahead, or stall if it falls behind.
          This keeps the reward signal grounded in the agent's actual state.

  3. Reference had trial-boundary discontinuities (413k concatenated frames).
     FIX: default to a single clean gait cycle (looped). The --ref_all flag
          re-enables the full concatenated reference if desired.

Usage
─────
  # Recommended: single gait cycle (extracted by extract_gait_cycle.py)
  python ppo_walker2d_phase.py --ref_cycle gait_cycle_reference.npy

  # Full Ulrich reference (handles discontinuities via gait-cycle wrapping)
  python ppo_walker2d_phase.py --ref_all --subjects 1 2 3

  # Finetune from pretrain_walker2d.py checkpoint
  python ppo_walker2d_phase.py --ref_cycle gait_cycle_reference.npy \\
      --finetune results/walker2d_pretrain_symmetry_*/model.zip

  # Quick smoke-test
  python ppo_walker2d_phase.py --ref_cycle gait_cycle_reference.npy \\
      --num_envs 4 --total_steps 2e5
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.walker2d_v4 import Walker2dEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, CallbackList
)

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
from ppo_walker2d import load_ulrich_reference   # reuse loader

CTRL_HZ = 125.0   # Walker2d-v4: frame_skip=4, dt=0.002s
REF_HZ  = 50.0    # Ulrich IK / extracted gait cycles


# ── reference loading ─────────────────────────────────────────────────────────

def load_ref_cycle(path: Path) -> np.ndarray:
    """Load a single gait cycle .npy, resample to CTRL_HZ with cubic spline."""
    from scipy.interpolate import CubicSpline
    raw   = np.load(path).astype(np.float32)
    n_in  = len(raw)
    n_out = int(round(n_in * CTRL_HZ / REF_HZ))
    x_in  = np.linspace(0, 1, n_in)
    x_out = np.linspace(0, 1, n_out)
    ref   = np.stack(
        [CubicSpline(x_in, raw[:, j])(x_out) for j in range(raw.shape[1])],
        axis=1,
    ).astype(np.float32)
    print(f"Gait cycle: {n_in} frames @ {REF_HZ}Hz -> {n_out} frames @ {CTRL_HZ}Hz (cubic spline)")
    return ref


# ── env ───────────────────────────────────────────────────────────────────────

# Walker2d joint limits (rad) — slightly relaxed at hip to cover Ulrich range
_JNT_LO = np.array([-2.618, -2.618, -0.785, -2.618, -2.618, -0.785], dtype=np.float32)
_JNT_HI = np.array([ 0.349,  0.000,  0.785,  0.349,  0.000,  0.785], dtype=np.float32)


class Walker2dPhaseAware(Walker2dEnv):
    """
    Walker2d-v4 with:
      • Phase-conditioned observations  [base(17) | q_ref(6) | sin φ | cos φ]
      • Adaptive phase tracking         (searches forward, picks best match)
      • DeepMimic-style imitation reward

    Reward
    ──────
      r = w_imit * Σ_j exp(-k * (q_j - q_ref_j)²)   ← joint tracking
        + w_fwd  * exp(-5 * (x_vel - v_target)²)      ← velocity target
        + w_vel  * Σ_j exp(-k * (dq_j - dq_ref_j)²)  ← velocity tracking
        - 1e-3   * ||action||²                         ← ctrl cost
      Early terminate if any non-ankle joint deviates > pose_term_thresh,
      or ankle deviates > ankle_term_thresh, or x_vel < -0.1.
    """

    # base Walker2d obs dim (qpos[1:] + qvel = 8 + 9 = 17)
    BASE_OBS  = 17
    N_REF     = 6   # hip/knee/ankle × 2
    N_PHASE   = 2   # sin φ, cos φ
    OBS_DIM   = BASE_OBS + N_REF + N_PHASE   # = 25

    def __init__(
        self,
        reference:          np.ndarray,   # (T, 6) float32 @ CTRL_HZ
        imitation_weight:   float = 4.0,
        vel_weight:         float = 1.0,
        ee_weight:          float = 4.0,  # end-effector foot position tracking
        root_weight:        float = 2.0,  # root height + pitch tracking
        contact_weight:     float = 1.0,
        imit_scale:         float = 8.0,  # sharpness of exp(-k·err²)
        max_phase_advance:  int   = 4,    # max frames to skip per step
        pose_term_thresh:   float = 0.9,  # rad — hip/knee termination
        ankle_term_thresh:  float = 0.40, # rad — ankle termination (looser)
        warm_start:         bool  = True,
        product_reward:     bool  = False,
        **kwargs,
    ):
        self._reference         = reference
        self._ref_len           = len(reference)
        self._imitation_weight  = imitation_weight
        self._vel_weight        = vel_weight
        self._ee_weight         = ee_weight
        self._root_weight       = root_weight
        self._contact_weight    = contact_weight
        self._imit_scale        = imit_scale
        self._max_phase_advance = max_phase_advance
        self._pose_term_thresh  = pose_term_thresh
        self._ankle_term_thresh = ankle_term_thresh
        self._warm_start        = warm_start
        self._product_reward    = product_reward
        self._phase             = 0

        # Pre-compute per-frame velocity from reference (for velocity tracking)
        # Shape (T, 6) — central differences, edges use forward/back
        self._ref_vel = np.gradient(reference, 1.0 / CTRL_HZ, axis=0).astype(np.float32)

        # Pre-compute stance side per frame from reference hip angles.
        # In Walker2d convention, the stance hip is more extended (less negative / more positive).
        # ref[:, 0] = hip_r,  ref[:, 3] = hip_l
        # stance_right[t] = True when right hip is more extended than left at frame t.
        self._stance_right = reference[:, 0] >= reference[:, 3]  # (T,) bool

        super().__init__(**kwargs)

        # Override observation_space after super().__init__ sets it —
        # MujocoEnv.__init__ assigns self.observation_space directly, so a
        # @property with no setter causes an AttributeError.
        self.observation_space = spaces.Box(
            low   = -np.inf,
            high  =  np.inf,
            shape = (self.OBS_DIM,),
            dtype = np.float32,
        )

        # Pre-compute reference foot positions (world z and root-relative x)
        # and root height via FK — used for EE and root reward terms.
        self._precompute_reference_kinematics()

    def _get_obs(self) -> np.ndarray:
        base    = super()._get_obs()                       # (17,)
        q_ref   = self._reference[self._phase]             # (6,)
        # Normalize phase to gait cycle period, not reference length.
        # With long references (7570 frames) the full sin/cos cycle would
        # take 60s — useless as a within-stride signal. Normalizing to
        # GAIT_CYCLE_FRAMES keeps the encoding meaningful regardless of
        # whether we use a single extracted cycle or a long continuous ref.
        GAIT_CYCLE_FRAMES = 140  # ~1.1s @ 125Hz
        phi = 2.0 * np.pi * (self._phase % GAIT_CYCLE_FRAMES) / GAIT_CYCLE_FRAMES
        phase_enc = np.array([np.sin(phi), np.cos(phi)], dtype=np.float32)
        return np.concatenate([base, q_ref, phase_enc])    # (25,)

    # ── reference FK pre-computation ──────────────────────────────────────

    def _precompute_reference_kinematics(self) -> None:
        """
        For each reference frame run FK to get:
          - ref_root_height: torso z in world frame
          - ref_foot_r/l_x_rel: foot x relative to root (forward placement)
          - ref_foot_r/l_z: foot z in world frame (swing elevation signal)

        Using world-z for feet rather than root-relative-z is critical:
        root-relative z is always negative (~-1.1m) with no swing signal.
        World-z is ~0 at stance and positive during swing.
        """
        n = self._ref_len
        self._ref_root_height  = np.zeros(n, dtype=np.float32)
        self._ref_foot_r_xrel  = np.zeros(n, dtype=np.float32)  # foot x relative to root
        self._ref_foot_l_xrel  = np.zeros(n, dtype=np.float32)
        self._ref_foot_r_z     = np.zeros(n, dtype=np.float32)  # foot world z
        self._ref_foot_l_z     = np.zeros(n, dtype=np.float32)

        qpos_save = self.data.qpos.copy()
        qvel_save = self.data.qvel.copy()

        for t in range(n):
            self.data.qpos[:] = 0.0
            self.data.qpos[1] = 1.28    # nominal standing height
            self.data.qpos[3:9] = self._reference[t]
            self.data.qvel[:] = 0.0
            mujoco.mj_kinematics(self.model, self.data)

            root = self.data.body("torso").xpos
            ftr  = self.data.body("foot").xpos
            ftl  = self.data.body("foot_left").xpos

            self._ref_root_height[t]  = float(root[2])
            self._ref_foot_r_xrel[t]  = float(ftr[0] - root[0])
            self._ref_foot_l_xrel[t]  = float(ftl[0] - root[0])
            self._ref_foot_r_z[t]     = float(ftr[2] - root[2])  # root-relative z
            self._ref_foot_l_z[t]     = float(ftl[2] - root[2])

        # Restore original state
        self.data.qpos[:] = qpos_save
        self.data.qvel[:] = qvel_save
        mujoco.mj_kinematics(self.model, self.data)

    # ── phase tracking ────────────────────────────────────────────────────

    def _advance_phase(self) -> None:
        """Fixed-rate phase clock — advances exactly 1 frame per env step.

        DeepMimic uses a fixed clock tied to real time, not joint matching.
        Adaptive phase lets the agent 'shop' for frames where its stiff legs
        match the reference (extended-knee phases), preventing it from ever
        learning knee flexion during swing. A fixed clock forces it to be at
        the correct phase regardless of its current state.
        """
        self._phase = (self._phase + 1) % self._ref_len

    # ── reset ─────────────────────────────────────────────────────────────

    def reset(self, **kwargs):
        self._phase = np.random.randint(0, self._ref_len)
        _, info = super().reset(**kwargs)

        if self._warm_start:
            qpos = self.data.qpos.copy()
            qvel = self.data.qvel.copy()
            # Set joint angles to reference at sampled phase (clamped to limits)
            qpos[3:9] = np.clip(self._reference[self._phase], _JNT_LO, _JNT_HI)
            # Set joint velocities to reference velocity
            qvel[3:9] = self._ref_vel[self._phase]
            self.set_state(qpos, qvel)

        return self._get_obs(), info

    # ── step ──────────────────────────────────────────────────────────────

    def step(self, action):
        _, _, terminated, truncated, info = super().step(action)

        q_sim  = self.data.qpos[3:9].astype(np.float32)
        dq_sim = self.data.qvel[3:9].astype(np.float32)
        q_ref  = self._reference[self._phase]
        dq_ref = self._ref_vel[self._phase]

        diff   = q_sim - q_ref
        diff_v = dq_sim - dq_ref

        k = self._imit_scale

        # ── joint pose tracking  (DeepMimic: k_p = 2 for 3D; k=8 fine for 6-DOF 2D)
        imit_r = float(np.mean(np.exp(-k * diff ** 2)))

        # ── joint velocity tracking  (DeepMimic: k_v = 0.1)
        vel_r = float(np.mean(np.exp(-0.1 * diff_v ** 2)))

        # ── end-effector (foot position) tracking ─────────────────────
        # Two components per foot:
        #   x_rel: foot x relative to root — forward placement signal
        #   z_world: foot z in world frame — swing clearance signal
        #            (world-z is ~0 at stance, >0 during swing;
        #             root-relative-z is always ~-1.1m with no swing signal)
        root_xpos = self.data.body("torso").xpos
        ftr_xpos  = self.data.body("foot").xpos
        ftl_xpos  = self.data.body("foot_left").xpos

        # Root-relative (x, z) — z goes from -1.29 (stance) to -0.89 (swing peak),
        # a 0.4m range that is the actual swing clearance signal.
        foot_r_xrel = ftr_xpos[0] - root_xpos[0]
        foot_r_zrel = ftr_xpos[2] - root_xpos[2]
        foot_l_xrel = ftl_xpos[0] - root_xpos[0]
        foot_l_zrel = ftl_xpos[2] - root_xpos[2]
        # x placement: k=40 (cm-level accuracy)
        # z clearance: k=40 during stance, k=200 during swing — much sharper
        # penalty when the reference says the foot should be elevated but it's dragging.
        SWING_CLEARANCE = -1.05  # root-relative z threshold: above this = swing phase
        r_is_swing = self._ref_foot_r_z[self._phase] > SWING_CLEARANCE
        l_is_swing = self._ref_foot_l_z[self._phase] > SWING_CLEARANCE
        kz_r = 200.0 if r_is_swing else 40.0
        kz_l = 200.0 if l_is_swing else 40.0

        ee_err_r_x = (foot_r_xrel - self._ref_foot_r_xrel[self._phase]) ** 2
        ee_err_l_x = (foot_l_xrel - self._ref_foot_l_xrel[self._phase]) ** 2
        ee_err_r_z = (foot_r_zrel - self._ref_foot_r_z[self._phase])    ** 2
        ee_err_l_z = (foot_l_zrel - self._ref_foot_l_z[self._phase])    ** 2
        ee_r = float(0.25 * (np.exp(-40.0 * ee_err_r_x) +
                              np.exp(-40.0 * ee_err_l_x) +
                              np.exp(-kz_r  * ee_err_r_z) +
                              np.exp(-kz_l  * ee_err_l_z)))

        # ── root tracking (height + pitch) ────────────────────────────
        # DeepMimic: k_root = 10, pitch coeff = 0.1 * root_rot_err.
        # Our pitch exploit needs coeff = 1.0 to actually penalise lean.
        root_height = float(root_xpos[2])
        root_pitch  = float(self.data.qpos[2])
        ref_height  = float(self._ref_root_height[self._phase])
        root_err = (root_height - ref_height) ** 2 + 3.0 * root_pitch ** 2
        root_r = float(np.exp(-10.0 * root_err))

        # ── contact alternation reward ────────────────────────────────
        foot_r_frc = float(np.linalg.norm(self.data.cfrc_ext[4]))
        foot_l_frc = float(np.linalg.norm(self.data.cfrc_ext[7]))
        if self._stance_right[self._phase]:
            contact_r = np.tanh(foot_r_frc / 50.0) - np.tanh(foot_l_frc / 50.0)
        else:
            contact_r = np.tanh(foot_l_frc / 50.0) - np.tanh(foot_r_frc / 50.0)
        contact_r = float(max(contact_r, 0.0))

        # ── ctrl cost ─────────────────────────────────────────────────
        ctrl_cost = -1e-3 * float(np.sum(np.square(self.data.ctrl)))

        # ── combine (DeepMimic-inspired weighted sum) ──────────────────
        # Scale by dt so returns are time-invariant across episode lengths.
        # Pose/vel scale by N_REF=6 (one term per joint); EE scales by 2
        # (one per foot); root and contact are scalar.
        # Approximate DeepMimic weight ratio: pose(0.65) vel(0.1) ee(0.15) root(0.1)
        reward = (self._imitation_weight * self.dt * self.N_REF * imit_r
                  + self._vel_weight     * self.dt * self.N_REF * vel_r
                  + self._ee_weight      * self.dt * 2          * ee_r
                  + self._root_weight    * self.dt              * root_r
                  + self._contact_weight * self.dt              * contact_r
                  + ctrl_cost)

        # ── termination ───────────────────────────────────────────────
        # super().step() already terminates on root height out of [0.8, 2.0].
        # Pitch termination: kill episode on forward/backward lean > 0.3 rad (~17°).
        # This forces the agent to maintain upright posture — without it the agent
        # learns controlled forward falling which is never penalized until height drops.
        if abs(root_pitch) > 0.3:
            terminated = True
        ankle_dev = max(abs(diff[2]), abs(diff[5]))
        other_dev = max(abs(diff[0]), abs(diff[1]), abs(diff[3]), abs(diff[4]))
        if ankle_dev > self._ankle_term_thresh or other_dev > self._pose_term_thresh:
            terminated = True
        x_vel = float(info.get("x_velocity", self.data.qvel[0]))
        if x_vel < -0.1:
            terminated = True

        # ── advance phase (after reward/termination use current phase) ─
        self._advance_phase()

        info.update(imit_r=imit_r, vel_r=vel_r, ee_r=ee_r, root_r=root_r, phase=self._phase)
        return self._get_obs(), reward, terminated, truncated, info


# ── callback ──────────────────────────────────────────────────────────────────

class LogCallback(BaseCallback):
    def __init__(self, log_interval: int = 50):
        super().__init__(verbose=0)
        self._interval = log_interval
        self._rollout  = 0
        self._ep_r: list[float] = []
        self._ep_l: list[int]   = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep:
                self._ep_r.append(float(ep["r"]))
                self._ep_l.append(int(ep["l"]))
        return True

    def _on_rollout_end(self) -> None:
        self._rollout += 1
        if self._rollout % self._interval == 0 and self._ep_r:
            print(
                f"[iter {self._rollout:5d} | steps {self.num_timesteps:>9,}]  "
                f"ep_r={np.mean(self._ep_r):8.1f}  "
                f"ep_len={np.mean(self._ep_l):6.0f}  "
                f"(n={len(self._ep_r)})"
            )
            self._ep_r.clear()
            self._ep_l.clear()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase-aware Walker2d imitation from Ulrich IK reference"
    )

    # reference
    ref_grp = parser.add_mutually_exclusive_group(required=True)
    ref_grp.add_argument("--ref_cycle", type=str,
                         help="Path to single gait-cycle .npy (recommended)")
    ref_grp.add_argument("--ref_all",   action="store_true",
                         help="Use full concatenated Ulrich reference")

    parser.add_argument("--subjects",     type=int, nargs="+", default=None)
    parser.add_argument("--trial_filter", type=str, default=None)

    # training
    parser.add_argument("--num_envs",    type=int,   default=32)
    parser.add_argument("--total_steps", type=float, default=5e6)
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--finetune",    default=None,
                        help="Pretrained .zip to finetune from")

    # reward weights (DeepMimic-style: pose, vel, ee, root, contact)
    parser.add_argument("--imit_weight",    type=float, default=4.0)
    parser.add_argument("--vel_weight",     type=float, default=1.0)
    parser.add_argument("--ee_weight",      type=float, default=4.0,
                        help="End-effector foot position tracking weight")
    parser.add_argument("--root_weight",    type=float, default=2.0,
                        help="Root height + pitch tracking weight")
    parser.add_argument("--contact_weight", type=float, default=1.0)

    # phase tracking
    parser.add_argument("--max_phase_advance", type=int,   default=4,
                        help="Max reference frames to skip per env step")

    # termination
    parser.add_argument("--pose_term",  type=float, default=0.9,
                        help="Hip/knee deviation threshold (rad)")
    parser.add_argument("--ankle_term", type=float, default=0.40,
                        help="Ankle deviation threshold (rad)")
    parser.add_argument("--no_pose_term", action="store_true",
                        help="Disable pose termination entirely")

    parser.add_argument("--product_reward", action="store_true",
                        help="DeepMimic product-of-exps reward: all components "
                             "must be satisfied simultaneously (geometric mean)")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    # ── load reference ────────────────────────────────────────────────
    if args.ref_cycle:
        reference = load_ref_cycle(Path(args.ref_cycle))
    else:
        print("Loading full Ulrich reference...")
        reference = load_ulrich_reference(
            subjects=args.subjects,
            trial_filter=args.trial_filter,
            control_hz=CTRL_HZ,
        )

    if args.no_pose_term:
        args.pose_term = 9999.0
        # ankle_term left as-is so --ankle_term still takes effect

    print(f"Reference shape: {reference.shape}  "
          f"({len(reference)/CTRL_HZ:.1f}s @ {CTRL_HZ}Hz)")

    # ── output dir ────────────────────────────────────────────────────
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag   = "cycle" if args.ref_cycle else "full"
    rform = "product" if args.product_reward else "sum"
    log_dir = PROJECT_ROOT / (args.out_dir or f"results/walker2d_phase_{tag}_{rform}_{stamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {log_dir}")
    np.save(log_dir / "reference.npy", reference)

    # ── env factory ───────────────────────────────────────────────────
    def make_env():
        def _init():
            return Walker2dPhaseAware(
                reference         = reference,
                imitation_weight  = args.imit_weight,
                vel_weight        = args.vel_weight,
                ee_weight         = args.ee_weight,
                root_weight       = args.root_weight,
                contact_weight    = args.contact_weight,
                max_phase_advance = args.max_phase_advance,
                pose_term_thresh  = args.pose_term,
                ankle_term_thresh = args.ankle_term,
                product_reward    = args.product_reward,
            )
        return _init

    env = SubprocVecEnv([make_env() for _ in range(args.num_envs)])
    env = VecMonitor(env)

    # ── model ─────────────────────────────────────────────────────────
    if args.finetune:
        path = str(Path(args.finetune).with_suffix(""))
        print(f"Finetuning from: {path}")
        model = PPO.load(path, env=env, device=args.device)
        model.learning_rate = 3e-5
        model.ent_coef      = 0.0
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate = 1e-4,
            n_steps       = 512,
            batch_size    = 4096,
            n_epochs      = 10,
            gamma         = 0.99,
            gae_lambda    = 0.95,
            clip_range    = 0.2,
            ent_coef      = 0.001,
            vf_coef       = 0.5,
            max_grad_norm = 0.5,
            target_kl     = 0.02,
            device        = args.device,
            policy_kwargs = {"net_arch": [256, 256]},
            verbose       = 0,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq  = max(5_000_000 // args.num_envs, 1),
        save_path  = str(log_dir / "checkpoints"),
        name_prefix= "model",
        verbose    = 0,
    )

    total_steps = int(args.total_steps)
    print(f"Training for {total_steps:,} steps with {args.num_envs} envs...")
    model.learn(
        total_timesteps = total_steps,
        callback        = CallbackList([LogCallback(), checkpoint_cb]),
        progress_bar    = True,
    )
    env.close()

    save_path = str(log_dir / "model")
    model.save(save_path)
    print(f"Model saved → {save_path}.zip")


if __name__ == "__main__":
    main()
