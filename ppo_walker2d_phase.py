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
    """Load a single gait cycle .npy, resample to CTRL_HZ."""
    raw   = np.load(path).astype(np.float32)
    n_in  = len(raw)
    n_out = int(round(n_in * CTRL_HZ / REF_HZ))
    x_in  = np.linspace(0, 1, n_in)
    x_out = np.linspace(0, 1, n_out)
    ref   = np.stack(
        [np.interp(x_out, x_in, raw[:, j]) for j in range(raw.shape[1])],
        axis=1,
    ).astype(np.float32)
    print(f"Gait cycle: {n_in} frames @ {REF_HZ}Hz -> {n_out} frames @ {CTRL_HZ}Hz")
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
        forward_weight:     float = 1.0,
        v_target:           float = 1.25,  # m/s
        imit_scale:         float = 8.0,   # sharpness of exp(-k·err²)
        max_phase_advance:  int   = 4,     # max frames to skip per step
        contact_weight:     float = 2.0,
        pose_term_thresh:   float = 0.9,   # rad — hip/knee termination
        ankle_term_thresh:  float = 0.40,  # rad — ankle termination (looser)
        warm_start:         bool  = True,
        product_reward:     bool  = False, # True → DeepMimic product-of-exps form
        **kwargs,
    ):
        self._reference         = reference
        self._ref_len           = len(reference)
        self._imitation_weight  = imitation_weight
        self._vel_weight        = vel_weight
        self._forward_weight    = forward_weight
        self._v_target          = v_target
        self._imit_scale        = imit_scale
        self._max_phase_advance = max_phase_advance
        self._contact_weight    = contact_weight
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

    def _get_obs(self) -> np.ndarray:
        base    = super()._get_obs()                       # (17,)
        q_ref   = self._reference[self._phase]             # (6,)
        phi     = 2.0 * np.pi * self._phase / self._ref_len
        phase_enc = np.array([np.sin(phi), np.cos(phi)], dtype=np.float32)
        return np.concatenate([base, q_ref, phase_enc])    # (25,)

    # ── phase tracking ────────────────────────────────────────────────────

    def _advance_phase(self) -> None:
        """
        Search forward up to max_phase_advance frames and lock to the
        candidate whose reference joints are closest to the current sim state.

        Always advances by at least 1 (phase never regresses).
        """
        q_sim = self.data.qpos[3:9].astype(np.float32)
        best_phase = (self._phase + 1) % self._ref_len
        best_err   = np.sum((q_sim - self._reference[best_phase]) ** 2)

        for dt in range(2, self._max_phase_advance + 1):
            candidate = (self._phase + dt) % self._ref_len
            err = np.sum((q_sim - self._reference[candidate]) ** 2)
            if err < best_err:
                best_err   = err
                best_phase = candidate

        self._phase = best_phase

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

        diff    = q_sim - q_ref
        diff_v  = dq_sim - dq_ref

        # ── sub-rewards, each normalised to [0, 1] ────────────────────
        k = self._imit_scale
        # Per-joint averages → [0, 1]; all joints must track to reach 1.
        imit_r   = float(np.mean(np.exp(-k       * diff   ** 2)))
        vel_r    = float(np.mean(np.exp(-k * 0.01 * diff_v ** 2)))

        # ── forward velocity reward ───────────────────────────────────
        x_vel = float(info.get("x_velocity", self.data.qvel[0]))
        fwd_r = float(np.exp(-5.0 * (x_vel - self._v_target) ** 2))

        # ── contact alternation reward ────────────────────────────────
        # Body indices: 4=foot (right), 7=foot_left
        foot_r_frc = float(np.linalg.norm(self.data.cfrc_ext[4]))
        foot_l_frc = float(np.linalg.norm(self.data.cfrc_ext[7]))
        if self._stance_right[self._phase]:
            contact_r = np.tanh(foot_r_frc / 50.0) - np.tanh(foot_l_frc / 50.0)
        else:
            contact_r = np.tanh(foot_l_frc / 50.0) - np.tanh(foot_r_frc / 50.0)
        contact_r = float(max(contact_r, 0.0))

        # ── ctrl cost ─────────────────────────────────────────────────
        ctrl_cost = -1e-3 * float(np.sum(np.square(self.data.ctrl)))

        if self._product_reward:
            # DeepMimic product form: all components must be satisfied simultaneously.
            # Each sub-reward is in [0,1]; product → geometric mean weighted by w_i.
            # r = dt * (imit^w_i * vel^w_v * fwd^w_f * contact^w_c)^(1/sum_w) + ctrl
            w_i = self._imitation_weight
            w_v = self._vel_weight
            w_f = self._forward_weight
            w_c = self._contact_weight
            total_w = w_i + w_v + w_f + w_c
            combined = (
                imit_r    ** w_i *
                vel_r     ** w_v *
                fwd_r     ** w_f *
                (contact_r + 1e-8) ** w_c   # avoid 0^w_c blowing up
            ) ** (1.0 / total_w)
            reward = self.dt * combined * total_w + ctrl_cost
        else:
            # Weighted sum form (original, but sub-rewards now averaged not summed).
            # Multiply by N_REF (6) to keep scale comparable to the old sum.
            reward = (self._imitation_weight * self.dt * self.N_REF * imit_r
                      + self._vel_weight     * self.dt * self.N_REF * vel_r
                      + self._forward_weight * self.dt              * fwd_r
                      + self._contact_weight * self.dt              * contact_r
                      + ctrl_cost)

        # ── termination ───────────────────────────────────────────────
        ankle_dev = max(abs(diff[2]), abs(diff[5]))
        other_dev = max(abs(diff[0]), abs(diff[1]), abs(diff[3]), abs(diff[4]))
        if ankle_dev > self._ankle_term_thresh or other_dev > self._pose_term_thresh:
            terminated = True
        if x_vel < -0.1:
            terminated = True

        # ── advance phase (after reward/termination use current phase) ─
        self._advance_phase()

        info.update(imit_r=imit_r, vel_r=vel_r, fwd_r=fwd_r, phase=self._phase)
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

    # reward weights
    parser.add_argument("--imit_weight",    type=float, default=4.0)
    parser.add_argument("--vel_weight",     type=float, default=1.0)
    parser.add_argument("--forward_weight", type=float, default=1.0)
    parser.add_argument("--contact_weight", type=float, default=2.0)

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
                forward_weight    = args.forward_weight,
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
        save_freq  = max(500_000 // args.num_envs, 1),
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
