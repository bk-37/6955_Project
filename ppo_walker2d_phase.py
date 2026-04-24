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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from gymnasium import spaces
from gymnasium.envs.mujoco.walker2d_v4 import Walker2dEnv, DEFAULT_CAMERA_CONFIG
from gymnasium.envs.mujoco import MujocoEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, CallbackList
)

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
from ppo_walker2d import load_ulrich_reference   # reuse loader

CTRL_HZ          = 125.0   # Walker2d-v4: frame_skip=4, dt=0.002s
REF_HZ           = 50.0    # Ulrich IK / extracted gait cycles
GAIT_CYCLE_FRAMES = 140    # ~1.1s @ 125Hz — used for sin/cos phase encoding


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

# Per-joint reward sharpness and weights — module-level to avoid per-step allocation.
# Order: [hip_r, knee_r, ankle_r, hip_l, knee_l, ankle_l]
#   k=10 → 50% reward at 0.26 rad (15°)  [hip   — slow postural joint]
#   k=20 → 50% reward at 0.19 rad (11°)  [knee  — moderate]
#   k=40 → 50% reward at 0.13 rad  (7°)  [ankle — tight, heel-strike timing]
_JSCALE   = np.array([10.0, 20.0, 40.0, 10.0, 20.0, 40.0], dtype=np.float32)
_JWEIGHTS = np.array([ 0.4,  1.0,  2.5,  0.4,  1.0,  2.5], dtype=np.float32)
_KVSCALE  = np.array([0.05,  0.1,  0.2, 0.05,  0.1,  0.2], dtype=np.float32)


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
        xml_file:           str   = "walker2d.xml",  # MuJoCo model to use
        imitation_weight:   float = 4.0,
        vel_weight:         float = 1.0,
        ee_weight:          float = 4.0,  # end-effector foot position tracking
        root_weight:        float = 2.0,  # root height + pitch tracking
        contact_weight:     float = 1.0,
        fwd_weight:         float = 0.0,  # forward velocity reward
        v_target:           float = 1.25, # target forward speed (m/s) — Ulrich treadmill
        swing_pen_weight:   float = 2.0,  # penalty for swing foot ground contact
        action_rate_weight: float = 0.0,  # penalty for large action changes (anti-jerk)
        peak_bonus_weight:  float = 0.0,  # bonus for matching ref at high-excursion phases
        imit_scale:         float = 20.0,  # sharpness of exp(-k·err²)
        max_phase_advance:  int   = 4,    # max frames to skip per step
        pose_term_thresh:   float = 0.9,  # rad — hip/knee termination
        ankle_term_thresh:  float = 0.40, # rad — ankle termination (looser)
        warm_start:         bool  = True,
        product_reward:     bool  = False,
        **kwargs,
    ):
        self._reference         = reference
        self._xml_file          = xml_file
        self._ref_len           = len(reference)
        self._imitation_weight  = imitation_weight
        self._vel_weight        = vel_weight
        self._ee_weight         = ee_weight
        self._root_weight       = root_weight
        self._contact_weight    = contact_weight
        self._fwd_weight        = fwd_weight
        self._v_target          = v_target
        self._swing_pen_weight  = swing_pen_weight
        self._action_rate_weight = action_rate_weight
        self._peak_bonus_weight  = peak_bonus_weight
        self._imit_scale        = imit_scale
        self._max_phase_advance = max_phase_advance
        self._pose_term_thresh  = pose_term_thresh
        self._ankle_term_thresh = ankle_term_thresh
        self._warm_start        = warm_start
        self._product_reward    = product_reward
        self._phase             = 0

        # Pre-compute per-frame velocity from reference (for velocity tracking)
        # Shape (T, 6) — central differences with periodic wrap for looping cycle.
        # np.gradient uses one-sided diffs at edges; for a looping reference we
        # want frame 0's velocity computed from frames [-1, 0, 1] (wrap-around).
        ref_pad = np.concatenate([reference[-1:], reference, reference[:1]], axis=0)
        self._ref_vel = (np.gradient(ref_pad, 1.0 / CTRL_HZ, axis=0)[1:-1]).astype(np.float32)

        # Pre-compute per-joint excursion normalizers for peak bonus.
        # excursion[t, j] ∈ [0,1]: 0 = joint at its neutral midpoint, 1 = at max/min extreme.
        # This gates the peak bonus on phases where the reference is near its range limits.
        _q_max    = reference.max(axis=0)   # (6,)
        _q_min    = reference.min(axis=0)   # (6,)
        _neutral  = (_q_max + _q_min) / 2.0
        _half_rng = (_q_max - _q_min) / 2.0 + 1e-6
        self._ref_excursion = (np.abs(reference - _neutral) / _half_rng).astype(np.float32)

        # Pre-compute stance side per frame from reference hip angles.
        # In Walker2d convention, the stance hip is more extended (less negative / more positive).
        # ref[:, 0] = hip_r,  ref[:, 3] = hip_l
        # stance_right[t] = True when right hip is more extended than left at frame t.
        self._stance_right = reference[:, 0] >= reference[:, 3]  # (T,) bool

        # Walker2dEnv.__init__ hardcodes "walker2d.xml" — to support custom XML files
        # we replicate its attribute setup and call MujocoEnv.__init__ directly.
        self._forward_reward_weight = 1.0
        self._ctrl_cost_weight      = 1e-3
        self._healthy_reward        = 1.0
        self._terminate_when_unhealthy = True
        self._healthy_z_range       = (0.8, 2.0)
        self._healthy_angle_range   = (-1.0, 1.0)
        self._reset_noise_scale     = 5e-3
        self._exclude_current_positions_from_observation = True

        # Resolve xml_file: relative names look in gymnasium assets dir,
        # absolute paths (or explicit PROJECT_ROOT paths) are used as-is.
        if xml_file == "walker2d.xml":
            resolved_xml = xml_file  # MujocoEnv finds this in its assets dir
        else:
            resolved_xml = str(PROJECT_ROOT / xml_file)

        _obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            resolved_xml,
            4,                               # frame_skip
            observation_space=_obs_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        # Previous action for action-rate penalty (anti-jerk)
        self._prev_action = np.zeros(6, dtype=np.float32)

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
        base    = super()._get_obs().astype(np.float32)    # (17,)
        q_ref   = self._reference[self._phase]             # (6,)
        # Normalize phase to gait cycle period, not reference length.
        # With long references (7570 frames) the full sin/cos cycle would
        # take 60s — useless as a within-stride signal. Normalizing to
        # GAIT_CYCLE_FRAMES keeps the encoding meaningful regardless of
        # whether we use a single extracted cycle or a long continuous ref.
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
        self._prev_action = np.zeros(6, dtype=np.float32)
        _, info = super().reset(**kwargs)

        if self._warm_start:
            qpos = self.data.qpos.copy()
            qvel = self.data.qvel.copy()
            # Set joint angles to reference at sampled phase (clamped to limits)
            qpos[3:9] = np.clip(self._reference[self._phase], _JNT_LO, _JNT_HI)
            # Set joint velocities to reference velocity
            qvel[3:9] = self._ref_vel[self._phase]
            # Forward velocity must match treadmill speed — otherwise both feet land
            # simultaneously (standing contact pattern) even though joints are mid-gait.
            # This is the primary cause of poor tracking: RSI without x_vel is internally
            # inconsistent with 1.25 m/s reference kinematics.
            qvel[0] = self._v_target
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

        # ── joint pose tracking ───────────────────────────────────────
        if self._product_reward:
            imit_r = float(np.exp(-np.sum(_JWEIGHTS * _JSCALE * diff ** 2) / np.sum(_JWEIGHTS)))
        else:
            imit_r = float(np.mean(_JWEIGHTS * np.exp(-_JSCALE * diff ** 2)))

        # ── peak excursion bonus ──────────────────────────────────────
        # Rewards the policy for matching the reference specifically when the
        # reference is near its range extremes (e.g. peak knee flex at mid-swing,
        # peak ankle push-off). excursion[phase] ∈ [0,1] gates the bonus.
        excursion = self._ref_excursion[self._phase]   # (6,) in [0,1]
        peak_bonus = float(np.mean(excursion * np.exp(-_JSCALE * diff ** 2)))

        # ── joint velocity tracking — tighter for ankles (push-off timing)
        vel_r = float(np.mean(np.exp(-_KVSCALE * diff_v ** 2)))

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
        SWING_CLEARANCE = -1.15  # root-relative z threshold: above this = swing phase
        # Lowered from -1.05: foot z rises from -1.29 (stance) to -0.89 (peak swing).
        # At -1.15 the high-k EE penalty + swing contact penalty trigger earlier in toe-off.
        r_is_swing = self._ref_foot_r_z[self._phase] > SWING_CLEARANCE
        l_is_swing = self._ref_foot_l_z[self._phase] > SWING_CLEARANCE
        kz_r = 40.0
        kz_l = 40.0

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
        root_err = (root_height - ref_height) ** 2 + 1.0 * root_pitch ** 2
        root_r = float(np.exp(-10.0 * root_err))

        # ── contact alternation reward ────────────────────────────────
        foot_r_frc = float(np.linalg.norm(self.data.cfrc_ext[4]))
        foot_l_frc = float(np.linalg.norm(self.data.cfrc_ext[7]))
        if self._stance_right[self._phase]:
            contact_r = np.tanh(foot_r_frc / 50.0) - np.tanh(foot_l_frc / 50.0)
        else:
            contact_r = np.tanh(foot_l_frc / 50.0) - np.tanh(foot_r_frc / 50.0)
        contact_r = float(max(contact_r, 0.0))

        # ── swing foot contact penalty ────────────────────────────────
        # Directly penalize ground contact on the swing foot — toe drag generates
        # small but non-zero contact forces that the alternation reward misses.
        swing_pen = 0.0
        if r_is_swing:
            swing_pen += float(np.tanh(foot_r_frc / 50.0))
        if l_is_swing:
            swing_pen += float(np.tanh(foot_l_frc / 50.0))

        # ── action rate penalty (anti-jerk) ──────────────────────────
        action_arr = np.asarray(action, dtype=np.float32)
        action_rate_pen = float(np.sum(np.square(action_arr - self._prev_action)))
        self._prev_action = action_arr.copy()

        # ── forward velocity reward ───────────────────────────────────
        x_vel = float(info.get("x_velocity", self.data.qvel[0]))
        fwd_r = float(np.exp(-3.0 * (x_vel - self._v_target) ** 2))

        # ── ctrl cost ─────────────────────────────────────────────────
        ctrl_cost = -1e-3 * float(np.sum(np.square(self.data.ctrl)))

        # ── combine (DeepMimic-inspired weighted sum) ──────────────────
        # Scale by dt so returns are time-invariant across episode lengths.
        # Pose/vel scale by N_REF=6 (one term per joint); EE scales by 2
        # (one per foot); root and contact are scalar.
        # Approximate DeepMimic weight ratio: pose(0.65) vel(0.1) ee(0.15) root(0.1)
        reward = (self._imitation_weight    * self.dt * self.N_REF * imit_r
                  + self._vel_weight        * self.dt * self.N_REF * vel_r
                  + self._peak_bonus_weight * self.dt * self.N_REF * peak_bonus
                  + self._fwd_weight        * self.dt              * fwd_r
                  + self._ee_weight         * self.dt * 2          * ee_r
                  + self._root_weight       * self.dt              * root_r
                  + self._contact_weight    * self.dt              * contact_r
                  - self._swing_pen_weight  * self.dt              * swing_pen
                  - self._action_rate_weight * self.dt             * action_rate_pen
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
        if x_vel < -0.1:
            terminated = True

        # ── advance phase (after reward/termination use current phase) ─
        self._advance_phase()

        info.update(imit_r=imit_r, vel_r=vel_r, ee_r=ee_r, root_r=root_r, phase=self._phase)
        return self._get_obs(), reward, terminated, truncated, info


# ── behavioral cloning warm-start ─────────────────────────────────────────────

def compute_bc_dataset(
    env:        "Walker2dPhaseAware",
    n_steps:    int   = 200_000,
    kp:         float = 200.0,
    kd:         float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect (obs, action) pairs by rolling out a PD tracking controller
    in the actual simulation.

    Why PD rollout instead of inverse dynamics
    ──────────────────────────────────────────
    mj_inverse computes torques assuming free-space dynamics — it has no
    knowledge of contact forces.  During stance phase, ground reaction forces
    do a large fraction of the work; without them the computed torques are
    systematically wrong for ~half the gait cycle, which is why the
    inverse-dynamics BC policy produces nothing like walking.

    The PD controller runs inside the full MuJoCo simulation where contact
    is computed every sub-step.  The torques it outputs are physically
    consistent with ground contact, gravity, and joint dynamics.  These are
    exactly the torques we want the policy to imitate.

    PD law (in torque space, then normalised by gear):
        τ = Kp·(q_ref − q) + Kd·(dq_ref − dq)
        action = clip(τ / gear, −1, 1)

    Kp=200, Kd=20 with gear=100 gives:
        action = 2·q_err + 0.2·dq_err
    A 0.1 rad error → 0.2 (20 Nm), which is firm tracking without saturation.
    """
    gear = float(env.model.actuator_gear[0, 0])   # all joints same gear in Walker2d

    obs_list:    list[np.ndarray] = []
    act_list:    list[np.ndarray] = []
    ep_lengths:  list[int]        = []
    ep_len = 0

    obs, _ = env.reset()
    for _ in range(n_steps):
        q_ref  = env._reference[env._phase]
        dq_ref = env._ref_vel[env._phase]
        q_sim  = env.data.qpos[3:9].astype(np.float32)
        dq_sim = env.data.qvel[3:9].astype(np.float32)

        torque = kp * (q_ref - q_sim) + kd * (dq_ref - dq_sim)
        action = np.clip(torque / gear, -1.0, 1.0).astype(np.float32)

        obs_list.append(obs.copy())
        act_list.append(action)
        ep_len += 1

        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            ep_lengths.append(ep_len)
            ep_len = 0
            obs, _ = env.reset()

    if ep_len > 0:
        ep_lengths.append(ep_len)

    ep_arr = np.array(ep_lengths)
    print(f"  PD episodes: {len(ep_arr)}  "
          f"length  mean={ep_arr.mean():.0f}  median={np.median(ep_arr):.0f}  "
          f"max={ep_arr.max()}  "
          f">140 frames: {(ep_arr > 140).mean()*100:.0f}%")

    return np.array(obs_list, dtype=np.float32), np.array(act_list, dtype=np.float32)


def pretrain_bc(
    model:      PPO,
    obs_arr:    np.ndarray,
    act_arr:    np.ndarray,
    n_epochs:   int   = 10,
    batch_size: int   = 512,
    lr:         float = 1e-3,
) -> None:
    """
    Supervised warm-start: minimise MSE( π_mean(obs), action ).
    Two-phase lr schedule: first half at lr, second half at lr/10.
    """
    device  = model.device
    obs_t   = torch.tensor(obs_arr, device=device)
    act_t   = torch.tensor(act_arr, device=device)
    dataset = data_utils.TensorDataset(obs_t, act_t)
    loader  = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt     = optim.Adam(model.policy.parameters(), lr=lr)
    mse     = nn.MSELoss()

    print(f"BC pre-training: {n_epochs} epochs × {len(dataset):,} frames")
    for epoch in range(n_epochs):
        if epoch == n_epochs // 2:
            for g in opt.param_groups:
                g["lr"] = lr / 10
            print(f"  [lr → {lr/10:.0e}]")
        total = 0.0
        for obs_b, act_b in loader:
            dist     = model.policy.get_distribution(obs_b)
            pred_act = dist.distribution.mean
            loss     = mse(pred_act, act_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"  epoch {epoch+1:3d}/{n_epochs}  mse={total/len(loader):.5f}")


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
    parser.add_argument("--bc_epochs", type=int, default=0,
                        help="BC warm-start epochs before PPO. Collects a PD-rollout "
                             "dataset (contact-aware) then trains policy via supervised MSE.")
    parser.add_argument("--bc_steps", type=int, default=200_000,
                        help="PD rollout steps to collect for BC dataset (default 200k)")
    parser.add_argument("--bc_kp",    type=float, default=200.0,
                        help="PD proportional gain for BC data collection (default 200)")
    parser.add_argument("--bc_kd",    type=float, default=20.0,
                        help="PD derivative gain for BC data collection (default 20)")
    parser.add_argument("--bc_only", action="store_true",
                        help="Stop after BC warm-start and save — skip PPO entirely.")

    # reward weights (DeepMimic-style: pose, vel, ee, root, contact)
    parser.add_argument("--imit_weight",    type=float, default=4.0)
    parser.add_argument("--vel_weight",     type=float, default=1.0)
    parser.add_argument("--ee_weight",      type=float, default=4.0,
                        help="End-effector foot position tracking weight")
    parser.add_argument("--root_weight",    type=float, default=2.0,
                        help="Root height + pitch tracking weight")
    parser.add_argument("--contact_weight", type=float, default=1.0)
    parser.add_argument("--fwd_weight", type=float, default=0.0,
                        help="Forward velocity reward weight")
    parser.add_argument("--v_target", type=float, default=1.25,
                        help="Target forward speed in m/s (used with --fwd_weight)")
    parser.add_argument("--swing_pen_weight", type=float, default=2.0,
                        help="Penalty weight for swing foot ground contact")
    parser.add_argument("--action_rate_weight", type=float, default=0.0,
                        help="Penalty weight for action rate of change (anti-jerk)")
    parser.add_argument("--peak_bonus_weight", type=float, default=0.0,
                        help="Bonus weight for matching ref at high-excursion phases (knee flex etc)")

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

    parser.add_argument("--scale_model", action="store_true",
                        help="Use Subject 1-scaled Walker2d geometry (walker2d_subject1.xml)")
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

    xml_file = "walker2d_subject1.xml" if args.scale_model else "walker2d.xml"
    if args.scale_model:
        xml_path = str(PROJECT_ROOT / "walker2d_subject1.xml")
        print(f"Using scaled model: {xml_path}")
    else:
        xml_path = "walker2d.xml"

    if args.no_pose_term:
        args.pose_term = 9999.0
        # ankle_term left as-is so --ankle_term still takes effect

    print(f"Reference shape: {reference.shape}  "
          f"({len(reference)/CTRL_HZ:.1f}s @ {CTRL_HZ}Hz)")

    # ── output dir ────────────────────────────────────────────────────
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag   = "cycle" if args.ref_cycle else "full"
    tag   = tag + "_s1scaled" if args.scale_model else tag
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
                xml_file          = xml_path,
                imitation_weight  = args.imit_weight,
                vel_weight        = args.vel_weight,
                ee_weight         = args.ee_weight,
                root_weight       = args.root_weight,
                contact_weight    = args.contact_weight,
                fwd_weight        = args.fwd_weight,
                v_target          = args.v_target,
                swing_pen_weight  = args.swing_pen_weight,
                action_rate_weight  = args.action_rate_weight,
                peak_bonus_weight   = args.peak_bonus_weight,
                max_phase_advance   = args.max_phase_advance,
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
        model.learning_rate = 1e-5
        model.ent_coef      = 0.0
        model.target_kl     = 0.005  # very conservative — small policy updates
    else:
        # Linear LR decay: 3e-4 → 3e-5 over training.
        # Prevents large destabilising updates once the policy finds a good gait.
        def lr_schedule(progress_remaining: float) -> float:
            # progress_remaining goes from 1.0 (start) → 0.0 (end)
            return 3e-5 + progress_remaining * (3e-4 - 3e-5)

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate = lr_schedule,
            n_steps       = 512,
            batch_size    = 4096,
            n_epochs      = 10,
            gamma         = 0.99,
            gae_lambda    = 0.95,
            clip_range    = 0.2,
            ent_coef      = 0.005,
            vf_coef       = 0.5,
            max_grad_norm = 0.5,
            target_kl     = 0.015,
            device        = args.device,
            policy_kwargs = {"net_arch": [256, 256]},
            verbose       = 0,
        )

    # ── BC warm-start ─────────────────────────────────────────────────
    if args.bc_epochs > 0 and not args.finetune:
        tmp_env = Walker2dPhaseAware(
            reference = reference,
            xml_file  = xml_path,
        )
        print(f"Collecting PD rollout dataset ({args.bc_steps:,} steps)...")
        obs_bc, act_bc = compute_bc_dataset(tmp_env, n_steps=args.bc_steps,
                                             kp=args.bc_kp, kd=args.bc_kd)
        tmp_env.close()
        print(f"  collected {len(obs_bc):,} (obs, action) pairs  "
              f"action range [{act_bc.min():.2f}, {act_bc.max():.2f}]")
        pretrain_bc(model, obs_bc, act_bc, n_epochs=args.bc_epochs)

        if args.bc_only:
            # Save as both model_bc.zip (labelled) and model.zip (for render_phase.py :final)
            model.save(str(log_dir / "model_bc"))
            model.save(str(log_dir / "model"))
            print(f"BC-only model saved → {log_dir}/model_bc.zip")
            print(f"  (also → {log_dir}/model.zip for render_phase.py :final)")
            env.close()
            return

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
