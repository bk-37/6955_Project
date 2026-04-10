"""
PPO imitation learning on Walker2d-v4 using Ulrich et al. IK data.

Trains a torque-actuated Walker2d to imitate the sagittal-plane joint
kinematics from 10 subjects × multiple walking trials (~413k frames total).

Phase-tracked DeepMimic-style reward over hip/knee/ankle.

Usage:
    python ppo_walker2d.py                          # train on all subjects
    python ppo_walker2d.py --subjects 1 2 3         # subset of subjects
    python ppo_walker2d.py --trial_filter baseline  # only baseline trials

After training, run extract_grf.py to pull foot contact forces for
downstream OpenSim static optimization comparison.
"""
import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco.walker2d_v4 import Walker2dEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
ULRICH_ROOT  = PROJECT_ROOT / "Ulrich_Treadmill_Data"

# ── IK loading ────────────────────────────────────────────────────────────────
def load_sto(path: Path) -> dict:
    """Parse an OpenSim .sto / .mot file into a column dict."""
    with open(path) as f:
        lines = f.readlines()
    for i, l in enumerate(lines):
        if l.strip() == "endheader":
            header_end = i
            break
    cols = lines[header_end + 1].split()
    data = np.array(
        [[float(x) for x in l.split()] for l in lines[header_end + 2:] if l.strip()]
    )
    return {c: data[:, i] for i, c in enumerate(cols)}


def load_ulrich_reference(subjects: list[int] | None = None,
                           trial_filter: str | None = None,
                           control_hz: float = 50.0) -> np.ndarray:
    """
    Load all Ulrich IK walking trials and concatenate into a single reference
    array of shape (T, 6):
        [hip_r, knee_r, ankle_r, hip_l, knee_l, ankle_l]
    Values are in RADIANS, Walker2d sign convention:
        hip:   walker = -opensim  (flex positive in OpenSim, negative in Walker2d)
        knee:  walker = -opensim  (same)
        ankle: walker = +opensim
    """
    if subjects is None:
        subjects = list(range(1, 11))  # Subject1..Subject10

    segments = []
    total_files = 0

    for subj_id in subjects:
        subj_dir = ULRICH_ROOT / f"Subject{subj_id}" / "IK"
        if not subj_dir.exists():
            print(f"  [warn] missing: {subj_dir}")
            continue

        for trial_dir in sorted(subj_dir.glob("walking_*")):
            if trial_filter and trial_filter not in trial_dir.name:
                continue
            ik_path = trial_dir / "output" / "results_ik.sto"
            if not ik_path.exists():
                continue

            d = load_sto(ik_path)
            # Ulrich IK is already in degrees at 50Hz
            orig_hz = 1.0 / (d["time"][1] - d["time"][0])
            orig_len = len(d["time"])
            new_len = int(orig_len * control_hz / orig_hz)
            orig_x = np.arange(orig_len)
            new_x = np.linspace(0, orig_len - 1, new_len)

            from scipy.interpolate import CubicSpline
            def resamp(key):
                return CubicSpline(orig_x, d[key])(new_x)

            # Walker2d joint axes are all [0, -1, 0] (negative Y).
            # A positive rotation in OpenSim around +Y maps to negative in Walker2d.
            # Hip: OpenSim hip_flexion_r range [-15, +30] deg
            #   Walker2d thigh_joint range [-150, 0] — flexion=negative, extension near 0
            #   Mapping: walker = -opensim  BUT extension (+0.35) slightly exceeds 0 limit
            #   Walker2d's default standing is thigh=-0.1 (slight flexion), so use -opensim
            #   and rely on joint limits to clamp the small extension excursion (~0.27 rad max)
            # Knee: OpenSim knee_angle_r always positive [0, 66] deg (always flexed)
            #   Walker2d leg_joint [-150, 0] — flexion=negative
            #   Mapping: walker = -opensim  ✓  (gives [-1.15, 0])
            # Ankle: OpenSim ankle_angle_r [-28, +12] deg (plantarflex=negative)
            #   Walker2d foot_joint [-45, 45] — same convention
            #   Mapping: walker = -opensim  (plantarflex positive in Walker2d)
            seg = np.stack([
                -np.deg2rad(resamp("hip_flexion_r")),
                -np.deg2rad(resamp("knee_angle_r")),
                -np.deg2rad(resamp("ankle_angle_r")),
                -np.deg2rad(resamp("hip_flexion_l")),
                -np.deg2rad(resamp("knee_angle_l")),
                -np.deg2rad(resamp("ankle_angle_l")),
            ], axis=1)
            segments.append(seg)
            total_files += 1

    ref = np.concatenate(segments, axis=0).astype(np.float32)
    duration = len(ref) / control_hz
    print(f"  Loaded {total_files} trials → {len(ref):,} frames @ {control_hz}Hz "
          f"({duration:.0f}s = {duration/60:.1f} min)")
    return ref


# ── imitation env ─────────────────────────────────────────────────────────────
class Walker2dImitation(Walker2dEnv):
    """
    Walker2d-v4 with phase-tracked imitation reward (DeepMimic style).

    r = w_fwd  * forward_reward
      + w_imit * sum_j( dt * exp(-8 * (q_j - q_ref_j)^2) )
      - ctrl_cost  (from base env)

    Phase index advances each step and loops over the full reference.
    On reset, phase is randomised so the agent sees all parts of gait.
    The sim is warm-started to the reference pose at the sampled phase.
    """
    def __init__(self, reference: np.ndarray,
                 imitation_weight: float = 3.0,
                 forward_weight: float = 1.0,
                 contact_weight: float = 1.0,
                 warm_start: bool = True,
                 pose_term_threshold: float = 0.8,
                 **kwargs):
        self._reference = reference
        self._ref_len = len(reference)
        self._imitation_weight = imitation_weight
        self._forward_weight = forward_weight
        self._contact_weight = contact_weight
        self._warm_start = warm_start
        self._pose_term_threshold = pose_term_threshold
        self._phase = 0
        super().__init__(**kwargs)

    def reset(self, **kwargs):
        self._phase = np.random.randint(0, self._ref_len)
        obs, info = super().reset(**kwargs)
        if self._warm_start:
            # Hip now allows +20° extension to match Ulrich reference data
            JNT_LO = np.array([-2.618, -2.618, -0.785, -2.618, -2.618, -0.785])
            JNT_HI = np.array([ 0.349,  0.,     0.785,  0.349,  0.,     0.785])
            qpos = self.data.qpos.copy()
            qpos[3:9] = np.clip(self._reference[self._phase], JNT_LO, JNT_HI)
            self.set_state(qpos, self.data.qvel.copy())
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        q_sim = self.data.qpos[3:9]
        q_ref = self._reference[self._phase]
        diff = q_sim - q_ref
        imitation_r = float(self.dt * np.sum(np.exp(-8.0 * diff ** 2)))

        # Walker2d-v4 exposes x_velocity in info, not reward_forward
        x_vel = info.get("x_velocity", self.data.qvel[0])

        # Exponential velocity reward
        fwd_r = float(self.dt * np.exp(-5.0 * (x_vel - 1.25) ** 2))

        # Hard terminate if walking backwards
        if x_vel < -0.1:
            terminated = True

        # DeepMimic-style early termination on pose deviation.
        # Use per-joint thresholds: ankles are tighter because hopping exploits
        # large ankle plantarflexion (deviation ~0.36 rad) while hips/knees stay near ref.
        ankle_dev = max(abs(diff[2]), abs(diff[5]))   # ankle_r, ankle_l
        other_dev = max(abs(diff[0]), abs(diff[1]), abs(diff[3]), abs(diff[4]))
        if ankle_dev > 0.25 or other_dev > self._pose_term_threshold:
            terminated = True

        # Foot contact alternation reward.
        # Walker2d contact forces: cfrc_ext shape (nbody, 6).
        # Body indices: 0=world,1=torso,2=thigh,3=leg,4=foot, 5=thigh_left,6=leg_left,7=foot_left
        # Use the z-force (index 2 in local frame) on each foot body.
        foot_r_frc = float(np.linalg.norm(self.data.cfrc_ext[4]))
        foot_l_frc = float(np.linalg.norm(self.data.cfrc_ext[7]))
        # Reference phase: first half of cycle = right stance, second half = left stance.
        # Reward contact that matches expected stance side.
        phase_frac = self._phase / self._ref_len
        if phase_frac < 0.5:
            # Right stance expected
            contact_r = np.tanh(foot_r_frc / 50.0) - np.tanh(foot_l_frc / 50.0)
        else:
            # Left stance expected
            contact_r = np.tanh(foot_l_frc / 50.0) - np.tanh(foot_r_frc / 50.0)
        contact_r = float(self.dt * max(contact_r, 0.0))

        # Height reward: penalise torso going too high (hopping) or too low (falling).
        # Walker2d standing height is ~1.25m (qpos[1]). Hopping launches it to ~1.5+.
        torso_z = float(self.data.qpos[1])
        height_r = float(self.dt * np.exp(-20.0 * (torso_z - 1.25) ** 2))

        # ctrl cost
        ctrl_cost = -1e-3 * float(np.sum(np.square(self.data.ctrl)))

        reward = (self._forward_weight   * fwd_r
                  + self._imitation_weight * imitation_r
                  + self._contact_weight  * contact_r
                  + 2.0                  * height_r
                  + ctrl_cost)

        self._phase = (self._phase + 1) % self._ref_len

        info["imitation_r"] = imitation_r
        info["contact_r"]   = contact_r
        info["phase"] = self._phase
        return obs, reward, terminated, truncated, info


# ── callback ──────────────────────────────────────────────────────────────────
class LogCallback(BaseCallback):
    def __init__(self, log_interval: int = 50):
        super().__init__(verbose=0)
        self._interval = log_interval
        self._rollout = 0
        self._ep_r: list[float] = []
        self._ep_l: list[int] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep:
                self._ep_r.append(float(ep["r"]))
                self._ep_l.append(int(ep["l"]))
        return True

    def _on_rollout_end(self) -> None:
        self._rollout += 1
        if self._rollout % self._interval == 0:
            if self._ep_r:
                print(
                    f"[iter {self._rollout:5d} | steps {self.num_timesteps:>9,}]  "
                    f"ep_r={np.mean(self._ep_r):8.1f}  "
                    f"ep_len={np.mean(self._ep_l):6.0f}  "
                    f"(n={len(self._ep_r)})"
                )
                self._ep_r.clear()
                self._ep_l.clear()
            else:
                print(
                    f"[iter {self._rollout:5d} | steps {self.num_timesteps:>9,}]  "
                    f"(no completed episodes)"
                )


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=None,
                        help="Subject IDs to include (default: all 1-10)")
    parser.add_argument("--trial_filter", default=None,
                        help="Only include trials whose name contains this string (e.g. 'baseline')")
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--total_steps", type=float, default=5e6)
    parser.add_argument("--device", default="cpu",
                        help="cpu recommended for MLP PPO (env steps are bottleneck)")
    parser.add_argument("--imitation_weight", type=float, default=5.0)
    parser.add_argument("--forward_weight", type=float, default=1.0)
    parser.add_argument("--contact_weight", type=float, default=2.0)
    parser.add_argument("--pose_term_threshold", type=float, default=0.8,
                        help="Max joint deviation (rad) before episode terminates (DeepMimic style)")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--ref_cycle", default=None,
                        help="Path to a pre-extracted single gait cycle .npy (overrides Ulrich loading)")
    parser.add_argument("--finetune", default=None,
                        help="Path to pretrained Walker2d model .zip to finetune from")
    parser.add_argument("--no_pose_term", action="store_true",
                        help="Disable pose termination (recommended when finetuning)")
    args = parser.parse_args()

    # Walker2d-v4: frame_skip=4, MuJoCo dt=0.002s → control_hz=125
    CTRL_HZ = 125.0
    REF_HZ  = 50.0   # Ulrich IK data and extracted gait cycles are at 50 Hz

    if args.ref_cycle:
        raw = np.load(args.ref_cycle).astype(np.float32)
        # Resample from REF_HZ to CTRL_HZ so phase advances 1 frame per env step
        n_in = len(raw)
        n_out = int(round(n_in * CTRL_HZ / REF_HZ))
        x_in  = np.linspace(0, 1, n_in)
        x_out = np.linspace(0, 1, n_out)
        reference = np.stack([np.interp(x_out, x_in, raw[:, j]) for j in range(raw.shape[1])], axis=1).astype(np.float32)
        print(f"Loaded gait cycle: {n_in} frames @ {REF_HZ}Hz → resampled to {n_out} frames @ {CTRL_HZ}Hz")
    else:
        print("Loading Ulrich reference data...")
        reference = load_ulrich_reference(
            subjects=args.subjects,
            trial_filter=args.trial_filter,
            control_hz=CTRL_HZ,
        )

    # output dir
    if args.out_dir:
        log_dir = PROJECT_ROOT / args.out_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        label = f"subj{'_'.join(map(str,args.subjects))}" if args.subjects else "all"
        log_dir = PROJECT_ROOT / "results" / f"walker2d_ulrich_{label}_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {log_dir}")
    np.save(log_dir / "reference.npy", reference)

    def make_env():
        def _init():
            return Walker2dImitation(
                reference=reference,
                imitation_weight=args.imitation_weight,
                forward_weight=args.forward_weight,
                contact_weight=args.contact_weight,
                pose_term_threshold=args.pose_term_threshold,
            )
        return _init

    if args.no_pose_term:
        args.pose_term_threshold = 9999.0

    env = SubprocVecEnv([make_env() for _ in range(args.num_envs)])
    env = VecMonitor(env)

    if args.finetune:
        finetune_path = str(Path(args.finetune).with_suffix(""))
        print(f"Finetuning from pretrained model: {finetune_path}")
        model = PPO.load(finetune_path, env=env, device=args.device)
        model.learning_rate = 3e-5   # lower lr for finetuning
        model.ent_coef = 0.0
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            n_steps=512,
            batch_size=4096,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.001,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.02,
            device=args.device,
            policy_kwargs={"net_arch": [256, 256]},
            verbose=0,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(500_000 // args.num_envs, 1),
        save_path=str(log_dir / "checkpoints"),
        name_prefix="model",
        verbose=0,
    )

    total_steps = int(args.total_steps)
    print(f"Training for {total_steps:,} steps with {args.num_envs} envs ...")
    model.learn(
        total_timesteps=total_steps,
        callback=CallbackList([LogCallback(log_interval=20), checkpoint_cb]),
        progress_bar=True,
    )
    env.close()

    save_path = str(log_dir / "model")
    model.save(save_path)
    print(f"Model saved → {save_path}.zip")


if __name__ == "__main__":
    main()
