"""
GAIL training on Walker2d-v4 using Ulrich IK data as demonstrations.

The discriminator learns "does this trajectory look like Ulrich walking?"
without any hand-crafted reward, phase tracking, or sign convention assumptions.
Only the 6 joint angles + their velocities are used in demonstrations —
the discriminator operates on the full Walker2d obs but demos are constructed
from IK data with reasonable constants for unmeasured dims.

Usage:
    python gail_walker2d.py
    python gail_walker2d.py --ref_cycle gait_cycle_reference.npy
    python gail_walker2d.py --subjects 1 2 3 --trial_filter baseline

Requirements:
    pip install imitation
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback

PROJECT_ROOT = Path(__file__).parent
ULRICH_ROOT  = PROJECT_ROOT / "Ulrich_Treadmill_Data"

REF_HZ  = 50.0
CTRL_HZ = 125.0  # Walker2d frame_skip=4, dt=0.002 → 125 Hz


# ── data loading (reused from ppo_walker2d) ───────────────────────────────────
def load_sto(path: Path) -> dict:
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


def load_ulrich_reference(subjects=None, trial_filter=None, control_hz=CTRL_HZ):
    if subjects is None:
        subjects = list(range(1, 11))
    segments = []
    for subj_id in subjects:
        subj_dir = ULRICH_ROOT / f"Subject{subj_id}" / "IK"
        if not subj_dir.exists():
            continue
        for trial_dir in sorted(subj_dir.glob("walking_*")):
            if trial_filter and trial_filter not in trial_dir.name:
                continue
            ik_path = trial_dir / "output" / "results_ik.sto"
            if not ik_path.exists():
                continue
            d = load_sto(ik_path)
            orig_hz = 1.0 / (d["time"][1] - d["time"][0])
            orig_len = len(d["time"])
            new_len = int(orig_len * control_hz / orig_hz)
            orig_x = np.arange(orig_len)
            new_x = np.linspace(0, orig_len - 1, new_len)
            def resamp(key): return np.interp(new_x, orig_x, d[key])
            seg = np.stack([
                -np.deg2rad(resamp("hip_flexion_r")),
                -np.deg2rad(resamp("knee_angle_r")),
                -np.deg2rad(resamp("ankle_angle_r")),
                -np.deg2rad(resamp("hip_flexion_l")),
                -np.deg2rad(resamp("knee_angle_l")),
                -np.deg2rad(resamp("ankle_angle_l")),
            ], axis=1)
            segments.append(seg)
    ref = np.concatenate(segments, axis=0).astype(np.float32)
    print(f"  Loaded {len(segments)} trials → {len(ref):,} frames @ {control_hz}Hz")
    return ref


def resample_cycle(raw, src_hz=REF_HZ, dst_hz=CTRL_HZ):
    n_in  = len(raw)
    n_out = int(round(n_in * dst_hz / src_hz))
    x_in  = np.linspace(0, 1, n_in)
    x_out = np.linspace(0, 1, n_out)
    return np.stack([np.interp(x_out, x_in, raw[:, j])
                     for j in range(raw.shape[1])], axis=1).astype(np.float32)


# ── build demonstrations ──────────────────────────────────────────────────────
def build_demonstrations(ref: np.ndarray, n_demo_transitions: int = 50_000):
    """
    Construct demonstration transitions using only the 6 joint angles + velocities (12-dim).
    We wrap the env in a JointOnlyWrapper so the discriminator only sees joints,
    avoiding the trivial separation between clean-constant demos and noisy full obs.
    """
    from imitation.data.types import Transitions

    T = len(ref)
    # Joint velocities via finite difference (cyclic)
    jvel = np.zeros_like(ref)
    for t in range(T):
        jvel[t] = (ref[(t + 1) % T] - ref[(t - 1) % T]) / (2.0 / CTRL_HZ)
    jvel = np.clip(jvel, -10.0, 10.0)

    # 12-dim obs: [6 joint angles, 6 joint velocities]
    obs_arr = np.concatenate([ref, jvel], axis=1).astype(np.float32)

    idx      = np.random.choice(T, size=n_demo_transitions, replace=True)
    obs      = obs_arr[idx]
    next_obs = obs_arr[(idx + 1) % T]
    acts     = np.zeros((n_demo_transitions, 6), dtype=np.float32)
    dones    = np.zeros(n_demo_transitions, dtype=bool)
    infos    = np.array([{}] * n_demo_transitions)

    transitions = Transitions(
        obs=obs,
        acts=acts,
        next_obs=next_obs,
        dones=dones,
        infos=infos,
    )
    print(f"  Built {n_demo_transitions:,} demonstration transitions (12-dim: joints + jvels)")
    return transitions


# ── callback ──────────────────────────────────────────────────────────────────
class LogCallback(BaseCallback):
    def __init__(self, log_interval=50):
        super().__init__(verbose=0)
        self._interval = log_interval
        self._rollout = 0
        self._ep_r: list = []
        self._ep_l: list = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep:
                self._ep_r.append(float(ep["r"]))
                self._ep_l.append(int(ep["l"]))
        return True

    def _on_rollout_end(self):
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
                print(f"[iter {self._rollout:5d} | steps {self.num_timesteps:>9,}]  (no completed episodes)")


# ── joint-only observation wrapper ───────────────────────────────────────────
class JointOnlyWrapper(gym.ObservationWrapper):
    """
    Reduces Walker2d-v4 obs (17-dim) to just the 6 joint angles + 6 joint velocities (12-dim).
    Walker2d obs layout:
        [z(0), torso(1), thigh(2), leg(3), foot(4), thigh_l(5), leg_l(6), foot_l(7),
         xdot(8), zdot(9), torso_av(10), vel_thigh(11), vel_leg(12), vel_foot(13),
         vel_thigh_l(14), vel_leg_l(15), vel_foot_l(16)]
    Joint slice: positions [2:8], velocities [11:17]
    """
    JOINT_IDX = list(range(2, 8)) + list(range(11, 17))  # 12 dims

    def __init__(self, env):
        super().__init__(env)
        low  = env.observation_space.low[self.JOINT_IDX]
        high = env.observation_space.high[self.JOINT_IDX]
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        return obs[self.JOINT_IDX].astype(np.float32)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_cycle", default=None,
                        help="Single gait cycle .npy (default: load all Ulrich trials)")
    parser.add_argument("--subjects", type=int, nargs="+", default=None)
    parser.add_argument("--trial_filter", default=None)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--total_steps", type=float, default=3e7)
    parser.add_argument("--n_demo", type=int, default=50_000,
                        help="Number of demonstration transitions to sample")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    try:
        from imitation.algorithms.adversarial.gail import GAIL
        from imitation.rewards.reward_nets import BasicRewardNet
        from imitation.util.networks import RunningNorm
    except ImportError:
        print("ERROR: 'imitation' package not found. Install with:")
        print("    pip install imitation")
        sys.exit(1)

    # Load reference
    if args.ref_cycle:
        raw = np.load(args.ref_cycle).astype(np.float32)
        ref = resample_cycle(raw)
        print(f"Loaded gait cycle: {len(raw)} frames → {len(ref)} @ {CTRL_HZ}Hz")
    else:
        print("Loading Ulrich reference data...")
        ref = load_ulrich_reference(args.subjects, args.trial_filter)

    # Output dir
    if args.out_dir:
        log_dir = PROJECT_ROOT / args.out_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = PROJECT_ROOT / "results" / f"gail_walker2d_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {log_dir}")
    np.save(log_dir / "reference.npy", ref)

    # Build demonstrations
    print("Building demonstrations...")
    demonstrations = build_demonstrations(ref, n_demo_transitions=args.n_demo)

    # Make envs — joint-only obs so discriminator sees same space as demos
    def make_env():
        def _init():
            return JointOnlyWrapper(gym.make("Walker2d-v4"))
        return _init

    venv = SubprocVecEnv([make_env() for _ in range(args.num_envs)])
    venv = VecMonitor(venv)

    # PPO learner
    learner = PPO(
        "MlpPolicy",
        venv,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=2048,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=args.device,
        policy_kwargs={"net_arch": [256, 256]},
        verbose=0,
    )

    # Discriminator reward network — built on the wrapped (12-dim) obs space
    base_env = JointOnlyWrapper(gym.make("Walker2d-v4"))
    # State-only discriminator with small network to slow down discriminator learning.
    # Smaller net = harder to memorize = longer competitive GAN dynamic.
    reward_net = BasicRewardNet(
        observation_space=base_env.observation_space,
        action_space=base_env.action_space,
        use_state=True,
        use_action=False,
        use_next_state=False,
        normalize_input_layer=RunningNorm,
        hid_sizes=(64, 64),
    )
    base_env.close()

    # Attach SB3 callbacks directly to the learner so they fire during gen training
    checkpoint_cb = CheckpointCallback(
        save_freq=max(500_000 // args.num_envs, 1),
        save_path=str(log_dir / "checkpoints"),
        name_prefix="model",
        verbose=0,
    )
    log_cb = LogCallback(log_interval=20)
    learner.set_env(venv)  # ensure callbacks can access env info

    # GAIL trainer — 1 disc update per round to prevent discriminator from winning too fast
    gail_trainer = GAIL(
        demonstrations=demonstrations,
        demo_batch_size=512,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=1,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
    )

    # Inject callbacks into the learner after GAIL wraps it
    learner._logger = learner._logger  # no-op, just ensuring init
    _sb3_cbs = CallbackList([log_cb, checkpoint_cb])

    total_steps = int(args.total_steps)
    print(f"Training GAIL for {total_steps:,} generator steps with {args.num_envs} envs...")

    # train_gen internally calls learner.learn(); pass callbacks via a round callback
    # that sets them before each gen phase
    def _round_cb(round_num: int) -> None:
        pass  # logging handled by SB3 callbacks injected below

    # Monkey-patch learn to inject our callbacks for every gen training call
    _orig_learn = learner.learn
    def _learn_with_cb(*a, **kw):
        kw.setdefault("callback", _sb3_cbs)
        kw.setdefault("reset_num_timesteps", False)
        return _orig_learn(*a, **kw)
    learner.learn = _learn_with_cb

    gail_trainer.train(total_timesteps=total_steps)

    venv.close()

    # Save policy weights only (avoids SubprocVecEnv cloudpickle auth token issue)
    save_path = str(log_dir / "policy")
    learner.policy.save(save_path)
    print(f"Policy saved → {save_path}.pt")
    # Also save via torch directly as a fallback
    import torch
    torch.save(learner.policy.state_dict(), str(log_dir / "policy_state_dict.pt"))
    print(f"State dict saved → {log_dir / 'policy_state_dict.pt'}")


if __name__ == "__main__":
    main()
