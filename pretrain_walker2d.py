"""
Stage 1: Train vanilla Walker2d-v4 with default rewards until it walks.
This is a solved benchmark — PPO reaches ep_len ~1000 in ~2M steps.

Save the checkpoint, then use ppo_walker2d.py --finetune to specialize
toward the Ulrich reference kinematics.

Usage:
    python pretrain_walker2d.py
    python pretrain_walker2d.py --total_steps 3e6 --num_envs 32
"""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

PROJECT_ROOT = Path(__file__).parent


class Walker2dContactWalk(gym.Wrapper):
    """
    Wraps Walker2d-v4 with two independent anti-hopping strategies (selectable):

    mode='contact':
      Height reward + foot alternation reward. No reference needed.
      Breaks hopping/flying by penalising torso rising above 1.25m.

    mode='symmetry':
      Phase-shifted bilateral symmetry loss (Yu et al. 2018).
      Stores a circular buffer of left-leg joint states; penalises the
      right-leg state diverging from the left-leg state half-a-stride ago.
      Half-stride ≈ 0.5s ≈ 62 steps at 125Hz. No reference needed.
      Much stronger signal for bilateral gait than contact alone.
    """
    HALF_STRIDE_STEPS = 62   # ~0.5s at 125Hz

    def __init__(self, env, weight=3.0, mode='symmetry'):
        super().__init__(env)
        self._w = weight
        self._mode = mode
        # contact mode state
        self._last_stance = 0
        self._right_steps = 0
        self._left_steps  = 0
        # symmetry mode state: circular buffer of past left-leg qpos [hip_l, knee_l, ankle_l]
        self._left_buf = np.zeros((self.HALF_STRIDE_STEPS, 3), dtype=np.float32)
        self._buf_idx  = 0

    def reset(self, **kwargs):
        self._last_stance = np.random.randint(0, 2)
        self._right_steps = 0
        self._left_steps  = 0
        self._left_buf[:] = 0.0
        self._buf_idx = 0
        self._right_air = 0
        self._left_air  = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        # Hard-cap ankle torques to 30% of max to force hip/knee to drive locomotion.
        # Walker2d action indices: 0=hip_r, 1=knee_r, 2=ankle_r, 3=hip_l, 4=knee_l, 5=ankle_l
        action = action.copy()
        action[2] = np.clip(action[2], -0.3, 0.3)
        action[5] = np.clip(action[5], -0.3, 0.3)
        obs, reward, terminated, truncated, info = self.env.step(action)
        uw = self.env.unwrapped
        dt = uw.dt

        torso_z = float(uw.data.qpos[1])

        # Use foot body world-z position — unambiguous, no force threshold needed.
        # Walker2d foot bodies: "foot" (right) and "foot_left".
        # Ground is z=0; feet are ~0.05-0.1m when planted, higher when airborne.
        right_z = float(uw.data.body("foot").xpos[2])
        left_z  = float(uw.data.body("foot_left").xpos[2])
        FOOT_GROUND = 0.08   # feet above this z = airborne (ground contact ~0.0-0.05)
        right_on = right_z < FOOT_GROUND
        left_on  = left_z  < FOOT_GROUND

        # Height reward: penalise torso rising above normal walking height
        height_r = dt * np.exp(-15.0 * (torso_z - 1.25) ** 2)

        if self._mode == 'contact':
            if right_on: self._right_steps += 1
            if left_on:  self._left_steps  += 1

            if self._last_stance == 0 and left_on:
                alt_r = dt
                self._last_stance = 1
            elif self._last_stance == 1 and right_on:
                alt_r = dt
                self._last_stance = 0
            else:
                alt_r = 0.0

            total = self._right_steps + self._left_steps + 1
            imbalance = abs(self._right_steps - self._left_steps) / total
            extra = height_r + alt_r - dt * imbalance

        else:  # symmetry mode
            hip_r_q = float(uw.data.qpos[3])
            hip_l_q = float(uw.data.qpos[6])

            # 1. Anti-phase hips: one must be flexed while other is extended.
            #    Score = cos of angle between them; max reward when 180° apart.
            #    Normalise by expected amplitude (~0.4 rad each side).
            hip_anti = dt * np.clip(-(hip_r_q * hip_l_q) / (0.4 ** 2), -1.0, 1.0)

            # 2. Hip range of motion: reward each hip reaching at least ±0.3 rad.
            #    Ankle-shuffling keeps hips near 0; walking requires ~±0.4 rad.
            hip_rom = dt * (np.clip(abs(hip_r_q) / 0.3, 0.0, 1.0) +
                            np.clip(abs(hip_l_q) / 0.3, 0.0, 1.0))

            # 3. Both-feet-airborne penalty (hop suppression)
            both_air = dt * (2.0 if (not right_on and not left_on) else 0.0)

            extra = height_r + hip_anti + hip_rom - both_air

        reward = reward + self._w * extra
        return obs, reward, terminated, truncated, info


class LogCallback(BaseCallback):
    def __init__(self, log_interval=20):
        super().__init__(verbose=0)
        self._interval = log_interval
        self._rollout = 0
        self._ep_r, self._ep_l = [], []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep:
                self._ep_r.append(float(ep["r"]))
                self._ep_l.append(int(ep["l"]))
        return True

    def _on_rollout_end(self):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--total_steps", type=float, default=5e6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--mode", default="symmetry", choices=["symmetry", "contact"],
                        help="Anti-hopping strategy: symmetry loss (Yu 2018) or contact alternation")
    parser.add_argument("--weight", type=float, default=3.0)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = PROJECT_ROOT / (args.out_dir or f"results/walker2d_pretrain_{args.mode}_{stamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {log_dir}")

    def make_env():
        def _init():
            # Reduce forward_reward_weight so shaped rewards can compete.
            # Default is 1.0 giving ~1.25/step at walking speed; our shaped
            # rewards max at ~0.13/step at weight=8 — reduce to 0.3 to balance.
            base = gym.make("Walker2d-v4", forward_reward_weight=0.5)
            # Increase gravity to penalise airborne time physically, not just via reward.
            # Default MuJoCo gravity is -9.81; 2x makes hopping energetically expensive.
            base.unwrapped.model.opt.gravity[2] = -24.525  # 2.5x normal gravity
            return Walker2dContactWalk(base, weight=args.weight, mode=args.mode)
        return _init

    env = SubprocVecEnv([make_env() for _ in range(args.num_envs)])
    env = VecMonitor(env)

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=4096,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
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
    print(f"Pretraining vanilla Walker2d for {total_steps:,} steps with {args.num_envs} envs...")
    model.learn(
        total_timesteps=total_steps,
        callback=CallbackList([LogCallback(), checkpoint_cb]),
        progress_bar=True,
    )
    env.close()

    save_path = str(log_dir / "model")
    model.save(save_path)
    print(f"\nModel saved → {save_path}.zip")
    print(f"\nNext step — finetune toward Ulrich reference:")
    print(f"  python ppo_walker2d.py --finetune {save_path}.zip --ref_cycle gait_cycle_reference.npy")


if __name__ == "__main__":
    main()
