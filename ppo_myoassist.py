"""
PPO training on the 22-muscle MyoAssist 2-D leg model using phase-tracked
imitation reward (DeepMimic style).

Run from project root:
    conda run -n OpenCap_RL python ppo_myoassist.py [--num_envs 8] [--total_steps 3e7]

The script changes cwd to myoassist/ so all relative paths inside that
package (model XML, reference data) resolve correctly.
"""
import argparse
import os
import sys
from datetime import datetime

# ── must happen before any myoassist import ──────────────────────────────────
MYOASSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "myoassist")
os.chdir(MYOASSIST_DIR)
sys.path.insert(0, MYOASSIST_DIR)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import VecMonitor

import rl_train.envs  # noqa: F401 – triggers gym env registration

from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig
from rl_train.train.policies.rl_agent_human import HumanActorCriticPolicy

# Observation/action index wiring for the no-exo 22-muscle imitation env.
# Taken directly from tutorial session_config.json (full_obs variant),
# minus the exo_actor block which doesn't apply to myoAssistLegImitation-v0.
NET_INDEXING_INFO = {
    "human_actor": {
        "observation": [
            {"type": "range", "range": [0,  8],  "comment": "8 qpos"},
            {"type": "range", "range": [8,  17], "comment": "9 qvel"},
            {"type": "range", "range": [17, 28], "comment": "11 left muscle activation"},
            {"type": "range", "range": [28, 39], "comment": "11 right muscle activation"},
            {"type": "range", "range": [39, 43], "comment": "4 foot force"},
            {"type": "range", "range": [43, 44], "comment": "target velocity"},
        ],
        "action": [
            {"type": "range_mapping", "range_net": [0,  11], "range_action": [0,  11], "comment": "11 right muscles"},
            {"type": "range_mapping", "range_net": [11, 22], "range_action": [11, 22], "comment": "11 left muscles"},
        ],
    },
    "common_critic": {
        "observation": [
            {"type": "range", "range": [0,  8],  "comment": "8 qpos"},
            {"type": "range", "range": [8,  17], "comment": "9 qvel"},
            {"type": "range", "range": [17, 28], "comment": "11 left muscle activation"},
            {"type": "range", "range": [28, 39], "comment": "11 right muscle activation"},
            {"type": "range", "range": [39, 43], "comment": "4 foot force"},
            {"type": "range", "range": [43, 44], "comment": "target velocity"},
        ],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# The partial_obs JSON has the correct 22-muscle model and reference data paths.
# Its env_id is the exo variant; we override it below to the base imitation env.
CONFIG_PATH = "rl_train/train/train_configs/imitation_tutorial_22_separated_net_partial_obs.json"

# ── callback: episode tracking + periodic logging ─────────────────────────────
class LogCallback(BaseCallback):
    """
    Tracks episode returns/lengths via VecMonitor's 'episode' info key.
    Prints a single summary line every `log_interval` rollouts.
    SB3's own per-rollout table is suppressed by setting PPO verbose=0.
    """
    def __init__(self, log_interval: int = 50, verbose: int = 1):
        super().__init__(verbose)
        self._log_interval = log_interval
        self._rollout = 0
        self._ep_rewards: list[float] = []
        self._ep_lens: list[int] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self._ep_rewards.append(float(ep["r"]))
                self._ep_lens.append(int(ep["l"]))
        return True

    def _on_rollout_end(self) -> None:
        self._rollout += 1
        if self._rollout % self._log_interval == 0:
            if self._ep_rewards:
                mean_r = float(np.mean(self._ep_rewards))
                mean_l = float(np.mean(self._ep_lens))
                print(
                    f"[iter {self._rollout:5d} | steps {self.num_timesteps:>10,}]  "
                    f"ep_r={mean_r:8.1f}  ep_len={mean_l:6.0f}  "
                    f"(n={len(self._ep_rewards)} eps)"
                )
                self._ep_rewards.clear()
                self._ep_lens.clear()
            else:
                print(
                    f"[iter {self._rollout:5d} | steps {self.num_timesteps:>10,}]  "
                    f"(no completed episodes yet)"
                )


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--num_envs", type=int, default=None,
                        help="Override number of parallel envs (default: from JSON = 32)")
    parser.add_argument("--total_steps", type=float, default=None,
                        help="Override total training steps (default: from JSON = 3e7)")
    parser.add_argument("--device", default=None, help="cpu / cuda (default: from JSON)")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: ../results/ppo_myoassist_YYYYMMDD-HHMMSS)")
    parser.add_argument("--target_kl", type=float, default=None,
                        help="PPO target KL (default: from JSON = 0.01). Try 0.03-0.05 if early-stopping at <10 epochs.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--resume", default=None,
                        help="Path to model.zip to resume training from")
    parser.add_argument("--checkpoint_freq", type=int, default=2_000_000,
                        help="Save a checkpoint every N steps (default: 2e6)")
    args = parser.parse_args()

    config: ImitationTrainSessionConfig = EnvironmentHandler.get_session_config_from_path(
        args.config, ImitationTrainSessionConfig
    )

    # Force no-exo env (22 muscles, no exo actuators)
    config.env_params.env_id = "myoAssistLegImitation-v0"

    # Loosen trajectory threshold — default 0.2 rad kills episodes on step 1-2
    # for a random policy. Set to 100 (effectively disabled) so the agent can
    # explore and learn; tighten later once policy is competent.
    config.env_params.out_of_trajectory_threshold = 100

    # ── apply CLI overrides ───────────────────────────────────────────────────
    if args.num_envs is not None:
        config.env_params.num_envs = args.num_envs
    if args.device is not None:
        config.ppo_params.device = args.device
    if args.target_kl is not None:
        config.ppo_params.target_kl = args.target_kl
    if args.lr is not None:
        config.ppo_params.learning_rate = args.lr
    total_steps = int(args.total_steps) if args.total_steps is not None else int(config.total_timesteps)

    # ── output directory ──────────────────────────────────────────────────────
    # Note: script chdirs to myoassist/, so resolve paths relative to project root
    project_root = os.path.dirname(MYOASSIST_DIR)
    if args.out_dir:
        log_dir = os.path.join(project_root, args.out_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(project_root, "results", f"ppo_myoassist_{stamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Saving to: {log_dir}")

    # ── create env ────────────────────────────────────────────────────────────
    print(f"Creating {config.env_params.num_envs} envs ({config.env_params.env_id}) ...")
    env = EnvironmentHandler.create_environment(config, is_rendering_on=False)
    env = VecMonitor(env)  # adds 'episode' key to infos so we can track ep_r / ep_len

    # ── build PPO with MlpPolicy ──────────────────────────────────────────────
    p = config.ppo_params
    policy_kwargs = {"custom_policy_params": {
        "net_arch": {"human_actor": [64, 64], "common_critic": [64, 64]},
        "net_indexing_info": NET_INDEXING_INFO,
        "log_std_init": 0.0,
    }}

    if args.resume:
        resume_path = args.resume if os.path.isabs(args.resume) else os.path.join(project_root, args.resume)
        if resume_path.endswith(".zip"):
            resume_path = resume_path[:-4]
        print(f"Resuming from {resume_path} ...")
        model = PPO.load(resume_path, env=env, device=p.device,
                         custom_objects={"policy_kwargs": policy_kwargs})
        model.learning_rate = p.learning_rate
    else:
        model = PPO(
            HumanActorCriticPolicy,
            env,
            learning_rate=p.learning_rate,
            n_steps=p.n_steps,
            batch_size=p.batch_size,
            n_epochs=p.n_epochs,
            gamma=p.gamma,
            gae_lambda=p.gae_lambda,
            clip_range=p.clip_range,
            clip_range_vf=p.clip_range_vf,
            ent_coef=p.ent_coef,
            vf_coef=p.vf_coef,
            max_grad_norm=p.max_grad_norm,
            target_kl=p.target_kl,
            device=p.device,
            policy_kwargs=policy_kwargs,
            verbose=0,
        )

    # ── train ─────────────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // config.env_params.num_envs, 1),
        save_path=log_dir,
        name_prefix="ckpt",
        verbose=1,
    )
    cb = CallbackList([LogCallback(log_interval=10), checkpoint_cb])
    print(f"Training for {total_steps:,} steps …")
    model.learn(
        total_timesteps=total_steps,
        callback=cb,
        progress_bar=True,
        reset_num_timesteps=not bool(args.resume),
    )
    env.close()

    save_path = os.path.join(log_dir, "final_model")
    model.save(save_path)
    print(f"Model saved → {save_path}.zip")


if __name__ == "__main__":
    main()
