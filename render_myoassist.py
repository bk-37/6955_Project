"""
Render a trained MyoAssist imitation policy.

Usage:
    python render_myoassist.py --model results/ppo_myoassist_<timestamp>/final_model.zip
"""
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MYOASSIST_DIR = os.path.join(PROJECT_ROOT, "myoassist")
os.chdir(MYOASSIST_DIR)
sys.path.insert(0, MYOASSIST_DIR)

import mujoco
import numpy as np
import imageio
from stable_baselines3 import PPO

import rl_train.envs  # noqa: F401
from rl_train.envs.environment_handler import EnvironmentHandler
from rl_train.train.train_configs.config_imitation import ImitationTrainSessionConfig

CONFIG_PATH = "rl_train/train/train_configs/imitation_tutorial_22_separated_net_partial_obs.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to final_model.zip or checkpoint")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--out", default=None, help="Output video path (default: model dir/render.mp4)")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    # resolve before chdir changes cwd
    model_path = Path(args.model) if os.path.isabs(args.model) else Path(PROJECT_ROOT) / args.model
    out_path = (Path(PROJECT_ROOT) / args.out if args.out and not os.path.isabs(args.out) else Path(args.out)) if args.out else model_path.parent / "render.mp4"
    out_path = str(out_path)

    # Use the session config from the same directory as the model if present
    model_session_config = model_path.parent.parent / "session_config.json"
    config_path = str(model_session_config) if model_session_config.exists() else CONFIG_PATH
    config: ImitationTrainSessionConfig = EnvironmentHandler.get_session_config_from_path(
        config_path, ImitationTrainSessionConfig
    )
    # Use the env_id from the session config (don't override — tutorial uses Exo variant)
    print(f"config.env_params.env_id='{config.env_params.env_id}'")
    config.env_params.num_envs = 1
    config.env_params.out_of_trajectory_threshold = 1_000_000
    config.env_params.custom_max_episode_steps = args.steps

    print("Creating env...")
    env = EnvironmentHandler.create_environment(config, is_rendering_on=False, is_evaluate_mode=True)

    # set up offscreen camera
    cam = mujoco.MjvCamera()
    env.unwrapped.sim.renderer.render_offscreen(camera_id=cam, width=args.width, height=args.height)

    print(f"Loading model from {model_path}...")
    load_path = str(model_path.with_suffix("")) if model_path.suffix == ".zip" else str(model_path)

    # The tutorial model was saved under 'myoassist_rl.rl_train.*'; our repo uses 'rl_train.*'.
    # Pre-import all rl_train submodules then alias them under myoassist_rl.rl_train.
    import sys, types, importlib, pkgutil, rl_train as _rl_train_pkg

    def _alias_package(real_name, alias_name):
        """Register real_name and all its submodules also under alias_name."""
        real_mod = sys.modules.get(real_name) or importlib.import_module(real_name)
        sys.modules.setdefault(alias_name, real_mod)
        if hasattr(real_mod, "__path__"):
            for _, subname, _ in pkgutil.walk_packages(real_mod.__path__, prefix=real_name + "."):
                alias_sub = alias_name + subname[len(real_name):]
                try:
                    sub = importlib.import_module(subname)
                    sys.modules.setdefault(alias_sub, sub)
                except Exception:
                    pass

    # Create top-level myoassist_rl namespace
    if "myoassist_rl" not in sys.modules:
        _ns = types.ModuleType("myoassist_rl")
        _ns.__path__ = []
        sys.modules["myoassist_rl"] = _ns

    _alias_package("rl_train", "myoassist_rl.rl_train")

    # Their old repo had rl_train/rl_agents/ — now split into rl_train/train/policies/
    import rl_train.train.policies as _policies_pkg
    sys.modules.setdefault("myoassist_rl.rl_train.rl_agents", _policies_pkg)
    _alias_package("rl_train.train.policies", "myoassist_rl.rl_train.rl_agents")

    model = PPO.load(load_path, device="cpu")

    obs, _ = env.reset()
    frames = []
    ep_r = 0.0

    for t in range(args.steps):
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_r += float(reward)

        # follow pelvis
        pelvis_pos = env.unwrapped.sim.data.body("pelvis").xpos.copy()
        cam.distance = 2.5
        cam.azimuth = 90
        cam.elevation = -10
        cam.lookat = np.array([pelvis_pos[0], pelvis_pos[1], 0.8])

        frame = env.unwrapped.sim.renderer.render_offscreen(
            camera_id=cam, width=args.width, height=args.height
        )
        frames.append(frame)

        if terminated or truncated:
            print(f"Episode ended at step {t+1}, return={ep_r:.1f}")
            break

    env.close()
    print(f"Writing {len(frames)} frames to {out_path} ...")
    writer = imageio.get_writer(out_path, fps=args.fps, codec="libx264", macro_block_size=None)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
