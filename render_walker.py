"""
Render a trained Walker2d imitation policy.

Usage:
    python render_walker.py --model results/walker2d_ulrich_all_<timestamp>/model.zip
"""
import argparse
from pathlib import Path

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from ppo_walker2d import Walker2dImitation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model.zip")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--vanilla", action="store_true",
                        help="Use plain Walker2d-v4 (for pretrained models without reference)")
    args = parser.parse_args()

    model_path = Path(args.model)
    load_path = str(model_path.with_suffix("")) if model_path.suffix == ".zip" else str(model_path)
    model = PPO.load(load_path, device="cpu")

    if args.vanilla:
        env = gym.make("Walker2d-v4", render_mode="rgb_array")
    else:
        ref_path = model_path.parent / "reference.npy"
        if not ref_path.exists():
            ref_path = model_path.parent.parent / "reference.npy"
        reference = np.load(ref_path)
        print(f"Reference: {len(reference):,} frames")
        env = Walker2dImitation(reference=reference, render_mode="rgb_array", warm_start=True)

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    all_frames = []
    ep_returns = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        ep_r = 0.0
        for t in range(args.steps):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_r += reward
            all_frames.append(env.render())
            if terminated or truncated:
                break
        ep_returns.append((ep + 1, ep_r, t + 1))
        print(f"Episode {ep+1}: return={ep_r:.1f}  length={t+1}")

    env.close()

    print(f"Rendering {len(all_frames)} frames ...")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    im = ax.imshow(all_frames[0])

    def update(i):
        im.set_data(all_frames[i])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(all_frames),
                                  interval=20, blit=True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
