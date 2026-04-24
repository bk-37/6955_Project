"""
render_phase.py — visualize phase-aware imitation policy episodes.

Usage:
  python render_phase.py <result_dir>:<checkpoint_steps>[:<label>] [<result_dir>:...] ...

  checkpoint_steps: integer (e.g. 15000000) or "final" (uses model.zip)

Optional flags:
  --xml   XML model file (default: walker2d_subject1.xml)
  --eps   Episodes per run (default: 3)
  --steps Max steps per episode (default: 2000)

Examples:
  python render_phase.py results/my_run:15000000:"15M checkpoint"
  python render_phase.py results/run1:final results/run2:10000000
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import PPO
from ppo_walker2d_phase import Walker2dPhaseAware, _JNT_LO, _JNT_HI


def parse_spec(spec: str, default_xml: str) -> dict:
    parts = spec.split(":")
    if len(parts) < 2:
        raise ValueError(f"Expected result_dir:checkpoint[:label], got: {spec!r}")
    result_dir = parts[0]
    ckpt_raw   = parts[1]
    label      = parts[2] if len(parts) > 2 else Path(result_dir).name + f":{ckpt_raw}"

    if ckpt_raw.lower() == "final":
        model_path = str(Path(result_dir) / "model")
    else:
        model_path = str(Path(result_dir) / "checkpoints" / f"model_{ckpt_raw}_steps")

    return {"label": label, "result_dir": result_dir,
            "model_path": model_path, "xml_file": default_xml}


def main():
    parser = argparse.ArgumentParser(description="Render phase-aware Walker2d policy")
    parser.add_argument("specs", nargs="+",
                        help="result_dir:checkpoint_steps[:label]")
    parser.add_argument("--xml",   default="walker2d_subject1.xml",
                        help="MuJoCo model XML (default: walker2d_subject1.xml)")
    parser.add_argument("--eps",   type=int, default=3,  help="Episodes per run")
    parser.add_argument("--steps", type=int, default=2000, help="Max steps per episode")
    args = parser.parse_args()

    runs = [parse_spec(s, args.xml) for s in args.specs]

    all_frames = []
    run_labels = []

    for run in runs:
        ref      = np.load(f"{run['result_dir']}/reference.npy")
        xml_file = run["xml_file"]
        env = Walker2dPhaseAware(
            reference=ref, xml_file=xml_file, render_mode="rgb_array",
            pose_term_thresh=9999.0, ankle_term_thresh=9999.0,
        )
        model = PPO.load(run["model_path"])

        for ep in range(args.eps):
            start_phase = ep * len(ref) // args.eps
            obs, _ = env.reset()
            env._phase = start_phase
            qpos = env.data.qpos.copy()
            qvel = env.data.qvel.copy()
            qpos[3:9] = np.clip(ref[start_phase], _JNT_LO, _JNT_HI)
            qvel[3:9] = 0.0
            env.set_state(qpos, qvel)
            obs = env._get_obs()

            ep_frames = []
            for _ in range(args.steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                ep_frames.append(env.render())
                if terminated or truncated:
                    break

            print(f"[{run['label']}] ep {ep+1}: {len(ep_frames)} steps")
            all_frames.extend(ep_frames)
            run_labels.append((len(all_frames) - len(ep_frames),
                               len(ep_frames), run["label"], ep + 1))

        env.close()

    print(f"\nTotal frames: {len(all_frames)}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    im    = ax.imshow(all_frames[0])
    title = ax.set_title("")

    def get_label(i):
        for start, n, label, ep in run_labels:
            if start <= i < start + n:
                return f"{label} — ep {ep} (frame {i - start + 1}/{n})"
        return ""

    def update(i):
        im.set_data(all_frames[i])
        title.set_text(get_label(i))
        return (im, title)

    ani = animation.FuncAnimation(
        fig, update, frames=len(all_frames), interval=8, blit=True
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
