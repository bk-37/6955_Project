"""
render_reference.py — kinematic playback of IK reference data.

Directly sets Walker2d joint angles from the reference each frame (no policy,
no dynamics). Shows what the reference motion actually looks like on the model.

Usage:
  python render_reference.py                        # uses gait_cycle_reference.npy
  python render_reference.py --ref_all              # streams through full Ulrich reference
  python render_reference.py --ref results/*/reference.npy
  python render_reference.py --ref_all --start 500 --n_frames 500
  python render_reference.py --speed 0.5            # half speed
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from ppo_walker2d_phase import Walker2dPhaseAware, CTRL_HZ, _JNT_LO, _JNT_HI


def main():
    parser = argparse.ArgumentParser()
    ref_grp = parser.add_mutually_exclusive_group()
    ref_grp.add_argument("--ref",     type=str, default=None,
                         help="Path to a reference .npy file")
    ref_grp.add_argument("--ref_all", action="store_true",
                         help="Load full Ulrich reference via load_ulrich_reference()")
    parser.add_argument("--subjects",  type=int, nargs="+", default=None)
    parser.add_argument("--xml",       default="walker2d.xml")
    parser.add_argument("--start",     type=int, default=0,
                        help="Start frame index (default 0)")
    parser.add_argument("--n_frames",  type=int, default=None,
                        help="Number of frames to play (default: one full gait cycle = 140)")
    parser.add_argument("--speed",    type=float, default=1.0,
                        help="Playback speed multiplier (default 1.0 = real time)")
    parser.add_argument("--pd_demo", action="store_true",
                        help="Run the PD tracking controller instead of kinematic playback. "
                             "Shows what the BC training data actually looks like in simulation.")
    parser.add_argument("--pd_kp",   type=float, default=200.0)
    parser.add_argument("--pd_kd",   type=float, default=20.0)
    args = parser.parse_args()

    # ── load reference ────────────────────────────────────────────────
    if args.ref_all:
        from ppo_walker2d import load_ulrich_reference
        print("Loading full Ulrich reference...")
        reference = load_ulrich_reference(
            subjects=args.subjects, control_hz=CTRL_HZ,
        )
    elif args.ref:
        reference = np.load(args.ref).astype(np.float32)
        if reference.ndim == 1:
            reference = reference.reshape(-1, 6)
    else:
        # default: look for a gait cycle in the project root
        candidates = list(PROJECT_ROOT.glob("gait_cycle*.npy"))
        if candidates:
            from ppo_walker2d_phase import load_ref_cycle
            reference = load_ref_cycle(candidates[0])
            print(f"Using: {candidates[0]}")
        else:
            # fall back to the most recent result's reference
            result_refs = sorted(PROJECT_ROOT.glob("results/*/reference.npy"),
                                 key=lambda p: p.stat().st_mtime)
            if not result_refs:
                print("No reference found. Pass --ref <path> or --ref_all.")
                return
            reference = np.load(result_refs[-1]).astype(np.float32)
            print(f"Using: {result_refs[-1]}")

    T = len(reference)
    start = args.start % T
    n = args.n_frames if args.n_frames is not None else min(280, T)  # default ~2 gait cycles
    print(f"Reference: {T} frames total — playing frames {start}..{start+n}")

    # ── build env for rendering only ─────────────────────────────────
    env = Walker2dPhaseAware(
        reference          = reference,
        xml_file           = args.xml,
        render_mode        = "rgb_array",
        pose_term_thresh   = 9999.0,   # never terminate
        ankle_term_thresh  = 9999.0,
    )

    # ── collect frames ────────────────────────────────────────────────
    frames = []

    if args.pd_demo:
        # Run PD tracking controller in live simulation — same as BC data collection.
        gear = float(env.model.actuator_gear[0, 0])
        obs, _ = env.reset()
        ep_steps = 0
        for i in range(n):
            q_ref  = env._reference[env._phase]
            dq_ref = env._ref_vel[env._phase]
            q_sim  = env.data.qpos[3:9].astype(np.float32)
            dq_sim = env.data.qvel[3:9].astype(np.float32)
            torque = args.pd_kp * (q_ref - q_sim) + args.pd_kd * (dq_ref - dq_sim)
            action = np.clip(torque / gear, -1.0, 1.0).astype(np.float32)
            frames.append(env.render())
            obs, _, terminated, truncated, _ = env.step(action)
            ep_steps += 1
            if terminated or truncated:
                print(f"  Episode ended at step {ep_steps} (frame {i})")
                obs, _ = env.reset()
                ep_steps = 0
        print(f"PD demo: {n} frames collected")
    else:
        # Pure kinematic playback — directly set joint angles, no dynamics.
        env.reset()
        for i in range(n):
            t = (start + i) % T
            qpos = env.data.qpos.copy()
            qvel = env.data.qvel.copy()
            qpos[1]   = 1.28
            qpos[2]   = 0.0
            qpos[3:9] = np.clip(reference[t], _JNT_LO, _JNT_HI)
            if i > 0:
                qpos[0] = env.data.qpos[0] + 1.25 / CTRL_HZ
            qvel[:] = 0.0
            env.set_state(qpos, qvel)
            frames.append(env.render())

    env.close()

    # ── animate ───────────────────────────────────────────────────────
    interval_ms = (1000.0 / CTRL_HZ) / args.speed   # real-time by default
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    im    = ax.imshow(frames[0])
    title = ax.set_title(f"Reference kinematic playback  (frame {start})")

    def update(i):
        im.set_data(frames[i])
        title.set_text(f"Reference playback — frame {start + i}/{T}  "
                       f"({(start + i) / CTRL_HZ:.2f}s)")
        return (im, title)

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=interval_ms, blit=True,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
