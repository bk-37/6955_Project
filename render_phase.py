import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import PPO
from ppo_walker2d_phase import Walker2dPhaseAware

RESULT_DIR = "results/walker2d_phase_full_sum_20260410-124935"
MODEL_PATH = f"{RESULT_DIR}/checkpoints/model_18000000_steps"
N_EPISODES = 5
MAX_STEPS  = 2000

ref = np.load(f"{RESULT_DIR}/reference.npy")
env = Walker2dPhaseAware(reference=ref, render_mode="rgb_array",
                         pose_term_thresh=9999.0, ankle_term_thresh=9999.0)
model = PPO.load(MODEL_PATH)

all_frames = []
for ep in range(N_EPISODES):
    # Fixed phase sweep: test phases 0, 28, 56, 84, 112 (evenly spaced through cycle)
    start_phase = (ep * len(ref) // N_EPISODES)
    obs, _ = env.reset()
    env._phase = start_phase
    from ppo_walker2d_phase import _JNT_LO, _JNT_HI
    qpos = env.data.qpos.copy(); qvel = env.data.qvel.copy()
    qpos[3:9] = np.clip(ref[start_phase], _JNT_LO, _JNT_HI)
    qvel[3:9] = 0.0
    env.set_state(qpos, qvel)
    obs = env._get_obs()

    ep_frames = []
    for _ in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        ep_frames.append(env.render())
        if terminated or truncated:
            break
    print(f"Episode {ep+1}: {len(ep_frames)} steps  (phase={start_phase})")
    all_frames.extend(ep_frames)

env.close()
print(f"Total frames: {len(all_frames)}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.axis("off")
im = ax.imshow(all_frames[0])

def update(i):
    im.set_data(all_frames[i])
    return (im,)

ani = animation.FuncAnimation(fig, update, frames=len(all_frames), interval=8, blit=True)
plt.tight_layout()
plt.show()
