import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import PPO
from ppo_walker2d_phase import Walker2dPhaseAware

RESULT_DIR = "results/poc_mocap/subject3"
ref = np.load(f"{RESULT_DIR}/reference.npy")
env = Walker2dPhaseAware(reference=ref, render_mode="rgb_array",
                         pose_term_thresh=9999.0, ankle_term_thresh=9999.0)
model = PPO.load(f"{RESULT_DIR}/checkpoints/model_13000000_steps")

frames = []
obs, _ = env.reset()
for _ in range(2000):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, truncated, _ = env.step(action)
    frames.append(env.render())
    if terminated or truncated:
        obs, _ = env.reset()
env.close()
print(f"Collected {len(frames)} frames")

fig, ax = plt.subplots(figsize=(8, 4))
ax.axis("off")
im = ax.imshow(frames[0])

def update(i):
    im.set_data(frames[i])
    return (im,)

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=8, blit=True)
plt.tight_layout()
plt.show()
