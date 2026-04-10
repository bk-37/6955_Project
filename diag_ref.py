import numpy as np, mujoco, gymnasium as gym

ref = np.load("gait_cycle_reference.npy")
n_in = len(ref); n_out = int(round(n_in * 125.0 / 50.0))
x_in = np.linspace(0,1,n_in); x_out = np.linspace(0,1,n_out)
ref125 = np.stack([np.interp(x_out,x_in,ref[:,j]) for j in range(6)],axis=1)

env = gym.make("Walker2d-v4"); env.reset()
m, d = env.unwrapped.model, env.unwrapped.data

print("=== Reference joint ranges ===")
names = ["hip_r","knee_r","ankle_r","hip_l","knee_l","ankle_l"]
for i,n in enumerate(names):
    print(f"  {n:10s}: [{ref125[:,i].min():.3f}, {ref125[:,i].max():.3f}]  mean={ref125[:,i].mean():.3f}")

print("\n=== Open-loop FK: does the reference stay upright? ===")
heights, pitches = [], []
for t in range(n_out):
    d.qpos[:] = 0.0; d.qpos[1] = 1.28; d.qpos[3:9] = ref125[t]
    mujoco.mj_kinematics(m, d)
    heights.append(float(d.body("torso").xpos[2]))

print(f"  Torso height range (FK at fixed pitch=0): [{min(heights):.3f}, {max(heights):.3f}]")
print(f"  This is the height the root term targets")

# Simulate open-loop playback — do the joints produce forward motion?
print("\n=== Open-loop playback (position control) ===")
env2 = gym.make("Walker2d-v4"); obs, _ = env2.reset()
d2 = env2.unwrapped.data
# Set initial state to reference frame 0
d2.qpos[3:9] = ref125[0]; d2.qvel[:] = 0.0
x_start = float(d2.qpos[0])
for t in range(min(n_out, 140)):
    # Just set joint angles directly (not physically valid, but diagnostic)
    target = ref125[t % n_out]
    # Use PD control approximation: action = target angles (Walker2d uses position targets)
    obs, r, term, trunc, info = env2.step(target[:6] * 0)  # zero action
    if term or trunc:
        print(f"  Terminated at step {t}")
        break
print(f"  x displacement with zero actions: {float(d2.qpos[0]) - x_start:.3f}m")
env2.close()
env.close()
