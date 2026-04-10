import gymnasium as gym
import numpy as np

env = gym.make("Walker2d-v4")
env.reset()
m = env.unwrapped.model

total_mass = float(m.body_mass.sum())
print(f"Total Walker2d mass: {total_mass:.2f} kg")
print(f"\nPer-body masses:")
for i in range(m.nbody):
    name = m.body(i).name
    mass = float(m.body_mass[i])
    if mass > 0:
        print(f"  {name:20s}: {mass:.3f} kg")

print(f"\nTo scale GRF from sim to match a {75:.0f}kg subject:")
print(f"  scale = {75/total_mass:.3f}  (multiply sim forces by this)")
print(f"\nWalker2d body height (torso z at standing): {float(env.unwrapped.data.body('torso').xpos[2]):.3f} m")
env.close()
