import numpy as np
import matplotlib.pyplot as plt

ref = np.load("gait_cycle_reference.npy")
n_in = len(ref)
n_out = int(round(n_in * 125.0 / 50.0))
x_in = np.linspace(0,1,n_in); x_out = np.linspace(0,1,n_out)
ref125 = np.stack([np.interp(x_out,x_in,ref[:,j]) for j in range(6)],axis=1)

names = ["hip_r","knee_r","ankle_r","hip_l","knee_l","ankle_l"]

# Check boundary discontinuity: how much does last frame differ from first?
print("=== Cycle boundary discontinuity (last frame -> first frame) ===")
for i, n in enumerate(names):
    jump = abs(ref125[-1, i] - ref125[0, i])
    print(f"  {n:10s}: {np.rad2deg(jump):.1f} deg jump")

# Check velocity continuity at boundary
dq = np.gradient(ref125, 1.0/125.0, axis=0)
print("\n=== Joint velocity at boundary ===")
for i, n in enumerate(names):
    print(f"  {n:10s}: vel[-1]={np.rad2deg(dq[-1,i]):6.1f}  vel[0]={np.rad2deg(dq[0,i]):6.1f}  deg/s")

# Plot 3 cycles to visualize
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
t = np.arange(n_out) / 125.0
for i, (ax, name) in enumerate(zip(axes.flat, names)):
    # Show 3 loops
    tt = np.concatenate([t, t + t[-1], t + 2*t[-1]])
    qq = np.tile(ref125[:, i], 3)
    ax.plot(np.rad2deg(qq))
    ax.axvline(n_out, color='r', ls='--', alpha=0.5, label='cycle boundary')
    ax.axvline(2*n_out, color='r', ls='--', alpha=0.5)
    ax.set_title(name); ax.set_ylabel('deg')
plt.suptitle("3 looped gait cycles — check for discontinuities at red lines")
plt.tight_layout()
plt.savefig("cycle_continuity.png", dpi=100)
print("\nSaved cycle_continuity.png")
