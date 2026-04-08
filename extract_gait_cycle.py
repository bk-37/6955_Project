"""
Extract one clean stride cycle from a single Ulrich baseline trial
and save it as a looping reference for Walker2d imitation.

Detects right heel strike events to find a complete stride cycle,
then saves ~100 frames (one stride at 50Hz ≈ 1s of walking).

Output: gait_cycle_reference.npy  shape (N, 6)
"""
import numpy as np
from pathlib import Path
from ppo_walker2d import load_sto, ULRICH_ROOT

# Use Subject1, first baseline trial
subj_dir = ULRICH_ROOT / "Subject1" / "IK"
trial_dirs = sorted(subj_dir.glob("walking_*baseline*"))
if not trial_dirs:
    trial_dirs = sorted(subj_dir.glob("walking_*"))
trial_dir = trial_dirs[0]
print(f"Using trial: {trial_dir.name}")

ik_path = trial_dir / "output" / "results_ik.sto"
d = load_sto(ik_path)

control_hz = 50.0
orig_hz = 1.0 / (d["time"][1] - d["time"][0])
orig_len = len(d["time"])
new_len = int(orig_len * control_hz / orig_hz)
orig_x = np.arange(orig_len)
new_x = np.linspace(0, orig_len - 1, new_len)

def resamp(key):
    return np.interp(new_x, orig_x, d[key])

hip_r   = -np.deg2rad(resamp("hip_flexion_r"))
knee_r  = -np.deg2rad(resamp("knee_angle_r"))
ankle_r = -np.deg2rad(resamp("ankle_angle_r"))
hip_l   = -np.deg2rad(resamp("hip_flexion_l"))
knee_l  = -np.deg2rad(resamp("knee_angle_l"))
ankle_l = -np.deg2rad(resamp("ankle_angle_l"))

ref = np.stack([hip_r, knee_r, ankle_r, hip_l, knee_l, ankle_l], axis=1).astype(np.float32)

# Detect right heel strike: hip_r crosses from extension (+) to flexion (-)
# i.e., hip_r goes from positive to negative — that's foot strike
# More robustly: find local minima of hip_r (most extended = heel strike)
from scipy.signal import find_peaks

# Right heel strike ≈ when hip_r is at minimum (most extended)
neg_hip = -hip_r  # invert so minima become maxima
peaks, props = find_peaks(neg_hip, distance=30, prominence=0.1)

print(f"Found {len(peaks)} right heel strikes at frames: {peaks[:5]} ...")

if len(peaks) >= 2:
    # Use 2nd to 3rd heel strike (skip first which may be partial)
    start = peaks[1] if len(peaks) > 2 else peaks[0]
    end   = peaks[2] if len(peaks) > 2 else peaks[1]
    cycle = ref[start:end]
    print(f"Stride cycle: frames {start}–{end} = {len(cycle)} frames = {len(cycle)/control_hz:.2f}s")
else:
    # Fallback: just use middle 100 frames
    mid = len(ref) // 2
    cycle = ref[mid:mid+100]
    print(f"Fallback: using frames {mid}–{mid+100}")

# Check joint limits
JNT_LO = np.array([-2.618, -2.618, -0.785, -2.618, -2.618, -0.785])
JNT_HI = np.array([ 0.349,  0.,     0.785,  0.349,  0.,     0.785])
clipped = np.any((cycle < JNT_LO) | (cycle > JNT_HI), axis=1)
print(f"Frames outside joint limits: {clipped.sum()} / {len(cycle)} ({100*clipped.mean():.1f}%)")

print(f"\nJoint ranges in cycle:")
names = ["hip_r", "knee_r", "ankle_r", "hip_l", "knee_l", "ankle_l"]
for i, name in enumerate(names):
    print(f"  {name:10s}: [{np.rad2deg(cycle[:,i].min()):.1f}°, {np.rad2deg(cycle[:,i].max()):.1f}°]")

out = Path("gait_cycle_reference.npy")
np.save(out, cycle)
print(f"\nSaved {len(cycle)}-frame gait cycle → {out}")

import matplotlib.pyplot as plt
t = np.arange(len(cycle)) / control_hz
fig, axes = plt.subplots(3, 2, figsize=(12, 7), sharex=True)
fig.suptitle(f"Single stride cycle — {trial_dir.name}")
for i, (ax, name) in enumerate(zip(axes.flat, names)):
    ax.plot(t, np.rad2deg(cycle[:, i]))
    ax.axhline(np.rad2deg(JNT_LO[i]), color='r', ls='--', alpha=0.4)
    ax.axhline(np.rad2deg(JNT_HI[i]), color='r', ls='--', alpha=0.4)
    ax.set_title(name); ax.set_ylabel("deg")
axes[-1,0].set_xlabel("time (s)")
axes[-1,1].set_xlabel("time (s)")
plt.tight_layout()
plt.savefig("gait_cycle_check.png", dpi=150)
print("Saved → gait_cycle_check.png")
plt.show()
