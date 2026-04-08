# Walker2d Run Log

---

## Background: why the first approach failed

The earliest runs (`walker2d_ulrich_all_*`) used `ppo_walker2d.py`, which fed the Ulrich
IK reference directly as a joint-angle imitation target. These all failed to produce walking
and are archived as a single representative example below.

Three compounding problems caused the failure:

1. **2.5× speed mismatch** — the Ulrich reference is at 50 Hz but Walker2d runs at 125 Hz
   (frame_skip=4). The reference was not resampled, so the gait cycle played out 2.5× too
   fast. The policy was chasing joint targets that changed faster than it could respond.

2. **Phase blindness** — the policy observation contained no information about where in the
   gait cycle the agent currently was. It could only learn an average response, which
   collapses to a partial-extension of the stance leg regardless of context.

3. **Full concatenated reference** — 413k frames from many trials concatenated end-to-end,
   creating discontinuous jumps at trial boundaries that corrupted the reward signal.

The result: the agent kicks one leg back (partial imitation reward for hip extension) and
falls over. The reward gradient is too noisy to climb toward actual walking.

**Representative failure — `walker2d_ulrich_all_20260406-221644`**
*(last and most developed run before switching approach)*
```bash
python render_walker.py --model results/walker2d_ulrich_all_20260406-221644/model.zip --steps 500
```

The symmetry reward-shaping experiments (`pretrain_walker2d.py`) were a detour to see if
walking could be bootstrapped without any reference, but hit their own local optima (hopping,
ankle-paddling, standing). The actual fix was `ppo_walker2d_phase.py`, which resampled the
reference to 125 Hz, added phase encoding to the observation, and used adaptive phase
tracking — producing the first walking policy (`walker2d_phase_cycle_20260408-115434`).

---

## Symmetry pretrain runs

Behavioral summary of the four keeper checkpoints from the `pretrain_walker2d.py`
symmetry-mode reward shaping experiments. All runs used PPO on `Walker2d-v4` with
the `Walker2dContactWalk` symmetry wrapper (no reference data required).

---

## Common setup

```bash
conda activate OpenCap_RL
cd c:/Users/bakel/Desktop/6955_Project
```

Render any checkpoint:
```bash
python render_walker.py --model results/walker2d_pretrain_symmetry_<timestamp>/model.zip --vanilla --steps 500
```

---

## Run 1 — Two-legged hopping with some swing
**Directory:** `results/walker2d_pretrain_symmetry_20260407-111838/`

**Behavior:** Bilateral hopping driven by ankle plantarflexion. Both feet leave the ground
simultaneously. Some hip/knee swing visible — first run to break out of one-legged hopping
into a symmetric (if still airborne) pattern. The hip anti-phase and hip ROM rewards were
being collected, but symmetry weight was too weak to overcome the hopping local optimum.

**Reward config:** `--mode symmetry --weight 3.0` (default), `forward_reward_weight=0.5`,
gravity = 2× normal (−19.62 m/s²), ankle torques hard-capped at ±0.3

**Training curve:**
| Steps | ep_r | ep_len |
|-------|------|--------|
| 327k  | 82.6 | 94     |
| 983k  | 300  | 198    |
| 1.97M | 430  | 226    |
| 3.28M | 1155 | 411    |
| 5.0M  | 1611 | 490    |

**Reproduce:**
```bash
python pretrain_walker2d.py --mode symmetry --num_envs 32 --total_steps 5e6
```

**Render:**
```bash
python render_walker.py --model results/walker2d_pretrain_symmetry_20260407-111838/model.zip --vanilla --steps 500
```

---

## Run 2 — One-legged hopping, back leg stabilizing
**Directory:** `results/walker2d_pretrain_symmetry_20260407-114136/`

**Behavior:** Strong one-legged hop: the dominant leg does all propulsion while the trailing
leg makes occasional stabilizing ground contacts but carries no load. Much higher return
(~3300) than Run 1 because the asymmetric strategy is more reward-efficient — the agent
maximizes forward velocity (high forward reward) without paying the symmetry penalty enough
to change strategy.

**Reward config:** Same as Run 1 — `--mode symmetry --weight 3.0`, same gravity/ankle cap.
Identical hyperparameters; different random seed led to a different local optimum.

**Training curve:**
| Steps | ep_r | ep_len |
|-------|------|--------|
| 327k  | 90.3 | 96     |
| 1.97M | 610  | 267    |
| 2.95M | 2026 | 698    |
| 3.93M | 2795 | 807    |
| 5.0M  | 3271 | 865    |

**Reproduce:**
```bash
python pretrain_walker2d.py --mode symmetry --num_envs 32 --total_steps 5e6
```
*(Same command as Run 1 — outcome varies by seed)*

**Render:**
```bash
python render_walker.py --model results/walker2d_pretrain_symmetry_20260407-114136/model.zip --vanilla --steps 500
```

---

## Run 3 — Ankle paddling
**Directory:** `results/walker2d_pretrain_symmetry_20260407-172719/`

**Behavior:** Legs stay low to the ground; hips and knees show some alternating motion but
all forward propulsion comes from ankle plantarflexion pushing backward against the ground
like paddles. No aerial phase — the increased gravity (2.5×) and hop suppression penalty
made leaving the ground too costly, so the agent found a flat-ground ankle-shuffle instead.
Fastest convergence of any run (ep_len ~850 by just 983k steps).

**Reward config:** `--mode symmetry --weight 8.0`, gravity = 2.5× normal (−24.525 m/s²),
ankle torques hard-capped at ±0.3, `forward_reward_weight=0.5`

**Training curve:**
| Steps | ep_r | ep_len |
|-------|------|--------|
| 327k  | 76.9 | 79     |
| 655k  | 351  | 343    |
| 983k  | 904  | 850    |
| 1.64M | 1075 | 891    |
| 3.28M | 1281 | 942    |
| 5.0M  | ~1280 | ~960  |

**Reproduce:**
```bash
python pretrain_walker2d.py --mode symmetry --num_envs 32 --total_steps 5e6 --weight 8.0
```

**Render:**
```bash
python render_walker.py --model results/walker2d_pretrain_symmetry_20260407-172719/model.zip --vanilla --steps 500
```

---

## Run 4 — Stands still and taps feet
**Directory:** `results/walker2d_pretrain_symmetry_20260408-110759/`

**Behavior:** Near-stationary policy: agent balances upright and alternates foot contacts
without meaningfully translating forward. The reduced forward velocity incentive (relative
to symmetry/height rewards) meant standing and tapping collected more reward per unit effort
than actually locomoting. A "reward hacking" equilibrium — maximizes height reward
(stays near z=1.25m), collects bilateral contact reward, avoids any risky movement.

**Reward config:** `--mode symmetry --weight 8.0`, gravity = 2.5× normal, ankle cap ±0.3.
Forward reward further reduced vs Run 3 during mid-session tuning.

**Training curve:**
| Steps | ep_r | ep_len |
|-------|------|--------|
| 327k  | 92.8 | 74     |
| 655k  | 177  | 108    |
| 1.64M | 872  | 738    |
| 1.97M | 1099 | 950    |
| 3.28M | 1138 | 969    |
| 5.0M  | ~1140 | ~970  |

**Reproduce:**
```bash
python pretrain_walker2d.py --mode symmetry --num_envs 32 --total_steps 5e6 --weight 8.0
```
*(Same command as Run 3 — the forward_reward_weight was reduced interactively during this
session; exact value not logged. To reproduce the standing behavior, try lowering it further:)*
```bash
python pretrain_walker2d.py --mode symmetry --num_envs 32 --total_steps 5e6 --weight 8.0
```
*(Edit `forward_reward_weight=0.1` in `pretrain_walker2d.py:make_env()` to reproduce)*

**Render:**
```bash
python render_walker.py --model results/walker2d_pretrain_symmetry_20260408-110759/model.zip --vanilla --steps 500
```

---

## Cleanup

Delete intermediate checkpoints (keep only `model.zip` for each keeper run):
```bash
rm -rf results/walker2d_pretrain_symmetry_20260407-111838/checkpoints
rm -rf results/walker2d_pretrain_symmetry_20260407-114136/checkpoints
rm -rf results/walker2d_pretrain_symmetry_20260407-172719/checkpoints
rm -rf results/walker2d_pretrain_symmetry_20260408-110759/checkpoints
```

Delete all other pretrain_symmetry runs:
```bash
for d in results/walker2d_pretrain_symmetry_*/; do
  case "$d" in
    *20260407-111838*|*20260407-114136*|*20260407-172719*|*20260408-110759*) ;;
    *) rm -rf "$d" ;;
  esac
done
```
