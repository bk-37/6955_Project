"""
airl_walker2d.py
────────────────
AIRL (Adversarial Inverse Reinforcement Learning) for Walker2d-v4,
treating IK reference transitions as expert s→s' pairs.

Key differences from ppo_walker2d_phase.py:
  - No hand-crafted reward. The discriminator learns r(s, s') from data.
  - Expert buffer: consecutive IK frames are (s_t, s_{t+1}) demonstrations.
  - State for discriminator: [q_joint(6), sin_φ, cos_φ] = 8-dim.
    This is directly comparable between expert and policy without any FK.
  - AIRL discriminator form: g(s,s') = f(s,s') + γ·h(s') - h(s)
    where f is the reward net and h is the shaping potential.
    The shaping term cancels environment dynamics so the recovered reward
    is interpretable and transferable (unlike plain GAIL).

Usage
─────
  python airl_walker2d.py --ref_cycle gait_cycle_reference.npy
  python airl_walker2d.py --ref_cycle gait_cycle_reference.npy --disc_updates 3 --num_envs 16
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import mujoco
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from ppo_walker2d_phase import (Walker2dPhaseAware, load_ref_cycle, CTRL_HZ,
                                 GAIT_CYCLE_FRAMES, compute_bc_dataset, pretrain_bc)

# ── constants ─────────────────────────────────────────────────────────────────

# Obs layout from Walker2dPhaseAware._get_obs():
#   obs[0:17]  = base Walker2d obs  (qpos[1:9] + qvel[0:9])
#     obs[0]   = torso z (height)
#     obs[1]   = torso pitch (qpos[2])
#     obs[2:8] = joint angles  (hip_r, knee_r, ankle_r, hip_l, knee_l, ankle_l)
#     obs[8]   = qvel[0] = x velocity
#     obs[9]   = qvel[1] = z velocity
#     obs[10]  = qvel[2] = pitch velocity
#     obs[11:17]= qvel[3:9] = 6 joint velocities
#   obs[17:23] = q_ref at current phase (6 joints)
#   obs[23:25] = [sin_φ, cos_φ]
_Q_JOINT_SLICE  = slice(2,  8)   # sim joint angles
_DQ_JOINT_SLICE = slice(11, 17)  # sim joint velocities
GAIT_CYCLE_FRAMES = 140          # must match ppo_walker2d_phase.py


# ── state extraction ──────────────────────────────────────────────────────────

def extract_airl_state(obs: np.ndarray, use_joint_vel: bool = True) -> np.ndarray:
    """
    Pull absolute joint state from the full 25-dim obs.

    State = [q_sim(6), (dq_sim(6),)]  — no phase encoding.

    Phase is already captured implicitly by the (s, s') transition structure:
    consecutive reference pairs encode "hip at angle A → hip at angle B" which
    IS the gait sequence. Adding explicit phase is redundant, and in error space
    it made all expert states identical zeros with no transition information.

    Both sides use the same representation (actual angles/velocities), so there
    is no domain gap from comparing kinematic features to simulation features.

    use_joint_vel=True  → 12-dim: [q(6), dq(6)]
    use_joint_vel=False →  6-dim: [q(6)]
    """
    q = obs[..., _Q_JOINT_SLICE]   # (..., 6)
    if use_joint_vel:
        dq = obs[..., _DQ_JOINT_SLICE]  # (..., 6)
        return np.concatenate([q, dq], axis=-1).astype(np.float32)
    return q.astype(np.float32)


def make_expert_buffer(
    reference:       np.ndarray,
    segment_lengths: list[int] | None = None,
    is_cycle:        bool = False,
    use_joint_vel:   bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (s, s_next) expert transition arrays from consecutive IK frames.

    Parameters
    ──────────
    reference       : (T, 6) reference array, already resampled to CTRL_HZ
    segment_lengths : lengths of each concatenated trial. Boundary pairs excluded.
    is_cycle        : include wraparound pair (valid for single extracted cycle only)
    use_joint_vel   : if True, state is [q(6), dq(6), phase(2)] = 14-dim
                      if False, state is [q(6), phase(2)] = 8-dim (ablation)
    """
    T = len(reference)

    # Expert states: absolute joint angles (+ velocities) from the reference.
    # No phase — the (s, s') transition structure captures gait sequence implicitly.
    ref_vel = np.gradient(reference, 1.0 / CTRL_HZ, axis=0).astype(np.float32)

    def ref_state(idx: np.ndarray) -> np.ndarray:
        parts = [reference[idx]]
        if use_joint_vel:
            parts.append(ref_vel[idx])
        return np.concatenate(parts, axis=1).astype(np.float32)

    # Build the set of boundary indices — the last frame of each trial segment.
    # A pair (t, t+1) is invalid when t is a boundary index.
    if segment_lengths is not None:
        boundary_ends = set(np.cumsum(segment_lengths) - 1)
    else:
        boundary_ends = set()

    # All candidate t values: 0 .. T-2 (consecutive pairs within the array)
    all_t = np.arange(T - 1)
    valid_mask = np.array([t not in boundary_ends for t in all_t])
    t_idx = all_t[valid_mask]

    s      = ref_state(t_idx)
    s_next = ref_state(t_idx + 1)

    # Optionally include the wraparound pair for a looping gait cycle.
    if is_cycle:
        wrap_s      = ref_state(np.array([T - 1]))
        wrap_s_next = ref_state(np.array([0]))
        s      = np.concatenate([s,      wrap_s],      axis=0)
        s_next = np.concatenate([s_next, wrap_s_next], axis=0)

    n_dropped = (T - 1) - len(t_idx)
    if n_dropped:
        print(f"Expert buffer: dropped {n_dropped} boundary transition(s) across {len(segment_lengths)} trials")

    return s, s_next


# ── AIRL discriminator ────────────────────────────────────────────────────────

class AIRLDiscriminator(nn.Module):
    """
    AIRL discriminator: D(s, s') = σ( g(s, s') )

    g(s, s') = f(s, s') + γ·h(s') - h(s)

    where
      f : reward network   — takes (s, s') concatenated
      h : shaping potential — takes s alone (like a state-value baseline)

    The shaping term makes the recovered reward invariant to environment
    dynamics so it can be used for transfer / interpretability.
    """

    def __init__(self, state_dim: int = 14, hidden: int = 256, gamma: float = 0.99):
        super().__init__()
        self.gamma = gamma

        self.f = nn.Sequential(
            nn.Linear(state_dim * 2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),        nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.h = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),    nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        """Return g(s, s') logits, shape (..., 1)."""
        f_val = self.f(torch.cat([s, s_next], dim=-1))
        h_s   = self.h(s)
        h_sp  = self.h(s_next)
        return f_val + self.gamma * h_sp - h_s

    @torch.no_grad()
    def reward(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        """
        AIRL reward: r = log D - log(1 - D) = g(s, s')

        The logit IS the reward — no sigmoid needed here.
        Shape: (..., 1).
        """
        return self(s, s_next)


# ── zero-reward env wrapper ───────────────────────────────────────────────────

class Walker2dAIRL(Walker2dPhaseAware):
    """
    Walker2dPhaseAware with the hand-crafted reward replaced by 0.
    AIRL reward is injected by AIRLCallback after each rollout.

    An optional tiny locomotion bonus (+0.1 * x_vel) prevents the agent
    from collapsing to stand-still in the early epochs before the
    discriminator gives a useful gradient. Disable with locomotion_bonus=0.
    """

    def __init__(self, *args, locomotion_bonus: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self._locomotion_bonus = locomotion_bonus

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        # Replace the hand-crafted reward with just a locomotion nudge.
        x_vel = float(info.get("x_velocity", self.data.qvel[0]))
        reward = self._locomotion_bonus * max(x_vel, 0.0)
        return obs, reward, terminated, truncated, info


# ── AIRL callback ─────────────────────────────────────────────────────────────

class AIRLCallback(BaseCallback):
    """
    After each PPO rollout:
      1. Extract (s, s_next) pairs from the rollout buffer (policy transitions).
      2. Sample a matching batch from the expert buffer (IK transitions).
      3. Update the discriminator with binary cross-entropy.
      4. Rewrite the rollout buffer rewards with g(s, s') from the discriminator.
         PPO then optimises the policy against the learned reward.

    Reward normalisation: running mean/std over a window to keep PPO stable.
    """

    def __init__(
        self,
        discriminator:   AIRLDiscriminator,
        expert_s:        np.ndarray,   # (N_E, 8)
        expert_s_next:   np.ndarray,   # (N_E, 8)
        disc_lr:          float = 1e-4,
        disc_updates:     int   = 1,
        label_smoothing:  float = 0.1,
        expert_noise:     float = 0.05,
        grad_penalty:     float = 10.0,
        min_frac_expert:  float = 0.05,
        use_joint_vel:    bool  = True,
        device:           str   = "cpu",
        log_interval:     int   = 50,
    ):
        super().__init__(verbose=0)
        self.disc         = discriminator.to(device)
        self.expert_s     = torch.tensor(expert_s,      device=device)
        self.expert_s_nxt = torch.tensor(expert_s_next, device=device)
        self.opt          = optim.Adam(discriminator.parameters(), lr=disc_lr)
        self.disc_updates     = disc_updates
        self.label_smoothing  = label_smoothing
        self.expert_noise     = expert_noise
        self.grad_penalty     = grad_penalty
        self.min_frac_expert  = min_frac_expert
        self.use_joint_vel    = use_joint_vel
        self.device           = device
        self._disc_frozen     = False
        self.log_interval = log_interval

        # Running reward normalisation (window = last 10k samples)
        self._rew_buf: list[float] = []
        self._rollout = 0

        # Logging
        self._disc_losses: list[float] = []
        self._raw_stats:   list[tuple] = []  # (raw_mean, raw_std, frac_pos) per rollout

    # ── helper: sample expert batch ───────────────────────────────────────────

    def _sample_expert(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, len(self.expert_s), (n,))
        return self.expert_s[idx], self.expert_s_nxt[idx]

    # ── helper: extract policy (s, s') from rollout buffer ────────────────────

    def _extract_policy_transitions(self):
        """
        Pull (s, s_next, is_terminal) from the SB3 rollout buffer.

        Buffer layout (after collect_rollouts):
          buffer.observations : (n_steps, n_envs, obs_dim) np.ndarray
          buffer.episode_starts: (n_steps, n_envs) — True at the first step
                                  of a new episode (i.e., after a terminal).

        s[t]     = observations[t]
        s_next[t]= observations[t+1]  — valid only when not a terminal step
        Terminal steps (where dones[t]=True) are masked out because
        s_next there belongs to a different episode.
        """
        buf  = self.model.rollout_buffer
        obs  = buf.observations   # (T, N, obs_dim)
        T, _, D = obs.shape

        # episode_starts[t] is True when step t begins a new episode.
        # A terminal at step t means episode_starts[t+1] is True.
        ep_starts = buf.episode_starts  # (T, N) bool

        # Build s[t], s_next[t] for t in [0, T-2] — skip last row (no next)
        s     = obs[:T-1].reshape(-1, D)          # ((T-1)*N, D)
        s_nxt = obs[1:T ].reshape(-1, D)          # ((T-1)*N, D)

        # Mask: True where transition is valid (s_next is in the same episode)
        # episode_starts[t+1] = True means a terminal happened at step t.
        valid = ~ep_starts[1:T].reshape(-1).astype(bool)  # ((T-1)*N,) bool

        s     = s[valid]
        s_nxt = s_nxt[valid]

        s_feat     = extract_airl_state(s,     self.use_joint_vel)
        s_nxt_feat = extract_airl_state(s_nxt, self.use_joint_vel)

        return (
            torch.tensor(s_feat,     device=self.device),
            torch.tensor(s_nxt_feat, device=self.device),
            valid,          # for rewriting rewards back into the full buffer
        )

    # ── discriminator update ──────────────────────────────────────────────────

    def _update_discriminator(
        self,
        pol_s:   torch.Tensor,
        pol_snxt:torch.Tensor,
    ) -> float:
        """
        Binary cross-entropy:
          expert  → label 1  (should look like demonstrations)
          policy  → label 0

        Loss = -E_E[log σ(g)] - E_π[log(1 - σ(g))]
             = -E_E[log σ(g)] - E_π[log σ(-g)]
        """
        bce        = nn.BCEWithLogitsLoss()
        smooth     = self.label_smoothing
        total_loss = 0.0

        for _ in range(self.disc_updates):
            n     = len(pol_s)
            e_s, e_snxt = self._sample_expert(n)

            # Expert noise augmentation: blur the expert manifold so the
            # discriminator can't cleanly separate near-expert policy transitions.
            # Applied only during training, not during reward computation.
            # Same noise offset for s and s' to preserve transition structure.
            if self.expert_noise > 0.0:
                noise   = self.expert_noise * torch.randn_like(e_s)
                e_s     = e_s     + noise
                e_snxt  = e_snxt  + noise

            g_expert = self.disc(e_s, e_snxt).squeeze(-1)
            g_policy = self.disc(pol_s, pol_snxt).squeeze(-1)

            expert_labels = torch.full_like(g_expert, 1.0 - smooth)
            policy_labels = torch.full_like(g_policy, smooth)
            loss = bce(g_expert, expert_labels) + bce(g_policy, policy_labels)

            # Gradient penalty (WGAN-GP): penalise large gradient norms on
            # interpolated expert-policy samples. This constrains the discriminator's
            # Lipschitz constant, preventing sharp decision boundaries that cause
            # frac_expert to collapse before the policy can adapt.
            if self.grad_penalty > 0.0:
                alpha   = torch.rand(n, 1, device=self.device)
                interp_s    = (alpha * e_s    + (1 - alpha) * pol_s   ).requires_grad_(True)
                interp_snxt = (alpha * e_snxt + (1 - alpha) * pol_snxt).requires_grad_(True)
                g_interp = self.disc(interp_s, interp_snxt).squeeze(-1)
                grads = torch.autograd.grad(
                    g_interp.sum(), interp_s,
                    create_graph=True, retain_graph=True,
                )[0]
                gp   = ((grads.norm(2, dim=1) - 1) ** 2).mean()
                loss = loss + self.grad_penalty * gp

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            total_loss += loss.item()

        return total_loss / self.disc_updates

    # ── reward rewriting ──────────────────────────────────────────────────────

    def _rewrite_rewards(
        self,
        pol_s:     torch.Tensor,
        pol_snxt:  torch.Tensor,
        valid:     np.ndarray,
    ) -> None:
        """
        Replace env rewards in the rollout buffer with g(s, s') from discriminator.

        Only valid (non-terminal) transitions get AIRL reward.
        Terminal transitions keep 0 (they already lost the episode bonus).

        Also applies running-mean normalisation so PPO reward scale is stable.
        """
        buf = self.model.rollout_buffer
        T, n_envs, _ = buf.observations.shape

        # Compute AIRL rewards for valid transitions
        with torch.no_grad():
            airl_r = self.disc.reward(pol_s, pol_snxt).squeeze(-1).cpu().numpy()

        # Log raw stats BEFORE normalisation — these are the diagnostically useful values.
        #   raw_mean: where the discriminator scores policy transitions on average
        #             (~-2.2 = perfect separation, ~0 = uncertain)
        #   raw_std:  variance in the reward signal across transitions
        #             (~0 = flat/useless, >0.5 = informative gradient)
        #   frac_pos: fraction of policy transitions the disc thinks look expert-like
        #             (0 = perfect separation, ~0.5 = disc confused = good signal)
        raw_mean = float(np.mean(airl_r))
        raw_std  = float(np.std(airl_r))
        frac_pos = float(np.mean(airl_r > 0))
        self._raw_stats.append((raw_mean, raw_std, frac_pos))

        # Normalise
        self._rew_buf.extend(airl_r.tolist())
        if len(self._rew_buf) > 50_000:
            self._rew_buf = self._rew_buf[-50_000:]
        if len(self._rew_buf) > 1:
            mu  = float(np.mean(self._rew_buf))
            std = float(np.std(self._rew_buf)) + 1e-8
            airl_r = (airl_r - mu) / std

        # Scatter back into the buffer's flat reward array.
        # buf.rewards is (T, n_envs) — we wrote (T-1)*n_envs rows (valid subset).
        full_rewards = np.zeros((T - 1) * n_envs, dtype=np.float32)
        full_rewards[valid] = airl_r

        # The last step's reward (row T-1) stays 0 (no s_next available).
        buf.rewards[:T-1] = full_rewards.reshape(T - 1, n_envs)
        buf.rewards[T-1]  = 0.0

    # ── callback hooks ────────────────────────────────────────────────────────

    def _on_rollout_end(self) -> None:
        self._rollout += 1

        pol_s, pol_snxt, valid = self._extract_policy_transitions()

        # Adaptive discriminator training: check frac_expert on the current
        # policy transitions BEFORE updating. If it's already below the floor,
        # freeze the discriminator and let the policy catch up first.
        # This directly prevents the race condition where the discriminator
        # achieves perfect separation before the policy can improve.
        with torch.no_grad():
            g_now     = self.disc(pol_s, pol_snxt).squeeze(-1)
            frac_now  = float((g_now > 0).float().mean())

        if frac_now > self.min_frac_expert:
            disc_loss = self._update_discriminator(pol_s, pol_snxt)
            self._disc_frozen = False
        else:
            disc_loss = float('nan')   # frozen — reward computed from current disc
            self._disc_frozen = True

        self._disc_losses.append(disc_loss if not self._disc_frozen else 0.0)

        self._rewrite_rewards(pol_s, pol_snxt, valid)

        if self._rollout % self.log_interval == 0:
            recent       = self._raw_stats[-self.log_interval:]
            avg_loss     = float(np.mean(self._disc_losses[-self.log_interval:]))
            avg_raw_mean = float(np.mean([s[0] for s in recent]))
            avg_raw_std  = float(np.mean([s[1] for s in recent]))
            avg_frac_pos = float(np.mean([s[2] for s in recent]))
            frozen_str = "  [DISC FROZEN]" if self._disc_frozen else ""
            print(
                f"[AIRL iter {self._rollout:5d} | steps {self.num_timesteps:>9,}]  "
                f"disc_loss={avg_loss:.4f}  "
                f"raw_r={avg_raw_mean:+.3f}±{avg_raw_std:.3f}  "
                f"frac_expert={avg_frac_pos:.3f}"
                f"{frozen_str}"
            )

    def _on_step(self) -> bool:
        return True


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AIRL Walker2d from IK reference")

    ref_grp = parser.add_mutually_exclusive_group(required=True)
    ref_grp.add_argument("--ref_cycle", type=str,
                         help="Path to single gait-cycle .npy")
    ref_grp.add_argument("--ref_all",   action="store_true",
                         help="Use full concatenated Ulrich reference (~400k transitions)")
    parser.add_argument("--subjects",      type=int, nargs="+", default=None)
    parser.add_argument("--trial_filter",  type=str, default=None)
    parser.add_argument("--num_envs",      type=int,   default=32)
    parser.add_argument("--total_steps",   type=float, default=5e6)
    parser.add_argument("--device",        default="cpu")
    parser.add_argument("--disc_lr",       type=float, default=3e-4)
    parser.add_argument("--disc_updates",  type=int,   default=5,
                        help="Discriminator gradient steps per PPO rollout")
    parser.add_argument("--disc_hidden",   type=int,   default=256)
    parser.add_argument("--label_smooth",  type=float, default=0.1,
                        help="Label smoothing — prevents discriminator saturating to ±inf")
    parser.add_argument("--expert_noise",  type=float, default=0.05,
                        help="Std of Gaussian noise added to expert states during disc training.")
    parser.add_argument("--grad_penalty",     type=float, default=10.0,
                        help="WGAN-GP gradient penalty coefficient. Set 0 to disable.")
    parser.add_argument("--min_frac_expert",  type=float, default=0.05,
                        help="Discriminator is frozen when frac_expert drops below this floor. "
                             "Forces the policy to catch up before the discriminator trains further.")
    parser.add_argument("--gamma",         type=float, default=0.99)
    parser.add_argument("--loco_bonus",    type=float, default=0.05,
                        help="Small forward-velocity bonus before discriminator warms up. Set 0 for pure AIRL.")
    parser.add_argument("--no_joint_vel",  action="store_true",
                        help="Ablation: use only joint positions + phase (8-dim), drop velocities")
    parser.add_argument("--bc_epochs", type=int, default=0,
                        help="BC warm-start epochs before AIRL (PD-rollout targets).")
    parser.add_argument("--bc_steps",  type=int, default=200_000,
                        help="PD rollout steps to collect for BC dataset (default 200k)")
    parser.add_argument("--bc_kp",     type=float, default=200.0)
    parser.add_argument("--bc_kd",     type=float, default=20.0)
    parser.add_argument("--finetune",      type=str, default=None,
                        help="Path to PPO .zip checkpoint to warm-start the policy from. "
                             "Recommended: AIRL from scratch fails because the discriminator "
                             "achieves perfect separation before the policy learns to walk. "
                             "A pre-trained walking policy generates transitions that overlap "
                             "with expert states, giving the discriminator a harder job and "
                             "the policy a useful reward gradient.")
    parser.add_argument("--scale_model",   action="store_true")
    parser.add_argument("--out_dir",       default=None)
    args = parser.parse_args()

    # ── load reference ────────────────────────────────────────────────
    segment_lengths = None
    if args.ref_cycle:
        reference = load_ref_cycle(Path(args.ref_cycle))
        is_cycle  = True   # clean loop — wraparound pair is valid
    else:
        from ppo_walker2d import load_ulrich_reference
        print("Loading full Ulrich reference...")
        reference, segment_lengths = load_ulrich_reference(
            subjects        = args.subjects,
            trial_filter    = args.trial_filter,
            control_hz      = CTRL_HZ,
            return_lengths  = True,
        )
        is_cycle = False   # concatenated trials — never wrap around
    print(f"Reference: {reference.shape}  ({len(reference)/CTRL_HZ:.1f}s @ {CTRL_HZ}Hz)")

    # ── build expert buffer ───────────────────────────────────────────
    use_joint_vel = not args.no_joint_vel
    expert_s, expert_s_next = make_expert_buffer(
        reference,
        segment_lengths = segment_lengths,
        is_cycle        = is_cycle,
        use_joint_vel   = use_joint_vel,
    )
    print(f"Expert transitions: {len(expert_s):,}  (state dim={expert_s.shape[1]})")

    # ── output dir ────────────────────────────────────────────────────
    stamp   = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = PROJECT_ROOT / (args.out_dir or f"results/walker2d_airl_{stamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {log_dir}")
    np.save(log_dir / "reference.npy", reference)

    # ── env factory ───────────────────────────────────────────────────
    xml_file = "walker2d_subject1.xml" if args.scale_model else "walker2d.xml"

    def make_env():
        def _init():
            return Walker2dAIRL(
                reference        = reference,
                xml_file         = xml_file,
                locomotion_bonus = args.loco_bonus,
                # Reward weights don't matter — env reward is overwritten.
                # Keep phase tracking / termination logic intact.
                imitation_weight = 0.0,
                vel_weight       = 0.0,
                ee_weight        = 0.0,
                root_weight      = 0.0,
                contact_weight   = 0.0,
                swing_pen_weight = 0.0,
            )
        return _init

    env = SubprocVecEnv([make_env() for _ in range(args.num_envs)])
    env = VecMonitor(env)

    # ── discriminator + callback ──────────────────────────────────────
    state_dim = expert_s.shape[1]  # 14 with vels, 8 without
    disc = AIRLDiscriminator(
        state_dim = state_dim,
        hidden    = args.disc_hidden,
        gamma     = args.gamma,
    )

    airl_cb = AIRLCallback(
        discriminator    = disc,
        expert_s         = expert_s,
        expert_s_next    = expert_s_next,
        disc_lr          = args.disc_lr,
        disc_updates     = args.disc_updates,
        label_smoothing  = args.label_smooth,
        expert_noise     = args.expert_noise,
        grad_penalty     = args.grad_penalty,
        min_frac_expert  = args.min_frac_expert,
        use_joint_vel    = use_joint_vel,
        device           = args.device,
    )

    # ── PPO ───────────────────────────────────────────────────────────
    if args.finetune:
        path = str(Path(args.finetune).with_suffix(""))
        print(f"Warm-starting policy from: {path}")
        model = PPO.load(path, env=env, device=args.device)
        model.learning_rate = 3e-5
        model.ent_coef      = 0.0
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate = 1e-4,
            n_steps       = 512,
            batch_size    = 4096,
            n_epochs      = 10,
            gamma         = args.gamma,
            gae_lambda    = 0.95,
            clip_range    = 0.2,
            ent_coef      = 0.001,
            vf_coef       = 0.5,
            max_grad_norm = 0.5,
            target_kl     = 0.02,
            device        = args.device,
            policy_kwargs = {"net_arch": [256, 256]},
            verbose       = 0,
        )

    # ── BC warm-start ─────────────────────────────────────────────────
    if args.bc_epochs > 0 and not args.finetune:
        tmp_env = Walker2dAIRL(reference=reference, xml_file=xml_file)
        print(f"Collecting PD rollout dataset ({args.bc_steps:,} steps)...")
        obs_bc, act_bc = compute_bc_dataset(tmp_env, n_steps=args.bc_steps,
                                             kp=args.bc_kp, kd=args.bc_kd)
        tmp_env.close()
        print(f"  collected {len(obs_bc):,} pairs  "
              f"action range [{act_bc.min():.2f}, {act_bc.max():.2f}]")
        pretrain_bc(model, obs_bc, act_bc, n_epochs=args.bc_epochs)

    checkpoint_cb = CheckpointCallback(
        save_freq   = max(5_000_000 // args.num_envs, 1),
        save_path   = str(log_dir / "checkpoints"),
        name_prefix = "model",
        verbose     = 0,
    )

    print(f"Training AIRL for {int(args.total_steps):,} steps / {args.num_envs} envs...")
    model.learn(
        total_timesteps = int(args.total_steps),
        callback        = CallbackList([airl_cb, checkpoint_cb]),
        progress_bar    = True,
    )
    env.close()

    save_path = str(log_dir / "model")
    model.save(save_path)
    torch.save(disc.state_dict(), str(log_dir / "discriminator.pt"))
    print(f"Policy saved  → {save_path}.zip")
    print(f"Discriminator → {log_dir}/discriminator.pt")


if __name__ == "__main__":
    main()
