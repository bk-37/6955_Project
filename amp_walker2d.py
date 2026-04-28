"""
amp_walker2d.py
───────────────
Adversarial Motion Priors (AMP) for Walker2d-v4.

Based on: Escontrela et al., "Adversarial Motion Priors Make Good
Substitutes for Complex Reward Functions" (ICRA 2022).

Key differences from airl_walker2d.py:
  - LSGAN discriminator loss (Eq. 3): (D-1)² + (D+1)² — minimises Pearson
    χ² divergence between reference and policy transition distributions.
    More stable than BCE/KL used in AIRL.
  - No AIRL shaping potential (f + γh' - h). Simpler D(s,s') architecture.
  - AMP-style bounded reward (Eq. 4): max(0, 1 - 0.25*(D-1)²) ∈ [0,1].
    No running normalisation required.
  - Task reward is NOT zeroed — combined as r = w_g*r_task + w_s*r_style.
    This prevents the cold-start collapse your AIRL hits without --finetune.
  - Gradient penalty applied to EXPERT samples only (zero-centered GP),
    not interpolated samples (WGAN-GP). Penalises non-zero gradients on
    the data manifold to prevent the generator overshooting.

Expert state representation: [q_joint(6), dq_joint(6)] = 12-dim.
Reuses make_expert_buffer / extract_airl_state from airl_walker2d.py.

Usage
─────
  # From scratch (task reward prevents cold-start collapse)
  python amp_walker2d.py --ref_cycle gait_cycle_reference.npy

  # Warm-start from a walking policy (recommended for fast convergence)
  python amp_walker2d.py --ref_cycle gait_cycle_reference.npy \\
      --finetune results/walker2d_phase_cycle_*/model.zip

  # GPU training
  python amp_walker2d.py --ref_cycle gait_cycle_reference.npy \\
      --num_envs 32 --device cuda

  # Ablation: discriminator on joint positions only (no velocities)
  python amp_walker2d.py --ref_cycle gait_cycle_reference.npy --no_joint_vel
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from ppo_walker2d_phase import (
    Walker2dPhaseAware, load_ref_cycle, CTRL_HZ,
    compute_bc_dataset, pretrain_bc, LogCallback,
)
from airl_walker2d import extract_airl_state, make_expert_buffer


# ── AMP discriminator ─────────────────────────────────────────────────────────

class AMPDiscriminator(nn.Module):
    """
    AMP discriminator: D(s, s') → scalar.

    Trained with LSGAN (Escontrela et al. Eq. 3):
      L = E_D[(D(s,s')-1)²] + E_π[(D(s,s')+1)²]
        + w_gp/2 * E_D[||∇_φ D(s,s')||²]

    The LSGAN formulation minimises Pearson χ² divergence.
    No shaping potential — unlike AIRL this is not trying to recover a
    transferable reward; it only needs to be a useful style signal.

    Architecture matches the paper: [1024, 512] for the discriminator.
    ELU activations (as used in the AMP paper).
    """

    def __init__(self, state_dim: int = 12, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden), nn.ELU(),
            nn.Linear(hidden, hidden // 2),   nn.ELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        """Return D(s, s') logit. Shape: (..., 1)."""
        return self.net(torch.cat([s, s_next], dim=-1))

    @torch.no_grad()
    def style_reward(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        """
        AMP style reward (Escontrela et al. Eq. 4):
          r_s = max(0, 1 - 0.25*(D(s,s') - 1)²) ∈ [0, 1]

        When D ≈ +1 (expert-like), r_s ≈ 1.
        When D ≈ -1 (policy-like), r_s = max(0, 1 - 0.25*4) = 0.
        The reward is naturally bounded without running normalisation.
        """
        d = self(s, s_next)
        return torch.clamp(1.0 - 0.25 * (d - 1.0) ** 2, min=0.0)


# ── task-reward-only env ──────────────────────────────────────────────────────

class Walker2dAMP(Walker2dPhaseAware):
    """
    Walker2dPhaseAware with the imitation reward replaced by a simple
    under-specified task reward. The style reward is injected by
    AMPCallback after each rollout.

    Task reward (analogous to velocity tracking in Escontrela et al. Eq. 1):
      r_g = exp(-5 * (x_vel - v_target)²)

    Without the AMP style reward, training with only r_g produces unnatural
    locomotion (exactly the motivating problem in the paper). The discriminator
    provides the naturalness signal, making the complex imitation reward
    in ppo_walker2d_phase.py unnecessary.

    Termination and phase logic from Walker2dPhaseAware are preserved.
    """

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        x_vel  = float(info.get("x_velocity", self.data.qvel[0]))
        task_r = float(np.exp(-5.0 * (x_vel - self._v_target) ** 2))
        return obs, task_r, terminated, truncated, info


# ── AMP callback ──────────────────────────────────────────────────────────────

class AMPCallback(BaseCallback):
    """
    After each PPO rollout:
      1. Extract policy (s, s') transitions from the rollout buffer.
      2. Update the discriminator with LSGAN + zero-centered GP.
      3. Compute style rewards: r_s = max(0, 1 - 0.25*(D-1)²).
      4. Rewrite buffer rewards: r = w_g * r_task + w_s * r_style.
         PPO then optimises the policy against the combined reward.

    The key difference from AIRLCallback: task rewards are SCALED (not
    zeroed). This keeps a meaningful gradient alive from step 1, so
    from-scratch training is feasible without a warm-start policy.
    """

    def __init__(
        self,
        discriminator:  AMPDiscriminator,
        expert_s:       np.ndarray,    # (N_E, state_dim)
        expert_s_next:  np.ndarray,    # (N_E, state_dim)
        w_task:         float = 0.35,  # paper: w_g = 0.35
        w_style:        float = 0.65,  # paper: w_s = 0.65
        disc_lr:        float = 1e-4,
        disc_updates:   int   = 1,
        disc_batch_size: int  = 4096,  # max samples per discriminator update step
        grad_penalty:   float = 10.0,  # paper: w_gp = 10
        expert_noise:   float = 0.05,  # Gaussian noise std on expert transitions during disc training.
                                       # Prevents memorisation of the small expert buffer (140 frames)
                                       # by blurring the expert manifold. Same noise applied to s and
                                       # s' to preserve transition structure (borrowed from airl_walker2d).
        use_joint_vel:  bool  = True,
        device:         str   = "cpu",
        log_interval:   int   = 50,
    ):
        super().__init__(verbose=0)
        self.disc          = discriminator.to(device)
        self.expert_s      = torch.tensor(expert_s,      device=device)
        self.expert_s_nxt  = torch.tensor(expert_s_next, device=device)
        self.w_task        = w_task
        self.w_style       = w_style
        self.opt           = optim.Adam(discriminator.parameters(), lr=disc_lr)
        self.disc_updates  = disc_updates
        self.expert_noise  = expert_noise
        self.grad_penalty  = grad_penalty
        self.use_joint_vel = use_joint_vel
        self.device        = device
        self.log_interval  = log_interval

        self.disc_batch_size = disc_batch_size
        self._rollout      = 0
        self._disc_losses:  list[float] = []
        self._style_stats:  list[tuple] = []  # (mean, std) of style_r per rollout

    # ── helpers ───────────────────────────────────────────────────────────────

    def _sample_expert(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, len(self.expert_s), (n,))
        return self.expert_s[idx], self.expert_s_nxt[idx]

    def _extract_policy_transitions(self):
        """
        Pull (s, s_next, valid_mask) from the SB3 rollout buffer.
        Masks out terminal transitions (where s_next belongs to a new episode).
        """
        buf = self.model.rollout_buffer
        obs = buf.observations        # (T, N_envs, obs_dim)
        T, _, D = obs.shape
        ep_starts = buf.episode_starts  # (T, N_envs) bool

        s     = obs[:T-1].reshape(-1, D)
        s_nxt = obs[1:T ].reshape(-1, D)
        valid = ~ep_starts[1:T].reshape(-1).astype(bool)

        s_feat     = extract_airl_state(s[valid],     self.use_joint_vel)
        s_nxt_feat = extract_airl_state(s_nxt[valid], self.use_joint_vel)

        return (
            torch.tensor(s_feat,     device=self.device),
            torch.tensor(s_nxt_feat, device=self.device),
            valid,
        )

    # ── discriminator update ──────────────────────────────────────────────────

    def _update_discriminator(
        self,
        pol_s:    torch.Tensor,
        pol_snxt: torch.Tensor,
    ) -> float:
        """
        LSGAN objective (Escontrela et al. Eq. 3):
          L = E_D[(D(s,s')-1)²] + E_π[(D(s,s')+1)²]
            + w_gp/2 * E_D[||∇_φ D(s,s')||²]

        Gradient penalty is on EXPERT (real) samples only — zero-centered GP.
        This differs from WGAN-GP (interpolated samples) and is what the paper
        specifies: "penalises nonzero gradients on the manifold of real data
        samples" to prevent the generator overshooting off the data manifold.

        Processed in mini-batches of disc_batch_size to avoid creating a huge
        computation graph (with create_graph=True) on the full rollout buffer,
        which caused a C-level crash on CPU with ~16K samples per rollout.
        """
        total_loss = 0.0
        for _ in range(self.disc_updates):
            # Subsample policy transitions to disc_batch_size
            n = min(len(pol_s), self.disc_batch_size)
            idx = torch.randperm(len(pol_s), device=self.device)[:n]
            pol_s_b   = pol_s[idx]
            pol_snxt_b = pol_snxt[idx]

            # Sample matching expert batch
            e_s, e_snxt = self._sample_expert(n)

            # Expert noise: blur the expert manifold so the discriminator
            # can't memorise the 140 exact expert transitions. Same offset
            # applied to s and s' to preserve transition structure.
            if self.expert_noise > 0.0:
                noise  = self.expert_noise * torch.randn_like(e_s)
                e_s    = e_s    + noise
                e_snxt = e_snxt + noise

            d_expert = self.disc(e_s,       e_snxt   ).squeeze(-1)
            d_policy = self.disc(pol_s_b,   pol_snxt_b).squeeze(-1)

            # LSGAN losses
            loss = ((d_expert - 1.0) ** 2).mean() + ((d_policy + 1.0) ** 2).mean()

            # Zero-centered gradient penalty on expert samples
            if self.grad_penalty > 0.0:
                e_s_gp  = e_s.detach().requires_grad_(True)
                e_sn_gp = e_snxt.detach().requires_grad_(True)
                d_gp = self.disc(e_s_gp, e_sn_gp).squeeze(-1)
                grads = torch.autograd.grad(
                    d_gp.sum(), e_s_gp,
                    create_graph=True,
                )[0]
                gp   = (grads.norm(2, dim=1) ** 2).mean()
                loss = loss + (self.grad_penalty / 2.0) * gp

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            total_loss += loss.item()

        return total_loss / self.disc_updates

    # ── reward rewriting ──────────────────────────────────────────────────────

    def _rewrite_rewards(
        self,
        pol_s:    torch.Tensor,
        pol_snxt: torch.Tensor,
        valid:    np.ndarray,
    ) -> None:
        """
        Combine task reward (already in buffer from env) with AMP style reward.

          r = w_g * r_task + w_s * r_style

        Only valid (non-terminal, non-last-step) transitions get style reward.
        All transitions get their task reward scaled by w_g.

        This is the key advantage over AIRL: the task reward stays alive
        throughout training. The discriminator doesn't need to warm up before
        the policy gets a useful gradient.
        """
        buf = self.model.rollout_buffer
        T, n_envs, _ = buf.observations.shape

        # Compute style rewards for valid transitions
        style_r = self.disc.style_reward(pol_s, pol_snxt).squeeze(-1).cpu().numpy()
        self._style_stats.append((float(np.mean(style_r)), float(np.std(style_r))))

        # Scale all task rewards by w_g in-place
        buf.rewards[:] *= self.w_task

        # Add w_s * r_style for valid transitions (last step row stays task-only)
        full_style = np.zeros((T - 1) * n_envs, dtype=np.float32)
        full_style[valid] = style_r
        buf.rewards[:T-1] += self.w_style * full_style.reshape(T - 1, n_envs)

    # ── callback hooks ────────────────────────────────────────────────────────

    def _on_rollout_end(self) -> None:
        self._rollout += 1
        pol_s, pol_snxt, valid = self._extract_policy_transitions()

        disc_loss = self._update_discriminator(pol_s, pol_snxt)
        self._disc_losses.append(disc_loss)

        self._rewrite_rewards(pol_s, pol_snxt, valid)

        if self._rollout % self.log_interval == 0:
            avg_loss  = float(np.mean(self._disc_losses[-self.log_interval:]))
            recent    = self._style_stats[-self.log_interval:]
            avg_style = float(np.mean([s[0] for s in recent]))
            std_style = float(np.mean([s[1] for s in recent]))
            print(
                f"[AMP iter {self._rollout:5d} | steps {self.num_timesteps:>9,}]  "
                f"disc_loss={avg_loss:.4f}  "
                f"style_r={avg_style:.3f}±{std_style:.3f}"
            )

    def _on_step(self) -> bool:
        return True


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AMP Walker2d from IK reference")

    ref_grp = parser.add_mutually_exclusive_group(required=True)
    ref_grp.add_argument("--ref_cycle", type=str,
                         help="Path to single gait-cycle .npy (recommended)")
    ref_grp.add_argument("--ref_all",   action="store_true",
                         help="Use full concatenated Ulrich reference")
    parser.add_argument("--subjects",     type=int, nargs="+", default=None)
    parser.add_argument("--trial_filter", type=str, default=None)

    # training
    parser.add_argument("--num_envs",    type=int,   default=32)
    parser.add_argument("--total_steps", type=float, default=5e6)
    parser.add_argument("--device",      default="cpu")

    # discriminator
    parser.add_argument("--disc_lr",      type=float, default=1e-4)
    parser.add_argument("--disc_updates",    type=int,   default=1,
                        help="Discriminator gradient steps per PPO rollout")
    parser.add_argument("--disc_batch_size", type=int,   default=4096,
                        help="Max samples per discriminator mini-batch (avoids large graph on CPU)")
    parser.add_argument("--disc_hidden",  type=int,   default=1024,
                        help="Discriminator hidden size (paper: [1024, 512])")
    parser.add_argument("--grad_penalty", type=float, default=10.0,
                        help="Zero-centered GP coefficient on expert samples (paper: 10)")
    parser.add_argument("--expert_noise", type=float, default=0.05,
                        help="Gaussian noise std on expert transitions during disc training "
                             "(prevents memorisation of small 140-frame expert buffer)")

    # reward
    parser.add_argument("--w_task",   type=float, default=0.35,
                        help="Task reward weight (paper: w_g=0.35)")
    parser.add_argument("--w_style",  type=float, default=0.65,
                        help="Style reward weight (paper: w_s=0.65)")
    parser.add_argument("--v_target", type=float, default=1.25,
                        help="Target forward speed for task reward (m/s)")

    # policy warm-start
    parser.add_argument("--finetune",  type=str, default=None,
                        help="PPO .zip to warm-start from (recommended for faster convergence)")
    parser.add_argument("--bc_epochs", type=int, default=0,
                        help="BC warm-start epochs before AMP (PD-rollout targets)")
    parser.add_argument("--bc_steps",  type=int, default=200_000)
    parser.add_argument("--bc_kp",     type=float, default=200.0)
    parser.add_argument("--bc_kd",     type=float, default=20.0)

    parser.add_argument("--no_joint_vel", action="store_true",
                        help="Ablation: use only joint positions for discriminator (6-dim vs 12-dim)")
    parser.add_argument("--scale_model",  action="store_true",
                        help="Use Subject 1-scaled Walker2d geometry (walker2d_subject1.xml)")
    parser.add_argument("--out_dir",      default=None)
    args = parser.parse_args()

    # ── load reference ────────────────────────────────────────────────
    segment_lengths = None
    if args.ref_cycle:
        reference = load_ref_cycle(Path(args.ref_cycle))
        is_cycle  = True
    else:
        from ppo_walker2d import load_ulrich_reference
        print("Loading full Ulrich reference...")
        reference, segment_lengths = load_ulrich_reference(
            subjects        = args.subjects,
            trial_filter    = args.trial_filter,
            control_hz      = CTRL_HZ,
            return_lengths  = True,
        )
        is_cycle = False
    print(f"Reference: {reference.shape}  ({len(reference)/CTRL_HZ:.1f}s @ {CTRL_HZ}Hz)")

    # ── build expert buffer ───────────────────────────────────────────
    use_joint_vel = not args.no_joint_vel
    expert_s, expert_s_next = make_expert_buffer(
        reference,
        segment_lengths = segment_lengths,
        is_cycle        = is_cycle,
        use_joint_vel   = use_joint_vel,
    )
    state_dim = expert_s.shape[1]
    print(f"Expert transitions: {len(expert_s):,}  (state dim={state_dim})")

    # ── output dir ────────────────────────────────────────────────────
    stamp   = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = PROJECT_ROOT / (args.out_dir or f"results/walker2d_amp_{stamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {log_dir}")
    np.save(log_dir / "reference.npy", reference)

    # ── env ───────────────────────────────────────────────────────────
    xml_file = "walker2d_subject1.xml" if args.scale_model else "walker2d.xml"

    def make_env():
        def _init():
            return Walker2dAMP(
                reference        = reference,
                xml_file         = xml_file,
                v_target         = args.v_target,
                # All imitation weights zeroed — style handled by AMP discriminator
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

    # ── discriminator ─────────────────────────────────────────────────
    disc = AMPDiscriminator(state_dim=state_dim, hidden=args.disc_hidden)

    amp_cb = AMPCallback(
        discriminator   = disc,
        expert_s        = expert_s,
        expert_s_next   = expert_s_next,
        w_task          = args.w_task,
        w_style         = args.w_style,
        disc_lr         = args.disc_lr,
        disc_updates    = args.disc_updates,
        disc_batch_size = args.disc_batch_size,
        grad_penalty    = args.grad_penalty,
        expert_noise    = args.expert_noise,
        use_joint_vel   = use_joint_vel,
        device          = args.device,
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
            gamma         = 0.99,
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
        tmp_env = Walker2dAMP(
            reference=reference, xml_file=xml_file, v_target=args.v_target,
            imitation_weight=0.0, vel_weight=0.0, ee_weight=0.0,
            root_weight=0.0, contact_weight=0.0, swing_pen_weight=0.0,
        )
        print(f"Collecting BC dataset ({args.bc_steps:,} steps)...")
        obs_bc, act_bc = compute_bc_dataset(
            tmp_env, n_steps=args.bc_steps, kp=args.bc_kp, kd=args.bc_kd,
        )
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

    print(f"Training AMP for {int(args.total_steps):,} steps / {args.num_envs} envs...")
    model.learn(
        total_timesteps = int(args.total_steps),
        callback        = CallbackList([amp_cb, checkpoint_cb, LogCallback()]),
        progress_bar    = True,
    )
    env.close()

    save_path = str(log_dir / "model")
    model.save(save_path)
    torch.save(disc.state_dict(), str(log_dir / "discriminator.pt"))
    print(f"Policy       → {save_path}.zip")
    print(f"Discriminator → {log_dir}/discriminator.pt")


if __name__ == "__main__":
    main()
