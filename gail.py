"""
gail.py
───────
Generative Adversarial Imitation Learning (GAIL) loop.

Goal
────
Given:
  • Expert demonstrations  (s, a) from EMG + IK data
  • A MyoSuite environment whose state = IK joint angles/velocities
    and whose actions = muscle activations

Train a policy π_θ to produce trajectories that the discriminator
D_φ cannot distinguish from the expert demonstrations.

Architecture
────────────
  Discriminator D_φ(s, a) → [0, 1]
    • Input  : concatenation of state and action
    • Output : probability that (s,a) is from the expert
    • Loss   : binary cross-entropy (WGAN-GP variant also available)

  Policy π_θ : see bc_policy.py  (pre-trained via BC)
    • Updated via PPO using −log D_φ(s, a) as reward signal

GAIL reward
───────────
  r_GAIL(s, a) = −log(1 − D_φ(s, a))   [original Ho & Ermon 2016]
  or equivalently  log(D_φ(s, a))  when D models P(expert).

Markered vs Markerless IK comparison
─────────────────────────────────────
  The discriminator can receive a "source" tag (0 = markered, 1 = markerless)
  concatenated to (s, a), allowing the same GAIL loop to learn from
  both IK sources while the policy optimises against the combined
  discriminator.  See `source_aware` flag.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Optional, Dict, List
from pathlib import Path

from bc_policy import BCPolicy


# ── discriminator ────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    D_φ(s, a) → scalar ∈ (0, 1)

    Gradient penalty (WGAN-GP style) is supported via `compute_gp()`.
    """

    def __init__(
        self,
        state_dim:   int,
        action_dim:  int,
        hidden_dims: tuple = (256, 256),
        source_aware: bool = False,   # +1 input dim for mocap-source tag
    ):
        super().__init__()
        extra = 1 if source_aware else 0
        in_dim = state_dim + action_dim + extra

        layers = []
        for h in hidden_dims:
            layers += [
                nn.utils.spectral_norm(nn.Linear(in_dim, h)),
                nn.LeakyReLU(0.2),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))

        self.net          = nn.Sequential(*layers)
        self.source_aware = source_aware

    def forward(
        self,
        state:  torch.Tensor,
        action: torch.Tensor,
        source: Optional[torch.Tensor] = None,   # float 0/1
    ) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        if self.source_aware and source is not None:
            src = source.float().unsqueeze(-1)
            x   = torch.cat([x, src], dim=-1)
        return self.net(x)   # raw logit; apply sigmoid externally

    def reward(self, state, action, source=None):
        """
        GAIL reward  r = −log(1 − D(s,a))
        Returns a detached scalar tensor per sample.
        """
        with torch.no_grad():
            logit = self.forward(state, action, source)
            prob  = torch.sigmoid(logit)
            # clamp to avoid log(0)
            prob  = prob.clamp(1e-6, 1 - 1e-6)
            return -torch.log(1.0 - prob).squeeze(-1)

    def compute_gp(self, real_sa, fake_sa, device):
        """Gradient penalty for WGAN-GP (optional)."""
        alpha  = torch.rand(real_sa.size(0), 1, device=device)
        interp = (alpha * real_sa + (1 - alpha) * fake_sa).requires_grad_(True)
        d_interp = self.net(interp)
        grads    = torch.autograd.grad(
            outputs=d_interp, inputs=interp,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True, retain_graph=True,
        )[0]
        gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
        return gp


# ── PPO buffer ───────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Stores one episode worth of (s, a, r, log_π, value, done) tuples.
    Not designed for parallelism — single environment rollout.
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.states     : List[torch.Tensor] = []
        self.actions    : List[torch.Tensor] = []
        self.rewards    : List[float]        = []
        self.log_probs  : List[torch.Tensor] = []
        self.values     : List[torch.Tensor] = []
        self.dones      : List[bool]         = []

    def add(self, s, a, r, lp, v, done):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(float(r))
        self.log_probs.append(lp)
        self.values.append(v)
        self.dones.append(done)

    def compute_returns(self, gamma=0.99, gae_lambda=0.95, last_value=0.0):
        """Generalised Advantage Estimation."""
        T         = len(self.rewards)
        advs      = torch.zeros(T)
        last_adv  = 0.0
        values    = [v.item() for v in self.values] + [last_value]

        for t in reversed(range(T)):
            mask    = 0.0 if self.dones[t] else 1.0
            delta   = self.rewards[t] + gamma * values[t + 1] * mask - values[t]
            last_adv = delta + gamma * gae_lambda * mask * last_adv
            advs[t] = last_adv

        returns = advs + torch.tensor(values[:-1])
        return advs, returns

    def to_tensors(self, device):
        return (
            torch.stack(self.states).to(device),
            torch.stack(self.actions).to(device),
            torch.tensor(self.rewards, dtype=torch.float32).to(device),
            torch.stack(self.log_probs).to(device),
            torch.stack(self.values).squeeze(-1).to(device),
        )


# ── value network (critic) ───────────────────────────────────────────────────

class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dims: tuple = (256, 256)):
        super().__init__()
        layers, in_d = [], state_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.Tanh()]
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        return self.net(state)


# ── GAIL trainer ─────────────────────────────────────────────────────────────

class GAILTrainer:
    """
    Full GAIL training loop.

    Parameters
    ──────────
    env            : A MyoSuite (or gym-compatible) environment.
                     state space = IK angles + velocities (matches ExpertData.S)
                     action space = muscle activations (matches ExpertData.A)
    policy         : BCPolicy (pre-trained)
    discriminator  : Discriminator
    expert_dataset : GAILDataset  (the ground-truth (s,a) pairs from EMG+IK)
    source_aware   : Whether the discriminator receives a mocap-source tag.
                     Pass source=0 for markered, source=1 for markerless.

    Training proceeds as:
      1. Collect rollout from π_θ in env (GAIL reward from D_φ)
      2. Update D_φ to distinguish expert vs policy rollouts
      3. Update π_θ + V via PPO using GAIL rewards
    """

    def __init__(
        self,
        env,
        policy:        BCPolicy,
        discriminator: Discriminator,
        expert_dataset,                   # GAILDataset
        device:        str   = "cpu",
        # PPO hyper-params
        ppo_epochs:    int   = 5,
        ppo_clip:      float = 0.2,
        vf_coef:       float = 0.5,
        ent_coef:      float = 0.01,
        gamma:         float = 0.99,
        gae_lambda:    float = 0.95,
        max_grad_norm: float = 0.5,
        # discriminator hyper-params
        disc_epochs:   int   = 3,
        disc_batch:    int   = 64,
        gp_lambda:     float = 10.0,       # 0 to disable GP
        # general
        rollout_len:   int   = 512,
        lr_policy:     float = 3e-4,
        lr_disc:       float = 1e-4,
        source_aware:  bool  = False,
    ):
        self.env           = env
        self.policy        = policy.to(device)
        self.disc          = discriminator.to(device)
        self.expert_ds     = expert_dataset
        self.device        = device
        self.ppo_epochs    = ppo_epochs
        self.ppo_clip      = ppo_clip
        self.vf_coef       = vf_coef
        self.ent_coef      = ent_coef
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.max_grad_norm = max_grad_norm
        self.disc_epochs   = disc_epochs
        self.disc_batch    = disc_batch
        self.gp_lambda     = gp_lambda
        self.rollout_len   = rollout_len
        self.source_aware  = source_aware

        # critic
        S = policy.trunk[0].in_features
        self.value_net = ValueNet(S).to(device)

        # optimisers
        self.opt_policy = Adam(
            list(policy.parameters()) + list(self.value_net.parameters()),
            lr=lr_policy,
        )
        self.opt_disc   = Adam(discriminator.parameters(), lr=lr_disc, betas=(0.5, 0.999))

        # expert DataLoader (infinite cycling)
        from torch.utils.data import DataLoader
        self.expert_loader = DataLoader(
            expert_dataset, batch_size=disc_batch, shuffle=True, drop_last=True
        )
        self._expert_iter  = iter(self.expert_loader)

        self.metrics: Dict[str, List] = {
            "disc_loss": [], "policy_loss": [], "value_loss": [],
            "mean_reward": [], "entropy": []
        }

    # ── expert batch ──────────────────────────────────────────────────────

    def _next_expert_batch(self):
        try:
            batch = next(self._expert_iter)
        except StopIteration:
            self._expert_iter = iter(self.expert_loader)
            batch = next(self._expert_iter)
        s, a, _ = batch
        return s.to(self.device), a.to(self.device)

    # ── rollout collection ────────────────────────────────────────────────

    def _collect_rollout(self, mocap_source: int = 0) -> RolloutBuffer:
        buf  = RolloutBuffer()
        obs, _ = self.env.reset()
        obs  = torch.tensor(obs, dtype=torch.float32, device=self.device)

        for _ in range(self.rollout_len):
            with torch.no_grad():
                action, log_p = self.policy.sample(obs.unsqueeze(0))
                value         = self.value_net(obs.unsqueeze(0))
                action        = action.squeeze(0)
                log_p         = log_p.squeeze(0)
                value         = value.squeeze(0)

                src = (torch.tensor([mocap_source], device=self.device)
                       if self.source_aware else None)
                gail_reward = self.disc.reward(
                    obs.unsqueeze(0), action.unsqueeze(0), src
                ).squeeze(0)

            next_obs, _env_reward, terminated, truncated, _ = self.env.step(
                action.cpu().numpy()
            )
            done = terminated or truncated

            buf.add(obs, action, gail_reward, log_p, value, done)
            obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)

            if done:
                obs, _ = self.env.reset()
                obs    = torch.tensor(obs, dtype=torch.float32, device=self.device)

        return buf

    # ── discriminator update ──────────────────────────────────────────────

    def _update_discriminator(self, policy_states, policy_actions):
        total_loss = 0.0

        for _ in range(self.disc_epochs):
            exp_s, exp_a  = self._next_expert_batch()

            # fake = policy rollout (random mini-batch)
            idx        = torch.randint(0, policy_states.shape[0],
                                       (self.disc_batch,), device=self.device)
            fake_s = policy_states[idx]
            fake_a = policy_actions[idx]

            real_logits = self.disc(exp_s,  exp_a)
            fake_logits = self.disc(fake_s, fake_a)

            real_labels = torch.ones_like(real_logits)
            fake_labels = torch.zeros_like(fake_logits)

            loss = (F.binary_cross_entropy_with_logits(real_logits, real_labels)
                  + F.binary_cross_entropy_with_logits(fake_logits, fake_labels))

            # optional gradient penalty
            if self.gp_lambda > 0:
                real_sa = torch.cat([exp_s,  exp_a],  dim=-1)
                fake_sa = torch.cat([fake_s, fake_a], dim=-1)
                gp      = self.disc.compute_gp(real_sa, fake_sa, self.device)
                loss    = loss + self.gp_lambda * gp

            self.opt_disc.zero_grad()
            loss.backward()
            self.opt_disc.step()
            total_loss += loss.item()

        return total_loss / self.disc_epochs

    # ── PPO update ────────────────────────────────────────────────────────

    def _update_policy(self, buf: RolloutBuffer):
        states, actions, rewards, old_log_probs, old_values = buf.to_tensors(self.device)
        advs, returns = buf.compute_returns(self.gamma, self.gae_lambda)
        advs    = advs.to(self.device)
        returns = returns.to(self.device)

        # normalise advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        T = states.shape[0]
        total_pl = total_vl = total_ent = 0.0

        for _ in range(self.ppo_epochs):
            # fresh shuffle
            perm = torch.randperm(T, device=self.device)
            for start in range(0, T, self.disc_batch):
                idx = perm[start: start + self.disc_batch]
                s   = states[idx];  a = actions[idx]
                adv = advs[idx];    ret = returns[idx]
                old_lp = old_log_probs[idx]

                new_lp = self.policy.log_prob(s, a)
                ratio  = (new_lp - old_lp).exp()

                surr1  = ratio * adv
                surr2  = ratio.clamp(1 - self.ppo_clip, 1 + self.ppo_clip) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                val    = self.value_net(s).squeeze(-1)
                vf_loss = F.mse_loss(val, ret)

                dist   = self.policy.get_distribution(s)
                entropy = dist.entropy().sum(-1).mean()

                loss = policy_loss + self.vf_coef * vf_loss - self.ent_coef * entropy

                self.opt_policy.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_net.parameters()),
                    self.max_grad_norm,
                )
                self.opt_policy.step()

                total_pl  += policy_loss.item()
                total_vl  += vf_loss.item()
                total_ent += entropy.item()

        n_batches = self.ppo_epochs * max(1, T // self.disc_batch)
        return total_pl / n_batches, total_vl / n_batches, total_ent / n_batches

    # ── main loop ─────────────────────────────────────────────────────────

    def train(
        self,
        total_steps:   int = 100_000,
        log_every:     int = 5_000,
        save_every:    int = 20_000,
        checkpoint_dir: Path = Path("checkpoints"),
        mocap_source:  int = 0,   # 0=markered, 1=markerless
    ):
        """
        Parameters
        ──────────
        mocap_source : int
            0 → expert trajectories come from markered IK
            1 → expert trajectories come from markerless IK
            (requires loading the appropriate ExpertData before calling train())
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        steps_done = 0

        print(f"[GAIL] Starting. total_steps={total_steps}, "
              f"rollout_len={self.rollout_len}, source={'markered' if mocap_source==0 else 'markerless'}")

        while steps_done < total_steps:
            # ── 1. collect rollout ────────────────────────────────────────
            buf = self._collect_rollout(mocap_source=mocap_source)
            states_t, actions_t, rewards_t, *_ = buf.to_tensors(self.device)
            steps_done += self.rollout_len

            # ── 2. update discriminator ───────────────────────────────────
            disc_loss = self._update_discriminator(states_t, actions_t)

            # ── 3. update policy (PPO) ────────────────────────────────────
            p_loss, v_loss, ent = self._update_policy(buf)

            mean_r = rewards_t.mean().item()
            self.metrics["disc_loss"].append(disc_loss)
            self.metrics["policy_loss"].append(p_loss)
            self.metrics["value_loss"].append(v_loss)
            self.metrics["mean_reward"].append(mean_r)
            self.metrics["entropy"].append(ent)

            if steps_done % log_every < self.rollout_len:
                print(
                    f"  step {steps_done:7d}  "
                    f"D={disc_loss:.4f}  π={p_loss:.4f}  "
                    f"V={v_loss:.4f}  r={mean_r:.4f}  H={ent:.3f}"
                )

            if steps_done % save_every < self.rollout_len:
                tag = f"step_{steps_done}"
                torch.save(self.policy.state_dict(),
                           checkpoint_dir / f"policy_{tag}.pt")
                torch.save(self.disc.state_dict(),
                           checkpoint_dir / f"disc_{tag}.pt")
                print(f"  [GAIL] Saved checkpoint @ step {steps_done}")

        print("[GAIL] Training complete.")
        return self.metrics
