"""
ppo_walk.py
───────────
Stage 1: Train a walking policy in MyoSuite using PPO on the native
myoLegWalk-v0 reward (forward velocity + cyclic hip + upright torso).

No expert demos needed — pure RL. The resulting policy is used as the
baseline for Stage 2 GAIL fine-tuning.

Usage
─────
  python ppo_walk.py                          # defaults
  python ppo_walk.py --total_steps 5000000 --device cuda
  python ppo_walk.py --resume checkpoints/ppo_walk_step_1000000.pt

The obs space of myoLegWalk-v0 is 403-dim (qpos, qvel, muscle states,
feet positions, phase variable). The action space is 80-dim muscle
activations ∈ [0, 1].
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Beta


# ── policy / value networks ───────────────────────────────────────────────────

class WalkPolicy(nn.Module):
    """
    Beta-distribution policy for bounded [0,1] muscle activations.
    Beta is more natural than Gaussian for actions that must stay in [0,1].
    Outputs concentration parameters (alpha, beta) > 1 so the distribution
    is unimodal.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 512):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden),  nn.ELU(),
            nn.Linear(hidden, hidden),  nn.ELU(),
        )
        self.alpha_head = nn.Linear(hidden, act_dim)
        self.beta_head  = nn.Linear(hidden, act_dim)

    def _concentration(self, obs):
        h = self.trunk(obs)
        # softplus + 1 keeps alpha, beta > 1 → unimodal Beta
        alpha = F.softplus(self.alpha_head(h)) + 1.0
        beta  = F.softplus(self.beta_head(h))  + 1.0
        return alpha, beta

    def get_distribution(self, obs):
        alpha, beta = self._concentration(obs)
        return Beta(alpha, beta)

    def forward(self, obs):
        """Deterministic mean action for evaluation."""
        alpha, beta = self._concentration(obs)
        return alpha / (alpha + beta)

    def sample(self, obs):
        dist   = self.get_distribution(obs)
        action = dist.rsample()
        log_p  = dist.log_prob(action.clamp(1e-6, 1 - 1e-6)).sum(-1)
        return action, log_p

    def log_prob(self, obs, action):
        dist = self.get_distribution(obs)
        return dist.log_prob(action.clamp(1e-6, 1 - 1e-6)).sum(-1)

    def entropy(self, obs):
        return self.get_distribution(obs).entropy().sum(-1)


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden),  nn.ELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


# ── rollout buffer ────────────────────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self):
        self.obs, self.actions, self.rewards = [], [], []
        self.log_probs, self.values, self.dones = [], [], []

    def add(self, obs, action, reward, log_p, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_p)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns(self, gamma=0.99, gae_lambda=0.95):
        T = len(self.rewards)
        advs    = torch.zeros(T)
        returns = torch.zeros(T)
        gae     = 0.0
        next_v  = 0.0
        for t in reversed(range(T)):
            mask   = 1.0 - float(self.dones[t])
            delta  = self.rewards[t] + gamma * next_v * mask - self.values[t].item()
            gae    = delta + gamma * gae_lambda * mask * gae
            advs[t]    = gae
            returns[t] = gae + self.values[t].item()
            next_v = self.values[t].item()
        return advs, returns

    def to_tensors(self, device):
        obs      = torch.stack(self.obs).to(device)
        actions  = torch.stack(self.actions).to(device)
        log_probs= torch.stack(self.log_probs).to(device)
        values   = torch.stack(self.values).to(device)
        return obs, actions, log_probs, values


# ── PPO trainer ───────────────────────────────────────────────────────────────

class PPOWalkTrainer:

    def __init__(
        self,
        env,
        policy:       WalkPolicy,
        value_net:    ValueNet,
        device:       str   = "cpu",
        rollout_len:  int   = 4096,
        ppo_epochs:   int   = 10,
        ppo_clip:     float = 0.2,
        vf_coef:      float = 0.5,
        ent_coef:     float = 0.01,
        gamma:        float = 0.99,
        gae_lambda:   float = 0.95,
        max_grad_norm:float = 0.5,
        lr:           float = 3e-4,
        batch_size:   int   = 256,
    ):
        self.env           = env
        self.policy        = policy.to(device)
        self.value_net     = value_net.to(device)
        self.device        = device
        self.rollout_len   = rollout_len
        self.ppo_epochs    = ppo_epochs
        self.ppo_clip      = ppo_clip
        self.vf_coef       = vf_coef
        self.ent_coef      = ent_coef
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.max_grad_norm = max_grad_norm
        self.batch_size    = batch_size

        self.lr_init = lr
        self.opt_policy = Adam(policy.parameters(),    lr=lr,       eps=1e-5)
        self.opt_value  = Adam(value_net.parameters(), lr=lr * 3.0, eps=1e-5)

    def _collect(self, obs):
        buf = RolloutBuffer()
        ep_rewards, ep_lens = [], []
        ep_r, ep_len = 0.0, 0

        for _ in range(self.rollout_len):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                action, log_p = self.policy.sample(obs_t.unsqueeze(0))
                value         = self.value_net(obs_t.unsqueeze(0))
                action = action.squeeze(0)
                log_p  = log_p.squeeze(0)
                value  = value.squeeze(0)

            next_obs, reward, terminated, truncated, _ = self.env.step(
                action.cpu().numpy()
            )
            done = terminated or truncated

            buf.add(obs_t, action, float(reward), log_p, value, done)
            ep_r   += reward
            ep_len += 1
            obs     = next_obs

            if done:
                ep_rewards.append(ep_r)
                ep_lens.append(ep_len)
                ep_r, ep_len = 0.0, 0
                obs, _ = self.env.reset()

        return buf, obs, ep_rewards, ep_lens

    def _update(self, buf, freeze_policy: bool = False):
        obs, actions, old_log_probs, old_values = buf.to_tensors(self.device)
        advs, returns = buf.compute_returns(self.gamma, self.gae_lambda)
        advs    = advs.to(self.device)
        returns = returns.to(self.device)
        advs    = (advs - advs.mean()) / (advs.std() + 1e-8)

        T = obs.shape[0]
        total_pl = total_vl = total_ent = 0.0
        n_batches = 0

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(T, device=self.device)
            for start in range(0, T, self.batch_size):
                idx = perm[start : start + self.batch_size]
                s   = obs[idx];     a   = actions[idx]
                adv = advs[idx];    ret = returns[idx]
                old_lp = old_log_probs[idx]

                # policy loss
                new_lp  = self.policy.log_prob(s, a)
                ratio   = (new_lp - old_lp).exp()
                surr1   = ratio * adv
                surr2   = ratio.clamp(1 - self.ppo_clip, 1 + self.ppo_clip) * adv
                pl      = -torch.min(surr1, surr2).mean()
                ent     = self.policy.entropy(s).mean()

                if not freeze_policy:
                    policy_loss = pl - self.ent_coef * ent
                    self.opt_policy.zero_grad()
                    policy_loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.opt_policy.step()

                # value loss with clipping
                val     = self.value_net(s)
                old_v   = old_values[idx]
                val_clipped = old_v + (val - old_v).clamp(-self.ppo_clip, self.ppo_clip)
                vl      = torch.max(F.mse_loss(val, ret), F.mse_loss(val_clipped, ret))

                self.opt_value.zero_grad()
                vl.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.opt_value.step()

                total_pl  += pl.item()
                total_vl  += vl.item()
                total_ent += ent.item()
                n_batches += 1

        n = max(1, n_batches)
        return total_pl / n, total_vl / n, total_ent / n

    def train(
        self,
        total_steps:      int  = 5_000_000,
        log_every:        int  = 50_000,
        save_every:       int  = 200_000,
        checkpoint_dir:   Path = Path("checkpoints"),
        value_warmup_steps: int = 200_000,
    ):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        obs, _ = self.env.reset()
        steps_done = 0
        all_ep_rewards = []

        print(f"[PPO] Starting. total_steps={total_steps}, rollout_len={self.rollout_len}")
        if value_warmup_steps > 0:
            print(f"[PPO] Value warmup for first {value_warmup_steps} steps (policy frozen)")

        while steps_done < total_steps:
            # linear LR annealing
            frac = 1.0 - steps_done / total_steps
            for pg in self.opt_policy.param_groups:
                pg["lr"] = self.lr_init * frac
            for pg in self.opt_value.param_groups:
                pg["lr"] = self.lr_init * 3.0 * frac

            buf, obs, ep_rewards, ep_lens = self._collect(obs)
            steps_done += self.rollout_len
            all_ep_rewards.extend(ep_rewards)

            policy_frozen = steps_done < value_warmup_steps
            pl, vl, ent = self._update(buf, freeze_policy=policy_frozen)

            if steps_done % log_every < self.rollout_len:
                mean_r = np.mean(all_ep_rewards[-20:]) if all_ep_rewards else 0.0
                mean_l = np.mean(ep_lens) if ep_lens else 0
                print(
                    f"  step {steps_done:8d}  "
                    f"ep_r={mean_r:7.2f}  ep_len={mean_l:5.0f}  "
                    f"π={pl:.4f}  V={vl:.4f}  H={ent:.3f}"
                )

            if steps_done % save_every < self.rollout_len:
                tag = f"ppo_walk_step_{steps_done}"
                torch.save(self.policy.state_dict(),    checkpoint_dir / f"policy_{tag}.pt")
                torch.save(self.value_net.state_dict(), checkpoint_dir / f"value_{tag}.pt")
                print(f"  [PPO] Saved checkpoint @ step {steps_done}")

        torch.save(self.policy.state_dict(),    checkpoint_dir / "ppo_walk_final.pt")
        torch.save(self.value_net.state_dict(), checkpoint_dir / "ppo_walk_value_final.pt")
        print("[PPO] Training complete.")


# ── render helper ─────────────────────────────────────────────────────────────

def render(policy: WalkPolicy, env, n_episodes: int = 3):
    import time
    policy.eval()
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        steps = 0
        while not done:
            env.unwrapped.mj_render()
            time.sleep(1 / 60)
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = policy(obs_t).squeeze(0).numpy()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_r += reward
            steps += 1
        print(f"  Episode {ep+1}: {steps} steps, total reward = {total_r:.2f}")
    env.close()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_steps", type=int,   default=5_000_000)
    parser.add_argument("--rollout_len", type=int,   default=4096)
    parser.add_argument("--ppo_epochs",  type=int,   default=10)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--ent_coef",    type=float, default=0.02)
    parser.add_argument("--device",      type=str,   default="cpu")
    parser.add_argument("--resume",      type=str,   default=None,
                        help="Path to policy checkpoint to resume from")
    parser.add_argument("--render",      action="store_true")
    parser.add_argument("--render_eps",  type=int,   default=3)
    parser.add_argument("--log_every",   type=int,   default=50_000)
    parser.add_argument("--save_every",  type=int,   default=200_000)
    args = parser.parse_args()

    import myosuite  # noqa: F401
    import gymnasium as gym

    env = gym.make("myoLegWalk-v0")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"[PPO] obs_dim={obs_dim}, act_dim={act_dim}")

    policy    = WalkPolicy(obs_dim, act_dim)
    value_net = ValueNet(obs_dim)

    if args.resume:
        policy.load_state_dict(torch.load(args.resume, map_location=args.device))
        # auto-load matching value checkpoint (policy_X.pt → value_X.pt)
        value_path = Path(args.resume).parent / Path(args.resume).name.replace("policy_", "value_").replace("ppo_walk_final", "ppo_walk_value_final")
        if value_path.exists():
            value_net.load_state_dict(torch.load(value_path, map_location=args.device))
            print(f"[PPO] Resumed from {args.resume} + {value_path.name}")
        else:
            print(f"[PPO] Resumed policy from {args.resume} (no matching value checkpoint found)")

    if args.render:
        render(policy, env, n_episodes=args.render_eps)
        return

    trainer = PPOWalkTrainer(
        env=env,
        policy=policy,
        value_net=value_net,
        device=args.device,
        rollout_len=args.rollout_len,
        ppo_epochs=args.ppo_epochs,
        ent_coef=args.ent_coef,
        lr=args.lr,
    )

    trainer.train(
        total_steps=args.total_steps,
        log_every=args.log_every,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
