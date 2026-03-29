"""
bc_policy.py
────────────
Behavioural Cloning (BC) pre-training stage.

Architecture
────────────
  Policy π_θ : state → action
    • Input  : IK joint angles + angular velocities  (S-dim)
    • Output : muscle activations ∈ [0, 1]           (A-dim)
    • Head   : Sigmoid (activations must be non-negative & ≤ 1)

  Optional auxiliary: GRF-conditioned variant
    (concatenate GRF features before the MLP trunk)

Training
────────
  Loss = MSE(π(s), a_expert)  +  λ_l1 * |π(s)|₁
  The small L1 term penalises spurious co-activation (empirically helpful
  for muscle-driven models).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from typing import Optional


# ── network ──────────────────────────────────────────────────────────────────

class BCPolicy(nn.Module):
    """
    Gaussian MLP policy with Sigmoid output for bounded muscle activations.

    For GAIL we'll wrap this in a stochastic policy layer; for BC training
    we use the deterministic mean output directly.
    """

    def __init__(
        self,
        state_dim:  int,
        action_dim: int,
        hidden_dims: tuple = (256, 256, 128),
        dropout:     float = 0.1,
        log_std_init: float = -1.0,
    ):
        super().__init__()

        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ELU(), nn.Dropout(dropout)]
            in_dim = h

        self.trunk = nn.Sequential(*layers)

        # deterministic mean head → Sigmoid output
        self.mean_head = nn.Sequential(
            nn.Linear(in_dim, action_dim),
            nn.Sigmoid(),
        )

        # learnable log-std for stochastic sampling (used in GAIL PPO step)
        self.log_std = nn.Parameter(
            torch.full((action_dim,), log_std_init)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns deterministic mean activation (for BC loss & rollout)."""
        return self.mean_head(self.trunk(state))

    def get_distribution(self, state: torch.Tensor):
        """Returns a Normal distribution clamped to [0,1] via sigmoid transform."""
        mean = self.forward(state)
        std  = self.log_std.exp().clamp(1e-4, 1.0)
        return torch.distributions.Normal(mean, std)

    def sample(self, state: torch.Tensor):
        """Sample + log-prob for PPO update."""
        dist   = self.get_distribution(state)
        sample = dist.rsample().clamp(0.0, 1.0)
        log_p  = dist.log_prob(sample).sum(-1)
        return sample, log_p

    def log_prob(self, state: torch.Tensor, action: torch.Tensor):
        dist = self.get_distribution(state)
        return dist.log_prob(action.clamp(1e-6, 1 - 1e-6)).sum(-1)


# ── BC trainer ───────────────────────────────────────────────────────────────

class BCTrainer:
    """
    Trains BCPolicy with MSE + optional L1 muscle sparsity loss.

    Usage
    ─────
        trainer = BCTrainer(policy, lr=3e-4)
        trainer.fit(train_loader, val_loader, epochs=200)
        trainer.save("checkpoints/bc_policy.pt")
    """

    def __init__(
        self,
        policy:      BCPolicy,
        lr:          float = 3e-4,
        l1_lambda:   float = 1e-3,
        device:      str   = "cpu",
    ):
        self.policy    = policy.to(device)
        self.device    = device
        self.l1_lambda = l1_lambda
        self.optimizer = Adam(policy.parameters(), lr=lr, weight_decay=1e-5)
        self.history   = {"train_loss": [], "val_loss": []}

    # ── single pass ──────────────────────────────────────────────────────

    def _step(self, batch, train: bool):
        states, actions, _ = batch
        states  = states.to(self.device)
        actions = actions.to(self.device)

        pred = self.policy(states)

        mse_loss = F.mse_loss(pred, actions)
        l1_loss  = pred.abs().mean()
        loss     = mse_loss + self.l1_lambda * l1_loss

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

        return loss.item(), mse_loss.item()

    # ── training loop ────────────────────────────────────────────────────

    def fit(
        self,
        train_loader,
        val_loader,
        epochs:        int   = 200,
        patience:      int   = 30,
        save_path:     Optional[Path] = None,
        verbose_every: int   = 10,
    ):
        scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)
        best_val  = float("inf")
        no_improve = 0

        for epoch in range(1, epochs + 1):
            # train
            self.policy.train()
            t_losses = [self._step(b, train=True)[0]  for b in train_loader]

            # validate
            self.policy.eval()
            with torch.no_grad():
                v_losses = [self._step(b, train=False)[0] for b in val_loader]

            t_loss = np.mean(t_losses)
            v_loss = np.mean(v_losses)
            self.history["train_loss"].append(t_loss)
            self.history["val_loss"].append(v_loss)
            scheduler.step()

            if v_loss < best_val:
                best_val   = v_loss
                no_improve = 0
                if save_path:
                    self.save(save_path)
            else:
                no_improve += 1

            if epoch % verbose_every == 0:
                print(f"  BC epoch {epoch:4d}/{epochs}  "
                      f"train={t_loss:.5f}  val={v_loss:.5f}  best={best_val:.5f}")

            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch} (no improvement for {patience} epochs)")
                break

        print(f"  BC training done. Best val loss: {best_val:.5f}")
        return self.history

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), path)
        print(f"  [BC] Saved → {path}")

    def load(self, path: Path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        print(f"  [BC] Loaded ← {path}")
