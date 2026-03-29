"""
train.py
────────
Entry point for:

  Phase 1 → Behavioural Cloning  (BC)
  Phase 2 → GAIL fine-tuning      (markered or markerless IK as expert)

Usage
─────
  # BC only:
  python train.py --mode bc

  # BC → GAIL with markered IK expert:
  python train.py --mode gail --source markered

  # BC → GAIL with markerless IK expert:
  python train.py --mode gail --source markerless

  # Skip BC and load existing checkpoint:
  python train.py --mode gail --bc_ckpt checkpoints/bc_policy.pt

Data
────
Put your files in a  data/  folder next to this script:
  data/walking1.mot           ← IK (markered, from OpenSim)
  data/walking1_EMG.sto       ← EMG activations
  data/walking1_forces.mot    ← GRFs (optional)

  data/walking1_markerless.mot  ← IK from markerless pipeline (if available)
  data/walking1_markerless_EMG.sto

MyoSuite environment
────────────────────
  Set ENV_ID to a registered MyoSuite env, e.g. "myoLegWalk-v0".
  If MyoSuite is not installed the script falls back to a lightweight
  DummyEnv that mirrors the data dimensions — useful for smoke-testing
  the pipeline without a physics engine.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# ── local modules ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from data_utils  import ExpertData, GAILDataset, make_dataloaders, IK_ALL_COLS, N_MUSCLES
from bc_policy   import BCPolicy, BCTrainer
from gail        import Discriminator, GAILTrainer


# ── environment ───────────────────────────────────────────────────────────────

ENV_ID = "myoLegWalk-v0"   # ← change to your MyoSuite env


def make_env(state_dim: int, action_dim: int):
    """
    Try to load MyoSuite; fall back to a DummyEnv with matching dimensions.
    """
    try:
        import myosuite  # noqa: F401
        import gymnasium as gym
        env = gym.make(ENV_ID)
        print(f"[env] Loaded MyoSuite env: {ENV_ID}")
        return env
    except ImportError:
        print("[env] MyoSuite / gymnasium not found — using DummyEnv "
              "(pipeline smoke-test only)")
        return DummyEnv(state_dim, action_dim)


class DummyEnv:
    """
    Minimal gym-compatible environment that echoes random states.
    Allows the full BC→GAIL code to run without MyoSuite installed.
    """

    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self._step_n    = 0
        self.max_steps  = 200

    def reset(self, **kwargs):
        self._step_n = 0
        obs = np.random.randn(self.state_dim).astype(np.float32) * 0.1
        return obs, {}

    def step(self, action):
        self._step_n += 1
        obs      = np.random.randn(self.state_dim).astype(np.float32) * 0.1
        reward   = float(np.random.randn())
        done     = self._step_n >= self.max_steps
        return obs, reward, done, False, {}

    @property
    def observation_space(self):
        class _Space:
            shape = (self.state_dim,)
        return _Space()

    @property
    def action_space(self):
        class _Space:
            shape = (self.action_dim,)
        return _Space()


# ── helpers ───────────────────────────────────────────────────────────────────

def build_expert(source: str, use_grf: bool = False) -> ExpertData:
    """
    Load expert data for the chosen mocap source.

    source = "markered"   → uses walking1.mot / walking1_EMG.sto
    source = "markerless" → uses walking1_markerless.mot / walking1_markerless_EMG.sto
                            (falls back to markered data if file is missing)
    """
    data_dir = Path(__file__).parent / "data"

    if source == "markerless":
        ik_path  = data_dir / "walking1_markerless.mot"
        emg_path = data_dir / "walking1_markerless_EMG.sto"
        if not ik_path.exists():
            print("[warn] Markerless IK not found — falling back to markered data.")
            ik_path  = data_dir / "walking1.mot"
            emg_path = data_dir / "walking1_EMG.sto"
    else:
        ik_path  = data_dir / "walking1.mot"
        emg_path = data_dir / "walking1_EMG.sto"

    grf_path = data_dir / "walking1_forces.mot"

    return ExpertData(
        ik_path=ik_path,
        emg_path=emg_path,
        grf_path=grf_path,
        use_grf=use_grf,
    )


def state_dim_from_expert(expert: ExpertData) -> int:
    return expert.S   # IK rotations + translations + velocities


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BC → GAIL training for MyoSuite")
    parser.add_argument("--mode",    choices=["bc", "gail"], default="gail")
    parser.add_argument("--source",  choices=["markered", "markerless"], default="markered",
                        help="Which mocap pipeline's IK to use as expert for GAIL")
    parser.add_argument("--bc_ckpt",  type=str, default=None,
                        help="Path to existing BC checkpoint (skips Phase 1)")
    parser.add_argument("--bc_epochs", type=int, default=200)
    parser.add_argument("--gail_steps", type=int, default=200_000)
    parser.add_argument("--use_grf",   action="store_true")
    parser.add_argument("--source_aware", action="store_true",
                        help="Condition discriminator on mocap source tag")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = args.device
    ckpt_dir = Path("checkpoints")

    # ── load expert data ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Loading expert data  (source: {args.source})")
    print(f"{'='*60}")
    expert = build_expert(args.source, use_grf=args.use_grf)

    S = state_dim_from_expert(expert)
    A = expert.A
    print(f"  State dim  S = {S}")
    print(f"  Action dim A = {A}")

    # ── build models ──────────────────────────────────────────────────────
    policy        = BCPolicy(state_dim=S, action_dim=A, hidden_dims=(256, 256, 128))
    discriminator = Discriminator(
        state_dim=S, action_dim=A,
        source_aware=args.source_aware,
    )

    # ── phase 1: BC ───────────────────────────────────────────────────────
    if args.bc_ckpt:
        print(f"\n[Phase 1] Loading BC checkpoint: {args.bc_ckpt}")
        policy.load_state_dict(torch.load(args.bc_ckpt, map_location=device))
    else:
        print(f"\n{'='*60}")
        print(f"  Phase 1: Behavioural Cloning")
        print(f"{'='*60}")

        train_loader, val_loader = make_dataloaders(expert, batch_size=32, val_frac=0.1)

        bc_trainer = BCTrainer(policy, lr=3e-4, l1_lambda=1e-3, device=device)
        bc_trainer.fit(
            train_loader, val_loader,
            epochs=args.bc_epochs,
            patience=30,
            save_path=ckpt_dir / "bc_policy_best.pt",
            verbose_every=20,
        )

    if args.mode == "bc":
        print("\n[Done] BC training only. Exiting.")
        return

    # ── phase 2: GAIL ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Phase 2: GAIL  (expert source: {args.source})")
    print(f"{'='*60}")

    expert_dataset = GAILDataset(expert)
    env            = make_env(state_dim=S, action_dim=A)

    gail_trainer = GAILTrainer(
        env=env,
        policy=policy,
        discriminator=discriminator,
        expert_dataset=expert_dataset,
        device=device,
        ppo_epochs=5,
        ppo_clip=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        disc_epochs=3,
        disc_batch=64,
        gp_lambda=10.0,
        rollout_len=512,
        lr_policy=3e-4,
        lr_disc=1e-4,
        source_aware=args.source_aware,
    )

    source_tag = 0 if args.source == "markered" else 1
    metrics    = gail_trainer.train(
        total_steps=args.gail_steps,
        log_every=5_000,
        save_every=20_000,
        checkpoint_dir=ckpt_dir,
        mocap_source=source_tag,
    )

    # ── save final ────────────────────────────────────────────────────────
    torch.save(policy.state_dict(),        ckpt_dir / "policy_final.pt")
    torch.save(discriminator.state_dict(), ckpt_dir / "disc_final.pt")
    print(f"\n[Done] Final models saved to {ckpt_dir}/")


if __name__ == "__main__":
    main()
