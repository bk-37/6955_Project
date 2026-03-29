"""
evaluate.py
───────────
Evaluate a trained BC / GAIL policy against the expert demonstration.

Metrics
───────
  • Activation MSE per muscle
  • Joint angle RMSE per DOF
  • Pearson r per muscle (temporal correlation with expert EMG)
  • GAIL discriminator score (how "expert-like" the policy is)

Supports side-by-side comparison of markered vs markerless expert IK,
useful for quantifying the sim-to-real gap introduced by the mocap pipeline.
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data_utils import ExpertData, parse_opensim
from bc_policy  import BCPolicy
from gail       import Discriminator
from train      import build_expert, state_dim_from_expert


def evaluate_bc(policy: BCPolicy, expert: ExpertData, device: str = "cpu") -> dict:
    """
    Run the policy on every expert state and compare predictions to
    expert muscle activations.

    Returns
    ───────
    metrics : dict with per-muscle and aggregate statistics.
    """
    policy.eval()
    states  = torch.from_numpy(expert.states).to(device)   # (T, S)
    actions = torch.from_numpy(expert.actions).to(device)  # (T, A)

    with torch.no_grad():
        pred = policy(states).cpu().numpy()                 # (T, A)

    gt   = actions.cpu().numpy()
    diff = pred - gt

    from data_utils import EMG_COLS
    muscle_names = [c.replace("_activation", "") for c in EMG_COLS]

    per_muscle_mse = (diff ** 2).mean(axis=0)
    per_muscle_r   = np.array([
        float(np.corrcoef(pred[:, i], gt[:, i])[0, 1])
        for i in range(gt.shape[1])
    ])

    print("\n── Muscle-level BC accuracy ──────────────────────────────────")
    print(f"  {'muscle':<24}  {'MSE':>8}  {'r':>6}")
    print(f"  {'-'*24}  {'-'*8}  {'-'*6}")
    for name, mse, r in zip(muscle_names, per_muscle_mse, per_muscle_r):
        print(f"  {name:<24}  {mse:8.5f}  {r:6.3f}")
    print(f"  {'AGGREGATE':<24}  {per_muscle_mse.mean():8.5f}  {per_muscle_r.mean():6.3f}")

    return {
        "per_muscle_mse":  per_muscle_mse,
        "per_muscle_r":    per_muscle_r,
        "aggregate_mse":   float(per_muscle_mse.mean()),
        "aggregate_r":     float(per_muscle_r.mean()),
        "pred_activations": pred,
        "gt_activations":   gt,
        "muscle_names":     muscle_names,
    }


def compare_sources(
    policy:      BCPolicy,
    markered:    ExpertData,
    markerless:  ExpertData,
    device:      str = "cpu",
) -> None:
    """
    Print a side-by-side comparison of how similar the policy rollouts are
    to markered vs markerless expert IK data.
    """
    print("\n── Source comparison: markered vs markerless ─────────────────")

    m1 = evaluate_bc(policy, markered,   device)
    m2 = evaluate_bc(policy, markerless, device)

    names = m1["muscle_names"]
    print(f"\n  {'muscle':<24}  {'MSE(marked)':>12}  {'MSE(markerless)':>15}  {'Δ':>8}")
    print(f"  {'-'*24}  {'-'*12}  {'-'*15}  {'-'*8}")
    for i, name in enumerate(names):
        delta = m2["per_muscle_mse"][i] - m1["per_muscle_mse"][i]
        print(
            f"  {name:<24}  {m1['per_muscle_mse'][i]:12.5f}  "
            f"{m2['per_muscle_mse'][i]:15.5f}  {delta:+8.5f}"
        )
    print(
        f"\n  Aggregate markered MSE   : {m1['aggregate_mse']:.5f}"
        f"   r={m1['aggregate_r']:.3f}"
    )
    print(
        f"  Aggregate markerless MSE : {m2['aggregate_mse']:.5f}"
        f"   r={m2['aggregate_r']:.3f}"
    )


def discriminator_score(
    disc:    Discriminator,
    policy:  BCPolicy,
    expert:  ExpertData,
    device:  str = "cpu",
    n_samples: int = 200,
) -> float:
    """
    Average D(s, π(s)) over expert states.
    Close to 1.0 = discriminator thinks the policy looks like the expert.
    """
    disc.eval(); policy.eval()
    idx     = np.random.choice(expert.T, n_samples, replace=False)
    states  = torch.from_numpy(expert.states[idx]).to(device)

    with torch.no_grad():
        pred_a  = policy(states)
        logits  = disc(states, pred_a)
        scores  = torch.sigmoid(logits).squeeze(-1).cpu().numpy()

    mean_score = scores.mean()
    print(f"\n── Discriminator score ───────────────────────────────────────")
    print(f"  Mean D(s, π(s)) = {mean_score:.4f}  "
          f"(1.0 = indistinguishable from expert)")
    return float(mean_score)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_ckpt", required=True)
    parser.add_argument("--disc_ckpt",   default=None)
    parser.add_argument("--source",      choices=["markered", "markerless", "both"],
                        default="markered")
    parser.add_argument("--device",      default="cpu")
    args = parser.parse_args()

    device  = args.device
    expert  = build_expert("markered")
    S, A    = state_dim_from_expert(expert), expert.A

    policy  = BCPolicy(state_dim=S, action_dim=A)
    policy.load_state_dict(torch.load(args.policy_ckpt, map_location=device))
    policy  = policy.to(device)

    if args.source == "both":
        markerless = build_expert("markerless")
        compare_sources(policy, expert, markerless, device)
    else:
        tgt = build_expert(args.source)
        evaluate_bc(policy, tgt, device)

    if args.disc_ckpt:
        disc = Discriminator(state_dim=S, action_dim=A)
        disc.load_state_dict(torch.load(args.disc_ckpt, map_location=device))
        disc = disc.to(device)
        discriminator_score(disc, policy, expert, device)


if __name__ == "__main__":
    main()
