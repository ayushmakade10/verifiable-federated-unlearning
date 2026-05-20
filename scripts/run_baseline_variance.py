"""
scripts/run_baseline_variance.py — Goal 3: Baseline Variance Measurement
===========================================================================

Loads all 10 gold-standard retraining models and computes pairwise
metrics to characterise natural variation:

  1. Pairwise L2 weight distances
  2. Pairwise logit KL divergences (on test set)
  3. Accuracy distribution

These numbers become the raw material for tolerance calibration
in Phase 3 (95th percentile thresholds per check).

Usage:
    python scripts/run_baseline_variance.py
    python scripts/run_baseline_variance.py --target-client 0
    python scripts/run_baseline_variance.py --output-dir results/

Requires: Goal 2 must be complete (10 gold models exist).
Phase 2, Goal 3 of the dissertation execution roadmap (Section 9.2).
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=import-error,wrong-import-position,no-member
from config.schemas import load_config
from models.resnet import build_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for baseline variance analysis."""
    parser = argparse.ArgumentParser(
        description="Goal 3: Baseline variance measurement across gold models.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to the project config YAML.",
    )
    parser.add_argument(
        "--target-client",
        type=int,
        default=0,
        help="Target client whose gold models to analyse (default: 0).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for variance analysis results. "
             "Defaults to outputs/gold/client_{k}/variance/.",
    )
    return parser.parse_args()


# ── Metric Computation ───────────────────────────────────────────


def compute_l2_weight_distance(
    sd1: dict[str, torch.Tensor],
    sd2: dict[str, torch.Tensor],
) -> float:
    """Compute the L2 (Euclidean) distance between two model state_dicts.

    Concatenates all parameter tensors into a single flat vector and
    computes the Euclidean norm of their difference. Only includes
    learnable parameters (skips BatchNorm running stats and counters).

    Args:
        sd1: First model state_dict.
        sd2: Second model state_dict.

    Returns:
        L2 distance as a float.
    """
    diff_sq_sum = 0.0
    for key in sorted(sd1.keys()):
        # Skip non-parameter buffers (running_mean, running_var, num_batches_tracked).
        if "running_" in key or "num_batches_tracked" in key:
            continue
        diff = sd1[key].float() - sd2[key].float()
        diff_sq_sum += (diff ** 2).sum().item()
    return float(np.sqrt(diff_sq_sum))


@torch.no_grad()
def compute_logit_kl_divergence(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Compute the mean KL divergence between two models' logit distributions.

    For each test sample, computes KL(softmax(logits1) || softmax(logits2))
    and returns the mean across all samples. Uses log-softmax for
    numerical stability.

    This is asymmetric: KL(P||Q) ≠ KL(Q||P). The caller should compute
    both directions for pairwise analysis.

    Args:
        model1: First model (provides P).
        model2: Second model (provides Q).
        dataloader: Test set DataLoader.
        device: Device for inference.

    Returns:
        Mean KL divergence as a float.
    """
    model1.eval()
    model2.eval()

    total_kl = 0.0
    total_samples = 0

    for inputs, _ in dataloader:
        inputs = inputs.to(device)
        logits1 = model1(inputs)
        logits2 = model2(inputs)

        log_p = F.log_softmax(logits1, dim=1)
        log_q = F.log_softmax(logits2, dim=1)

        # KL(P || Q) = sum(P * (log_P - log_Q))
        kl = F.kl_div(log_q, log_p, log_target=True, reduction="sum")
        total_kl += kl.item()
        total_samples += inputs.size(0)

    return total_kl / total_samples if total_samples > 0 else 0.0


@torch.no_grad()
def compute_test_accuracy(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Compute test accuracy for a single model."""
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        _, predicted = model(inputs).max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    return correct / total if total > 0 else 0.0


# ── Main ─────────────────────────────────────────────────────────


def main() -> None:
    """Run Goal 3: pairwise variance analysis across gold-standard models."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("baseline_variance")

    config_path = str(REPO_ROOT / args.config)
    cfg = load_config(config_path)
    target_client = args.target_client
    num_trials = cfg.gold_standard.num_trials
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gold_base = (
        Path(cfg.checkpoint.output_dir) / "gold" / f"client_{target_client}"
    )
    output_dir = Path(args.output_dir) if args.output_dir else gold_base / "variance"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load all gold models ─────────────────────────────────────
    logger.info("Loading %d gold models for client %d...", num_trials, target_client)
    models = []
    state_dicts = []

    for trial in range(num_trials):
        trial_dir = gold_base / f"trial_{trial:02d}"
        model_path = trial_dir / "final_model.pt"
        if not model_path.exists():
            logger.error("Missing model: %s", model_path)
            sys.exit(1)

        sd = torch.load(model_path, weights_only=True)
        state_dicts.append(sd)

        model = build_model(num_classes=cfg.model.num_classes)
        model.load_state_dict(sd)
        model = model.to(device)
        models.append(model)
        logger.info("  Loaded trial %d", trial)

    # ── Test DataLoader ──────────────────────────────────────────
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=cfg.data.data_root,
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=0,
    )

    # ── 1. Per-model accuracy ────────────────────────────────────
    logger.info("Computing per-model accuracies...")
    accuracies = []
    for trial, model in enumerate(models):
        acc = compute_test_accuracy(model, test_loader, device)
        accuracies.append(acc)
        logger.info("  Trial %d: accuracy = %.4f", trial, acc)

    # ── 2. Pairwise L2 weight distances ──────────────────────────
    logger.info("Computing pairwise L2 weight distances...")
    pairs = list(itertools.combinations(range(num_trials), 2))
    l2_distances = []

    for i, j in pairs:
        dist = compute_l2_weight_distance(state_dicts[i], state_dicts[j])
        l2_distances.append({"trial_i": i, "trial_j": j, "l2_distance": dist})

    l2_values = [d["l2_distance"] for d in l2_distances]
    logger.info(
        "  L2 distances: min=%.4f, max=%.4f, mean=%.4f, median=%.4f",
        min(l2_values), max(l2_values),
        np.mean(l2_values), np.median(l2_values),
    )

    # ── 3. Pairwise logit KL divergences ─────────────────────────
    logger.info("Computing pairwise logit KL divergences...")
    kl_divergences = []

    for i, j in pairs:
        # Compute both directions.
        kl_ij = compute_logit_kl_divergence(
            models[i], models[j], test_loader, device,
        )
        kl_ji = compute_logit_kl_divergence(
            models[j], models[i], test_loader, device,
        )
        # Symmetric KL = (KL(P||Q) + KL(Q||P)) / 2.
        symmetric_kl = (kl_ij + kl_ji) / 2.0
        kl_divergences.append({
            "trial_i": i,
            "trial_j": j,
            "kl_ij": kl_ij,
            "kl_ji": kl_ji,
            "symmetric_kl": symmetric_kl,
        })

    sym_kl_values = [d["symmetric_kl"] for d in kl_divergences]
    logger.info(
        "  Symmetric KL: min=%.6f, max=%.6f, mean=%.6f, median=%.6f",
        min(sym_kl_values), max(sym_kl_values),
        np.mean(sym_kl_values), np.median(sym_kl_values),
    )

    # ── 4. Percentile analysis ───────────────────────────────────
    l2_p95 = float(np.percentile(l2_values, 95))
    kl_p95 = float(np.percentile(sym_kl_values, 95))

    # ── Results ──────────────────────────────────────────────────
    results = {
        "target_client": target_client,
        "num_trials": num_trials,
        "num_pairs": len(pairs),
        "accuracy": {
            "per_trial": accuracies,
            "mean": float(np.mean(accuracies)),
            "std": float(np.std(accuracies)),
            "min": float(min(accuracies)),
            "max": float(max(accuracies)),
            "range": float(max(accuracies) - min(accuracies)),
        },
        "l2_weight_distance": {
            "pairwise": l2_distances,
            "mean": float(np.mean(l2_values)),
            "std": float(np.std(l2_values)),
            "median": float(np.median(l2_values)),
            "min": float(min(l2_values)),
            "max": float(max(l2_values)),
            "p95": l2_p95,
        },
        "logit_kl_divergence": {
            "pairwise": kl_divergences,
            "mean": float(np.mean(sym_kl_values)),
            "std": float(np.std(sym_kl_values)),
            "median": float(np.median(sym_kl_values)),
            "min": float(min(sym_kl_values)),
            "max": float(max(sym_kl_values)),
            "p95": kl_p95,
        },
    }

    results_path = output_dir / "variance_analysis.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # ── Print summary ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"BASELINE VARIANCE ANALYSIS — Target Client {target_client}")
    print("=" * 70)

    print(f"\n  Accuracy Distribution ({num_trials} trials):")
    print(f"    Mean:  {np.mean(accuracies):.4f}")
    print(f"    Std:   {np.std(accuracies):.4f}")
    print(f"    Range: [{min(accuracies):.4f}, {max(accuracies):.4f}] "
          f"(spread: {max(accuracies) - min(accuracies):.4f})")

    print(f"\n  L2 Weight Distance ({len(pairs)} pairs):")
    print(f"    Mean:   {np.mean(l2_values):.4f}")
    print(f"    Median: {np.median(l2_values):.4f}")
    print(f"    95th %%: {l2_p95:.4f}")
    print(f"    Range:  [{min(l2_values):.4f}, {max(l2_values):.4f}]")

    print(f"\n  Symmetric KL Divergence ({len(pairs)} pairs):")
    print(f"    Mean:   {np.mean(sym_kl_values):.6f}")
    print(f"    Median: {np.median(sym_kl_values):.6f}")
    print(f"    95th %%: {kl_p95:.6f}")
    print(f"    Range:  [{min(sym_kl_values):.6f}, {max(sym_kl_values):.6f}]")

    # Sanity checks.
    print("\n  Sanity Checks:")
    acc_range = max(accuracies) - min(accuracies)
    if acc_range <= 0.02:
        print(f"    ✓ Accuracy spread {acc_range:.4f} ≤ 2% — well-behaved")
    elif acc_range <= 0.05:
        print(f"    ⚠ Accuracy spread {acc_range:.4f} in 2-5% range — monitor")
    else:
        print(f"    ✗ Accuracy spread {acc_range:.4f} > 5% — investigate")

    l2_cv = np.std(l2_values) / np.mean(l2_values) if np.mean(l2_values) > 0 else 0
    if l2_cv < 0.5:
        print(f"    ✓ L2 distance CV {l2_cv:.2f} < 0.5 — roughly bounded")
    else:
        print(f"    ⚠ L2 distance CV {l2_cv:.2f} ≥ 0.5 — high variance")

    print(f"\n  Results saved: {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
