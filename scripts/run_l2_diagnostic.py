"""
scripts/run_l2_diagnostic.py — PIN-2: L2 Discrimination Validation
=====================================================================

Phase 3 diagnostic (must run BEFORE implementing Check 3).

Loads the original model (run_001, trained on all 50 clients) and all
10 gold-standard models (trained without client 0), then computes:

  1. L2 distance from the original model to each gold model (10 values).
  2. Comparison against the gold-vs-gold distribution (from variance_analysis.json).

Decision logic:
  - If original-to-gold distances are clearly outside the gold-to-gold range,
    L2 discriminates — use it for Check 3.
  - If they overlap significantly, L2 cannot distinguish no-unlearning from
    correct unlearning — pivot to an alternative metric.

Also computes layer-group L2 distances (conv-stem, layer1–4, final FC) to
identify which layers contribute most to any discrimination, in case a
layer-specific metric is needed as a fallback.

Usage:
    python scripts/run_l2_diagnostic.py
    python scripts/run_l2_diagnostic.py --target-client 0

Requires:
  - outputs/run_001/final_model.pt (original training run)
  - outputs/gold/client_0/trial_XX/final_model.pt (10 gold models)
  - outputs/gold/client_0/variance/variance_analysis.json (baseline stats)

Phase 3, PIN-2 of the dissertation execution roadmap (Section 9.3).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=import-error,wrong-import-position,no-member
from config.schemas import load_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the L2 diagnostic."""
    parser = argparse.ArgumentParser(
        description="PIN-2: L2 Discrimination Validation (Phase 3 diagnostic).",
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
        help="Target client whose gold models to compare against (default: 0).",
    )
    parser.add_argument(
        "--original-run",
        type=str,
        default="run_001",
        help="Run ID for the original training (default: run_001).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for diagnostic results. "
             "Defaults to outputs/gold/client_{k}/variance/.",
    )
    return parser.parse_args()


# ── Layer-group L2 Analysis ──────────────────────────────────────


# ResNet-18 layer groups for CIFAR-adapted architecture.
LAYER_GROUPS = {
    "conv_stem": ["conv1.", "bn1."],
    "layer1": ["layer1."],
    "layer2": ["layer2."],
    "layer3": ["layer3."],
    "layer4": ["layer4."],
    "fc": ["fc."],
}


def compute_l2_weight_distance(
    sd1: dict[str, torch.Tensor],
    sd2: dict[str, torch.Tensor],
) -> float:
    """Compute the L2 (Euclidean) distance between two model state_dicts.

    Skips BatchNorm running stats and counters (non-learnable buffers).
    Identical to run_baseline_variance.py for consistency.
    """
    diff_sq_sum = 0.0
    for key in sorted(sd1.keys()):
        if "running_" in key or "num_batches_tracked" in key:
            continue
        diff = sd1[key].float() - sd2[key].float()
        diff_sq_sum += (diff ** 2).sum().item()
    return float(np.sqrt(diff_sq_sum))


def compute_layer_group_l2(
    sd1: dict[str, torch.Tensor],
    sd2: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Compute per-layer-group L2 distances.

    Groups parameters by ResNet-18 architectural blocks to identify
    which layers contribute most to overall weight distance.

    Returns:
        Dict mapping group name to L2 distance for that group.
    """
    group_sums: dict[str, float] = {name: 0.0 for name in LAYER_GROUPS}
    unmatched_sum = 0.0

    for key in sorted(sd1.keys()):
        if "running_" in key or "num_batches_tracked" in key:
            continue
        diff = sd1[key].float() - sd2[key].float()
        sq_sum = (diff ** 2).sum().item()

        matched = False
        for group_name, prefixes in LAYER_GROUPS.items():
            if any(key.startswith(p) for p in prefixes):
                group_sums[group_name] += sq_sum
                matched = True
                break
        if not matched:
            unmatched_sum += sq_sum

    result = {name: float(np.sqrt(val)) for name, val in group_sums.items()}
    if unmatched_sum > 0:
        result["other"] = float(np.sqrt(unmatched_sum))
    return result


# ── Main ─────────────────────────────────────────────────────────


def main() -> None:
    """Run PIN-2: L2 discrimination diagnostic."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("l2_diagnostic")

    config_path = str(REPO_ROOT / args.config)
    cfg = load_config(config_path)
    target_client = args.target_client
    num_trials = cfg.gold_standard.num_trials

    # ── Paths ────────────────────────────────────────────────────
    output_base = Path(cfg.checkpoint.output_dir)
    original_model_path = output_base / args.original_run / "final_model.pt"
    gold_base = output_base / "gold" / f"client_{target_client}"
    variance_path = gold_base / "variance" / "variance_analysis.json"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else gold_base / "variance"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Validate paths ───────────────────────────────────────────
    if not original_model_path.exists():
        logger.error("Original model not found: %s", original_model_path)
        sys.exit(1)
    if not variance_path.exists():
        logger.error("Variance analysis not found: %s", variance_path)
        logger.error("Run 'python scripts/run_baseline_variance.py' first.")
        sys.exit(1)

    # ── Load gold-vs-gold baseline stats ─────────────────────────
    with open(variance_path, encoding="utf-8") as f:
        baseline = json.load(f)

    gold_l2_mean = baseline["l2_weight_distance"]["mean"]
    gold_l2_std = baseline["l2_weight_distance"]["std"]
    gold_l2_p95 = baseline["l2_weight_distance"]["p95"]
    gold_l2_min = baseline["l2_weight_distance"]["min"]
    gold_l2_max = baseline["l2_weight_distance"]["max"]

    logger.info(
        "Gold-vs-gold L2 baseline: mean=%.4f, std=%.4f, p95=%.4f, range=[%.4f, %.4f]",
        gold_l2_mean, gold_l2_std, gold_l2_p95, gold_l2_min, gold_l2_max,
    )

    # ── Load original model ──────────────────────────────────────
    logger.info("Loading original model: %s", original_model_path)
    original_sd = torch.load(original_model_path, weights_only=True)

    # ── Load gold models and compute distances ───────────────────
    logger.info("Loading %d gold models and computing L2 distances...", num_trials)
    original_to_gold_distances = []
    layer_group_distances = []

    for trial in range(num_trials):
        trial_dir = gold_base / f"trial_{trial:02d}"
        model_path = trial_dir / "final_model.pt"
        if not model_path.exists():
            logger.error("Missing gold model: %s", model_path)
            sys.exit(1)

        gold_sd = torch.load(model_path, weights_only=True)

        # Overall L2 distance.
        dist = compute_l2_weight_distance(original_sd, gold_sd)
        original_to_gold_distances.append(dist)

        # Per-layer-group L2 distances.
        layer_dists = compute_layer_group_l2(original_sd, gold_sd)
        layer_group_distances.append({"trial": trial, **layer_dists})

        logger.info("  Original → Gold trial %d: L2 = %.4f", trial, dist)

    # ── Statistical comparison ───────────────────────────────────
    orig_to_gold = np.array(original_to_gold_distances)
    orig_mean = float(np.mean(orig_to_gold))
    orig_std = float(np.std(orig_to_gold))
    orig_min = float(np.min(orig_to_gold))
    orig_max = float(np.max(orig_to_gold))
    orig_median = float(np.median(orig_to_gold))

    # How many standard deviations away from the gold-vs-gold mean?
    z_score = (orig_mean - gold_l2_mean) / gold_l2_std if gold_l2_std > 0 else 0.0

    # Does the original-to-gold range overlap with gold-to-gold range?
    overlap_lower = max(orig_min, gold_l2_min)
    overlap_upper = min(orig_max, gold_l2_max)
    ranges_overlap = overlap_lower <= overlap_upper

    # Separation ratio: gap between distributions / spread of distributions.
    if orig_min > gold_l2_max:
        separation_gap = orig_min - gold_l2_max
    elif gold_l2_min > orig_max:
        separation_gap = gold_l2_min - orig_max
    else:
        separation_gap = 0.0  # Overlapping.

    combined_spread = (gold_l2_max - gold_l2_min) + (orig_max - orig_min)
    separation_ratio = (
        separation_gap / (combined_spread / 2)
        if combined_spread > 0
        else 0.0
    )

    # ── Decision ─────────────────────────────────────────────────
    # L2 discriminates if original-to-gold distances are clearly outside
    # the gold-to-gold distribution (e.g. all above p95, or z-score > 3).
    all_above_p95 = bool(np.all(orig_to_gold > gold_l2_p95))
    all_below_min = bool(np.all(orig_to_gold < gold_l2_min))
    clearly_different = all_above_p95 or all_below_min or abs(z_score) > 3.0

    # ── Layer-group analysis ─────────────────────────────────────
    # Average per-group distances across all 10 original-to-gold comparisons.
    group_names = list(LAYER_GROUPS.keys())
    avg_layer_dists = {}
    for group in group_names:
        vals = [ld[group] for ld in layer_group_distances if group in ld]
        if vals:
            avg_layer_dists[group] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

    # ── Save results ─────────────────────────────────────────────
    results = {
        "diagnostic": "PIN-2: L2 Discrimination Validation",
        "original_run": args.original_run,
        "target_client": target_client,
        "num_gold_trials": num_trials,
        "original_to_gold_l2": {
            "per_trial": original_to_gold_distances,
            "mean": orig_mean,
            "std": orig_std,
            "median": orig_median,
            "min": orig_min,
            "max": orig_max,
        },
        "gold_to_gold_l2_baseline": {
            "mean": gold_l2_mean,
            "std": gold_l2_std,
            "p95": gold_l2_p95,
            "min": gold_l2_min,
            "max": gold_l2_max,
        },
        "comparison": {
            "z_score": z_score,
            "ranges_overlap": ranges_overlap,
            "separation_gap": separation_gap,
            "separation_ratio": separation_ratio,
            "all_above_gold_p95": all_above_p95,
            "all_below_gold_min": all_below_min,
        },
        "decision": {
            "l2_discriminates": clearly_different,
            "recommendation": (
                "Use L2 for Check 3"
                if clearly_different
                else "L2 does NOT discriminate — pivot to alternative metric"
            ),
        },
        "layer_group_analysis": {
            "original_to_gold_averages": avg_layer_dists,
            "per_trial_detail": layer_group_distances,
        },
    }

    results_path = output_dir / "l2_diagnostic.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    _print_report(results, results_path)


def _print_report(results: dict, results_path: Path) -> None:
    """Print a human-readable summary of the L2 diagnostic results."""
    orig = results["original_to_gold_l2"]
    gold = results["gold_to_gold_l2_baseline"]
    comp = results["comparison"]
    decision = results["decision"]
    layer_avgs = results["layer_group_analysis"]["original_to_gold_averages"]

    print("\n" + "=" * 72)
    print("PIN-2: L2 DISCRIMINATION DIAGNOSTIC")
    print("=" * 72)

    print(f"\n  Original model: {results['original_run']}")
    print(f"  Target client:  {results['target_client']}")
    print(f"  Gold trials:    {results['num_gold_trials']}")

    print("\n  Original-to-Gold L2 distances:")
    for trial, dist in enumerate(orig["per_trial"]):
        marker = "  ← above gold p95" if dist > gold["p95"] else ""
        print(f"    Trial {trial}: {dist:.4f}{marker}")
    print(f"    Mean:   {orig['mean']:.4f}")
    print(f"    Std:    {orig['std']:.4f}")
    print(f"    Range:  [{orig['min']:.4f}, {orig['max']:.4f}]")

    print("\n  Gold-to-Gold L2 baseline (45 pairs):")
    print(f"    Mean:   {gold['mean']:.4f}")
    print(f"    Std:    {gold['std']:.4f}")
    print(f"    p95:    {gold['p95']:.4f}")
    print(f"    Range:  [{gold['min']:.4f}, {gold['max']:.4f}]")

    print("\n  Statistical comparison:")
    print(f"    Z-score (orig mean vs gold mean): {comp['z_score']:+.2f}")
    print(f"    Ranges overlap:                   {comp['ranges_overlap']}")
    print(f"    Separation gap:                   {comp['separation_gap']:.4f}")
    print(f"    Separation ratio:                 {comp['separation_ratio']:.4f}")
    print(f"    All orig-to-gold > gold p95:      {comp['all_above_gold_p95']}")

    print("\n  Layer-group analysis (original-to-gold averages):")
    for group, stats in layer_avgs.items():
        print(f"    {group:12s}: mean={stats['mean']:.4f}, "
              f"std={stats['std']:.4f}")

    separator = "=" * 50
    print(f"\n  {separator}")
    if decision["l2_discriminates"]:
        print("  ✓ DECISION: L2 DISCRIMINATES")
        print("    → Use L2 weight distance for Check 3")
    else:
        print("  ✗ DECISION: L2 DOES NOT DISCRIMINATE")
        print("    → Original-to-gold distances overlap with gold-to-gold")
        print("    → Pivot options: layer-specific L2, cosine similarity,")
        print("      or final-layer Frobenius norm (see master doc PIN-2)")
    print(f"  {separator}")

    print(f"\n  Results saved: {results_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
