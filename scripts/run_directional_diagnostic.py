"""
scripts/run_directional_diagnostic.py — Directional Consistency Validation
============================================================================

Phase 3 diagnostic for the pivoted Check 3 metric.

After L2 failed to discriminate (PIN-2), this diagnostic validates the
**directional consistency** metric:

  For each (provider, gold_i) pair:
    delta_provider = provider_weights - gold_i_weights
    delta_original = original_weights - gold_i_weights
    metric = cosine_similarity(delta_provider, delta_original)

Expected results:
  - No-unlearning (provider = original): cosine = 1.0 (by construction)
  - Legitimate retraining (provider = gold_j): cosine ≈ 0 (random direction)
  - Fine-tuning masquerade: cosine close to 1.0 but not exactly

If gold-to-gold cosines are well below 1.0, the metric discriminates.

Usage:
    python scripts/run_directional_diagnostic.py
    python scripts/run_directional_diagnostic.py --target-client 0

Requires:
  - outputs/run_001/final_model.pt (original training run)
  - outputs/gold/client_0/trial_XX/final_model.pt (10 gold models)
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# pylint: disable=import-error,wrong-import-position,no-member
from config.schemas import load_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the directional diagnostic."""
    parser = argparse.ArgumentParser(
        description="Directional consistency metric validation (Check 3 pivot).",
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="Path to the project config YAML.",
    )
    parser.add_argument(
        "--target-client", type=int, default=0,
        help="Target client whose gold models to use (default: 0).",
    )
    parser.add_argument(
        "--original-run", type=str, default="run_001",
        help="Run ID for the original training (default: run_001).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for diagnostic results.",
    )
    return parser.parse_args()


def flatten_learnable_params(state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten all learnable parameters into a single 1-D vector.

    Skips BatchNorm running stats and counters.

    Args:
        state_dict: Model state_dict.

    Returns:
        1-D float32 tensor of all learnable parameters concatenated.
    """
    parts = []
    for key in sorted(state_dict.keys()):
        if "running_" in key or "num_batches_tracked" in key:
            continue
        parts.append(state_dict[key].float().flatten())
    return torch.cat(parts)


def directional_cosine(
    provider_vec: torch.Tensor,
    gold_i_vec: torch.Tensor,
    original_vec: torch.Tensor,
) -> float:
    """Compute cosine similarity between deviation vectors.

    delta_provider = provider_vec - gold_i_vec
    delta_original = original_vec - gold_i_vec
    returns cosine_similarity(delta_provider, delta_original)

    Args:
        provider_vec: Flattened provider model parameters.
        gold_i_vec: Flattened gold model i parameters.
        original_vec: Flattened original model parameters.

    Returns:
        Cosine similarity as a float in [-1, 1].
    """
    delta_provider = provider_vec - gold_i_vec
    delta_original = original_vec - gold_i_vec
    cos = torch.nn.functional.cosine_similarity(  # pylint: disable=not-callable
        delta_provider.unsqueeze(0),
        delta_original.unsqueeze(0),
    )
    return cos.item()


def main() -> None:
    """Run directional consistency diagnostic."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("directional_diagnostic")

    config_path = str(REPO_ROOT / args.config)
    cfg = load_config(config_path)
    target_client = args.target_client
    num_trials = cfg.gold_standard.num_trials

    output_base = Path(cfg.checkpoint.output_dir)
    original_model_path = output_base / args.original_run / "final_model.pt"
    gold_base = output_base / "gold" / f"client_{target_client}"
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else gold_base / "variance"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load original model ──────────────────────────────────────
    if not original_model_path.exists():
        logger.error("Original model not found: %s", original_model_path)
        sys.exit(1)

    logger.info("Loading original model: %s", original_model_path)
    original_sd = torch.load(original_model_path, weights_only=True)
    original_vec = flatten_learnable_params(original_sd)

    # ── Load gold models ─────────────────────────────────────────
    logger.info("Loading %d gold models...", num_trials)
    gold_vecs = []
    for trial in range(num_trials):
        model_path = gold_base / f"trial_{trial:02d}" / "final_model.pt"
        if not model_path.exists():
            logger.error("Missing gold model: %s", model_path)
            sys.exit(1)
        gold_sd = torch.load(model_path, weights_only=True)
        gold_vecs.append(flatten_learnable_params(gold_sd))
        logger.info("  Loaded trial %d", trial)

    # ── Case 1: Original model as provider ───────────────────────
    # cosine(original - gold_i, original - gold_i) = 1.0 by definition.
    logger.info("Case 1: Original model as provider (expect cosine = 1.0)...")
    original_cosines = []
    for trial in range(num_trials):
        cos = directional_cosine(original_vec, gold_vecs[trial], original_vec)
        original_cosines.append(cos)
        logger.info("  Original → Gold %d: cosine = %.6f", trial, cos)

    # ── Case 2: Each gold model as provider ──────────────────────
    # cosine(gold_j - gold_i, original - gold_i) — expect low values.
    logger.info("Case 2: Gold model j as provider (expect cosine ≈ 0)...")
    gold_cross_cosines = []
    pairs = list(itertools.combinations(range(num_trials), 2))

    for i, j in pairs:
        # gold_j as provider, measured against gold_i reference.
        cos_ji = directional_cosine(gold_vecs[j], gold_vecs[i], original_vec)
        # gold_i as provider, measured against gold_j reference.
        cos_ij = directional_cosine(gold_vecs[i], gold_vecs[j], original_vec)
        avg_cos = (cos_ji + cos_ij) / 2.0
        gold_cross_cosines.append({
            "trial_i": i, "trial_j": j,
            "cos_j_vs_i": cos_ji, "cos_i_vs_j": cos_ij,
            "symmetric": avg_cos,
        })

    symmetric_values = [d["symmetric"] for d in gold_cross_cosines]

    # ── Case 3: Per-gold-model median cosine (comparison method) ─
    # For the group-membership method, each gold model j acts as
    # "pseudo-provider" and computes its cosine against each gold_i
    # (for i ≠ j). Its median is one reference value.
    logger.info("Case 3: Per-gold-model median cosine (group membership)...")
    gold_reference_medians = []
    for j in range(num_trials):
        cosines_for_j = []
        for i in range(num_trials):
            if i == j:
                continue
            cos = directional_cosine(gold_vecs[j], gold_vecs[i], original_vec)
            cosines_for_j.append(cos)
        median_cos = float(np.median(cosines_for_j))
        gold_reference_medians.append(median_cos)
        logger.info("  Gold %d median cosine: %.6f", j, median_cos)

    threshold = float(np.percentile(gold_reference_medians, 95))

    # Original model's median cosine (should be ~1.0, well above threshold).
    original_median = float(np.median(original_cosines))

    # ── Decision ─────────────────────────────────────────────────
    discriminates = (
        original_median > threshold
        and max(gold_reference_medians) < original_median
    )

    # ── Save results ─────────────────────────────────────────────
    results = {
        "diagnostic": "Directional Consistency Metric Validation",
        "target_client": target_client,
        "original_run": args.original_run,
        "case_1_original_as_provider": {
            "per_trial_cosines": original_cosines,
            "mean": float(np.mean(original_cosines)),
            "median": original_median,
            "note": "Expected 1.0 by construction",
        },
        "case_2_gold_cross_cosines": {
            "pairwise": gold_cross_cosines,
            "symmetric_mean": float(np.mean(symmetric_values)),
            "symmetric_std": float(np.std(symmetric_values)),
            "symmetric_median": float(np.median(symmetric_values)),
            "symmetric_min": float(min(symmetric_values)),
            "symmetric_max": float(max(symmetric_values)),
            "symmetric_p95": float(np.percentile(symmetric_values, 95)),
        },
        "case_3_group_membership": {
            "gold_reference_medians": gold_reference_medians,
            "threshold_p95": threshold,
            "original_median_cosine": original_median,
            "original_above_threshold": bool(original_median > threshold),
        },
        "decision": {
            "discriminates": discriminates,
            "separation": original_median - threshold,
            "recommendation": (
                "Directional cosine discriminates — use for Check 3"
                if discriminates
                else "Directional cosine does NOT discriminate — investigate further"
            ),
        },
    }

    results_path = output_dir / "directional_diagnostic.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    _print_report(results, results_path)


def _print_report(results: dict, results_path: Path) -> None:
    """Print a human-readable summary of the directional diagnostic."""
    case1 = results["case_1_original_as_provider"]
    case2 = results["case_2_gold_cross_cosines"]
    case3 = results["case_3_group_membership"]
    decision = results["decision"]

    print("\n" + "=" * 72)
    print("DIRECTIONAL CONSISTENCY METRIC DIAGNOSTIC")
    print("=" * 72)

    print("\n  Case 1: Original model as provider (no-unlearning)")
    print(f"    Median cosine: {case1['median']:.6f} (expect 1.0)")

    print("\n  Case 2: Gold-to-gold cross-cosines (legitimate retraining)")
    print(f"    Symmetric mean:   {case2['symmetric_mean']:.6f}")
    print(f"    Symmetric median: {case2['symmetric_median']:.6f}")
    print(f"    Symmetric p95:    {case2['symmetric_p95']:.6f}")
    print(f"    Range: [{case2['symmetric_min']:.6f}, "
          f"{case2['symmetric_max']:.6f}]")

    print("\n  Case 3: Group membership (comparison method)")
    print("    Gold reference medians:")
    for trial, med in enumerate(case3["gold_reference_medians"]):
        print(f"      Gold {trial}: {med:.6f}")
    print(f"    Threshold (p95): {case3['threshold_p95']:.6f}")
    print(f"    Original median: {case3['original_median_cosine']:.6f}")

    separator = "=" * 50
    print(f"\n  {separator}")
    if decision["discriminates"]:
        print("  ✓ DECISION: DIRECTIONAL COSINE DISCRIMINATES")
        print(f"    Separation: {decision['separation']:.6f}")
        print("    → Use directional consistency for Check 3")
    else:
        print("  ✗ DECISION: DOES NOT DISCRIMINATE")
        print("    → Investigate further alternatives")
    print(f"  {separator}")

    print(f"\n  Results saved: {results_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
