"""
verification/comparison.py — "Does It Belong to the Group?" Comparison
========================================================================

Implements the gold model comparison method from Section 4.7 of the
master document. This is the single place where the pass/fail logic
for quantitative checks lives. Every check computes its own distance
metric, produces raw distance vectors, and then calls this module
for the pass/fail decision.

Method:
  Step 1: For each gold model i, compute its median distance to the
          other 9 gold models → 10 reference values.
  Step 2: Threshold = 95th percentile of these 10 reference values.
  Step 3: Compute the provider model's median distance to all 10 gold
          models. Pass if provider_median ≤ threshold.

The same mechanics apply regardless of the distance metric (KL, L2,
accuracy gap, directional cosine, velocity curve RMS).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class ComparisonResult:
    """Outcome of the group-membership comparison for one check.

    Attributes:
        provider_median: Provider model's median distance to all gold models.
        threshold: 95th percentile of gold reference medians.
        passed: True if provider_median ≤ threshold.
        deviation_ratio: provider_median / threshold (< 1.0 means pass).
        provider_distances: All individual provider-to-gold distances.
        gold_reference_medians: Each gold model's median-to-group value.
    """
    provider_median: float
    threshold: float
    passed: bool
    deviation_ratio: float
    provider_distances: List[float]
    gold_reference_medians: List[float] = field(repr=False)


def compute_gold_reference(
    gold_pairwise: List[List[float]],
    percentile: float = 95.0,
) -> tuple[List[float], float]:
    """Compute gold reference medians and threshold from pairwise distances.

    For each gold model i, computes the median of its distances to the
    other gold models. Returns all reference medians and the threshold
    at the given percentile.

    Args:
        gold_pairwise: N×N matrix where gold_pairwise[i][j] is the
            distance between gold model i and gold model j. Diagonal
            entries (i==j) should be 0.0 and are excluded.
        percentile: Percentile for the threshold (default 95.0).

    Returns:
        Tuple of (reference_medians, threshold).
        reference_medians: List of N median-to-group values.
        threshold: The percentile of these N values.
    """
    num_gold = len(gold_pairwise)
    reference_medians = []

    for i in range(num_gold):
        distances_to_others = [
            gold_pairwise[i][j]
            for j in range(num_gold)
            if j != i
        ]
        reference_medians.append(float(np.median(distances_to_others)))

    threshold = float(np.percentile(reference_medians, percentile))
    return reference_medians, threshold


def evaluate_provider(
    provider_distances: List[float],
    gold_reference_medians: List[float],
    threshold: float,
) -> ComparisonResult:
    """Evaluate whether the provider model belongs to the gold group.

    Args:
        provider_distances: Distance from provider to each gold model
            (N values, one per gold model).
        gold_reference_medians: Each gold model's median-to-group value
            (from compute_gold_reference).
        threshold: The calibrated threshold (p95 of gold reference medians).

    Returns:
        ComparisonResult with pass/fail decision and diagnostic values.
    """
    provider_median = float(np.median(provider_distances))
    passed = provider_median <= threshold
    deviation_ratio = provider_median / threshold if threshold > 0 else float("inf")

    return ComparisonResult(
        provider_median=provider_median,
        threshold=threshold,
        passed=passed,
        deviation_ratio=deviation_ratio,
        provider_distances=list(provider_distances),
        gold_reference_medians=list(gold_reference_medians),
    )


def compare_against_gold(
    provider_distances: List[float],
    gold_pairwise: List[List[float]],
    percentile: float = 95.0,
) -> ComparisonResult:
    """Full comparison: compute reference, threshold, and evaluate provider.

    Convenience function that combines compute_gold_reference and
    evaluate_provider into a single call.

    Args:
        provider_distances: Distance from provider to each gold model.
        gold_pairwise: N×N pairwise distance matrix for gold models.
        percentile: Percentile for the threshold (default 95.0).

    Returns:
        ComparisonResult with pass/fail decision.
    """
    reference_medians, threshold = compute_gold_reference(
        gold_pairwise, percentile,
    )
    return evaluate_provider(
        provider_distances, reference_medians, threshold,
    )
