"""
verification/checks/check_weight_distance.py — Check 3: Weight Distance
==========================================================================

Measures whether the provider model's weight-space position is consistent
with a correctly-unlearned model.

After the L2 diagnostic (PIN-2) showed that raw L2 distance does not
discriminate between the original model and gold models, this check
uses the **directional consistency** metric:

  For each (provider, gold_i) pair:
    delta_provider = provider_weights - gold_i_weights
    delta_original = original_weights - gold_i_weights
    metric = cosine_similarity(delta_provider, delta_original)

Interpretation:
  ~1.0 → provider deviates from gold in the same direction as the
         original model (strong indicator of no-unlearning)
  ~0.0 → provider deviates in an unrelated direction (consistent
         with independent retraining)

The comparison method is identical to other checks: the provider's
median directional cosine must be ≤ the p95 threshold of gold
reference medians. A no-unlearning provider has cosine ≈ 1.0 (well
above any legitimate gold-to-gold value) and fails.

Requires access to the original model's weights, which the auditor
already has as part of the evidence from the original training run.

Detects: no-unlearning, fine-tuning masquerade.
Specification reference: Section 4.8 (Check 3), PIN-2 pivot.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from verification.checks import CheckResult
from verification.comparison import compare_against_gold
from verification.metrics import directional_cosine


def run_check(
    provider_sd: Dict[str, torch.Tensor],
    gold_state_dicts: List[Dict[str, torch.Tensor]],
    original_sd: Dict[str, torch.Tensor],
    gold_pairwise: List[List[float]],
) -> CheckResult:
    """Run Check 3: Weight Distance (directional consistency).

    Args:
        provider_sd: Provider model state_dict.
        gold_state_dicts: List of gold-standard model state_dicts.
        original_sd: Original (pre-unlearning) model state_dict.
        gold_pairwise: Pre-computed gold-vs-gold directional cosine
            matrix (N×N, diagonal = 0). Entry [i][j] is
            cosine(gold_j - gold_i, original - gold_i).

    Returns:
        CheckResult with pass/fail based on directional consistency.
    """
    # Compute provider's directional cosine against each gold model.
    provider_distances = []
    for gold_sd in gold_state_dicts:
        cos = directional_cosine(provider_sd, gold_sd, original_sd)
        provider_distances.append(cos)

    result = compare_against_gold(provider_distances, gold_pairwise)

    return CheckResult(
        check_name="weight_distance",
        passed=result.passed,
        measured_value=result.provider_median,
        threshold=result.threshold,
        deviation_ratio=result.deviation_ratio,
        individual_distances=result.provider_distances,
        gold_reference_medians=result.gold_reference_medians,
        metadata={
            "metric": "directional_cosine",
            "note": (
                "Higher cosine means provider deviates in the same direction "
                "as the original model — indicative of no-unlearning. "
                "Legitimate retraining produces cosine near 0."
            ),
            "pivot_reason": (
                "Raw L2 distance did not discriminate (PIN-2 diagnostic). "
                "Directional consistency measures deviation direction, "
                "not magnitude."
            ),
        },
    )
