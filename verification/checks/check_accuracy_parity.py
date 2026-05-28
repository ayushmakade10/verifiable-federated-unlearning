"""
verification/checks/check_accuracy_parity.py — Check 2: Accuracy Parity
==========================================================================

Measures whether the provider model's accuracy is consistent with a
correctly-unlearned model by comparing absolute accuracy gaps against
gold-standard reference models.

Evaluated on two data sets:
  - Primary: class-weighted probe set
  - Secondary: full test set

Detects: rollback (massive accuracy drop), degraded unlearning,
gross failures that affect model utility.

Specification reference: Sections 4.8 (Check 2), 4.7 (comparison method).
"""

from __future__ import annotations

from typing import List

import torch

from verification.checks import CheckResult
from verification.comparison import compare_against_gold
from verification.metrics import accuracy_gap


def run_check(
    provider_model: torch.nn.Module,
    gold_models: List[torch.nn.Module],
    probe_loader: torch.utils.data.DataLoader,
    full_loader: torch.utils.data.DataLoader,
    device: torch.device,
    gold_pairwise_probe: List[List[float]],
    gold_pairwise_full: List[List[float]],
) -> CheckResult:
    """Run Check 2: Accuracy Parity.

    Args:
        provider_model: The provider's model to verify.
        gold_models: List of gold-standard reference models.
        probe_loader: Class-weighted probe set DataLoader.
        full_loader: Full test set DataLoader.
        device: Device for inference.
        gold_pairwise_probe: Pre-computed gold-vs-gold accuracy gap
            matrix on the probe set (N×N, diagonal = 0).
        gold_pairwise_full: Pre-computed gold-vs-gold accuracy gap
            matrix on the full test set (N×N, diagonal = 0).

    Returns:
        CheckResult with pass/fail based on the probe set comparison.
    """
    num_gold = len(gold_models)

    # ── Provider-to-gold accuracy gaps on probe set ──────────────
    provider_probe_distances = []
    for gold_model in gold_models:
        gap = accuracy_gap(
            provider_model, gold_model, probe_loader, device,
        )
        provider_probe_distances.append(gap)

    probe_result = compare_against_gold(
        provider_probe_distances, gold_pairwise_probe,
    )

    # ── Provider-to-gold accuracy gaps on full test set ──────────
    provider_full_distances = []
    for gold_model in gold_models:
        gap = accuracy_gap(
            provider_model, gold_model, full_loader, device,
        )
        provider_full_distances.append(gap)

    full_result = compare_against_gold(
        provider_full_distances, gold_pairwise_full,
    )

    # ── Assemble result (probe set is primary) ───────────────────
    return CheckResult(
        check_name="accuracy_parity",
        passed=probe_result.passed,
        measured_value=probe_result.provider_median,
        threshold=probe_result.threshold,
        deviation_ratio=probe_result.deviation_ratio,
        individual_distances=probe_result.provider_distances,
        gold_reference_medians=probe_result.gold_reference_medians,
        metadata={
            "evaluation_sets": {
                "probe": {
                    "passed": probe_result.passed,
                    "provider_median": probe_result.provider_median,
                    "threshold": probe_result.threshold,
                    "deviation_ratio": probe_result.deviation_ratio,
                    "distances": probe_result.provider_distances,
                },
                "full_test": {
                    "passed": full_result.passed,
                    "provider_median": full_result.provider_median,
                    "threshold": full_result.threshold,
                    "deviation_ratio": full_result.deviation_ratio,
                    "distances": full_result.provider_distances,
                },
            },
            "num_gold_models": num_gold,
        },
    )
