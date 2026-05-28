"""
verification/checks/ — Verification Check Interface & Results
================================================================

Defines the CheckResult dataclass returned by every verification check,
and the ASSURANCE_CHECKS mapping that determines which checks are
active at each assurance level.

Every check module in this package exposes a single public function:

    def run_check(...) -> CheckResult

The specific arguments vary by check (some need model inference, some
operate on files only), but the return type is always CheckResult.

Specification references: Sections 4.1, 4.2, 4.8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class CheckResult:
    """Outcome of a single verification check.

    Attributes:
        check_name: Identifier (e.g. "logit_divergence").
        passed: True if the provider passes this check.
        measured_value: Provider's median distance to the gold group
            (or equivalent scalar for non-quantitative checks).
        threshold: The calibrated p95 threshold (0.0 for binary checks
            like evidence consistency).
        deviation_ratio: measured_value / threshold. Values < 1.0 mean
            the provider is within tolerance. Set to 0.0 for binary checks.
        individual_distances: All provider-to-gold distances (empty for
            binary checks like Check 5).
        gold_reference_medians: Each gold model's median-to-group value
            (empty for binary checks).
        metadata: Check-specific extras (e.g. sub-results for probe vs
            full test set, per-tier details for Check 5).
    """
    check_name: str
    passed: bool
    measured_value: float
    threshold: float
    deviation_ratio: float
    individual_distances: List[float] = field(default_factory=list)
    gold_reference_medians: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# Assurance level → active check names (Section 4.2).
ASSURANCE_CHECKS: Dict[str, List[str]] = {
    "basic": [
        "logit_divergence",
        "accuracy_parity",
        "evidence_consistency",
    ],
    "strong": [
        "logit_divergence",
        "accuracy_parity",
        "weight_distance",
        "checkpoint_trajectory",
        "evidence_consistency",
    ],
    "high": [
        "logit_divergence",
        "accuracy_parity",
        "weight_distance",
        "checkpoint_trajectory",
        "evidence_consistency",
    ],
}
