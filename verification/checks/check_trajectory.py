"""
verification/checks/check_trajectory.py — Check 4: Checkpoint Trajectory
===========================================================================

Compares the provider's checkpoint velocity curve against gold-standard
velocity curves to detect training process anomalies.

The velocity curve is the series of consecutive-checkpoint L2 distances
(a 19-point profile for 20 checkpoints). Legitimate training shows a
characteristic decelerating convergence pattern. Manipulation produces
anomalous patterns:
  - Partial retraining: spike at the retraining start point
  - Fine-tuning masquerade: flat pattern (no convergence dynamics)
  - Rollback: trajectory matches original run, not gold

Uses the same median-to-group comparison method as all other checks.
The distance between two velocity curves is the RMS of point-by-point
differences.

Limitation: Only gold trials with full checkpoints (trials 0–2) are
available, giving n=3 for the comparison. This is documented as a
limitation — methodological consistency across all checks is prioritised
over statistical power from a small sample.

Graceful degradation: If the provider doesn't supply checkpoints,
the check reports "insufficient_evidence" rather than forcing a failure.

Specification reference: Sections 4.8 (Check 4), PIN-1.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from verification.checks import CheckResult
from verification.comparison import compare_against_gold
from verification.metrics import checkpoint_velocity, velocity_curve_distance


def run_check(
    provider_checkpoints: List[Dict[str, torch.Tensor]],
    gold_checkpoint_sets: List[List[Dict[str, torch.Tensor]]],
    gold_pairwise: List[List[float]],
) -> CheckResult:
    """Run Check 4: Checkpoint Trajectory Consistency.

    Args:
        provider_checkpoints: Ordered list of provider checkpoint
            state_dicts (e.g. rounds 10, 20, ..., 200).
        gold_checkpoint_sets: List of gold checkpoint sequences.
            Each entry is an ordered list of state_dicts for one
            gold trial. Only trials with full checkpoints are included
            (typically trials 0–2, giving n=3).
        gold_pairwise: Pre-computed gold-vs-gold velocity curve RMS
            distance matrix (n×n, diagonal = 0).

    Returns:
        CheckResult. If provider has insufficient checkpoints,
        returns a result with passed=False and metadata explaining why.
    """
    num_gold_trajectories = len(gold_checkpoint_sets)

    # ── Graceful degradation: insufficient evidence ──────────────
    if len(provider_checkpoints) < 2:
        return CheckResult(
            check_name="checkpoint_trajectory",
            passed=False,
            measured_value=0.0,
            threshold=0.0,
            deviation_ratio=0.0,
            metadata={
                "status": "insufficient_evidence",
                "reason": (
                    f"Provider supplied {len(provider_checkpoints)} "
                    f"checkpoint(s); minimum 2 required for trajectory "
                    f"analysis."
                ),
                "num_gold_trajectories": num_gold_trajectories,
            },
        )

    # ── Compute provider velocity curve ──────────────────────────
    provider_velocity = checkpoint_velocity(provider_checkpoints)

    # Verify curve lengths match.
    gold_velocities = []
    for gold_ckpts in gold_checkpoint_sets:
        gold_vel = checkpoint_velocity(gold_ckpts)
        gold_velocities.append(gold_vel)

    expected_length = len(gold_velocities[0]) if gold_velocities else 0
    if expected_length > 0 and len(provider_velocity) != expected_length:
        return CheckResult(
            check_name="checkpoint_trajectory",
            passed=False,
            measured_value=0.0,
            threshold=0.0,
            deviation_ratio=0.0,
            metadata={
                "status": "length_mismatch",
                "reason": (
                    f"Provider velocity curve has {len(provider_velocity)} "
                    f"points; gold curves have {expected_length}. "
                    f"Checkpoint counts differ."
                ),
                "provider_curve_length": len(provider_velocity),
                "gold_curve_length": expected_length,
            },
        )

    # ── Provider-to-gold velocity curve distances ────────────────
    provider_distances = []
    for gold_vel in gold_velocities:
        rms = velocity_curve_distance(provider_velocity, gold_vel)
        provider_distances.append(rms)

    result = compare_against_gold(provider_distances, gold_pairwise)

    return CheckResult(
        check_name="checkpoint_trajectory",
        passed=result.passed,
        measured_value=result.provider_median,
        threshold=result.threshold,
        deviation_ratio=result.deviation_ratio,
        individual_distances=result.provider_distances,
        gold_reference_medians=result.gold_reference_medians,
        metadata={
            "provider_velocity_curve": provider_velocity,
            "num_gold_trajectories": num_gold_trajectories,
            "limitation": (
                f"Only {num_gold_trajectories} gold trajectories available "
                f"for comparison (trials with full checkpoints). Small "
                f"sample size limits threshold robustness."
            ),
        },
    )
