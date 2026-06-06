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

**Round-based alignment:** When the provider's checkpoint sequence
covers a different range than the gold sequence (e.g. partial retraining
from round 50 produces checkpoints at rounds 60–200, not 10–200), the
check aligns sequences by round number instead of array position. It
compares velocity curves only on the overlapping rounds and reports
the coverage percentage and missing rounds.

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

from typing import Dict, List, Optional

import torch

from verification.checks import CheckResult
from verification.comparison import compare_against_gold, evaluate_provider
from verification.metrics import checkpoint_velocity, velocity_curve_distance

# Minimum fraction of gold velocity points that must be covered
# by the provider's checkpoints for a meaningful comparison.
# Below this, the check auto-fails with "insufficient_coverage".
MIN_COVERAGE_FRACTION = 0.25


def run_check(
    provider_checkpoints: List[Dict[str, torch.Tensor]],
    gold_checkpoint_sets: List[List[Dict[str, torch.Tensor]]],
    gold_pairwise: List[List[float]],
    provider_rounds: Optional[List[int]] = None,
    gold_rounds: Optional[List[int]] = None,
    calibrated_threshold: Optional[float] = None,
    calibrated_medians: Optional[List[float]] = None,
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
        provider_rounds: Round numbers corresponding to
            ``provider_checkpoints``. When provided together with
            ``gold_rounds``, enables round-based alignment instead
            of requiring equal-length sequences.
        gold_rounds: Round numbers corresponding to each gold
            checkpoint sequence (all gold trials share the same
            round numbers).
        calibrated_threshold: Pre-computed threshold from
            calibration. Used when round-based alignment produces
            a subset comparison.
        calibrated_medians: Pre-computed gold reference medians
            from calibration.

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

    # ── Determine alignment mode ─────────────────────────────────
    use_alignment = (
        provider_rounds is not None
        and gold_rounds is not None
        and len(provider_rounds) == len(provider_checkpoints)
    )

    if use_alignment:
        return _run_with_alignment(
            provider_checkpoints, gold_checkpoint_sets,
            gold_pairwise,
            provider_rounds, gold_rounds,
            calibrated_threshold, calibrated_medians,
            num_gold_trajectories,
        )

    return _run_legacy(
        provider_checkpoints, gold_checkpoint_sets,
        gold_pairwise, num_gold_trajectories,
    )


def _run_legacy(
    provider_checkpoints: List[Dict[str, torch.Tensor]],
    gold_checkpoint_sets: List[List[Dict[str, torch.Tensor]]],
    gold_pairwise: List[List[float]],
    num_gold_trajectories: int,
) -> CheckResult:
    """Original comparison path: require matching sequence lengths."""
    provider_velocity = checkpoint_velocity(provider_checkpoints)

    gold_velocities = [
        checkpoint_velocity(gold_ckpts)
        for gold_ckpts in gold_checkpoint_sets
    ]

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
                    f"Provider velocity curve has "
                    f"{len(provider_velocity)} points; gold curves "
                    f"have {expected_length}. Checkpoint counts "
                    f"differ. Supply round numbers for alignment."
                ),
                "provider_curve_length": len(provider_velocity),
                "gold_curve_length": expected_length,
            },
        )

    provider_distances = [
        velocity_curve_distance(provider_velocity, gv)
        for gv in gold_velocities
    ]

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
            "alignment": "positional",
            "coverage_pct": 100.0,
        },
    )


def _run_with_alignment(
    provider_checkpoints: List[Dict[str, torch.Tensor]],
    gold_checkpoint_sets: List[List[Dict[str, torch.Tensor]]],
    gold_pairwise: List[List[float]],
    provider_rounds: List[int],
    gold_rounds: List[int],
    calibrated_threshold: Optional[float],
    calibrated_medians: Optional[List[float]],
    num_gold_trajectories: int,
) -> CheckResult:
    """Round-aligned comparison for mismatched checkpoint sequences.

    Finds overlapping rounds, computes velocity curves on the overlap,
    and compares using the pre-calibrated threshold.
    """
    provider_set = set(provider_rounds)
    gold_set = set(gold_rounds)
    overlap_rounds = sorted(provider_set & gold_set)
    missing_rounds = sorted(gold_set - provider_set)

    total_gold_velocity_points = len(gold_rounds) - 1
    overlap_velocity_points = max(0, len(overlap_rounds) - 1)

    coverage = (
        overlap_velocity_points / total_gold_velocity_points
        if total_gold_velocity_points > 0 else 0.0
    )

    # ── Insufficient overlap ─────────────────────────────────────
    if len(overlap_rounds) < 2:
        return CheckResult(
            check_name="checkpoint_trajectory",
            passed=False,
            measured_value=0.0,
            threshold=calibrated_threshold or 0.0,
            deviation_ratio=0.0,
            metadata={
                "status": "insufficient_overlap",
                "reason": (
                    f"Only {len(overlap_rounds)} overlapping round(s) "
                    f"between provider and gold checkpoints; minimum "
                    f"2 required for velocity computation."
                ),
                "provider_rounds": provider_rounds,
                "gold_rounds": gold_rounds,
                "overlap_rounds": overlap_rounds,
                "coverage_pct": coverage * 100,
                "missing_rounds": missing_rounds,
                "num_gold_trajectories": num_gold_trajectories,
            },
        )

    if coverage < MIN_COVERAGE_FRACTION:
        return CheckResult(
            check_name="checkpoint_trajectory",
            passed=False,
            measured_value=0.0,
            threshold=calibrated_threshold or 0.0,
            deviation_ratio=0.0,
            metadata={
                "status": "insufficient_coverage",
                "reason": (
                    f"Provider checkpoints cover {coverage:.1%} of "
                    f"the gold trajectory ({overlap_velocity_points}/"
                    f"{total_gold_velocity_points} velocity points). "
                    f"Minimum {MIN_COVERAGE_FRACTION:.0%} required."
                ),
                "provider_rounds": provider_rounds,
                "gold_rounds": gold_rounds,
                "overlap_rounds": overlap_rounds,
                "coverage_pct": coverage * 100,
                "missing_rounds": missing_rounds,
                "num_gold_trajectories": num_gold_trajectories,
            },
        )

    # ── Build round-to-index maps ────────────────────────────────
    provider_idx = {r: i for i, r in enumerate(provider_rounds)}
    gold_idx = {r: i for i, r in enumerate(gold_rounds)}

    # ── Extract aligned subsequences ─────────────────────────────
    provider_overlap = [
        provider_checkpoints[provider_idx[r]] for r in overlap_rounds
    ]
    provider_velocity = checkpoint_velocity(provider_overlap)

    gold_velocities = []
    for gold_ckpts in gold_checkpoint_sets:
        gold_overlap = [gold_ckpts[gold_idx[r]] for r in overlap_rounds]
        gold_vel = checkpoint_velocity(gold_overlap)
        gold_velocities.append(gold_vel)

    # ── Provider-to-gold distances on the overlap ────────────────
    provider_distances = [
        velocity_curve_distance(provider_velocity, gv)
        for gv in gold_velocities
    ]

    # ── Pass/fail using pre-calibrated threshold ─────────────────
    if calibrated_threshold is not None and calibrated_medians is not None:
        result = evaluate_provider(
            provider_distances, calibrated_medians,
            calibrated_threshold,
        )
    else:
        # Fallback: recompute from gold pairwise (full-curve).
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
            "alignment": "round_based",
            "overlap_rounds": overlap_rounds,
            "coverage_pct": coverage * 100,
            "missing_rounds": missing_rounds,
            "overlap_velocity_points": overlap_velocity_points,
            "total_gold_velocity_points": total_gold_velocity_points,
            "provider_velocity_curve": provider_velocity,
            "num_gold_trajectories": num_gold_trajectories,
        },
    )
