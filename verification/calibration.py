"""
verification/calibration.py — Threshold Calibration from Gold Models
=======================================================================

Computes pairwise distance matrices across the gold-standard reference
models for each verification check, then derives the p95 threshold
via the group-membership method (Section 4.7).

Calibration runs once and produces a frozen CalibrationBundle. This
bundle is then consumed by each check during provider evaluation.

The calibration must be frozen BEFORE any provider model is evaluated
(Section 4.4). The same thresholds apply to all providers and all
assurance levels — profiles differ by check subset, not by threshold.

Specification references: Sections 4.4, 4.7, 9.2 (Goal 4).
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import torch

from verification.comparison import compute_gold_reference
from verification.metrics import (
    checkpoint_velocity,
    compute_accuracy,
    directional_cosine,
    symmetric_kl_divergence,
    velocity_curve_distance,
)

logger = logging.getLogger(__name__)


@dataclass
class CheckCalibration:
    """Calibration data for a single check.

    Attributes:
        check_name: Identifier matching the check module.
        pairwise_matrix: N×N distance matrix (diagonal = 0).
        reference_medians: N median-to-group values.
        threshold: p95 of reference_medians.
    """
    check_name: str
    pairwise_matrix: List[List[float]]
    reference_medians: List[float]
    threshold: float


@dataclass
class CalibrationBundle:
    """Complete calibration data for all checks.

    Attributes:
        checks: Dict mapping check name to its calibration data.
        num_gold_models: Number of gold models used.
        target_client: Target client ID.
    """
    checks: Dict[str, CheckCalibration] = field(default_factory=dict)
    num_gold_models: int = 0
    target_client: int = 0

    def save(self, path: Path) -> None:
        """Serialise the calibration bundle to JSON."""
        data = {
            "num_gold_models": self.num_gold_models,
            "target_client": self.target_client,
            "checks": {},
        }
        for name, cal in self.checks.items():
            data["checks"][name] = {
                "pairwise_matrix": cal.pairwise_matrix,
                "reference_medians": cal.reference_medians,
                "threshold": cal.threshold,
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Calibration saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "CalibrationBundle":
        """Load a calibration bundle from JSON."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        bundle = cls(
            num_gold_models=data["num_gold_models"],
            target_client=data["target_client"],
        )
        for name, cal_data in data["checks"].items():
            bundle.checks[name] = CheckCalibration(
                check_name=name,
                pairwise_matrix=cal_data["pairwise_matrix"],
                reference_medians=cal_data["reference_medians"],
                threshold=cal_data["threshold"],
            )
        return bundle


# ── Calibration Functions ────────────────────────────────────────


def calibrate_logit_divergence(
    gold_models: List[torch.nn.Module],
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    label: str = "logit_divergence",
) -> CheckCalibration:
    """Calibrate Check 1 (or any KL-based check) on a given evaluation set.

    Args:
        gold_models: List of gold-standard models.
        dataloader: Evaluation DataLoader (probe or full test).
        device: Device for inference.
        label: Check name label for the calibration.

    Returns:
        CheckCalibration with pairwise symmetric KL matrix and threshold.
    """
    num_gold = len(gold_models)
    matrix = _init_matrix(num_gold)

    pairs = list(itertools.combinations(range(num_gold), 2))
    for i, j in pairs:
        kl = symmetric_kl_divergence(
            gold_models[i], gold_models[j], dataloader, device,
        )
        matrix[i][j] = kl
        matrix[j][i] = kl
        logger.debug("  KL(%d, %d) = %.6f", i, j, kl)

    medians, threshold = compute_gold_reference(matrix)
    return CheckCalibration(
        check_name=label,
        pairwise_matrix=matrix,
        reference_medians=medians,
        threshold=threshold,
    )


def calibrate_accuracy_parity(
    gold_models: List[torch.nn.Module],
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    label: str = "accuracy_parity",
) -> CheckCalibration:
    """Calibrate Check 2 on a given evaluation set.

    Args:
        gold_models: List of gold-standard models.
        dataloader: Evaluation DataLoader.
        device: Device for inference.
        label: Check name label.

    Returns:
        CheckCalibration with pairwise accuracy gap matrix and threshold.
    """
    num_gold = len(gold_models)

    # Pre-compute all accuracies (each model evaluated once).
    accuracies = []
    for idx, model in enumerate(gold_models):
        acc = compute_accuracy(model, dataloader, device)
        accuracies.append(acc)
        logger.debug("  Gold %d accuracy = %.4f", idx, acc)

    # Build pairwise accuracy gap matrix.
    matrix = _init_matrix(num_gold)
    for i, j in itertools.combinations(range(num_gold), 2):
        gap = abs(accuracies[i] - accuracies[j])
        matrix[i][j] = gap
        matrix[j][i] = gap

    medians, threshold = compute_gold_reference(matrix)
    return CheckCalibration(
        check_name=label,
        pairwise_matrix=matrix,
        reference_medians=medians,
        threshold=threshold,
    )


def calibrate_weight_distance(
    gold_state_dicts: List[Dict[str, torch.Tensor]],
    original_sd: Dict[str, torch.Tensor],
) -> CheckCalibration:
    """Calibrate Check 3 using directional consistency metric.

    For each pair (gold_i, gold_j):
        metric = cosine(gold_j - gold_i, original - gold_i)

    Note: the matrix is NOT symmetric — cosine(gold_j - gold_i,
    original - gold_i) ≠ cosine(gold_i - gold_j, original - gold_j).
    We store the full non-symmetric matrix.

    Args:
        gold_state_dicts: List of gold-standard model state_dicts.
        original_sd: Original (pre-unlearning) model state_dict.

    Returns:
        CheckCalibration with pairwise directional cosine matrix.
    """
    num_gold = len(gold_state_dicts)
    matrix = _init_matrix(num_gold)

    for i in range(num_gold):
        for j in range(num_gold):
            if i == j:
                continue
            # gold_j as "pseudo-provider", measured against gold_i.
            cos = directional_cosine(
                gold_state_dicts[j], gold_state_dicts[i], original_sd,
            )
            matrix[j][i] = cos
            logger.debug("  cosine(gold_%d, ref=gold_%d) = %.6f", j, i, cos)

    medians, threshold = compute_gold_reference(matrix)
    return CheckCalibration(
        check_name="weight_distance",
        pairwise_matrix=matrix,
        reference_medians=medians,
        threshold=threshold,
    )


def calibrate_trajectory(
    gold_checkpoint_sets: List[List[Dict[str, torch.Tensor]]],
) -> CheckCalibration:
    """Calibrate Check 4 from gold checkpoint velocity curves.

    Only gold trials with full checkpoints are included (typically
    trials 0–2, giving n=3). The pairwise matrix is 3×3.

    Args:
        gold_checkpoint_sets: List of checkpoint sequences. Each is
            an ordered list of state_dicts for one gold trial.

    Returns:
        CheckCalibration with pairwise velocity curve RMS matrix.
    """
    num_traj = len(gold_checkpoint_sets)

    # Compute velocity curves.
    velocities = []
    for idx, ckpts in enumerate(gold_checkpoint_sets):
        vel = checkpoint_velocity(ckpts)
        velocities.append(vel)
        logger.debug("  Gold trajectory %d: %d velocity points", idx, len(vel))

    # Pairwise RMS distances.
    matrix = _init_matrix(num_traj)
    for i, j in itertools.combinations(range(num_traj), 2):
        rms = velocity_curve_distance(velocities[i], velocities[j])
        matrix[i][j] = rms
        matrix[j][i] = rms
        logger.debug("  Velocity RMS(%d, %d) = %.6f", i, j, rms)

    medians, threshold = compute_gold_reference(matrix)
    return CheckCalibration(
        check_name="checkpoint_trajectory",
        pairwise_matrix=matrix,
        reference_medians=medians,
        threshold=threshold,
    )


# ── Helpers ──────────────────────────────────────────────────────


def _init_matrix(size: int) -> List[List[float]]:
    """Create an N×N matrix initialised to 0.0."""
    return [[0.0] * size for _ in range(size)]
