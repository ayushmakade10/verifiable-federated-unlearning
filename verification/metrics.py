"""
verification/metrics.py — Shared Distance & Divergence Functions
==================================================================

Canonical implementations of every distance metric used by the
verification checks. Each check calls these rather than reimplementing
its own distance computation.

Metrics provided:
  - l2_weight_distance: Euclidean distance between two state_dicts
  - directional_cosine: cosine similarity between deviation vectors
  - symmetric_kl_divergence: symmetric KL on logit distributions
  - accuracy_gap: absolute accuracy difference between two models
  - checkpoint_velocity: consecutive-checkpoint L2 distances
  - flatten_learnable_params: helper to flatten state_dict to 1-D vector

Specification references: Sections 4.7, 4.8 of the master document.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F


# ── Weight-Space Metrics ─────────────────────────────────────────


def flatten_learnable_params(
    state_dict: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Flatten all learnable parameters into a single 1-D vector.

    Skips BatchNorm running stats (running_mean, running_var) and
    batch counters (num_batches_tracked).

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


def l2_weight_distance(
    sd1: Dict[str, torch.Tensor],
    sd2: Dict[str, torch.Tensor],
) -> float:
    """Compute the L2 (Euclidean) distance between two model state_dicts.

    Skips BatchNorm running stats and counters.

    Args:
        sd1: First model state_dict.
        sd2: Second model state_dict.

    Returns:
        L2 distance as a float.
    """
    diff_sq_sum = 0.0
    for key in sorted(sd1.keys()):
        if "running_" in key or "num_batches_tracked" in key:
            continue
        diff = sd1[key].float() - sd2[key].float()
        diff_sq_sum += (diff ** 2).sum().item()
    return float(np.sqrt(diff_sq_sum))


def directional_cosine(
    provider_sd: Dict[str, torch.Tensor],
    gold_sd: Dict[str, torch.Tensor],
    original_sd: Dict[str, torch.Tensor],
) -> float:
    """Compute directional consistency between provider and original model.

    Measures whether the provider model deviates from a gold model in the
    same direction as the original model does. Used for Check 3 after L2
    was found not to discriminate.

    For the (provider, gold_i) pair:
        delta_provider = provider - gold_i
        delta_original = original - gold_i
        metric = cosine_similarity(delta_provider, delta_original)

    Interpretation:
        ~1.0 → provider deviates in same direction as original (no-unlearning)
        ~0.0 → provider deviates in unrelated direction (legitimate retraining)
        <0   → provider deviates in opposite direction

    Args:
        provider_sd: Provider model state_dict.
        gold_sd: Gold reference model state_dict.
        original_sd: Original (pre-unlearning) model state_dict.

    Returns:
        Cosine similarity as a float in [-1, 1].
    """
    provider_vec = flatten_learnable_params(provider_sd)
    # Co-locate all vectors on the provider's device before the vector math.
    # This is a no-op whenever the three state_dicts already share a device
    # (the ResNet path, and the gold-vs-gold calibration where every input is
    # co-located) — `.to(same_device)` returns the same tensor, so existing
    # results stay byte-identical. It only changes the previously-broken case
    # where artifacts were saved on different devices (e.g. a CPU provider
    # bundle vs CUDA gold/original for the ConvNet/QuickDrop path).
    gold_vec = flatten_learnable_params(gold_sd).to(provider_vec.device)
    original_vec = flatten_learnable_params(original_sd).to(provider_vec.device)

    delta_provider = provider_vec - gold_vec
    delta_original = original_vec - gold_vec

    cos = F.cosine_similarity(  # pylint: disable=not-callable
        delta_provider.unsqueeze(0),
        delta_original.unsqueeze(0),
    )
    return cos.item()


# ── Logit-Space Metrics ──────────────────────────────────────────


@torch.no_grad()
def symmetric_kl_divergence(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Compute symmetric KL divergence between two models' logit distributions.

    For each sample, computes (KL(P||Q) + KL(Q||P)) / 2 where P and Q
    are the softmax distributions from model1 and model2 respectively.

    Args:
        model1: First model.
        model2: Second model.
        dataloader: Evaluation DataLoader.
        device: Device for inference.

    Returns:
        Mean symmetric KL divergence across all samples.
    """
    model1.eval()
    model2.eval()

    total_kl_pq = 0.0
    total_kl_qp = 0.0
    total_samples = 0

    for inputs, _ in dataloader:
        inputs = inputs.to(device)
        logits1 = model1(inputs)
        logits2 = model2(inputs)

        log_p = F.log_softmax(logits1, dim=1)
        log_q = F.log_softmax(logits2, dim=1)

        kl_pq = F.kl_div(log_q, log_p, log_target=True, reduction="sum")
        kl_qp = F.kl_div(log_p, log_q, log_target=True, reduction="sum")

        total_kl_pq += kl_pq.item()
        total_kl_qp += kl_qp.item()
        total_samples += inputs.size(0)

    if total_samples == 0:
        return 0.0
    return (total_kl_pq + total_kl_qp) / (2.0 * total_samples)


@torch.no_grad()
def compute_accuracy(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Compute classification accuracy for a single model.

    Args:
        model: The model to evaluate.
        dataloader: Evaluation DataLoader.
        device: Device for inference.

    Returns:
        Accuracy as a float in [0, 1].
    """
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        _, predicted = model(inputs).max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    return correct / total if total > 0 else 0.0


def accuracy_gap(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Compute the absolute accuracy difference between two models.

    Args:
        model1: First model.
        model2: Second model.
        dataloader: Evaluation DataLoader.
        device: Device for inference.

    Returns:
        Absolute accuracy difference as a float.
    """
    acc1 = compute_accuracy(model1, dataloader, device)
    acc2 = compute_accuracy(model2, dataloader, device)
    return abs(acc1 - acc2)


# ── Checkpoint Trajectory Metrics ────────────────────────────────


def checkpoint_velocity(
    checkpoint_state_dicts: List[Dict[str, torch.Tensor]],
) -> List[float]:
    """Compute the velocity curve from an ordered sequence of checkpoints.

    The velocity curve is the series of consecutive L2 distances between
    adjacent checkpoints. For N checkpoints, this produces N-1 values.

    A legitimate training run shows a characteristic decelerating curve.
    Manipulation produces anomalous patterns: sudden spikes (partial
    retraining) or flat lines (fine-tuning masquerade).

    Args:
        checkpoint_state_dicts: Ordered list of state_dicts from
            consecutive checkpoints (e.g. rounds 10, 20, ..., 200).

    Returns:
        List of L2 distances between consecutive checkpoints.

    Raises:
        ValueError: If fewer than 2 checkpoints are provided.
    """
    if len(checkpoint_state_dicts) < 2:
        raise ValueError(
            f"Need at least 2 checkpoints for velocity, got "
            f"{len(checkpoint_state_dicts)}"
        )

    velocities = []
    for idx in range(len(checkpoint_state_dicts) - 1):
        dist = l2_weight_distance(
            checkpoint_state_dicts[idx],
            checkpoint_state_dicts[idx + 1],
        )
        velocities.append(dist)
    return velocities


def velocity_curve_distance(
    curve1: List[float],
    curve2: List[float],
) -> float:
    """Compute the RMS distance between two velocity curves.

    Both curves must have the same length (same number of checkpoint
    intervals).

    Args:
        curve1: First velocity curve.
        curve2: Second velocity curve.

    Returns:
        Root-mean-square of point-by-point differences.

    Raises:
        ValueError: If the curves have different lengths.
    """
    if len(curve1) != len(curve2):
        raise ValueError(
            f"Velocity curves must have equal length: "
            f"{len(curve1)} vs {len(curve2)}"
        )
    arr1 = np.array(curve1)
    arr2 = np.array(curve2)
    return float(np.sqrt(np.mean((arr1 - arr2) ** 2)))
